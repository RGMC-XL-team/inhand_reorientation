# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/HaozhiQi/hora/blob/main/hora/algo/deploy/deploy.py
# --------------------------------------------------------


import pdb
import math
import os
import random
import xml.etree.ElementTree as ET
from collections import deque
import rospy, rospkg
from leap_hardware.hardware_controller import LeapHand

import hydra
import matplotlib.pyplot as plt
import numpy as np
import isaacgym
from isaacgym.torch_utils import quat_from_euler_xyz
import torch
from gym import spaces
from omegaconf import DictConfig
from rl_games.algos_torch import model_builder
from rl_games.torch_runner import Runner, _override_sigma, _restore

from leapsim.learning.lfd_model import LfDAgent
from leapsim.utils.reformat import omegaconf_to_dict

from leap_hardware.srv import object_state
from leap_task_B.leap_grasp_cube import LeapGraspCommander
from leapsim.utils.utils import get_finger_indice_real


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class HardwarePlayer:
    def __init__(self, config, extra_args):
        self.config = omegaconf_to_dict(config)
        self.extra_args = extra_args

        self.set_defaults()
        self.action_scale = 1 / 24
        self.actions_num = 16

        # hand setting
        rospack = rospkg.RosPack()
        canonical_pose_path = os.path.join(rospack.get_path("leap_sim"), "leapsim/cache", "leap_canonical_pose.npy")
        self.init_pose = np.load(canonical_pose_path)
        self.get_dof_limits()

        self.debug_viz = self.config["task"]["env"]["enableDebugVis"]
        if self.debug_viz:
            self.setup_plot()

    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()

        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()

        return values[:, self.sim_to_real_indices]

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = self.config["task"]["env"]["sim_to_real_indices"]
        self.real_to_sim_indices = self.config["task"]["env"]["real_to_sim_indices"]

    def get_dof_limits(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        hand_asset_file = self.config["task"]["env"]["asset"]["handAsset"]

        tree = ET.parse(os.path.join(asset_root, hand_asset_file))
        root = tree.getroot()

        self.leap_dof_lower = [0 for _ in range(16)]
        self.leap_dof_upper = [0 for _ in range(16)]

        for child in root.getchildren():
            if child.tag == "joint" and child.attrib["type"] == "revolute":
                joint_idx = int(child.attrib["name"])

                for gchild in child.getchildren():
                    if gchild.tag == "limit":
                        lower = float(gchild.attrib["lower"])
                        upper = float(gchild.attrib["upper"])

                        self.leap_dof_lower[joint_idx] = lower
                        self.leap_dof_upper[joint_idx] = upper

        self.leap_dof_lower = torch.tensor(self.leap_dof_lower).to(self.device)[None, :]
        self.leap_dof_upper = torch.tensor(self.leap_dof_upper).to(self.device)[None, :]

        self.full_leap_dof_lower = self.real_to_sim(self.leap_dof_lower).squeeze()
        self.full_leap_dof_upper = self.real_to_sim(self.leap_dof_upper).squeeze()

        self.leap_dof_lower = self.full_leap_dof_lower[self.free_finger_indices]
        self.leap_dof_upper = self.full_leap_dof_upper[self.free_finger_indices]

        self.leap_dof_lower = torch.tensor([-0.27349405,  1.44280052,  0.8606107 ,  1.07633467, -0.24649895, 0.27324316,  0.19108043,  1.08493021]).to(self.device)
        self.leap_dof_upper = torch.tensor([0.2964683 , 1.98224004, 1.60876273, 1.6441488 , 0.15939786, 1.00356965, 0.86435953, 1.61686065]).to(self.device)
        self.action_scale = 1 / 6

    def plot_callback(self):
        self.fig.canvas.restore_region(self.bg)

        self.ydata.append(self.cur_obs_joint_angles[0, 9].item())
        self.ydata2.append(self.cur_obs_joint_angles[0, 4].item())

        self.ln.set_ydata(list(self.ydata))
        self.ln.set_xdata(range(len(self.ydata)))

        self.ln2.set_ydata(list(self.ydata2))
        self.ln2.set_xdata(range(len(self.ydata2)))

        self.ax.draw_artist(self.ln)
        self.ax.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-1, 1)
        self.ydata = deque(maxlen=100)  # Plot 5 seconds of data
        self.ydata2 = deque(maxlen=100)
        (self.ln,) = self.ax.plot(range(len(self.ydata)), list(self.ydata), animated=True)
        (self.ln2,) = self.ax.plot(range(len(self.ydata2)), list(self.ydata2), animated=True)
        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)

    def set_defaults(self):
        self.device = "cuda:0"

        self.frozen_finger_indices = get_finger_indice_real(self.extra_args["frozen_fingers"])
        self.free_finger_indices = get_finger_indice_real(self.extra_args["free_fingers"])

        if "include_obj_pose" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_obj_pose"] = True

        # deploy options
        self.include_obj_pose = self.extra_args["deploy_options"]["include_obj_pose"]
        self.enable_hardware = self.extra_args["deploy_options"]["enable_hardware"]
        if not self.enable_hardware:
            self.saved_init_pose = np.array(self.extra_args["deploy_options"]["sample_initial_pose"])
            self.debug_mode = self.extra_args["deploy_options"]["debug_mode"]

        self.obj_pos_lower = torch.tensor([
            self.extra_args["obj_pos_range"]["x"][0], \
            self.extra_args["obj_pos_range"]["y"][0], \
            self.extra_args["obj_pos_range"]["z"][0]
        ], dtype=torch.float, device=self.device)

        self.obj_pos_upper = torch.tensor([
            self.extra_args["obj_pos_range"]["x"][1], \
            self.extra_args["obj_pos_range"]["y"][1], \
            self.extra_args["obj_pos_range"]["z"][1]
        ], dtype=torch.float, device=self.device)

        self.player_model_config = self.extra_args["model_config"]

        if self.include_obj_pose:
            self.player_model_config["input_size"]  = 4 * len(self.extra_args["free_fingers"]) + 7
        else:
            self.player_model_config["input_size"] = 4 * len(self.extra_args["free_fingers"])
        self.player_model_config["output_size"] = 4 * len(self.extra_args["free_fingers"])

        self.seq_length = self.extra_args["rnn_seq_length"]
        self.player_model_path = self.extra_args["model_path"]
        self.num_obs = self.player_model_config["input_size"]
        self.num_action = self.player_model_config["output_size"]

    def get_unscaled_obj_poses(self):
        if self.enable_hardware:
            obj_pose = torch.tensor(self.object_state_proxy().pose, dtype=torch.float32, device=self.device)
            obj_pose[0:3] = unscale(obj_pose[0:3], self.obj_pos_lower, self.obj_pos_upper)
        else:
            obj_pose = torch.zeros(7, dtype=torch.float32, device=self.device)
        return obj_pose

    def deploy(self):
        def move_hand_to_pose(leap: LeapHand, start_position: np.ndarray, goal_position: np.ndarray):
            _rate = rospy.Rate(20)
            _iters = 4 * 20
            for i in range(_iters + 1):
                _progress = i / _iters
                _position = (1 - _progress) * start_position + _progress * goal_position
                leap.command_joint_position(_position)
                _rate.sleep()

        # generate object flip target
        flip_pitch_target = -np.pi/2
        self.goal_rot = quat_from_euler_xyz(
            torch.zeros(1), torch.tensor(flip_pitch_target), torch.zeros(1)
        ).to(self.device)

        if self.enable_hardware:
            # try to set up rospy
            rospy.init_node("example")
            leap = LeapHand()
            leap.leap_dof_lower = self.full_leap_dof_lower.cpu().numpy()
            leap.leap_dof_upper = self.full_leap_dof_upper.cpu().numpy()
            leap.sim_to_real_indices = self.sim_to_real_indices
            leap.real_to_sim_indices = self.real_to_sim_indices

            # Wait for connections.
            rospy.wait_for_service("/leap_position")

            pdb.set_trace()

            hz = 20
            self.control_dt = 1 / hz
            ros_rate = rospy.Rate(hz)

            print("command to the initial position")

            # move to initial pose
            current_position = leap.poll_joint_position()[0]
            move_hand_to_pose(leap, start_position=current_position, goal_position=self.init_pose)

            # move to grasp pose
            rospy.wait_for_service("/object_state")
            self.object_state_proxy = rospy.ServiceProxy("/object_state", object_state)
            curr_object_state = self.object_state_proxy().pose
            grasp_commander = LeapGraspCommander()
            grasp_commander.set_cube_transform_from_pos_quat(pos=curr_object_state[0:3], quat=curr_object_state[3:7])
            self.grasp_pose = grasp_commander.solve_hand_grasp_IK()

            current_position = leap.poll_joint_position()[0]
            print(f"Hand will move {self.grasp_pose - current_position} in joint space!")

            pdb.set_trace()
            move_hand_to_pose(leap, start_position=current_position, goal_position=self.grasp_pose)

            print("done, policy deployment will start")
            pdb.set_trace()

        if self.enable_hardware:
            obses, _ = leap.poll_joint_position()
        else:
            obses = self.saved_init_pose.copy()
        prev_target = torch.from_numpy(obses.astype(np.float32)).cuda()[None].clone()
        obses = obses[self.free_finger_indices]

        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 0, self.num_obs)).astype(np.float32)).cuda()

        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]
        cur_obj_pose = self.get_unscaled_obj_poses()[None]
        if self.include_obj_pose:
            one_step_input = torch.cat([cur_obs_buf.clone(), cur_obj_pose.clone()], dim=-1).unsqueeze(0)
        else:
            one_step_input = cur_obs_buf.clone().unsqueeze(0)

        # if self.config["task"]["env"]["include_history"]:
        #     num_append_iters = 3
        # else:
        #     num_append_iters = 1
        num_append_iters = 3

        for _ in range(num_append_iters):
            obs_buf = torch.cat([obs_buf, one_step_input.clone()], dim=1)

            # if self.config["task"]["env"]["include_targets"]:
            #     obs_buf = torch.cat([obs_buf, prev_target.clone()], dim=-1)

            # if self.config["task"]["env"]["include_obj_target"]:
            #     obs_buf = torch.cat([obs_buf, self.goal_rot.clone()], dim=-1)

            # if "phase_period" in self.config["task"]["env"]:
            #     phase = torch.tensor([[0.0, 1.0]], device=self.device)
            #     obs_buf = torch.cat([obs_buf, phase], dim=-1)

        obs_buf = obs_buf.float()

        counter = 0

        # if "debug" in self.config["task"]["env"]:
        #     self.obs_list = []
        #     self.target_list = []

        #     if "record" in self.config["task"]["env"]["debug"]:
        #         self.record_duration = int(self.config["task"]["env"]["debug"]["record"]["duration"] / self.control_dt)

        #     if "actions_file" in self.config["task"]["env"]["debug"]:
        #         self.actions_list = torch.from_numpy(
        #             np.load(self.config["task"]["env"]["debug"]["actions_file"])
        #         ).cuda()
        #         self.record_duration = self.actions_list.shape[0]

        q_command = []
        q_reach = []
        if not self.enable_hardware:
            _fake_hand_dof_pos = prev_target.clone()

        while True:
            counter += 1
            # obs = self.running_mean_std(obs_buf.clone()) # ! Need to check if this is implemented
            if counter >= 1000:
                q_command = np.array(q_command)
                q_reach = np.array(q_reach)
                np.save("debug/q_command.npy", q_command)
                np.save("debug/q_reach.npy", q_reach)
                break

            if hasattr(self, "actions_list"):
                action = self.actions_list[counter - 1][None, :]
            else:
                action = self.forward_network(obs_buf)

            action = torch.clamp(action, -1.0, 1.0)
            full_action = torch.zeros((1, 16), device=self.device)
            full_action[0, self.free_finger_indices] = action.clone()

            if "actions_mask" in self.config["task"]["env"]:
                action = action * torch.tensor(self.config["task"]["env"]["actions_mask"]).cuda()[None, :]

            target = prev_target + self.action_scale * full_action
            target = torch.clip(target, self.full_leap_dof_lower, self.full_leap_dof_upper)
            prev_target = target.clone()

            # interact with the hardware
            commands = target.cpu().numpy()[0]
            q_command.append(commands.tolist())

            if self.enable_hardware:
                q_reach.append(leap.poll_joint_position()[0].tolist())
                leap.command_joint_position(commands)
                # command_list.append(commands)
                # get o_{t+1}
                obses, _ = leap.poll_joint_position()
                ros_rate.sleep()  # keep 20 Hz command
            else:
                q_reach.append(target.cpu().numpy()[0].tolist())
                _fake_hand_dof_pos = target.clone()
                obses = _fake_hand_dof_pos.cpu().numpy().flatten()

            obses = obses[self.free_finger_indices]
            obses = torch.from_numpy(obses.astype(np.float32)).cuda()

            cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]
            self.cur_obs_joint_angles = cur_obs_buf.clone()
            cur_obj_pose = self.get_unscaled_obj_poses()[None]
            if self.include_obj_pose:
                one_step_input = torch.cat([cur_obs_buf.clone(), cur_obj_pose.clone()], dim=-1).unsqueeze(0)
            else:
                one_step_input = cur_obs_buf.clone().unsqueeze(0)

            if self.debug_viz:
                self.plot_callback()

            if hasattr(self, "obs_list"):
                self.obs_list.append(cur_obs_buf[0].clone())
                self.target_list.append(target[0].clone().squeeze())

                if counter == self.record_duration - 1:
                    self.obs_list = torch.stack(self.obs_list, dim=0)
                    self.obs_list = self.obs_list.cpu().numpy()

                    self.target_list = torch.stack(self.target_list, dim=0)
                    self.target_list = self.target_list.cpu().numpy()

                    if "actions_file" in self.config["task"]["env"]["debug"]:
                        actions_file = os.path.basename(self.config["task"]["env"]["debug"]["actions_file"])
                        folder = os.path.dirname(self.config["task"]["env"]["debug"]["actions_file"])
                        suffix = "_".join(actions_file.split("_")[1:])
                        joints_file = os.path.join(folder, f"joints_real_{suffix}")
                        target_file = os.path.join(folder, f"targets_real_{suffix}")
                    else:
                        suffix = self.config["task"]["env"]["debug"]["record"]["suffix"]
                        joints_file = f"debug/joints_real_{suffix}.npy"
                        target_file = f"debug/targets_real_{suffix}.npy"

                    np.save(joints_file, self.obs_list)
                    np.save(target_file, self.target_list)
                    exit()

            # if self.config["task"]["env"]["include_history"]:
            #     obs_buf = obs_buf[:, num_obs_single:].clone()
            # else:
            #     obs_buf = torch.zeros((1, 0), device=self.device)

            obs_buf = obs_buf[:, 1:].clone()
            obs_buf = torch.cat([obs_buf, one_step_input.clone()], dim=1)

            # if self.config["task"]["env"]["include_targets"]:
            #     obs_buf = torch.cat([obs_buf, target.clone()], dim=-1)

            # if self.config["task"]["env"]["include_obj_target"]:
            #     obs_buf = torch.cat([obs_buf, self.goal_rot.clone()], dim=-1)

            # if "phase_period" in self.config["task"]["env"]:
            #     omega = 2 * math.pi / self.config["task"]["env"]["phase_period"]
            #     phase_angle = (counter - 1) * omega / hz
            #     num_envs = obs_buf.shape[0]
            #     phase = torch.zeros((num_envs, 2), device=obs_buf.device)
            #     phase[:, 0] = math.sin(phase_angle)
            #     phase[:, 1] = math.cos(phase_angle)
            #     obs_buf = torch.cat([obs_buf, phase.clone()], dim=-1)

            # if "obs_mask" in self.config["task"]["env"]:
            #     obs_buf = obs_buf * torch.tensor(self.config["task"]["env"]["obs_mask"]).cuda()[None, :]

            obs_buf = obs_buf.float()

    def forward_network(self, obs):
        return self.player(obs).detach()

    def restore(self):
        _input_size = self.player_model_config["input_size"]
        _hidden_size = self.player_model_config["hidden_size"]
        _rnn_layers = self.player_model_config["rnn_layers"]
        _mlp_hidden_sizes = self.player_model_config["mlp_hidden_sizes"]
        _output_size = self.player_model_config["output_size"]
        
        self.player = LfDAgent(
            input_size=_input_size, hidden_size=_hidden_size, rnn_layers=_rnn_layers, \
            mlp_hidden_sizes=_mlp_hidden_sizes, output_size=_output_size
        ).to(self.device)

        _saved_model_weights = torch.load(self.player_model_path)
        self.player.load_state_dict(_saved_model_weights)


@hydra.main(config_name="config", config_path="cfg")
def main(config: DictConfig):
    extra_args = {
        "model_path": "debug/lfd/best_rnn_model.pth",
        "model_config": {
            "input_size": 23,
            "hidden_size": 256,
            "rnn_layers": 1,
            "mlp_hidden_sizes": [512, 256, 128],
            "output_size": 16
        },
        "rnn_seq_length": 3,
        "obj_pos_range": {
            "x": [-0.08, 0.0],
            "y": [0.01, 0.07],
            "z": [0.06, 0.11]
        },
        "frozen_fingers": ["finger1", "finger3"],
        "free_fingers": ["thumb", "finger2"],
        "deploy_options": {
            "include_obj_pose": False,
            "enable_hardware": False,
            "sample_initial_pose": [
                0.6719424, -1.05685414, 1.66289367, 0.53081591, \
                0.33446655, 1.59693277,  1.25332098,  1.21190344, \
                0.39735977,  0.02000032, 1.25178708,  1.07384522, \
                0.78085487,  1.05236946,  1.13827266, 1.09838899
            ],
            "debug_mode": True
        }
    }
    agent = HardwarePlayer(config, extra_args)
    agent.restore()
    agent.deploy()


if __name__ == "__main__":
    main()
