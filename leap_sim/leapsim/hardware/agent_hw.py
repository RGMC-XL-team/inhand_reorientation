#!/home/yongpeng/anaconda3/envs/rlgpu/bin/python
"""
    This script is modified from deploy.py, with the
    removal of all Hydra related stuff.
    This script is a ROS node file.
    This script runs all agents used in leap_task_B.
"""


import pdb
from copy import deepcopy
import math
import os
import xml.etree.ElementTree as ET

import yaml
import rospy
import rospkg
import actionlib
from typing import Dict

import numpy as np
from isaacgym.torch_utils import quat_from_euler_xyz
import torch
from gym import spaces
from rl_games.algos_torch import model_builder
from rl_games.torch_runner import Runner, _override_sigma, _restore

from leapsim.learning import amp_continuous, amp_models, amp_network_builder, amp_players
from leapsim.utils.rlgames_utils import RLGPUAlgoObserver

from leap_hardware.srv import object_state
from leap_task_B.leap_grasp_cube import LeapGraspCommander
# from leapsim.hardware_controller import LeapHand
from leap_hardware.hardware_controller import LeapHand

from leap_task_B.msg import RunPolicyAction, RunPolicyActionFeedback, RunPolicyActionResult
from leap_task_B.taskB_utils import compute_angle_distance, get_euler_from_quat, \
    get_quat_from_euler, get_z_up_quat


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def fill_config(config):
    """Set config default"""
    if "include_history" not in config["task"]["env"]:
        config["task"]["env"]["include_history"] = True

    if "include_targets" not in config["task"]["env"]:
        config["task"]["env"]["include_targets"] = True

    if "history_length" not in config["task"]["env"]:
        config["task"]["env"]["history_length"] = 3

    if "include_obj_target" not in config["task"]["env"]:
        config["task"]["env"]["include_obj_target"] = False

    return config

def restore(policy_config):
    """Restore the RL network from config"""
    # set default
    policy_config = fill_config(policy_config)

    rlg_config_dict = policy_config["train"]
    rlg_config_dict["params"]["config"]["env_info"] = {}
    
    # adjust number of observations
    if policy_config["task"]["env"]["include_obj_target"]:
        policy_config["task"]["env"]["numObservations"] += \
            4 * policy_config["task"]["env"]["history_length"]
    _num_obs = policy_config["task"]["env"]["numObservations"]

    _num_actions = 16
    observation_space = spaces.Box(np.ones(_num_obs) * -np.Inf, np.ones(_num_obs) * np.Inf)
    rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
    action_space = spaces.Box(np.ones(_num_actions) * -1.0, np.ones(_num_actions) * 1.0)
    rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
    rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder("amp_continuous", lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder(
            "amp_continuous", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs)
        )
        model_builder.register_model(
            "continuous_amp", lambda network, **kwargs: amp_models.ModelAMPContinuous(network)
        )
        model_builder.register_network("amp", lambda **kwargs: amp_network_builder.AMPBuilder())

        return runner

    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    args = {"train": False, "play": True, "checkpoint": policy_config["checkpoint"], "sigma": None}

    _player = runner.create_player()
    _restore(_player, args)
    _override_sigma(_player, args)

    extra_args = {"num_obs": _num_obs, "num_actions": _num_actions}
    return _player, extra_args

def get_nearest_yaw_pi2(yaw):
    """Get the nearest yaw in [-pi/2, 0.0, pi/2, pi]"""
    _candidate_yaws = [-np.pi/2, 0.0, np.pi/2, np.pi]
    _distance = np.inf * np.ones(4,)
    for i in range(4):
        _distance[i] = compute_angle_distance(yaw, _candidate_yaws[i])
    return _candidate_yaws[np.argmin(_distance)]
    

class MultiHardwarePlayer(object):
    """Restores multiple agent networks, and provides leaphand control"""
    def __init__(self) -> None:
        rospy.init_node("hardware_agent_node")
        self.device = rospy.get_param("policy_device", "cuda:0")
        self.control_hz = rospy.get_param("control_hz", 20)
        self.action_name = rospy.get_param("action_name", "run_policy")
        self.control_rate = rospy.Rate(self.control_hz)

        rospack = rospkg.RosPack()
        self.package_paths = {
            "leap_sim": rospack.get_path("leap_sim")
        }

        self.load_and_initialize_agent_config()
        self.initialize_software()
        self.initialize_hardware()

    ## Setup APIs
    # ----------------------------------------

    def load_reset_via_points(self, cfg):
        """Load hand pos via points when reset"""
        # only thumb and finger3 are allowed to move for these via points
        self.reset_finger_up_pos = cfg["finger_up_pos"]
        self.reset_finger_reach_out_pos = cfg["finger_reach_out_pos"]
        self.reset_finger_back_pos = cfg["finger_back_pos"]
        self.reset_finger_up_pos[0:4] = self.init_pose[0:4].copy()
        self.reset_finger_up_pos[8:12] = self.init_pose[8:12].copy()
        self.reset_finger_reach_out_pos[0:4] = self.init_pose[0:4].copy()
        self.reset_finger_reach_out_pos[8:12] = self.init_pose[8:12].copy()
        self.reset_finger_back_pos[0:4] = self.init_pose[0:4].copy()
        self.reset_finger_back_pos[8:12] = self.init_pose[8:12].copy()

    def load_and_initialize_agent_config(self):
        """Load RL agent config and restore the networks"""
        self.policy_name_to_config_map = {}
        self.policy_name_to_player_map = {}
        self.policy_name_to_extra_args_map = {}
        self.hand_asset_name = ""
        self.sim_to_real_indices = list(range(16))
        self.real_to_sim_indices = list(range(16))

        # load all plain-text configs
        self.rl_agent_index = yaml.safe_load(
            open(os.path.join(self.package_paths["leap_sim"], "leapsim/cfg/dict", "task_B_config.yaml"))
        )
        for task in self.rl_agent_index:
            for sub_task in self.rl_agent_index[task]:
                policy_name = f"{task.upper()}_{sub_task.upper()}"
                if self.rl_agent_index[task][sub_task] == "None":
                    self.policy_name_to_config_map[policy_name] = None
                    continue

                checkpoint_name = self.rl_agent_index[task][sub_task]
                config_yaml_path = os.path.join(self.package_paths["leap_sim"], "leapsim/cfg/dict", \
                                               f"leap-{task}-{checkpoint_name}.yaml") 
                policy_config = yaml.safe_load(open(config_yaml_path))
                self.policy_name_to_config_map[policy_name] = policy_config

                rospy.loginfo(f"Load {policy_name} from {config_yaml_path}!")

                # use information from the very first config
                if self.hand_asset_name == "":
                    self.hand_asset_name = policy_config["task"]["env"]["asset"]["handAsset"]
                    self.sim_to_real_indices = policy_config["task"]["env"]["sim_to_real_indices"]
                    self.real_to_sim_indices = policy_config["task"]["env"]["real_to_sim_indices"]

        # get dof limits
        self.get_dof_limits()

        # restore all networks
        for name, policy_config in self.policy_name_to_config_map.items():
            if policy_config is None:
                player, extra_args = None, {}
            else:
                player, extra_args = restore(policy_config)
            self.policy_name_to_player_map[name] = player
            self.policy_name_to_extra_args_map[name] = extra_args

        rospy.loginfo("All agents are restored successfully!")

        # others
        _taskB_config_path = os.path.join(rospkg.RosPack().get_path("leap_task_B"), "config/taskB_XL.yaml")
        _taskB_config = yaml.safe_load(
            open(_taskB_config_path)
        )
        self.action_scale = _taskB_config["control"]["action_scale"]
        self.smooth_move_time = _taskB_config["control"]["smooth_move_time"]
        self.rl_cube_length = _taskB_config["control"]["rl_cube_length"]
        self.model_based_cube_length = _taskB_config["control"]["model_based_cube_length"]
        self.flip_fingertip_radius = _taskB_config["control"]["flip_fingertip_radius"]
        self.rot_fingertip_radius = _taskB_config["control"]["rot_fingertip_radius"]

        ## hand reset pose
        # canonical_pose_path = os.path.join(self.package_paths["leap_sim"], "leapsim/cache", "leap_canonical_pose_v2.npy")
        self.init_pose = np.array(_taskB_config["leap_canonical_pose_v2"])
        if "reset_via_hand_pos" in _taskB_config:
            self.load_reset_via_points(_taskB_config["reset_via_hand_pos"])

        ## object should be reset to this position for better manipulation
        # self.object_center_pose = np.array([-0.0542509-0.005, 0.04004605-0.005, 0.08418902+0.005, 0.0, 0.0, 0.0, 1.0])
        self.object_center_pose = np.concatenate((_taskB_config["object_center_pos"], [0, 0, 0, 1]))
        self.rot_reset_pos = _taskB_config["reset_object_pos"]["rot"]
        self.flip_reset_pos = _taskB_config["reset_object_pos"]["flip"]
        
        self.finger_to_sim_index_map = {
            "finger1": [0, 1, 2, 3],
            "thumb": [4, 5, 6, 7],
            "finger2": [8, 9, 10, 11],
            "finger3": [12, 13, 14, 15]
        }
        # keep the disabled fingers at init_pose (finger1, finger2, finger3, thumb)
        self.disabled_fingers = []

        rospy.loginfo(f"Load config from {_taskB_config_path}!")

    def initialize_hardware(self):
        """Initialize LeapHand and object state service"""
        # Start LeapHand API
        rospy.wait_for_service("/leap_position")
        rospy.wait_for_service("/leap_velocity")
        self.leap_hw = LeapHand()
        self.leap_hw.leap_dof_lower = self.leap_dof_lower.cpu().numpy()
        self.leap_hw.leap_dof_upper = self.leap_dof_upper.cpu().numpy()
        self.leap_hw.sim_to_real_indices = self.sim_to_real_indices
        self.leap_hw.real_to_sim_indices = self.real_to_sim_indices

        # Start object state service
        rospy.wait_for_service("/object_state")
        self.object_state_proxy = rospy.ServiceProxy("/object_state", object_state)

        rospy.loginfo("Hardware is initialized successfully!")

    def initialize_software(self):
        """Initialize grasp commander and other stuff"""
        self.grasp_commander = LeapGraspCommander(visualize=False)
        self.current_player = None
        self.current_player_config = None

        # set up action server
        self._action_feedback = RunPolicyActionFeedback()
        self._action_result = RunPolicyActionResult()
        self._action_server = actionlib.SimpleActionServer(
            self.action_name, RunPolicyAction, execute_cb=self.execute_cb, auto_start=False
        )
        self._action_server.start()

        rospy.loginfo("Software is initialized successfully!")

    def switch_to_policy(self, policy_name):
        """Switch to another policy"""

        # check policy illegal
        if not policy_name in self.policy_name_to_player_map or \
            self.policy_name_to_player_map[policy_name] is None:
            rospy.logerr(f"Policy {policy_name} is illegal, cannot switch!")
            return
        
        self.current_player = self.policy_name_to_player_map[policy_name]
        self.current_player_config = self.policy_name_to_config_map[policy_name]
        self.current_player_extra_args = self.policy_name_to_extra_args_map[policy_name]
        
        # set defaults
        self.current_player_config = fill_config(self.current_player_config)

        self.num_obs_single = self.current_player_extra_args["num_obs"] // 3

        # generate auxiliary goal if needed
        if "FLIP" in policy_name:
            if "IN" in policy_name:
                flip_pitch_target = -np.pi/2
            elif "OUT" in policy_name:
                flip_pitch_target = np.pi/2
            else:
                flip_pitch_target = 0.0
            self.goal_rot = quat_from_euler_xyz(
                torch.zeros(1), torch.tensor(flip_pitch_target), torch.zeros(1)
            ).to(self.device)

    def real_to_sim(self, values):
        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        return values[:, self.sim_to_real_indices]

    def get_dof_limits(self):
        asset_root = self.package_paths["leap_sim"]

        tree = ET.parse(os.path.join(asset_root, self.hand_asset_name))
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

        self.leap_dof_lower = self.real_to_sim(self.leap_dof_lower).squeeze()
        self.leap_dof_upper = self.real_to_sim(self.leap_dof_upper).squeeze()

    # ----------------------------------------

    ## Motor control APIs
    # ----------------------------------------

    def move_hand_to_pose(
            self, 
            start_position: np.ndarray, 
            goal_position: np.ndarray, 
            total_time: float=2.0, 
            extend_ratio: float=0.0,
            enable_preempt: bool=True
        ):
        """
        total_time should be in seconds
        extend_ratio: percentage
        """
        start_position = np.array(start_position).copy()
        goal_position = np.array(goal_position).copy()
        
        # handle disabled fingers
        for finger_name in self.disabled_fingers:
            sim_indices = self.finger_to_sim_index_map[finger_name]
            goal_position[sim_indices] = self.init_pose[sim_indices].copy()

        total_iters = round((total_time * self.control_hz))
        for i in range(round((total_iters + 1) * (1 + extend_ratio))):
            # cancel current motion
            if enable_preempt:
                if self._action_server.is_preempt_requested():
                    self.preempt_cb()
                    rospy.loginfo("Current goal is preempted")
                    self._action_server.set_preempted()
                    break

            progress = i / total_iters
            position = (1 - progress) * start_position + progress * goal_position
            self.leap_hw.command_joint_position(position)
            self.control_rate.sleep()

    def smooth_move_hand_to_pose(self, goal_position: np.ndarray, total_time: float=2.0):
        """Interpolate from current hand pose to goal"""
        current_position = self.leap_hw.poll_joint_position()[0]
        self.move_hand_to_pose(start_position=current_position, goal_position=goal_position.copy(), total_time=total_time)

    # ----------------------------------------

    ## Reset APIs
    # ----------------------------------------

    def set_run_model_based_policy(self):
        self.grasp_commander.set_cube_length(self.model_based_cube_length)

    def set_run_rl_policy(self):
        self.grasp_commander.set_cube_length(self.rl_cube_length)

    def reset_hand(self, enable_preempt=True):
        """move hand to pre-grasp position"""
        self.disabled_fingers.clear()

        current_position = self.leap_hw.poll_joint_position()[0]
        self.move_hand_to_pose(
            start_position=current_position, 
            goal_position=self.init_pose, 
            total_time=self.smooth_move_time,
            enable_preempt=enable_preempt
        )
        # TODO(yongpeng): add via points
        """add via-points here"""

    def reset_object(self, yaw=None, current_policy="ROT"):
        """reset object to center position"""
        self.grasp_commander.set_fingertip_height_bias(0.03)

        self.reset_hand()

        curr_object_state = np.array(self.object_state_proxy().pose)
        
        # disable fingers based on object position
        if curr_object_state[0] < -0.07 and curr_object_state[1] > 0.05:
            self.disabled_fingers = ["finger1", "finger2"]
        elif curr_object_state[0] > -0.045 and curr_object_state[1] > 0.045:
            self.disabled_fingers = ["thumb", "finger2"]
        elif curr_object_state[0] > -0.045 and curr_object_state[1] <= 0.045:
            self.disabled_fingers = ["thumb"]
        else:
            self.disabled_fingers = ["thumb"]
        
        self.grasp_commander.set_fake_cube_transform_from_pos_quat(pos=curr_object_state[0:3], quat=curr_object_state[3:7])
        self.grasp_pose = self.grasp_commander.solve_hand_grasp_IK(use_saved_ftip_pos=False)
        self.smooth_move_hand_to_pose(self.grasp_pose, total_time=self.smooth_move_time)

        if yaw is None:
            current_yaw = get_euler_from_quat(get_z_up_quat(curr_object_state[3:7]))[2]
            goal_yaw = get_nearest_yaw_pi2(current_yaw)
        else:
            goal_yaw = yaw
        desired_object_quat = get_quat_from_euler([0, 0, goal_yaw])

        if current_policy == "ROT":
            desired_object_pose = np.concatenate(
                [self.rot_reset_pos[0:2].copy(),
                [curr_object_state[2]],
                desired_object_quat])
        else:
            desired_object_pose = np.concatenate(
                [self.flip_reset_pos[0:2].copy(),
                [curr_object_state[2]],
                desired_object_quat])
            
        # current_hand_pose = self.grasp_pose.copy()
        self.grasp_commander.set_fake_cube_transform_from_pos_quat(pos=desired_object_pose[0:3], quat=desired_object_pose[3:7])
        self.grasp_pose = self.grasp_commander.solve_hand_grasp_IK(use_saved_ftip_pos=True)
        self.smooth_move_hand_to_pose(self.grasp_pose, total_time=self.smooth_move_time)

        self.reset_hand()

    def reset_object_aggressive(self):
        """reset the object aggressively with thumb and finger3"""
        # _finger_up_pos = np.array([-0.34508689,  0.15192281, -0.41258202,  0.15805872,
        #                            1.27939877, 1.57545698,  0.19180612,  0.01539837,
        #                            -0.34508689,  0.1733986 , -0.42025195,  0.18106826,
        #                            1.10759299, -0.02755298,  0.15345682, 0.11970909])
        # _finger_reach_out_pos = np.array([-0.34508689,  0.15959273, -0.41258202,  0.15499075,
        #                                   0.51087436, 1.87611711,  1.58772878,  0.88976751,
        #                                   -0.34508689,  0.1733986 , -0.42025195,  0.18106826,
        #                                   1.36376755,  0.89590345,  0.84528221, 1.03549559])
        # _finger_back_pos = np.array([-0.34508689,  0.17646657, -0.41258202,  0.15499075,
        #                              -0.378834634,  1.59233081,  1.89452486,  1.09225307,
        #                              -0.34508689,  0.1733986 , -0.42025195,  0.18106826,
        #                              0.949592993,  1.01095186,  1.35763158,  0.978738621])
        self.reset_hand()
        # _finger_up_pos[0:4] = current_position[0:4].copy(); _finger_up_pos[8:12] = current_position[8:12].copy()
        # _finger_reach_out_pos[0:4] = current_position[0:4].copy(); _finger_reach_out_pos[8:12] = current_position[8:12].copy()
        # _finger_back_pos[0:4] = current_position[0:4].copy(); _finger_back_pos[8:12] = current_position[8:12].copy()
        self.smooth_move_hand_to_pose(
            self.reset_finger_up_pos, total_time=self.smooth_move_time
        )
        self.smooth_move_hand_to_pose(
            self.reset_finger_reach_out_pos, total_time=self.smooth_move_time
        )
        self.smooth_move_hand_to_pose(
            self.reset_finger_back_pos, total_time=self.smooth_move_time
        )
        self.reset_hand()

    def grasp_object(self):
        self.grasp_commander.reset_fingertip_height_bias()
        curr_object_state = np.array(self.object_state_proxy().pose)
        self.grasp_commander.set_fake_cube_transform_from_pos_quat(pos=curr_object_state[0:3], quat=curr_object_state[3:7])
        self.grasp_pose = self.grasp_commander.solve_hand_grasp_IK(use_saved_ftip_pos=False)
        self.smooth_move_hand_to_pose(self.grasp_pose, total_time=self.smooth_move_time)


    # ----------------------------------------

    ## RL APIs
    # ----------------------------------------

    def forward_network(self, obs):
        return self.current_player.get_action(obs, True)
    
    def deploy_current_policy(self):
        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 0)).astype(np.float32)).cuda()

        # get the very first observation
        obses, _ = self.leap_hw.poll_joint_position()
        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]

        if self.current_player_config["task"]["env"]["include_history"]:
            num_append_iters = 3
        else:
            num_append_iters = 1

        for i in range(num_append_iters):
            obs_buf = torch.cat([obs_buf, cur_obs_buf.clone()], dim=-1)

            if self.current_player_config["task"]["env"]["include_targets"]:
                obs_buf = torch.cat([obs_buf, prev_target.clone()], dim=-1)

            if self.current_player_config["task"]["env"]["include_obj_target"]:
                obs_buf = torch.cat([obs_buf, self.goal_rot.clone()], dim=-1)

            if "phase_period" in self.current_player_config["task"]["env"]:
                phase = torch.tensor([[0.0, 1.0]], device=self.device)
                obs_buf = torch.cat([obs_buf, phase], dim=-1)

        if "obs_mask" in self.current_player_config["task"]["env"]:
            obs_buf = obs_buf * torch.tensor(self.current_player_config["task"]["env"]["obs_mask"]).cuda()[None, :]

        obs_buf = obs_buf.float()

        counter = 0

        if self.current_player.is_rnn:
            self.current_player.init_rnn()

        while True:
            counter += 1
            # obs = self.running_mean_std(obs_buf.clone()) # ! Need to check if this is implemented
            """Implement terminate condition here"""
            # if counter >= 200:
            #     break
            # cancel current motion
            if self._action_server.is_preempt_requested():
                self.preempt_cb()
                rospy.loginfo("Current goal is preempted")
                self._action_server.set_preempted()
                break

            action = self.forward_network(obs_buf)
            action = torch.clamp(action, -1.0, 1.0)

            if "actions_mask" in self.current_player_config["task"]["env"]:
                action = action * torch.tensor(self.current_player_config["task"]["env"]["actions_mask"]).cuda()[None, :]

            target = prev_target + self.action_scale * action
            target = torch.clip(target, self.leap_dof_lower, self.leap_dof_upper)
            prev_target = target.clone()

            # interact with the hardware
            commands = target.cpu().numpy()[0]

            if "disable_actions" not in self.current_player_config["task"]["env"]:
                self.leap_hw.command_joint_position(commands)

            self.control_rate.sleep()  # keep 20 Hz command

            # command_list.append(commands)
            # get o_{t+1}
            obses, _ = self.leap_hw.poll_joint_position()
            obses = torch.from_numpy(obses.astype(np.float32)).cuda()

            # obs_buf_list.append(obses.cpu().numpy().squeeze())
            cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]
            self.cur_obs_joint_angles = cur_obs_buf.clone()

            if self.current_player_config["task"]["env"]["include_history"]:
                obs_buf = obs_buf[:, self.num_obs_single:].clone()
            else:
                obs_buf = torch.zeros((1, 0), device=self.device)

            obs_buf = torch.cat([obs_buf, cur_obs_buf.clone()], dim=-1)

            if self.current_player_config["task"]["env"]["include_targets"]:
                obs_buf = torch.cat([obs_buf, target.clone()], dim=-1)

            if self.current_player_config["task"]["env"]["include_obj_target"]:
                obs_buf = torch.cat([obs_buf, self.goal_rot.clone()], dim=-1)

            if "phase_period" in self.current_player_config["task"]["env"]:
                omega = 2 * math.pi / self.current_player_config["task"]["env"]["phase_period"]
                phase_angle = (counter - 1) * omega / self.control_hz   # default hz: 20
                num_envs = obs_buf.shape[0]
                phase = torch.zeros((num_envs, 2), device=obs_buf.device)
                phase[:, 0] = math.sin(phase_angle)
                phase[:, 1] = math.cos(phase_angle)
                obs_buf = torch.cat([obs_buf, phase.clone()], dim=-1)

            if "obs_mask" in self.current_player_config["task"]["env"]:
                obs_buf = obs_buf * torch.tensor(self.current_player_config["task"]["env"]["obs_mask"]).cuda()[None, :]

            obs_buf = obs_buf.float()

    # ----------------------------------------

    ## Actionlib APIs
    # ----------------------------------------
    def execute_cb(self, goal):
        rospy.loginfo(f"Receiving new action goal, preparing to execute policy {goal.policy_name}")

        _requested_policy = goal.policy_name
        _success = False
        if _requested_policy == "RESET_HAND":
            self.set_run_model_based_policy()
            self.reset_hand()
            _success = True
        elif _requested_policy in ["RESET_OBJECT", "RESET_OBJECT_ROT"]:
            self.set_run_model_based_policy()
            self.reset_object(current_policy="ROT")
            _success = True
        elif _requested_policy == "RESET_OBJECT_FLIP":
            self.set_run_model_based_policy()
            self.reset_object(current_policy="FLIP")
            _success = True
        elif _requested_policy == "RESET_OBJECT_AGGRESSIVE":
            self.set_run_model_based_policy()
            self.reset_object_aggressive()
            _success = True
        elif _requested_policy in ["ROT_CW", "ROT_CCW", "FLIP_IN", "FLIP_OUT"]:
            self.set_run_rl_policy()
            self.switch_to_policy(_requested_policy)
            if "ROT" in _requested_policy:
                self.grasp_commander.set_fingertip_radius(self.rot_fingertip_radius)
                self.grasp_object()
                self.grasp_commander.reset_fingertip_radius()
            else:
                self.grasp_commander.set_fingertip_radius(self.flip_fingertip_radius)
                self.grasp_object()
                self.grasp_commander.reset_fingertip_radius()
            self.deploy_current_policy()
            _success = False
        else:
            raise NotImplementedError(f"Policy {_requested_policy} is not implemented!")
    
        if _success:
            self._action_server.set_succeeded(self._action_result)

    def preempt_cb(self):
        self.reset_hand(enable_preempt=False)


    ## Test APIs
    # ----------------------------------------

    def test_deploy(self):
        # pdb.set_trace()
        # self.reset_hand()

        # curr_object_state = self.object_state_proxy().pose
        # self.grasp_commander.set_cube_transform_from_pos_quat(pos=curr_object_state[0:3], quat=curr_object_state[3:7])
        # self.grasp_pose = self.grasp_commander.solve_hand_grasp_IK()

        # current_position = self.leap_hw.poll_joint_position()[0]
        # print(f"Hand will move {self.grasp_pose - current_position} in joint space!")

        # pdb.set_trace()
        # self.move_hand_to_pose(start_position=current_position, goal_position=self.grasp_pose, total_time=2.0)
        # pdb.set_trace()

        # self.grasp_commander.set_cube_transform_from_pos_quat(pos=curr_object_state[0:3]+np.array([-0.025, 0.0, 0.0]), quat=curr_object_state[3:7])
        # self.grasp_pose = self.grasp_commander.solve_hand_grasp_IK()
        # current_position = self.leap_hw.poll_joint_position()[0]
        # self.move_hand_to_pose(start_position=current_position, goal_position=self.grasp_pose, total_time=2.0)
        # pdb.set_trace()

        while not rospy.is_shutdown():
            pdb.set_trace()
            self.reset_object()

    # ----------------------------------------
    

if __name__ == "__main__":
    player = MultiHardwarePlayer()
    # player.test_deploy()
    rospy.spin()
    
