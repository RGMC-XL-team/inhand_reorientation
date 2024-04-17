import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import rospkg
import rospy
import torch
import yaml
from gym import spaces
from isaacgym.torch_utils import quat_conjugate, quat_from_euler_xyz, quat_mul, to_torch, torch_rand_float, unscale
from leapsim.learning import amp_continuous, amp_models, amp_network_builder, amp_players
from leapsim.utils.rlgames_utils import RLGPUAlgoObserver
from rl_games.algos_torch import model_builder
from rl_games.torch_runner import Runner, _override_sigma, _restore

from leap_hardware.hardware_controller import LeapHand
from leap_hardware.srv import object_state

LEAP_HAND_DEFAULT_DOF_POS = [
    1.0783312,
    -0.6933233999999999,
    0.4955898999999999,
    1.6027808000000001,
    -0.2334461,
    1.9132094000000005,
    1.6278199999999996,
    1.2707759999999995,
    0.4939744,
    0.0,
    0.7217784999999999,
    1.2285776,
    1.0783312,
    0.6933233999999999,
    0.4955898999999999,
    1.6027808000000001,
]


class LeapHardwareAgent:
    """
    This class handles Leap Hand sim2real
    :param config: dict dumped from hydra, i.e., returned by omegaconf_to_dict
    """

    def __init__(self, config) -> None:
        self.config = config
        self.get_package_paths()
        self.set_defaults()
        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = "cuda"

        self.debug_viz = self.config["task"]["env"]["enableDebugVis"]

        # hand setting
        self.init_pose = self.fetch_grasp_state()
        self.get_dof_limits()
        self.construct_sim_to_real_transformation()

        # prepare hardware
        self.prepare_hardware()
        self.prepare_states()

    def set_defaults(self):
        self.num_envs = 1
        self.num_fingertips = 4
        self.ordered_fingertip_link_names = [
            "finger1_tip_center",
            "thumb_tip_center",
            "finger2_tip_center",
            "finger3_tip_center",
        ]

        self.vel_obs_scale = 0.2
        self.override_object_z = True  # to align with training input range

        # update paths in config
        self.config["checkpoint"] = os.path.join(self.package_paths["leap_sim"], "leapsim", self.config["checkpoint"])
        self.config["train"]["params"]["load_path"] = os.path.join(
            self.package_paths["leap_sim"], "leapsim", self.config["train"]["params"]["load_path"]
        )

        # control
        self.control_hz = 20
        self.control_dt = 1 / self.control_hz

    def get_package_paths(self):
        rospack = rospkg.RosPack()
        self.package_paths = {}
        self.package_paths["leap_sim"] = rospack.get_path("leap_sim")

    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()

        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()

        return values[:, self.sim_to_real_indices]

    def get_dof_limits(self):
        asset_root = self.package_paths["leap_sim"]
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

        self.leap_dof_lower = self.real_to_sim(self.leap_dof_lower).squeeze()
        self.leap_dof_upper = self.real_to_sim(self.leap_dof_upper).squeeze()

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = self.config["task"]["env"]["sim_to_real_indices"]
        self.real_to_sim_indices = self.config["task"]["env"]["real_to_sim_indices"]

    def prepare_hardware(self):
        # prepare Leap Hand
        rospy.wait_for_service("/leap_position")
        rospy.wait_for_service("/leap_velocity")
        rospy.wait_for_service("/leap_effort")

        self.leap_hardware = LeapHand()
        self.leap_hardware.leap_dof_lower = self.leap_dof_lower.cpu().numpy()
        self.leap_hardware.leap_dof_upper = self.leap_dof_upper.cpu().numpy()
        self.leap_hardware.sim_to_real_indices = self.sim_to_real_indices
        self.leap_hardware.real_to_sim_indices = self.real_to_sim_indices

        # preapre perception
        rospy.wait_for_service("/object_state")
        self.object_state_proxy = rospy.ServiceProxy("/object_state", object_state)

    def prepare_states(self):
        self.leap_hand_dof_pos = torch.zeros((1, 16), dtype=torch.float).to(self.device)
        self.leap_hand_dof_vel = torch.zeros((1, 16), dtype=torch.float).to(self.device)
        self.actions = torch.zeros((1, 16), dtype=torch.float).to(self.device)
        self.object_pose = torch.zeros((1, 7), dtype=torch.float).to(self.device)
        self.object_linvel = torch.zeros((1, 3), dtype=torch.float).to(self.device)
        self.object_angvel = torch.zeros((1, 3), dtype=torch.float).to(self.device)
        self.goal_states = torch.zeros((1, 7), dtype=torch.float).to(self.device)
        self.goal_rot = torch.zeros((1, 4), dtype=torch.float).to(self.device)
        self.fingertip_state = torch.zeros((1, 4, 13), dtype=torch.float).to(self.device)

    def fetch_grasp_state(self):
        # return torch.tensor(LEAP_HAND_DEFAULT_DOF_POS).to(self.device)[None, :]
        return np.array(LEAP_HAND_DEFAULT_DOF_POS)

    def get_random_goal(self):
        _offset = self.config["task"]["env"]["override_dist"]["cube_small"]

        rand_integer = torch_rand_float(0.0, 4.0, (1, 1), device=self.device).squeeze().long() - 2
        rand_goal_rot = torch.pi / 2 * rand_integer

        goal_rot = quat_from_euler_xyz(
            torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device), rand_goal_rot
        )
        self.goal_rot[0, :] = goal_rot

        goal_pos = torch.tensor(
            [
                _offset["override_object_init_x"],
                _offset["override_object_init_y"],
                _offset["override_object_init_z"] - 0.5,
            ],
            dtype=torch.float,
            device=self.device,
        )
        self.goal_states[0, 0:3] = goal_pos
        self.goal_states[0, 3:7] = goal_rot

        # debug publish goal
        import tf2_ros

        from leap_hardware.ros_utils import pos_quat_to_ros_transform

        ros_transform = pos_quat_to_ros_transform(
            goal_pos.cpu().numpy(), goal_rot.cpu().numpy()[[-1, 0, 1, 2]], child_frame="cube_goal"
        )
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        tf_broadcaster.sendTransform(ros_transform)

    def get_hand_state(self):
        """
        Return hand positions and velocities
        """
        # joint states
        joint_positions, joint_velocities = self.leap_hardware.poll_joint_state()
        self.leap_hand_dof_pos[0, :] = to_torch(joint_positions, dtype=torch.float, device=self.device)
        self.leap_hand_dof_vel[0, :] = to_torch(joint_velocities, dtype=torch.float, device=self.device)

        # fingertip states
        fingertip_state, fingertip_link_names = self.leap_hardware.poll_fingertip_state(
            ordered_link_names=self.ordered_fingertip_link_names
        )
        self.fingertip_state[0, ...] = to_torch(fingertip_state, dtype=torch.float, device=self.device).reshape(4, 13)

    def get_object_state(self):
        """
        Return object pose and velocity
        """
        object_state = self.object_state_proxy()
        self.object_pose[0, :] = to_torch(object_state.pose, dtype=torch.float, device=self.device)
        self.object_linvel[0, :] = to_torch(object_state.velocity[:3], dtype=torch.float, device=self.device)
        self.object_angvel[0, :] = to_torch(object_state.velocity[3:], dtype=torch.float, device=self.device)

    def refresh_hw(self):
        """
        refresh all the states on hardware
        """
        self.get_hand_state()
        self.get_object_state()

    def move_hand_to_pose(self, leap_start_positions, leap_end_positions):
        assert isinstance(leap_start_positions, np.ndarray) and leap_start_positions.shape == (16,)
        assert isinstance(leap_end_positions, np.ndarray) and leap_end_positions.shape == (16,)

        _rate = rospy.Rate(self.control_hz)
        _iters = 4 * self.control_hz
        for i in range(_iters + 1):
            _progress = i / _iters
            _position = (1 - _progress) * leap_start_positions + _progress * leap_end_positions
            self.leap_hardware.command_joint_position(_position)
            _rate.sleep()
        rospy.loginfo("Hand moved to initial pose!")

    def compute_full_observation_hw(self, no_vel=False):
        """
        compute the full observation on hardware
        """
        scaled_dof_pos = unscale(self.leap_hand_dof_pos, self.leap_dof_lower, self.leap_dof_upper)
        # TODO(yongpeng): for debug purpose, we scale twice, the same as training
        scaled_dof_pos = unscale(scaled_dof_pos, self.leap_dof_lower, self.leap_dof_upper)

        object_rot = self.object_pose[0, 3:7].unsqueeze(0)
        object_pose_override = self.object_pose.clone()
        fingertip_state_override = self.fingertip_state.clone()
        if self.override_object_z:
            object_pose_override[0, 2] = self.config["task"]["env"]["override_dist"]["cube_small"][
                "override_object_init_z"
            ]
            fingertip_state_override[0, :, 2] += 0.5
        quat_dist = quat_mul(object_rot, quat_conjugate(self.goal_rot))

        if no_vel:
            fingertip_pos = self.fingertip_state[:, :, :3]
            out = torch.cat(
                [
                    scaled_dof_pos,
                    object_pose_override,
                    self.goal_rot,
                    quat_dist,
                    fingertip_pos.reshape(1, 3 * self.num_fingertips),
                    self.actions,
                ],
                dim=-1,
            )
        else:
            out = torch.cat(
                [
                    scaled_dof_pos,
                    self.vel_obs_scale * self.leap_hand_dof_vel,
                    object_pose_override,
                    self.object_linvel,
                    self.vel_obs_scale * self.object_angvel,
                    self.goal_rot,
                    quat_dist,
                    fingertip_state_override.reshape(self.num_envs, 13 * self.num_fingertips),
                    self.actions,
                ],
                dim=-1,
            )
        return out

    def forward_network(self, obs):
        return self.player.get_action(obs, True)

    def restore(self):
        rlg_config_dict = self.config["train"]
        rlg_config_dict["params"]["config"]["env_info"] = {}
        self.num_obs = self.config["task"]["env"]["numObservations"]
        self.num_actions = 16
        observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
        action_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
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

        args = {"train": False, "play": True, "checkpoint": self.config["checkpoint"], "sigma": None}

        self.player = runner.create_player()
        _restore(self.player, args)
        _override_sigma(self.player, args)

    def deploy(self):
        self.get_random_goal()
        import pdb

        pdb.set_trace()

        self.refresh_hw()
        self.move_hand_to_pose(
            leap_start_positions=self.leap_hand_dof_pos.squeeze().cpu().numpy(), leap_end_positions=self.init_pose
        )
        import pdb

        pdb.set_trace()

        self.refresh_hw()
        self.obs_buf = torch.zeros((1, self.num_obs), dtype=torch.float).to(self.device)
        prev_target = self.leap_hand_dof_pos.clone()

        counter = 0
        _rate = rospy.Rate(self.control_hz)

        _debug_start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if (rospy.Time.now() - _debug_start_time).secs >= 3.0:
                break

            time_0 = time.time()
            self.refresh_hw()
            time_1 = time.time()
            self.obs_buf = self.compute_full_observation_hw().clone().squeeze(1)
            time_2 = time.time()
            print("refresh_hw time: ", time_1 - time_0)
            print("compute_obs time: ", time_2 - time_1)

            if "obs_mask" in self.config["task"]["env"]:
                self.obs_buf = self.obs_buf * torch.tensor(self.config["task"]["env"]["obs_mask"]).cuda()[None, :]
            counter += 1

            action = self.forward_network(self.obs_buf)
            action = torch.clamp(action, -1.0, 1.0)
            self.actions = action.clone().unsqueeze(0).to(self.device)

            if "actions_mask" in self.config["task"]["env"]:
                action = action * torch.tensor(self.config["task"]["env"]["actions_mask"]).cuda()[None, :]

            target = prev_target + self.action_scale * action
            target = torch.clip(target, self.leap_dof_lower, self.leap_dof_upper)
            prev_target = target.clone()

            # interact with the hardware
            commands = target.cpu().numpy()[0]

            # send command to hand
            self.leap_hardware.command_joint_position(commands)

            _rate.sleep()

        print("deploy_done")


if __name__ == "__main__":
    with open("/home/yongpeng/competition/RGMC_XL/leap_ws/src/leap_sim/leapsim/cfg/dict/leaphand_rot_z_goal.yaml") as f:
        config = yaml.safe_load(f)
    rospy.init_node("leap_hardware_agent_node")
    agent = LeapHardwareAgent(config)
    agent.restore()
    agent.deploy()
