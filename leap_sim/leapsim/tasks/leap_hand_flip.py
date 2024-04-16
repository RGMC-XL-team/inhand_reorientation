# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: 
# https://github.com/HaozhiQi/hora/blob/main/hora/tasks/leap_hand_hora.py
# --------------------------------------------------------

import os
import sys
from attr import has
from importlib_metadata import itertools
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate, to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float, scale, quat_from_euler_xyz

from leapsim.tasks.rewards.rewards_flip import compute_leaphand_reward
from leapsim.utils.torch_jit_utils import quat_diff_rad, random_quaternions, randomize_rotation_three_axis

from glob import glob
import math
import torchvision
import warnings
import matplotlib.pyplot as plt
from .base.vec_task import VecTaskRotGoal
from collections import deque

class LeapHandFlip(VecTaskRotGoal):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=None, force_render=None):
        self.cfg = cfg
        self.set_defaults()
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_cfg(cfg['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_cfg(cfg['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(cfg['env']['object'])
        # 4. setup reward
        self._setup_reward_cfg(cfg['env']['reward'])
        # 5. setup reset
        if 'random_reset_grasps' not in cfg['env']:
            cfg['env']['random_reset_grasps'] = True
        self.reset_dof_pos_noise = cfg['env']['reset_dof_pos_noise']
        self.reset_dof_vel_noise = cfg['env']['reset_dof_vel_noise']
        self.reset_position_noise = cfg['env']['reset_position_noise']
        self.reset_position_noise_scale = cfg['env']['reset_position_noise_scale']
        self.reset_goal_position_noise = cfg['env']['reset_goal_position_noise']
        self.reset_goal_position_noise_scale = cfg['env']['reset_goal_position_noise_scale']
        self.aware_ftip_pos_names = cfg['env']['reward']['aware_ftip_pos_names']

        self.base_obj_scale = cfg['env']['baseObjScale']
        self.save_init_pose = cfg['env']['genGrasps']
        self.aggregate_mode = self.cfg['env']['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.cfg['env']['reset_height_threshold']
        self.grasp_cache_name = self.cfg['env']['grasp_cache_name']
        self.evaluate = self.cfg['on_evaluation']

        # goal config
        self.enbale_goal_based_reward = ('goalReward' in self.cfg['env'])

        # observation type
        self.obs_type = self.cfg['env']['observation_type']
        self.vel_obs_scale = 0.2 # scale factor of velocity based observations

        ## TODO: change value here
        self.num_obs_dict = {
            "proprioception": 120,
            "full_state": 121
        }
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]

        self.reward_cfg = self.cfg['env']['reward'].copy()
        self.reward_cfg.update(self.cfg['env']['goalReward'])

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless)

        self.debug_viz = self.cfg['env']['enableDebugVis']
        self.max_episode_length = self.cfg['env']['episodeLength']
        self.dt = self.sim_params.dt
        self.control_dt = self.sim_params.dt * self.control_freq_inv # This is the actual control frequency

        if self.viewer:
            self.default_cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            self.default_cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, self.default_cam_pos, self.default_cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        # self.leap_hand_default_dof_pos = torch.zeros(self.num_leap_hand_dofs, dtype=torch.float, device=self.device)
        # TODO(yongpeng): add this to the .yaml files
        self.leap_hand_default_dof_pos = torch.tensor([
             1.0783312, -0.6933233999999999, 0.4955898999999999, 1.6027808000000001,
             -0.2334461, 1.9132094000000005, 1.6278199999999996, 1.2707759999999995,
             0.4939744, 0.0, 0.7217784999999999, 1.2285776, 
             1.0783312, 0.6933233999999999, 0.4955898999999999, 1.6027808000000001], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.leap_hand_default_dof_vel = torch.zeros(self.num_leap_hand_dofs, dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.leap_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_leap_hand_dofs]
        self.leap_hand_dof_pos = self.leap_hand_dof_state[..., 0]
        self.leap_hand_dof_vel = self.leap_hand_dof_state[..., 1]

        self.object_rpy = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel_finite_diff = torch.zeros((self.num_envs, 3), device=self.device)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.torques = gymtorch.wrap_tensor(dof_force_tensor).view(-1, self.num_leap_hand_dofs)

        # palm contact force (if needed)
        if self.cfg['env']['reward']['pen_palm_contact']:
            leap_hand_handle = self.gym.find_actor_handle(self.envs[0], 'hand')
            self.palm_body_index = self.gym.find_actor_rigid_body_index(self.envs[0],
                                                                         leap_hand_handle,
                                                                         'palm_lower',
                                                                         gymapi.DOMAIN_SIM)     # which domain should be used here?
            print(f'Palm body index:{self.palm_body_index}')
            self.palm_contact_force = self.contact_forces[:, self.palm_body_index]

        self.global_counter = 0
        self.prev_global_counter = 0

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        # object apply random forces parameters
        self.force_scale = self.cfg['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.cfg['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.cfg['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.cfg['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)    
        self.early_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.randomize_scale and self.scale_list_init:
            self.saved_grasping_states = {}
            for s in self.randomize_scale_list:
                self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(
                    f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy'
                )).float().to(self.device)
        else:
            assert self.save_init_pose

        self.reset_goal_type = self.cfg['env']['desired_motion']['resetGoalType']
        assert self.reset_goal_type in ['random_rot', 'random_quat']

        assert 'flipAxis' in self.cfg['env']['desired_motion']
        self.flip_axis = self.cfg['env']['desired_motion']['flipAxis']
        assert self.flip_axis in ['x', 'y', 'z']
        print("Will reset goal using flip axis {}!".format(self.flip_axis))

        # set dof pos mask (by experience)
        # |--- finger1 ---|--- thumb ---|--- finger2 ---|--- finger3 ---|
        if self.flip_axis == "z":
            self.dof_pos_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif self.flip_axis == "y":
            self.dof_pos_mask = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        elif self.flip_axis == "x":
            self.dof_pos_mask = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

        # for axis="x" or "y", set delay_reset_goal to allow pi/2 rotation
        if self.flip_axis in ['x', 'y']:
            self.delay_reset_goal = True
        else:
            self.delay_reset_goal = False

        if self.num_envs > 1:
            self.enable_joint_limits_report = False

        self.goal_based_reward_cfg = self.cfg['env']['goalReward']
        self.success_tolerance = self.cfg['env']['successTolerance']

        # useful buffers
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.object_init_pose_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)

        self.total_successes = 0

        self.reset_goal_buf = self.reset_buf.clone()

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_consecutive_successes = cfg['env']['goalReward']['maxConsecutiveSuccesses']
        print("max consecutive successes: ", self.max_consecutive_successes)

        self.previous_object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_torques = self.torques.clone()
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain
        self.resample_randomizations(None)

        # debug and understanding statistics
        self.env_timeout_counter = to_torch(np.zeros((len(self.envs)))).long().to(self.device)  # max 10 (10000 envs)
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000
        self.object_angvel_finite_diff_ep_buf = deque(maxlen=1000)
        self.object_angvel_finite_diff_mean = torch.zeros(self.num_envs, device=self.device)
        self.setup_keyboard_events()

        if "actions_mask" in self.cfg["env"]:
            self.actions_mask = torch.tensor(self.cfg["env"]["actions_mask"], device=self.device)[None, :]
        else:
            self.actions_mask = torch.ones((1, self.num_leap_hand_dofs), device=self.device)
        
        if self.debug_viz:
            self.setup_plot()

        if "debug" in self.cfg["env"]:
            self.obs_list = []          # origin joint pos
            self.obs_full_list = []     # full observation
            self.target_list = []

            if "record" in self.cfg["env"]["debug"]:
                self.record_duration = int(self.cfg["env"]["debug"]["record"]["duration"] / self.control_dt)

            if "actions_file" in self.cfg["env"]["debug"]:
                self.actions_list = torch.from_numpy(np.load(self.cfg["env"]["debug"]["actions_file"])).cuda()        
                self.record_duration = self.actions_list.shape[0]

        print("Finished initializing simulation envs!")
    
    def set_camera(self, position, lookat):
        """ 
        Set camera position and direction
        """

        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, self.envs[self.lookat_id], cam_pos, cam_target)

    def lookat(self, i):
        look_at_pos = self.hand_pos[i, :].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    def render(self):
        super().render()

        if self.viewer:
            if not self.free_cam:
                self.lookat(self.lookat_id)
            
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if not self.free_cam:
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs
                        self.lookat(self.lookat_id)

                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    
                    if self.free_cam:
                        self.gym.viewer_camera_look_at(self.viewer, None, self.default_cam_pos, self.default_cam_target)

    def setup_keyboard_events(self):
        self.lookat_id = 0
        self.free_cam = False
        self.lookat_vec = torch.tensor([0.4, -0.2, 0.1], requires_grad=False, device=self.device)

        if self.viewer is None:
            return
        
        # subscribe to keyboard shortcuts
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_F, "free_cam")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")

    def resample_randomizations(self, env_ids):
        if "joint_noise" not in self.cfg["env"]["randomization"]:
            return

        self.joint_noise_cfg = self.cfg["env"]["randomization"]["joint_noise"]

        if env_ids is None:
            self.joint_noise_iid_scale = torch.zeros((self.num_envs, self.num_leap_hand_dofs), device=self.device)
            self.joint_noise_constant_offset = torch.zeros((self.num_envs, self.num_leap_hand_dofs), device=self.device)
            self.joint_noise_outlier_scale = torch.zeros((self.num_envs, self.num_leap_hand_dofs), device=self.device)
            self.joint_noise_outlier_rate = torch.zeros((self.num_envs, self.num_leap_hand_dofs), device=self.device)
            env_ids = torch.arange(self.num_envs, device=self.device)

        if "iid" in self.joint_noise_cfg:
            low, high = self.joint_noise_cfg["iid"]["scale_range"]
            self.joint_noise_iid_scale[env_ids] = torch.rand((env_ids.shape[0], self.num_leap_hand_dofs), device=self.device) * (high - low) + low
            self.joint_noise_iid_type = self.joint_noise_cfg["iid"]["type"]

        if "constant_offset" in self.joint_noise_cfg:
            low, high = self.joint_noise_cfg["constant_offset"]["range"]
            self.joint_noise_constant_offset[env_ids] = torch.rand((env_ids.shape[0], self.num_leap_hand_dofs), device=self.device) * (high - low) + low

        if "outlier" in self.joint_noise_cfg:
            low, high = self.joint_noise_cfg["outlier"]["scale_range"]
            self.joint_noise_outlier_scale[env_ids] = torch.rand((env_ids.shape[0], self.num_leap_hand_dofs), device=self.device) * (high - low) + low
            
            low, high = self.joint_noise_cfg["outlier"]["rate_range"]
            self.joint_noise_outlier_rate[env_ids] = torch.rand((env_ids.shape[0], self.num_leap_hand_dofs), device=self.device) * (high - low) + low
            
            self.joint_noise_outlier_type = self.joint_noise_cfg["outlier"]["type"]

    def setup_plot(self):   
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-20, 20)
        self.ydata = deque(maxlen=100) # Plot 5 seconds of data
        self.ydata2 = deque(maxlen=100)
        (self.ln,) = self.ax.plot(range(len(self.ydata)), list(self.ydata), animated=True)
        (self.ln2,) = self.ax.plot(range(len(self.ydata2)), list(self.ydata2), animated=True)
        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)

    def set_defaults(self):
        if "include_pd_gains" not in self.cfg["env"]:
            self.cfg["env"]["include_pd_gains"] = False

        if "include_friction_coefficient" not in self.cfg["env"]:
            self.cfg["env"]["include_friction_coefficient"] = False

        if "include_obj_scales" not in self.cfg["env"]:
            self.cfg["env"]["include_obj_scales"] = False

        if "leap_hand_start_z" not in self.cfg["env"]:
            self.cfg["env"]["leap_hand_start_z"] = 0.5
        
        if "grasp_dof_search_radius" not in self.cfg["env"]:
            self.cfg["env"]["grasp_dof_search_radius"] = 0.5

        if "obs_mask" not in self.cfg["env"]:
            self.cfg["env"]["obs_mask"] = None

        if "include_targets" not in self.cfg["env"]:
            self.cfg["env"]["include_targets"] = True
        
        if "include_obj_pose" not in self.cfg["env"]:
            self.cfg["env"]["include_obj_pose"] = False

        if "include_history" not in self.cfg["env"]:
            self.cfg["env"]["include_history"] = True

        if "joint_limits" not in self.cfg["env"]["randomization"]:
            self.cfg["env"]["randomization"]["joint_limits"] = 0

        if "mask_body_collision" not in self.cfg["env"]:
            self.cfg["env"]["mask_body_collision"] = {}        
    
        if "disable_actions" not in self.cfg["env"]:
            self.cfg["env"]["disable_actions"] = False

        if "disable_gravity" not in self.cfg["env"]:
            self.cfg["env"]["disable_gravity"] = False

        if "disable_object_collision" not in self.cfg["env"]:
            self.cfg["env"]["disable_object_collision"] = False

        if "disable_resets" not in self.cfg["env"]:
            self.cfg["env"]["disable_resets"] = False

        if "disable_self_collision" not in self.cfg["env"]:
            self.cfg["env"]["disable_self_collision"] = False

        if "rotation_axis" not in self.cfg["env"]:
            self.rotation_axis = torch.tensor([0., 0., 1.])
        else:
            self.rotation_axis = torch.tensor(self.cfg["env"]["rotation_axis"])

        if "enableJointLimitsReport" not in self.cfg['env']:
            self.enable_joint_limits_report = False

        # Multiple rigid shapes correspond to a rigid body, the indices can be found using get_asset_rigid_body_shape_indices
        self.body_shape_indices = [ 
            ( 0 ,  17 ),
            ( 17 ,  1 ),
            ( 18 ,  5 ),
            ( 23 ,  5 ),
            ( 28 ,  3 ),
            ( 31 ,  1 ),
            ( 32 ,  4 ),
            ( 36 ,  9 ),
            ( 45 ,  3 ),
            ( 48 ,  1 ),
            ( 49 ,  5 ),
            ( 54 ,  5 ),
            ( 59 ,  3 ),
            ( 62 ,  1 ),
            ( 63 ,  5 ),
            ( 68 ,  5 ),
            ( 73 ,  3 )
        ]

    def _get_leap_asset(self, asset_root=None, asset_options=None):
        # enable and disable ftip pos reward
        if len(self.aware_ftip_pos_names) == 0:
            self.cgf['env']['reward']['ftipRewardScale'] = 0
            print("No fingertip pos is awared, set ftipRewardScale to 0!")

        rb_links = self.gym.get_asset_rigid_body_names(self.hand_asset)
        self.fingertips = [x for x in rb_links if 'tip_center' in x]  # ["finger1_tip_center", "finger2_tip_center", "finger3_tip_center", "thumb_tip_center"]
        self.ftip_pos_mask = []

        for name in self.fingertips:
            if name in self.aware_ftip_pos_names:
                self.ftip_pos_mask.append(1)
            else:
                self.ftip_pos_mask.append(0)

        self.num_fingertips = len(self.fingertips)

        print(f'Number of fingertips:{self.num_fingertips}  Fingertips:{self.fingertips}')

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()

        # set leap_hand dof properties
        self.num_leap_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        leap_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.leap_hand_dof_lower_limits = []
        self.leap_hand_dof_upper_limits = []

        for i in range(self.num_leap_hand_dofs):
            self.leap_hand_dof_lower_limits.append(leap_hand_dof_props['lower'][i])
            self.leap_hand_dof_upper_limits.append(leap_hand_dof_props['upper'][i])
            leap_hand_dof_props['effort'][i] = 0.5
            leap_hand_dof_props['stiffness'][i] = self.cfg['env']['controller']['pgain']
            leap_hand_dof_props['damping'][i] = self.cfg['env']['controller']['dgain']
            leap_hand_dof_props['friction'][i] = 0.01
            leap_hand_dof_props['armature'][i] = 0.001

        self.leap_hand_dof_lower_limits = to_torch(self.leap_hand_dof_lower_limits, device=self.device)
        self.leap_hand_dof_upper_limits = to_torch(self.leap_hand_dof_upper_limits, device=self.device)

        self.leap_hand_dof_lower_limits = self.leap_hand_dof_lower_limits.repeat((self.num_envs, 1))  
        self.leap_hand_dof_lower_limits += (2 * torch.rand_like(self.leap_hand_dof_lower_limits) - 1) * self.cfg["env"]["randomization"]["joint_limits"]
        self.leap_hand_dof_upper_limits = self.leap_hand_dof_upper_limits.repeat((self.num_envs, 1))
        self.leap_hand_dof_upper_limits += (2 * torch.rand_like(self.leap_hand_dof_upper_limits) - 1) * self.cfg["env"]["randomization"]["joint_limits"]

        hand_pose, obj_pose = self._init_object_pose()

        goal_start_pose = self.get_goal_object_start_pose(object_start_pose=obj_pose)

        # compute aggregate size
        self.num_leap_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_leap_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_leap_hand_bodies + 2
        max_agg_shapes = self.num_leap_hand_shapes + 2

        self.envs = []

        self.object_init_state = []

        self.hand_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        # get fingertip handles
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in
                                  self.fingertips]

        leap_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(leap_hand_rb_count, leap_hand_rb_count + object_rb_count))
        self.obj_scales = []
        self.object_friction_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, leap_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            if self.cfg["env"]["disable_object_collision"]:
                collision_group = -(i+2)
            else:
                collision_group = i

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object_cube', collision_group, 0, 0)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            obj_scale = self.base_obj_scale
            
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(self.randomize_scale_list[i % num_scales] - 0.025, self.randomize_scale_list[i % num_scales] + 0.025)
                
                if "randomize_scale_factor" in self.cfg["env"]:
                    obj_scale *= np.random.uniform(*self.cfg["env"]["randomize_scale_factor"])
                
                self.obj_scales.append(obj_scale)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self.object_friction_buf[i] = obj_friction

            # create goal object
            goal_asset = self.goal_asset_list[object_type_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs,
                                                0, 2)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.obj_scales = torch.tensor(self.obj_scales, device=self.device)
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
    
        self.setup_torch_states()

        print("Finished setting up simulation envs!")

    def setup_torch_states(self):
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        # TODO(yongpeng): check if we need any goal bias here
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

        # unit tensors
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def get_goal_object_start_pose(self, object_start_pose):
        self.goal_displacement = gymapi.Vec3(0., 0, 0.25)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        return goal_start_pose

    def reset_target_pose(self, env_ids, apply_reset=False):
        """
            generate new target pose for env_ids
        """
        # reset_target_pose should be called after reset_idx
        reset_without_restart = (self.progress_buf[env_ids] > 0)
        reset_from_current_rot_ids = env_ids[reset_without_restart]
        reset_from_init_rot_ids = env_ids[~reset_without_restart]
        current_rot = torch.zeros((len(env_ids), 4), dtype=torch.float, device=self.device)
        current_rot[reset_without_restart] = self.object_rot[reset_from_current_rot_ids]
        current_rot[~reset_without_restart] = self.object_init_pose_buf[reset_from_init_rot_ids, 3:7]
        
        if self.reset_goal_type == "random_quat":
            # random 3D rotations
            goal_rot = random_quaternions(num=len(env_ids), device=self.device, order='xyzw')
        elif self.reset_goal_type == "random_rot":
            # random goals with single-axis rotation
            rand_angles = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device) * torch.pi

            # for ROTATION case, the desired orientation is 0, pi/2, -pi/2, -pi
            rand_integer = torch_rand_float(0.0, 4.0, (len(env_ids), 1), device=self.device).squeeze().long() - 2
            rand_goal_rot = torch.pi/2 * rand_integer + torch_rand_float(-0.26, 0.26, (len(env_ids), 1), device=self.device).squeeze()

            # for FLIPPING case, the desired delta orientation is 
            rand_bool = (torch.rand((len(env_ids), 1), device=self.device) > 0.5).squeeze()
            rand_goal_rot_pi = torch.where(rand_bool, torch.ones_like(rand_bool, dtype=torch.int), -torch.ones_like(rand_bool, dtype=torch.int))
            rand_goal_rot_pi = rand_goal_rot_pi * \
                torch_rand_float(0.9*np.pi/2, 1.1*np.pi/2, (len(env_ids), 1), device=self.device).squeeze()

            # ROTATION case
            if self.flip_axis == 'z':
                goal_rot = quat_from_euler_xyz(0.025*rand_angles[:, 0], 0.025*rand_angles[:, 1], rand_goal_rot)
            # FLIPPING cases
            elif self.flip_axis == 'y':
                delta_rot = quat_from_euler_xyz(0.025*rand_angles[:, 0], rand_goal_rot_pi, 0.025*rand_angles[:, 2])
                goal_rot = quat_mul(current_rot, delta_rot)
            elif self.flip_axis == 'x':
                delta_rot = quat_from_euler_xyz(rand_goal_rot_pi, 0.025*rand_angles[:, 1], 0.025*rand_angles[:, 2])
                goal_rot = quat_mul(current_rot, delta_rot)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        rand_floats[:, 0] = rand_floats[:, 0] * self.reset_goal_position_noise_scale[0]
        rand_floats[:, 1] = rand_floats[:, 1] * self.reset_goal_position_noise_scale[1]
        rand_floats[:, 2] = rand_floats[:, 2] * self.reset_goal_position_noise_scale[2]

        # add position noise to goal
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3] + self.reset_goal_position_noise * rand_floats
        self.goal_states[env_ids, 3:7] = goal_rot
        
        # apply the changes to goal object here
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))

        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids=None):
        """
            (1) apply dr_randomization
            (2) reset target pose, but do not apply
            (3) reset object pose, reset goal pose if not applied
            (4) delay reset target pose
            (5) reset hand dof pos and vel
            (6) clear related buffers
        """
        # print("> > reset_idx called")

        # (1)
        if self.randomize_mass:
            lower, upper = self.randomize_mass_lower, self.randomize_mass_upper

            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                for p in prop:
                    p.mass = np.random.uniform(lower, upper)
                self.gym.set_actor_rigid_body_properties(env, handle, prop)
        else:
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        # randomize joint noise matrix
        self.resample_randomizations(env_ids)

        # (2)
        if not self.delay_reset_goal:
            self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # (3)
        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            
            if "sampled_pose_idx" in self.cfg["env"]:
                sampled_pose_idx = np.ones(len(s_ids), dtype=np.int32) * self.cfg["env"]["sampled_pose_idx"]
            else:
                sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))

            if self.cfg["env"]["random_reset_grasps"]:
                # if len(self.reset_state_replay_buffers) < 1024 or torch.rand(1) >= self.prob_use_reset_state_buffer:
                if True:
                    rand_floats = torch_rand_float(-1.0, 1.0, (len(s_ids), self.num_leap_hand_dofs * 2 + 6), device=self.device)

                    # object position
                    rand_floats[:, 0] *= self.reset_position_noise_scale[0]
                    rand_floats[:, 1] *= self.reset_position_noise_scale[1]
                    rand_floats[:, 2] *= self.reset_position_noise_scale[2]

                    self.root_state_tensor[self.object_indices[s_ids]] = self.object_init_state[s_ids].clone()
                    self.root_state_tensor[self.object_indices[s_ids], 0:3] = self.object_init_state[s_ids, 0:3] + \
                                                                                self.reset_position_noise * rand_floats[:, 0:3]
                    
                    # object orientation
                    # for axis=z, rpy=(noise, noise, -pi~pi)
                    # for axis=y, rpy=(noise, [-pi/2, 0, pi/2, pi], noise)
                    # for axis=x, rpy=([-pi/2, 0, pi/2, pi], noise, noise)
                    rand_floats[: 3:6] *= np.pi
                    random_rot_pi = (torch_rand_float(0, 4, (len(s_ids), 1), device=self.device).squeeze().to(torch.long)-2) * torch.pi / 2
                    if self.flip_axis == "z":
                        new_object_rot = quat_from_euler_xyz(0.025*rand_floats[:, 3], 0.025*rand_floats[:, 4], rand_floats[:, 5])
                    elif self.flip_axis == "y":
                        new_object_rot = quat_from_euler_xyz(0.025*rand_floats[:, 3], random_rot_pi, 0.1*rand_floats[:, 5])
                    elif self.flip_axis == "x":
                        new_object_rot = quat_from_euler_xyz(random_rot_pi, 0.025*rand_floats[:, 4], 0.1*rand_floats[:, 5])

                    self.root_state_tensor[self.object_indices[s_ids], 3:7] = new_object_rot
                    self.root_state_tensor[self.object_indices[s_ids], 7:13] = torch.zeros_like(
                        self.root_state_tensor[self.object_indices[s_ids], 7:13])
                    
                    # hand dof
                    delta_max = self.leap_hand_dof_upper_limits - self.leap_hand_default_dof_pos
                    delta_min = self.leap_hand_dof_lower_limits - self.leap_hand_default_dof_pos
                    # rand_floats[:, 6:6 + self.num_leap_hand_dofs] = torch.abs(rand_floats[:, 6:6 + self.num_leap_hand_dofs])
                    rand_delta = delta_min[s_ids, :] + (delta_max[s_ids, :] - delta_min[s_ids, :]) * rand_floats[:, 6:6 + self.num_leap_hand_dofs]

                    pos = self.leap_hand_default_dof_pos[s_ids, :] + self.reset_dof_pos_noise * rand_delta
                    self.leap_hand_dof_pos[s_ids, :] = pos
                    self.leap_hand_dof_vel[s_ids, :] = self.leap_hand_default_dof_vel + \
                                                        self.reset_dof_vel_noise * rand_floats[:,
                                                                                    6 + self.num_leap_hand_dofs:6 + self.num_leap_hand_dofs * 2]
                    self.prev_targets[s_ids, :self.num_leap_hand_dofs] = pos
                    self.cur_targets[s_ids, :self.num_leap_hand_dofs] = pos

                    self.init_pose_buf[s_ids, :] = pos.clone()
                    self.object_init_pose_buf[s_ids, :] = self.root_state_tensor[self.object_indices[s_ids], :7].clone()
                else:
                    sampled_pose = torch.zeros((len(s_ids), 23), device=self.device, dtype=torch.float)
                    # sampled_pose = self.reset_state_replay_buffers.sample(len(s_ids))
                    self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, 16:]
                    self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
                    pos = sampled_pose[:, :16]
                    self.leap_hand_dof_pos[s_ids, :] = pos
                    self.leap_hand_dof_vel[s_ids, :] = 0
                    self.prev_targets[s_ids, :self.num_leap_hand_dofs] = pos
                    self.cur_targets[s_ids, :self.num_leap_hand_dofs] = pos
                    self.init_pose_buf[s_ids, :] = pos.clone()
                    self.object_init_pose_buf[s_ids, :] = sampled_pose[:, 16:].clone()
            else:
                sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()
                self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, 16:]
                self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
                pos = sampled_pose[:, :16]
                self.leap_hand_dof_pos[s_ids, :] = pos
                self.leap_hand_dof_vel[s_ids, :] = 0
                self.prev_targets[s_ids, :self.num_leap_hand_dofs] = pos
                self.cur_targets[s_ids, :self.num_leap_hand_dofs] = pos
                self.init_pose_buf[s_ids, :] = pos.clone()
                self.object_init_pose_buf[s_ids, :] = sampled_pose[:, 16:].clone()

        # (4)
        if self.delay_reset_goal:
            self.reset_target_pose(env_ids)

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                self.goal_object_indices[env_ids],
                                                self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        
        # (5)
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        mask = self.progress_buf[env_ids] > 0
        self.object_angvel_finite_diff_ep_buf.extend(list(self.object_angvel_finite_diff_mean[env_ids][mask]))
        self.object_angvel_finite_diff_mean[env_ids] = 0

        if "print_object_angvel" in self.cfg["env"] and len(self.object_angvel_finite_diff_ep_buf) > 0:
            print("mean object angvel: ", sum(self.object_angvel_finite_diff_ep_buf) / len(self.object_angvel_finite_diff_ep_buf))

        self.progress_buf[env_ids] = 0
        # self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        # self.at_reset_buf[env_ids] = 1
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.consecutive_successes[env_ids] = 0
        
    def get_joint_noise(self):
        tensor = torch.zeros_like(self.leap_hand_dof_pos)

        if "joint_noise" not in self.cfg["env"]["randomization"]:
            return tensor

        if not self.joint_noise_cfg["add_noise"]:
            return tensor

        if "iid" in self.joint_noise_cfg:
            if self.joint_noise_iid_type == "gaussian":
                tensor = tensor + torch.randn_like(tensor) * self.joint_noise_iid_scale
            elif self.joint_noise_iid_type == "uniform":
                tensor = tensor + (2 * torch.rand(tensor) - 1) * self.joint_noise_iid_scale
            
        if "constant_offset" in self.joint_noise_cfg:
            tensor = tensor + self.joint_noise_constant_offset

        if "outlier" in self.joint_noise_cfg:
            outlier_noise_prob = self.joint_noise_outlier_rate * self.control_dt 
            outlier_mask = torch.rand_like(outlier_noise_prob) <= outlier_noise_prob
            
            if self.joint_noise_outlier_type == "gaussian":
                tensor = tensor + torch.randn_like(tensor) * self.joint_noise_outlier_scale * outlier_mask
            elif self.joint_noise_outlier_type == "uniform":
                tensor = tensor + (2 * torch.rand(tensor) - 1) * self.joint_noise_outlier_scale * outlier_mask

        return tensor
    
    def compute_full_observations(self, no_vel=False):
        scaled_dof_pos = unscale(
            self.leap_hand_dof_pos_noisy,
            self.leap_hand_dof_lower_limits,
            self.leap_hand_dof_upper_limits
        )
        # self.object_pose[:, :3] += torch_rand_float(-0.01, 0.01, (self.object_pose.shape[0], 3), device=self.device)
        # self.object_pose[:, 3:] += torch_rand_float(-0.1, 0.1, (self.object_pose.shape[0], 4), device=self.device)
        self.object_rot = self.object_pose[:, 3:7]
        quat_dist = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

        if no_vel:
            out = torch.cat(
                [
                    scaled_dof_pos,
                    self.object_pose,
                    self.goal_rot,
                    quat_dist,
                    self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips),
                    self.actions
                ],
                dim=-1
            )
        else:
            # 'robust' means success after the 1st trial with seed=123
            # self.fingertip_state[0, :, 7:10] += torch_rand_float(-0.02, 0.02, (4, 3), device=self.device)
            # self.fingertip_state[0, :, 10:13] += torch_rand_float(-0.1, 0.1, (4, 3), device=self.device)
            import pdb; pdb.set_trace()
            out = torch.cat(
                [
                    scaled_dof_pos,  # robust under perturbation ( +U[-0.1, 0.1])
                    self.vel_obs_scale * self.leap_hand_dof_vel,   # robust under perturbation ( + torch_rand_float(-0.05, 0.05, self.leap_hand_dof_vel.shape, device=self.device))
                    # robust under perturbation ( +U[-0.01,0.01] for xyz and +U[-0.1,0.1] for quat)
                    self.object_pose,
                    self.object_linvel,  # robust under perturbation ( + torch_rand_float(-0.02, 0.02, self.object_linvel.shape, device=self.device))
                    self.vel_obs_scale * self.object_angvel,    # robust under perturbation ( + torch_rand_float(-0.1, 0.1, self.object_angvel.shape, device=self.device))
                    self.goal_rot,
                    quat_dist,
                    # robust under xyz perturbation ( + torch_rand_float(-0.05, 0.05, (self.fingertip_state.shape[1], 3), device=self.device))
                    # robust under quat perturbation ( + torch_rand_float(-0.2, 0.2, (self.fingertip_state.shape[1], 4), device=self.device))
                    # robust under perturbation ( +U[-0.02,0.02] for linvel and +U[-0.1,0.1] for angvel)
                    self.fingertip_state.reshape(self.num_envs, 13 * self.num_fingertips),
                    # robust under perturbation ( + torch_rand_float(-0.5, 0.5, self.actions.shape, device=self.device))
                    self.actions
                ],
                dim=-1
            )
        return out

    def compute_observations(self):
        # print("> > > compute_observations called")

        self._refresh_gym()

        # deal with normal observation, do sliding window
        joint_noise_matrix = self.get_joint_noise()
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.leap_hand_dof_pos, self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        self.leap_hand_dof_pos_noisy = cur_obs_buf.squeeze(1).clone()

        self.obs_buf = self.compute_full_observations().clone().squeeze(1)

        if hasattr(self, "obs_list"):
            self.obs_list.append(self.leap_hand_dof_pos.clone())
            self.obs_full_list.append(self.obs_buf.clone())
            self.target_list.append(self.cur_targets[0].clone().squeeze())

            print("global counter: ", self.global_counter)
            if self.global_counter == self.record_duration - 1:
                self.obs_list = torch.stack(self.obs_list, dim=0)
                self.obs_list = self.obs_list.cpu().numpy()

                self.obs_full_list = torch.stack(self.obs_full_list, dim=0)
                self.obs_full_list = self.obs_full_list.cpu().numpy()

                self.target_list = torch.stack(self.target_list, dim=0)
                self.target_list = self.target_list.cpu().numpy()

                if "actions_file" in self.cfg["env"]["debug"]:
                    actions_file = os.path.basename(self.cfg["env"]["debug"]["actions_file"])
                    folder = os.path.dirname(self.cfg["env"]["debug"]["actions_file"])
                    suffix = "_".join(actions_file.split("_")[1:])
                    joints_file = os.path.join(folder, "joints_sim_{}".format(suffix)) 
                    target_file = os.path.join(folder, "targets_sim_{}".format(suffix))
                else:
                    suffix = self.cfg["env"]["debug"]["record"]["suffix"]
                    joints_file = "debug/joints_sim_{}.npy".format(suffix)
                    obs_full_file = "debug/obs_full_sim_{}.npy".format(suffix)
                    target_file = "debug/targets_sim_{}.npy".format(suffix)

                np.save(joints_file, self.obs_list)
                np.save(obs_full_file, self.obs_full_list)
                np.save(target_file, self.target_list) 
                exit()

        if self.cfg["env"]["obs_mask"] is not None:
            self.obs_buf = self.obs_buf * torch.tensor(self.cfg["env"]["obs_mask"], device=self.device)[None, :]

    def compute_reward(self, actions):
        """
            - sparse task reward (rot_dist <= 0.4)
            - dense task reward
            - keep fingertip close to the object
            - energy reward
            - penalty for pushing the object away (object fallen penalty)
        """
        # print("> > > compute_reward called")
        
        """
            - reset_buf: [input>>] all zero | [>>output] fallen + timeout
            - reset_goal_buf: [input>>] all zero | [>>output] goal reach
            - done_buf: [input>>] N/A | [>>output] fallen + goal reach + timeout
        """

        res = compute_leaphand_reward(
            self.reset_buf, self.reset_goal_buf, self.progress_buf,
            # self.successes,
            self.max_episode_length,
            self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.reward_cfg, self.actions,
            self.fingertip_pos, self.fingertip_vel, self.ftip_pos_mask,
            self.object_linvel, self.object_angvel,
            self.leap_hand_dof_vel, self.torques,
            dof_pos=self.leap_hand_dof_pos, target_dof_pos=self.init_pose_buf, dof_pos_mask=self.dof_pos_mask,
            palm_cf=self.palm_contact_force if self.cfg['env']['reward']['pen_palm_contact'] else None
        )
        
        self.rew_buf[:] = res[0] * self.cfg['env']['rew_scale']
        self.done_buf[:] = res[1]
        self.reset_buf[:] = res[2]
        self.reset_goal_buf[:] = res[3]
        self.progress_buf[:] = res[4]
        self.successes[:] = res[5]
        abs_rot_dist = res[6]
        abs_pos_dict = res[7]
        reward_terms = res[8]
        timeout_envs = res[9]

        # compute consecutive successes
        self.consecutive_successes = self.consecutive_successes + self.successes
        max_consecutive_successes_reached = (self.consecutive_successes >= self.max_consecutive_successes)
        self.reset_buf = torch.where(max_consecutive_successes_reached, \
                                        torch.ones_like(self.reset_buf), self.reset_buf)

        # compute mean reward values for logging (add prefix 'rew' to enable wandb logging)
        self.extras['rew_rot_reward'] = reward_terms['rot_reward'].mean()
        self.extras['rew_pos_reward'] = reward_terms['pos_reward'].mean()
        self.extras['rew_dof_pos_reward'] = reward_terms['dof_pos_reward'].mean()
        self.extras['rew_ftip_reward'] = reward_terms['ftip_reward'].mean()
        self.extras['rew_energy_reward'] = reward_terms['energy_reward'].mean()
        self.extras['rew_object_fallen'] = reward_terms['object_fallen'].mean()

        self.extras['success'] = self.reset_goal_buf.detach().to(self.rl_device).flatten()
        self.extras['consecutive_success'] = self.consecutive_successes.detach().to(self.rl_device).flatten()
        self.extras['abs_rot_dist'] = abs_rot_dist.detach().to(self.rl_device)
        self.extras['abs_pos_dist'] = abs_pos_dict.detach().to(self.rl_device)
        self.extras['TimeLimit.truncated'] = timeout_envs.detach().to(self.rl_device)
        for reward_key, reward_val in reward_terms.items():
            if reward_key == 'object_fallen':
                self.extras[reward_key] = reward_val.detach()
    
    def post_physics_step(self):
        # print("> > post_physics_step called")

        self.progress_buf += 1

        # self.reset_buf[:] = 0
        # self.early_termination_buf[:] = 0
        # self._refresh_gym()

        self.compute_reward(self.actions)
        self.compute_observations()
        
        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
                
            self.plot_callback()

    def plot_callback(self):
        self.fig.canvas.restore_region(self.bg)

        # self.ydata.append(self.object_rpy[0, 2].item())
        self.ydata.append(self.object_angvel_finite_diff[0, 2].item())
        self.ydata2.append(self.object_rpy[0, 2].item())

        self.ln.set_ydata(list(self.ydata))
        self.ln.set_xdata(range(len(self.ydata)))

        self.ln2.set_ydata(list(self.ydata2))
        self.ln2.set_xdata(range(len(self.ydata2)))

        self.ax.draw_artist(self.ln)
        self.ax.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        """
            (1) reset pose and/or goal pose
            (2) set actions without apply
            (3) apply random forces
        """
        # print("> > pre_physics_step called")

        # (1)
        # reset and reset goals
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.global_counter += 1
        
        # (2)
        if hasattr(self, "actions_list"):
            actions = self.actions_list[self.global_counter-1].repeat((self.num_envs, 1))

        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions.clone().to(self.device)
        self.actions *= self.actions_mask

        targets = self.prev_targets + 1 / 24 * self.actions
        self.cur_targets[:] = tensor_clamp(targets, self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits)
        
        # Code for debugging joint angles
        # self.cur_targets = torch.zeros_like(self.cur_targets)
        # self.cur_targets[:, 5] = math.sin(self.global_counter / 20 * 2 * math.pi / 2)
        # self.cur_targets = scale(self.cur_targets, self.leap_hand_dof_upper_limits, self.leap_hand_dof_lower_limits)
        
        self.prev_targets[:] = self.cur_targets.clone()

        if self.enable_joint_limits_report:
            near_joint_limits = self.LEAPhand_limits_check(self.cur_targets.clone().flatten())
            if near_joint_limits:
                print("WARNING: leap hand near joint limits!")

        # (3)
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object_cube'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def reset(self):
        # print("> reset called")

        self.reset_buf.fill_(1)
        self.reset_goal_buf.fill_(1)

        super().reset()
        return self.obs_dict
    
    def construct_sim_to_real_transformation(self):
        self.sim_dof_order = self.gym.get_actor_dof_names(self.envs[0], 0)
        self.sim_dof_order = [int(x) for x in self.sim_dof_order]
        self.real_dof_order = list(range(16))
        self.sim_to_real_indices = [] # Value at i is the location of ith real index in the sim list

        for x in self.real_dof_order:
            self.sim_to_real_indices.append(self.sim_dof_order.index(x))
        
        self.real_to_sim_indices = []

        for x in self.sim_dof_order:
            self.real_to_sim_indices.append(self.real_dof_order.index(x))
        
        import pdb; pdb.set_trace()
        assert(self.sim_to_real_indices == self.cfg["env"]["sim_to_real_indices"])
        assert(self.real_to_sim_indices == self.cfg["env"]["real_to_sim_indices"])

    def real_to_sim(self, values):
        if not hasattr(self, "sim_dof_order"):
            self.construct_sim_to_real_transformation()

        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_dof_order"):
            self.construct_sim_to_real_transformation()
        
        return values[:, self.sim_to_real_indices]

    def update_low_level_control(self):
        previous_dof_pos = self.leap_hand_dof_pos.clone()
        self._refresh_gym()      
        if os.getenv("RVIZ") is None and not self.cfg["env"]["disable_actions"]:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def check_termination(self, object_pos):
        resets = torch.logical_or(
            torch.less(object_pos[:, -1], self.reset_z_threshold),
            torch.greater_equal(self.progress_buf, self.max_episode_length),
        )

        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.hand_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # update fingertip information
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_vel = self.rigid_body_states[:, self.fingertip_handles][:, :, 7:13]

        # update goal information
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        if self.prev_global_counter != self.global_counter: # This is required since sometimes _refresh_gym is called multiple times within same step
            # TODO(yongpeng): make sure the anular velocity computation here avoids singularity problem
            new_object_roll, new_object_pitch, new_object_yaw = euler_from_quaternion(self.object_rot)
            new_object_rpy = torch.stack((new_object_roll, new_object_pitch, new_object_yaw), dim=1) 
            delta_counter = self.global_counter - self.prev_global_counter
            self.object_rpy = new_object_rpy
            self.prev_global_counter = self.global_counter
            
            dr, dp, dy = euler_from_quaternion(quat_mul(self.object_rot, quat_conjugate(self.previous_object_rot)))
            self.object_angvel_finite_diff = torch.stack([dr, dp, dy], dim=-1)
            self.object_angvel_finite_diff /= (self.control_dt * delta_counter)
            self.previous_object_rot = self.object_rot.clone() 

        if "phase_period" in self.cfg["env"]:
            omega = 2 * math.pi / self.cfg["env"]["phase_period"]
            phase_angle = (self.progress_buf - 1) * self.control_dt * omega
            self.phase = torch.stack([torch.sin(phase_angle), torch.cos(phase_angle)], dim=-1)

    def _setup_domain_rand_cfg(self, rand_cfg):
        self.randomize_mass = rand_cfg['randomizeMass']
        self.randomize_mass_lower = rand_cfg['randomizeMassLower']
        self.randomize_mass_upper = rand_cfg['randomizeMassUpper']
        self.randomize_com = rand_cfg['randomizeCOM']
        self.randomize_com_lower = rand_cfg['randomizeCOMLower']
        self.randomize_com_upper = rand_cfg['randomizeCOMUpper']
        self.randomize_friction = rand_cfg['randomizeFriction']
        self.randomize_friction_lower = rand_cfg['randomizeFrictionLower']
        self.randomize_friction_upper = rand_cfg['randomizeFrictionUpper']
        self.randomize_scale = rand_cfg['randomizeScale']
        self.scale_list_init = rand_cfg['scaleListInit']
        self.randomize_scale_list = rand_cfg['randomizeScaleList']
        self.randomize_scale_lower = rand_cfg['randomizeScaleLower']
        self.randomize_scale_upper = rand_cfg['randomizeScaleUpper']
        self.randomize_pd_gains = rand_cfg['randomizePDGains']
        self.randomize_p_gain_lower = rand_cfg['randomizePGainLower']
        self.randomize_p_gain_upper = rand_cfg['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_cfg['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_cfg['randomizeDGainUpper']

    def _setup_priv_option_cfg(self, p_cfg):
        self.enable_priv_obj_position = p_cfg['enableObjPos']
        self.enable_priv_obj_mass = p_cfg['enableObjMass']
        self.enable_priv_obj_scale = p_cfg['enableObjScale']
        self.enable_priv_obj_com = p_cfg['enableObjCOM']
        self.enable_priv_obj_friction = p_cfg['enableObjFriction']

    def _setup_object_info(self, o_cfg):
        self.object_type = o_cfg['type']
        raw_prob = o_cfg['sampleProb']
        assert (sum(raw_prob) == 1)

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'simple_tennis_ball': 'assets/ball.urdf',
            'cube': 'assets/cube.urdf',
            'cube_small': 'assets/cube_50mm.urdf'
        }
        for p_id, prim in enumerate(primitive_list):
            if 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'../assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))]
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', 'assets/')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'../assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', 'assets/')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(self.object_type_list)
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _setup_reward_cfg(self, r_cfg):
        pass

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.cfg['env']['asset']['handAsset']
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False        # True -> False to enable xxx_tip_center frames
        hand_asset_options.disable_gravity = False
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        # Convex decomposition
        hand_asset_options.vhacd_enabled = True
        hand_asset_options.vhacd_params.resolution = 300000
        # hand_asset_options.vhacd_params.max_convex_hulls = 30
        # hand_asset_options.vhacd_params.max_num_vertices_per_ch = 64

        hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)
        
        if "leap_hand" in hand_asset_file:
            rsp = self.gym.get_asset_rigid_shape_properties(self.hand_asset)   

            for i, (_, body_group) in enumerate(self.cfg["env"]["mask_body_collision"].items()):
                filter_value = 2 ** i

                for body_idx in body_group:
                    start, count = self.body_shape_indices[body_idx]
                    
                    for idx in range(count):
                        rsp[idx + start].filter = rsp[idx + start].filter | filter_value 

            if self.cfg["env"]["disable_self_collision"]: # Disable all collisions
                for i in range(len(rsp)):
                    rsp[i].filter = 1

            self.gym.set_asset_rigid_shape_properties(self.hand_asset, rsp)

        self._get_leap_asset()

        # load object asset
        self.object_asset_list = []
        self.goal_asset_list = []

        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()

            if self.cfg["env"]["disable_gravity"]:
                object_asset_options.disable_gravity = True

            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)

            object_asset_options.disable_gravity = True
            goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.goal_asset_list.append(goal_asset)

        print("Finished loading object assets!")

    def _init_object_pose(self):
        assert len(self.object_type_list) == 1
        object_type = self.object_type_list[0]

        leap_hand_start_pose = gymapi.Transform()
        leap_hand_start_pose.p = gymapi.Vec3(0, 0, self.cfg["env"]["leap_hand_start_z"])

        leap_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi) 
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = leap_hand_start_pose.p.x
        pose_dx, pose_dy, pose_dz = -0.01, -0.04, 0.15

        override_cfg = self.cfg["env"]["override_dist"]

        if "override_object_init_x" in override_cfg[object_type]:
            pose_dx = override_cfg[object_type]["override_object_init_x"]
            print("override object init x with delta: {}".format(pose_dx))

        if "override_object_init_y" in override_cfg[object_type]:
            pose_dy = override_cfg[object_type]["override_object_init_y"]
            print("override object init y with delta: {}".format(pose_dy))

        object_start_pose.p.x = leap_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = leap_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = leap_hand_start_pose.p.z + pose_dz

        # for grasp pose generation, it is used to initialize the object
        # it should be slightly higher than the fingertip
        # so it is set to be 0.66 for internal leap and 0.64 for the public leap
        # ----
        # for in-hand object rotation, the initialization of z is only used in the first step
        # it is set to be 0.65 for backward compatibility
        object_z = 0.66 if self.save_init_pose else 0.65
        if 'internal' not in self.grasp_cache_name:
            object_z -= 0.02
        object_start_pose.p.z = object_z

        if "override_object_init_z" in override_cfg[object_type]:
            object_start_pose.p.z = override_cfg[object_type]["override_object_init_z"] 
            print("override object init z with: {}".format(object_start_pose.p.z))

        return leap_hand_start_pose, object_start_pose

    def LEAPhand_limits_check(self, joints):
        near_upper = torch.less(torch.abs(joints-self.leap_hand_dof_upper_limits), 0.002).any()
        near_lower = torch.less(torch.abs(joints-self.leap_hand_dof_lower_limits), 0.002).any()

        return torch.logical_or(near_upper, near_lower)

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_hand_dof_lower_limits).squeeze().cpu().numpy()
        sim_max = self.sim_to_real(self.leap_hand_dof_upper_limits).squeeze().cpu().numpy()
        
        return sim_min, sim_max

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)

        return joints
    
    def LEAPhand_to_LEAPsim(self, joints):
        joints = np.array(joints)
        ret_joints = joints - 3.14159
        
        return ret_joints

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def quaternion_from_euler(euler_angle):
    """
        This is the reverse operation of euler_from_quaternion
        (reference: https://blog.csdn.net/xiaoma_bk/article/details/79082629)
    """
    roll = euler_angle[:,0]; pitch = euler_angle[:,1]; yaw = euler_angle[:,2]
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)

    quat_w = cy * cp * cr + sy * sp * sr
    quat_x = cy * cp * sr - sy * sp * cr
    quat_y = sy * cp * sr + cy * sp * cr
    quat_z = sy * cp * cr - cy * sp * sr

    return quat_w, quat_x, quat_y, quat_z

def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower)/(upper - lower)

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
