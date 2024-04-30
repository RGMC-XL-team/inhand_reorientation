# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/HaozhiQi/hora/blob/main/hora/tasks/leap_hand_grasp.py
# --------------------------------------------------------

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    tensor_clamp,
    to_torch,
    torch_rand_float,
)
from scipy.spatial.transform import Rotation as R

from leapsim.tasks.leap_hand_rot import LeapHandRot


class LeapHandGrasp(LeapHandRot):
    def __init__(
        self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=None, force_render=None
    ):
        cfg["env"]["forceScale"] = 2.0

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless)
        self.saved_grasping_states = torch.zeros((0, 23), dtype=torch.float, device=self.device)
        self.parse_handles()

        if "canonical_pose" in cfg["env"]:
            self.canonical_pose = cfg["env"]["canonical_pose"]
            print("use custom canonical pose: ", self.canonical_pose)
        else:
            self.canonical_pose = [
                0.082,
                1.244,
                0.265,
                0.298,
                1.104,
                1.163,
                0.953,
                -0.138,
                0.005,
                1.096,
                0.080,
                0.150,
                0.029,
                1.337,
                0.285,
                0.317,
            ]

        if "num_contact_fingers" in cfg["env"]:
            self.num_contact_fingers = cfg["env"]["num_contact_fingers"]
        else:
            self.num_contact_fingers = 2

        if "finger_dist_threshold" in cfg["env"]:
            self.finger_dist_threshold = cfg["env"]["finger_dist_threshold"]
        else:
            self.finger_dist_threshold = 0.1

        if "finger_above_obj_center_z" in cfg["env"]:
            self.finger_above_obj_center_z = cfg["env"]["finger_above_obj_center_z"]
        else:
            self.finger_above_obj_center_z = [-0.1, 0.1]

        if "finger_near_obj_min_num" in cfg["env"]:
            self.finger_near_obj_min_num = cfg["env"]["finger_near_obj_min_num"]
        else:
            self.finger_near_obj_min_num = 4

        if "random_xy_bias_limit" in cfg["env"]:
            self.random_xy_bias_limit = cfg["env"]["random_xy_bias_limit"]
        else:
            self.random_xy_bias_limit = 0.0

        if "fingertip_dist_min" in cfg["env"]:
            self.fingertip_dist_min = cfg["env"]["fingertip_dist_min"]
        else:
            self.fingertip_dist_min = 0.0

        if "grasp_cache_len" not in self.cfg["env"]:
            self.cfg["env"]["grasp_cache_len"] = 5e4

        self.random_reset_method = self.cfg["env"]["randomGraspMethod"]
        assert self.random_reset_method in ["euler_angle", "euler_angle_for_flip", "euler_angle_for_rot", "original", "rot_free_yaw"]

        # self.fix_reset_quat = self.cfg['env']['fix_reset_quat']

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def parse_handles(self):
        rb_links = self.gym.get_asset_rigid_body_names(self.hand_asset)
        self.fingertips = [x for x in rb_links if 'tip_center' in x]  # ["finger1_tip_center", "thumb_tip_center", "finger2_tip_center", "finger3_tip_center"]

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in
                                  self.fingertips]

    def reset_idx(self, env_ids, goal_env_ids=None):
        if self.randomize_mass:
            lower, upper = self.randomize_mass_lower, self.randomize_mass_upper
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, "object")
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                for p in prop:
                    p.mass = np.random.uniform(lower, upper)
                self.gym.set_actor_rigid_body_properties(env, handle, prop)
        else:
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, "object")
                prop = self.gym.get_actor_rigid_body_properties(env, handle)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower,
                self.randomize_p_gain_upper,
                (len(env_ids), self.num_actions),
                device=self.device,
            ).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower,
                self.randomize_d_gain_upper,
                (len(env_ids), self.num_actions),
                device=self.device,
            ).squeeze(1)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_leap_hand_dofs * 2 + 5), device=self.device)

        # generate random euler angle
        rand_rpy = torch.tensor(
            R.random(len(env_ids)).as_euler("xyz", degrees=False), dtype=torch.float, device=self.device
        )

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0
        success = self.progress_buf[env_ids] == self.max_episode_length
        all_states = torch.cat([self.leap_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]], dim=1)
        self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success]])
        print("current cache size:", self.saved_grasping_states.shape[0])
        if len(self.saved_grasping_states) >= self.cfg["env"]["grasp_cache_len"]:
            name = f'cache/{self.grasp_cache_name}_grasp_50k_s{str(self.base_obj_scale).replace(".", "")}.npy'
            np.save(name, self.saved_grasping_states[: self.cfg["env"]["grasp_cache_len"]].cpu().numpy())
            exit()

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2]
        rand_xy_bias = torch_rand_float(-self.random_xy_bias_limit, self.random_xy_bias_limit, (len(env_ids), 2), device=self.device)
        self.root_state_tensor[self.object_indices[env_ids], 0:2] += rand_xy_bias
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[
            env_ids, self.up_axis_idx
        ] - 0.015

        if self.random_reset_method == "original":
            # new_object_rot = randomize_rotation(
            #     rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
            # )
            new_object_rot = torch.zeros((len(env_ids), 4), device=self.device)
            new_object_rot[:] = 0
            new_object_rot[:, -1] = 1
        elif self.random_reset_method == "rot_free_yaw":
            zero_roll = torch.zeros(len(env_ids),)
            zero_pitch = torch.zeros(len(env_ids),)
            random_yaw = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze() * torch.pi
            new_object_rot = quat_from_euler_xyz(zero_pitch, zero_roll, random_yaw)
        elif self.random_reset_method == "euler_angle":
            new_object_rot = randomize_rotation_from_euler(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
        elif self.random_reset_method == "euler_angle_for_rot":
            new_object_rot = randomize_rotation_from_euler(0.05 * rand_rpy[:, 0], 0.05 * rand_rpy[:, 1], rand_rpy[:, 2])
        elif self.random_reset_method == "euler_angle_for_flip":
            new_object_rot = randomize_rotation_from_euler(0.05 * rand_rpy[:, 0], rand_rpy[:, 1], 0.05 * rand_rpy[:, 2])
            delta_rpy = torch_rand_float(0, 4, (len(env_ids), 3), device=self.device).int() * (torch.pi / 2.0)
            delta_rot = randomize_rotation_three_axis(
                delta_rpy[:, 0],
                delta_rpy[:, 1],
                delta_rpy[:, 2],
                self.x_unit_tensor[env_ids],
                self.y_unit_tensor[env_ids],
                self.z_unit_tensor[env_ids],
            )
            new_object_rot = quat_mul(new_object_rot, delta_rot)

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices),
        )

        pos = to_torch(self.canonical_pose, device=self.device)[None].repeat(len(env_ids), 1)
        pos += self.cfg["env"]["grasp_dof_search_radius"] * rand_floats[:, 5 : 5 + self.num_leap_hand_dofs]
        pos = tensor_clamp(pos, self.leap_hand_dof_lower_limits[env_ids], self.leap_hand_dof_upper_limits[env_ids])

        self.leap_hand_dof_pos[env_ids, :] = pos
        self.leap_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, : self.num_leap_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_leap_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids)
            )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_reward(self, actions):
        def list_intersect(li, hash_num):
            # 17 is the object index
            # 4, 8, 12, 16 are fingertip index
            # return number of contact with obj_id
            obj_id = 17
            query_list = [
                obj_id * hash_num + self.fingertip_handles[0],
                obj_id * hash_num + self.fingertip_handles[1],
                obj_id * hash_num + self.fingertip_handles[2],
                obj_id * hash_num + self.fingertip_handles[3]
            ]
            return len(np.intersect1d(query_list, li))

        assert self.device == "cpu"
        contacts = [self.gym.get_env_rigid_contacts(env) for env in self.envs]
        contact_list = [
            list_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000) for contact in contacts
        ]
        contact_condition = to_torch(contact_list, device=self.device)

        obj_pos = self.rigid_body_states[:, [-1], :3]
        object_quat = self.rigid_body_states[:, [-1], 3:7]
        finger_pos = self.rigid_body_states[:, self.fingertip_handles, :3]

        # compute pairwise fingertip distance
        fingertip_pairwise_dist = torch.zeros((self.num_envs, 6), device=self.device)
        pairwise_counter = 0
        for i in range(4):
            for j in range(i+1, 4):
                fingertip_pairwise_dist[:, pairwise_counter] = torch.norm(finger_pos[:, i, :] - finger_pos[:, j, :], dim=-1)
                pairwise_counter += 1

        # the sampled pose need to satisfy (check 1 here):
        # 1) at least {finger_near_obj_min_num} fingertips is nearby objects (distance <= {finger_dist_threshold})
        # cond1 = (torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1)) < self.finger_dist_threshold).all(-1)
        cond1 = (torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1)) < self.finger_dist_threshold).sum(-1) >= self.finger_near_obj_min_num
        # 2) at least two fingers are in contact with object (num_contact_fingers=0 in config)
        cond2 = contact_condition >= self.num_contact_fingers
        # 3) object does not fall after a few iterations
        # 0.645 for internal leap
        # 0.625 for public leap
        cond3 = torch.bitwise_and(
            torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold),
            torch.less(obj_pos[:, -1, -1], 0.65)
        )
        # 4) object's z-axis should point upwards (the object should not tilt too much)
        object_euler = get_euler_xyz(object_quat.squeeze(1))
        cond4 = torch.logical_and(
            torch.logical_or(torch.abs(object_euler[0]) <= 0.1, torch.abs(object_euler[0] - 2 * torch.pi) <= 0.1),
            torch.logical_or(torch.abs(object_euler[1]) <= 0.1, torch.abs(object_euler[1] - 2 * torch.pi) <= 0.1),
        )
        # 5) all fingers should not be too high
        cond5 = torch.logical_and(
            (((obj_pos[..., 2] + self.finger_above_obj_center_z[0]) - finger_pos[..., 2]) <= 0).all(-1),
            (((obj_pos[..., 2] + self.finger_above_obj_center_z[1]) - finger_pos[..., 2]) >= 0).all(-1)
        )
        # 6) all fingertips should not be too near (avoid collisions)
        cond6 = (fingertip_pairwise_dist >= self.fingertip_dist_min).all(-1)

        cond = cond1.float() * cond2.float() * cond3.float() * cond4.float() * cond5.float() * cond6.float()
        # reset if any of the above condition does not hold
        self.reset_buf[cond < 1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def randomize_rotation_three_axis(rand0, rand1, rand2, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    return quat_mul(
        quat_mul(
            quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
        ),
        quat_from_angle_axis(rand2 * np.pi, z_unit_tensor),
    )


@torch.jit.script
def randomize_rotation_from_euler(roll_tensor, pitch_tensor, yaw_tensor):
    return quat_from_euler_xyz(roll_tensor, pitch_tensor, yaw_tensor)
