import torch.cuda

# from isaacgymenvs.utils.torch_jit_utils import *
from isaacgym.torch_utils import *


# @torch.jit.script
def compute_reward(
    reset_buf,
    progress_buf,
    max_episode_length: float,
    object_pos,
    target_pos,
    object_rot,
    actions,
    object_linvel,
    object_linvel_penalty_scale: float,
    object_angvel,
    object_angvel_finite_diff,
    angvel_lower_bound: float,
    angvel_upper_bound: float,
    rotation_axis,
    dof_vel,
    dof_vel_finite_diff,
    dof_torque,
    dof_pos,
    target_dof_pos,
    dof_pos_mask,
    rot_reward_scale: float,
    dof_pos_reward_scale: float,
    fall_dist: float,
    fall_penalty: float,
    success_tolerance_pos: float,
    energy_scale: float,
    torque_scale: float,
    reset_when_tilted: bool,
    reset_warm_start_length: int,
    reset_roll_pitch_thresh: float,
):
    num_envs = object_pos.shape[0]

    # goal mask controls the consideration of (x, y, z) in goal_dist calculation
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    roll_x, pitch_y, yaw_z = get_euler_xyz(object_rot)
    roll_x, pitch_y, yaw_z = roll_x.unsqueeze(-1), pitch_y.unsqueeze(-1), yaw_z.unsqueeze(-1)
    object_euler = torch.concatenate((roll_x, pitch_y, yaw_z), dim=-1)

    # fingertip-object distance
    reward_terms = dict()

    # rotate reward
    angvel_clipped = torch.clip(object_angvel_finite_diff[:, 2], min=angvel_lower_bound, max=angvel_upper_bound)
    angvel_clipped *= rotation_axis[:, 2]
    reward_terms["rot_reward"] = rot_reward_scale * angvel_clipped

    # object linvel penalty
    object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)
    reward_terms["obj_linvel_penalty"] = object_linvel_penalty * object_linvel_penalty_scale

    # dof pos diff penalty
    if dof_pos_mask is None:
        dof_pos_mask = torch.ones((num_envs, 1), dtype=torch.float, device=target_dof_pos.device)
    else:
        dof_pos_mask = torch.where(
                        dof_pos_mask > 0,
                        torch.ones_like(dof_pos_mask),
                        torch.zeros_like(dof_pos_mask)
                    ).to(target_dof_pos.device).repeat(num_envs, 1)
    dof_pos_diff = (((dof_pos - target_dof_pos) * dof_pos_mask) ** 2).sum(-1)
    dof_pos_diff_rew = dof_pos_diff * dof_pos_reward_scale
    reward_terms["dof_pos_reward"] = dof_pos_diff_rew

    # energy penalty
    energy_cost = ((dof_vel_finite_diff * dof_torque).sum(-1)) ** 2
    reward_terms["energy_reward"] = energy_cost * energy_scale

    # dof torque penalty
    torque_cost = (dof_torque ** 2).sum(-1)
    reward_terms["torque_reward"] = torque_cost * torque_scale

    # reset if object falls
    fall_envs = goal_dist >= fall_dist
    # reset if object deviates large from palm center
    moveout_envs = goal_dist >= success_tolerance_pos
    fall_envs = torch.where(moveout_envs, torch.ones_like(fall_envs), fall_envs)
    # reset if object tilt too much (large roll/pitch detected)
    if reset_when_tilted:
        reset_tilted_condition = progress_buf >= reset_warm_start_length
        roll_exceed_limit = torch.min(
            torch.concatenate((torch.abs(roll_x), torch.abs(roll_x - 2*torch.pi)), dim=-1), dim=-1
        )[0] >= reset_roll_pitch_thresh
        pitch_exceed_limit = torch.min(
            torch.concatenate((torch.abs(pitch_y), torch.abs(pitch_y - 2*torch.pi)), dim=-1), dim=-1
        )[0] >= reset_roll_pitch_thresh
        tilt_envs = torch.logical_and(
            torch.logical_or(roll_exceed_limit, pitch_exceed_limit), reset_tilted_condition
        )
        fall_envs = torch.where(tilt_envs, torch.ones_like(fall_envs), fall_envs)

    dones = fall_envs.clone()

    resets = torch.where(fall_envs, torch.ones_like(reset_buf), reset_buf)

    reward_terms["object_fallen"] = fall_envs.float() * fall_penalty

    reward = torch.sum(torch.stack(list(reward_terms.values())), dim=0)
    reward = torch.where(fall_envs, reward + fall_penalty, reward)
    time_due_envs = progress_buf >= max_episode_length - 1
    # resets: 1) reset_buf, 2) time due, 3) fall, 4) goal reach
    resets = torch.where(time_due_envs, torch.ones_like(resets), resets)
    # dones: 1) goal reach, 2) fall (include move far from palm center), 3) time due
    dones = torch.logical_or(dones, time_due_envs)

    # regularize roll, pitch to [-pi, pi]
    abs_roll_x = torch.abs(torch.where(roll_x > torch.pi, roll_x - 2 * torch.pi, roll_x))
    abs_pitch_y = torch.abs(torch.where(pitch_y > torch.pi, pitch_y - 2 * torch.pi, pitch_y))

    return (
        reward,
        dones.int(),
        resets,
        progress_buf,
        abs_roll_x,
        abs_pitch_y,
        reward_terms,
        time_due_envs,
    )


@torch.no_grad()
def compute_leaphand_reward(
    reset_buf,
    progress_buf,
    max_episode_length: float,
    object_pos,
    target_pos,
    object_rot,
    reward_cfg,
    actions,
    object_linvel,
    object_angvel,
    object_angvel_finite_diff,
    rotation_axis,
    dof_vel,
    dof_vel_finite_diff,
    dof_torque,
    dof_pos,
    target_dof_pos,
    dof_pos_mask=None,
):
    rot_reward_scale = reward_cfg["rotate_finite_diff"]
    fall_dist = reward_cfg["fallDistance"]
    fall_penalty = reward_cfg["object_fallen"]
    success_tolerance_pos = reward_cfg["successTolerancePos"]
    dof_pos_reward_scale = reward_cfg["poseDiffPenaltyScale"]
    torque_penalty_scale = reward_cfg["torquePenaltyScale"]
    work_penalty_scale = reward_cfg["workPenaltyScale"]

    object_linvel_penalty_scale = reward_cfg["objLinvelPenaltyScale"]
    angvel_lower_bound = reward_cfg["angvelClipMin"]
    angvel_upper_bound = reward_cfg["angvelClipMax"]
    reset_when_tilted = reward_cfg["resetWhenTilted"]
    reset_warm_start_length = reward_cfg["resetWarmStartLength"]
    reset_roll_pitch_thresh = reward_cfg["resetRollPitchThresh"]
    kwargs = dict(
        reset_buf=reset_buf,
        progress_buf=progress_buf,
        max_episode_length=max_episode_length,
        object_pos=object_pos,
        target_pos=target_pos,
        object_rot=object_rot,
        actions=actions,
        object_linvel=object_linvel,
        object_linvel_penalty_scale=object_linvel_penalty_scale,
        object_angvel=object_angvel,
        object_angvel_finite_diff=object_angvel_finite_diff,
        angvel_lower_bound=angvel_lower_bound,
        angvel_upper_bound=angvel_upper_bound,
        rotation_axis=rotation_axis,
        dof_vel=dof_vel,
        dof_vel_finite_diff=dof_vel_finite_diff,
        dof_torque=dof_torque,
        dof_pos=dof_pos,
        target_dof_pos=target_dof_pos,
        dof_pos_mask=dof_pos_mask,
        rot_reward_scale=rot_reward_scale,
        dof_pos_reward_scale=dof_pos_reward_scale,
        fall_dist=fall_dist,
        fall_penalty=fall_penalty,
        success_tolerance_pos=success_tolerance_pos,
        energy_scale=work_penalty_scale,
        torque_scale=torque_penalty_scale,
        reset_when_tilted=reset_when_tilted,
        reset_warm_start_length=reset_warm_start_length,
        reset_roll_pitch_thresh=reset_roll_pitch_thresh,
    )
    out = compute_reward(**kwargs)
    return out
