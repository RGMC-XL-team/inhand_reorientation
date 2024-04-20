import torch.cuda

# from isaacgymenvs.utils.torch_jit_utils import *
from isaacgym.torch_utils import *


# @torch.jit.script
def compute_reward(
    reset_buf,
    reset_goal_buf,
    progress_buf,
    # successes,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    actions,
    fingertip_pos, ftip_pos_mask,
    object_linvel,
    object_angvel,
    dof_vel,
    dof_torque,
    dof_pos,
    target_dof_pos,
    dof_pos_mask,
    rot_reward_scale: float,
    rot_eps: float,
    pos_reward_scale: float,
    pos_eps: float,
    dof_pos_reward_scale: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    success_tolerance: float,
    success_tolerance_pos: float,
    ftip_reward_scale: float,
    energy_scale: float,
    dof_pos_diff_thresh: float,
    dof_vel_thresh: float,
    obj_lin_vel_thresh: float,
    obj_ang_vel_thresh: float,
    action_norm_thresh: float,
    penalize_palm_contact: bool,
    # palm_cf, palm_cf_scale: float,
    clip_energy_reward: bool,
    energy_upper_bound: float,
    ftip_cf, ftip_cf_scale, ftip_cf_reward_scale: float
):
    num_envs = object_pos.shape[0]

    # goal mask controls the consideration of (x, y, z) in goal_dist calculation
    # use the mask [1, 1, 0] will lead to fingers lifting the object up
    goal_mask = torch.tensor([1, 1, 1], device=target_pos.device).repeat(num_envs, 1)
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    masked_goal_dist = torch.norm((object_pos - target_pos) * goal_mask, p=2, dim=-1)

    # fingertip-object distance
    reward_terms = dict()
    if ftip_reward_scale is not None and ftip_reward_scale < 0:
        ftip_pos_mask = torch.tensor(ftip_pos_mask, dtype=torch.float).to(fingertip_pos.device).repeat(num_envs, 1)
        ftip_diff = (fingertip_pos.view(num_envs, -1, 3) - object_pos[:, None, :]) * ftip_pos_mask[:, :, None]
        ftip_dist = torch.linalg.norm(ftip_diff, dim=-1).view(num_envs, -1)
        ftip_dist_mean = ftip_dist.mean(dim=-1)
        ftip_reward = ftip_dist_mean * ftip_reward_scale
        reward_terms['ftip_reward'] = ftip_reward

    # fingertip contact force
    if ftip_cf is not None and ftip_cf_scale is not None:
        ftip_cf_scale = torch.tensor(ftip_cf_scale, dtype=torch.float).to(ftip_cf.device).repeat(num_envs, 1)
        ftip_cf_norm = torch.linalg.norm(ftip_cf, dim=-1).view(num_envs, -1)
        ftip_in_contact = ftip_cf_norm > 0.5
        ftip_cf_reward = (ftip_in_contact * ftip_cf_scale).mean(dim=-1) * ftip_cf_reward_scale
        reward_terms['ftip_cf_reward'] = ftip_cf_reward

    object_linvel_norm = torch.linalg.norm(object_linvel, dim=-1)
    object_angvel_norm = torch.linalg.norm(object_angvel, dim=-1)

    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    abs_rot_dist = torch.abs(rot_dist)
    abs_pos_dist = torch.abs(goal_dist)

    # print("[debug] abs_rot_dist: ", abs_rot_dist.sum().item())

    # rotation reward
    rot_rew = 1.0 / (abs_rot_dist + rot_eps) * rot_reward_scale
    reward_terms["rot_reward"] = rot_rew

    # position reward
    # pos_rew = -masked_goal_dist * pos_reward_scale
    pos_rew = 1.0 / (masked_goal_dist + pos_eps) * pos_reward_scale
    reward_terms["pos_reward"] = pos_rew

    # dof pos diff penalty
    if dof_pos_mask is None:
        dof_pos_mask = torch.zeros((num_envs, 1), dtype=torch.float, device=target_dof_pos.device)
    else:
        dof_pos_mask = torch.tensor(dof_pos_mask, dtype=torch.float).to(target_dof_pos.device).repeat(num_envs, 1)
    dof_pos_diff = torch.norm((dof_pos - target_dof_pos) * dof_pos_mask, p=2, dim=-1) ** 2
    dof_pos_diff_rew = dof_pos_diff * dof_pos_reward_scale
    reward_terms["dof_pos_reward"] = dof_pos_diff_rew

    action_norm = torch.linalg.norm(actions, dim=-1)
    energy_cost = torch.abs(dof_vel * dof_torque).sum(dim=-1)
    if clip_energy_reward:
        energy_cost = torch.clamp(energy_cost, max=energy_upper_bound)
    reward_terms["energy_reward"] = -energy_cost * energy_scale

    # if penalize_palm_contact:
    #     in_contact = torch.abs(palm_cf).sum(-1) > 0.5                       # 0.2
    #     reward_terms['palm_contact_reward'] = -in_contact.float() * palm_cf_scale

    dof_vel_norm = torch.linalg.norm(dof_vel, dim=-1)

    goal_reach = (
        (abs_rot_dist <= success_tolerance)
        & (abs_pos_dist <= success_tolerance_pos)
        & (dof_vel_norm <= dof_vel_thresh)
        & (object_linvel_norm <= obj_lin_vel_thresh)
        & (object_angvel_norm <= obj_ang_vel_thresh)
        & (dof_pos_diff <= dof_pos_diff_thresh)
        & (torch.min(fingertip_pos[..., 2], dim=1).values >= object_pos[..., 2] - 0.02)
    )  # this forces the fingers to reset after manipulation

    # if penalize_palm_contact:
    #     goal_reach = goal_reach & (torch.abs(palm_cf).sum(-1) < 0.5)        # 0.2
    # goal_reach = goal_reach & (action_norm <= action_norm_thresh)
    # goal_resets: 1) reset_goal_buf, 2) goal reach?

    # debug success conditions
    # if len(abs_rot_dist) == 1 and abs_rot_dist.sum() <= success_tolerance:
    #     print("[debug] dof_vel condition: {0} <= {1}".format(dof_vel_norm.sum().item(), dof_vel_thresh))
    #     print("[debug] object_linvel condition: {0} <= {1}".format(object_linvel_norm.sum().item(), obj_lin_vel_thresh))
    #     print("[debug] object_angvel condition: {0} <= {1}".format(object_angvel_norm.sum().item(), obj_ang_vel_thresh))
    #     print("[debug] action_norm condition: {0} <= {1}".format(action_norm.sum().item(), action_norm_thresh))

    goal_resets = torch.where(goal_reach, torch.ones_like(reset_goal_buf), reset_goal_buf)

    # reset if object falls
    fall_envs = goal_dist >= fall_dist
    # reset if object deviates large from palm center
    moveout_envs = masked_goal_dist >= success_tolerance_pos
    fall_envs = torch.where(moveout_envs, torch.ones_like(fall_envs), fall_envs)
    # reset if any finger is below the object
    ftip_below_obj = torch.min(fingertip_pos[..., 2], dim=1).values < object_pos[..., 2] - 0.02
    fall_envs = torch.where(ftip_below_obj, torch.ones_like(fall_envs), fall_envs)

    dones = torch.logical_or(goal_reach, fall_envs)

    resets = torch.where(fall_envs, torch.ones_like(reset_buf), reset_buf)
    successes = goal_resets.clone()

    reward_terms["object_fallen"] = fall_envs.float() * fall_penalty

    reward = torch.sum(torch.stack(list(reward_terms.values())), dim=0)
    reward = torch.where(goal_reach, reward + reach_goal_bonus, reward)
    reward = torch.where(fall_envs, reward + fall_penalty, reward)
    time_due_envs = progress_buf >= max_episode_length - 1
    # resets: 1) reset_buf, 2) time due, 3) fall, 4) goal reach
    resets = torch.where(time_due_envs, torch.ones_like(resets), resets)
    resets = torch.where(goal_reach, torch.ones_like(resets), resets)
    # dones: 1) goal reach, 2) fall (include move far from palm center), 3) time due
    dones = torch.logical_or(dones, time_due_envs)
    return (
        reward,
        dones.int(),
        resets,
        goal_resets,
        progress_buf,
        successes,
        abs_rot_dist,
        abs_pos_dist,
        reward_terms,
        time_due_envs,
    )


@torch.no_grad()
def compute_leaphand_reward(
    reset_buf,
    reset_goal_buf,
    progress_buf,
    # successes,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    reward_cfg,
    actions,
    fingertip_pos=None, fingertip_vel=None, ftip_pos_mask=None,
    object_linvel=None,
    object_angvel=None,
    dof_vel=None,
    dof_torque=None,
    dof_pos=None,
    target_dof_pos=None,
    dof_pos_mask=None,
    ftip_cf=None
    # palm_cf=None
):
    rot_reward_scale = reward_cfg["rotRewardScale"]
    rot_eps = reward_cfg["rotEps"]
    pos_reward_scale = reward_cfg["posRewardScale"]
    pos_eps = reward_cfg["posEps"]
    reach_goal_bonus = reward_cfg["reachGoalBonus"]
    fall_dist = reward_cfg["fallDistance"]
    fall_penalty = reward_cfg["fallPenalty"]
    success_tolerance = reward_cfg["successTolerance"]
    success_tolerance_pos = reward_cfg["successTolerancePos"]
    ftip_reward_scale = reward_cfg['ftipRewardScale']
    penalize_palm_contact = reward_cfg["pen_palm_contact"]
    # dof_pos_mask = reward_cfg['dof_pos_mask'] if 'dof_pos_mask' in reward_cfg else None
    dof_pos_reward_scale = reward_cfg["poseDiffPenaltyScale"]
    kwargs = dict(
        reset_buf=reset_buf,
        reset_goal_buf=reset_goal_buf,
        progress_buf=progress_buf,
        # successes=successes,
        max_episode_length=max_episode_length,
        object_pos=object_pos,
        object_rot=object_rot,
        target_pos=target_pos,
        target_rot=target_rot,
        actions=actions,
        fingertip_pos=fingertip_pos,
        ftip_pos_mask=ftip_pos_mask,
        object_linvel=object_linvel,
        object_angvel=object_angvel,
        dof_vel=dof_vel,
        dof_torque=dof_torque,
        dof_pos=dof_pos,
        target_dof_pos=target_dof_pos,
        dof_pos_mask=dof_pos_mask,
        rot_reward_scale=rot_reward_scale,
        rot_eps=rot_eps,
        pos_reward_scale=pos_reward_scale,
        pos_eps=pos_eps,
        dof_pos_reward_scale=dof_pos_reward_scale,
        reach_goal_bonus=reach_goal_bonus,
        fall_dist=fall_dist,
        fall_penalty=fall_penalty,
        success_tolerance=success_tolerance,
        success_tolerance_pos=success_tolerance_pos,
        ftip_reward_scale=ftip_reward_scale,
        energy_scale=reward_cfg["energy_scale"],
        dof_pos_diff_thresh=reward_cfg["dof_pos_diff_thresh"],
        dof_vel_thresh=reward_cfg["dof_vel_thresh"],
        obj_lin_vel_thresh=reward_cfg["obj_lin_vel_thresh"],
        obj_ang_vel_thresh=reward_cfg["obj_ang_vel_thresh"],
        action_norm_thresh=reward_cfg["action_norm_thresh"],
        penalize_palm_contact=penalize_palm_contact,
        # palm_cf=palm_cf if palm_cf is not None else torch.ones(1),
        # palm_cf_scale=reward_cfg['palm_cf_scale'],
        clip_energy_reward=reward_cfg["clip_energy_reward"],
        energy_upper_bound=reward_cfg["energy_upper_bound"],
        ftip_cf=ftip_cf,
        ftip_cf_scale=reward_cfg["ftip_cf_scale"],
        ftip_cf_reward_scale=reward_cfg["ftip_cf_reward_scale"]
    )
    out = compute_reward(**kwargs)
    return out
