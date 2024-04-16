# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
from leapsim.utils.utils import set_seed
import torch
import numpy as np
from typing import Callable

from leapsim.tasks import isaacgym_task_map


def get_rlgames_env_creator(
        # used to create the vec task
        seed: int,
        task_config: dict,
        task_name: str,
        sim_device: str,
        rl_device: str,
        graphics_device_id: int,
        headless: bool,
        # Used to handle multi-gpu case
        multi_gpu: bool = False,
        post_create_hook: Callable = None,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        headless: Whether to run in headless mode.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
        virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
        force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
    Returns:
        A VecTaskPython object.
    """
    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """

        # create native task and pass custom config
        env = isaacgym_task_map[task_name](
            cfg=task_config,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if post_create_hook is not None:
            post_create_hook()

        return env
    return create_rlgpu_env


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.mean_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.mean_consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.mean_abs_rot = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.mean_abs_pos = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.mean_rew_object_fallen = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])

            if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                        if 'rew' in k:
                            self.direct_info['reward/{}'.format(k.replace("rew_", ""))] = v
                        else:
                            self.direct_info[k] = v

                    # the following metrics are averaged only among the done envs
                    # -----------------------------------------------------------
                    # log successes
                    if k == "success":
                        self.mean_successes.update(v[done_indices])
                    # log consecutive successes
                    if k == "consecutive_success":
                        self.mean_consecutive_successes.update(v[done_indices])
                    # log absolute rotate distance
                    if k == "abs_rot_dist":
                        self.mean_abs_rot.update(v[done_indices])
                    # log absolute position distance
                    if k == "abs_pos_dist":
                        self.mean_abs_pos.update(v[done_indices])
                    # log object fallen reward
                    if k == "object_fallen":
                        self.mean_rew_object_fallen.update(v[done_indices])
                    # -----------------------------------------------------------

    def after_clear_stats(self):
        self.mean_scores.clear()
        self.mean_successes.clear()
        self.mean_consecutive_successes.clear()
        self.mean_abs_rot.clear()
        self.mean_abs_pos.clear()
        self.mean_rew_object_fallen.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                    infotensor = torch.tensor([], device=self.algo.device)
                    for ep_info in self.ep_infos:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                    value = torch.mean(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, epoch_num)
            self.ep_infos.clear()
        
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, epoch_num)
            self.writer.add_scalar(f'{k}/time', v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)

        if self.mean_successes.current_size > 0:
            mean_successes = self.mean_successes.get_mean()
            self.writer.add_scalar('successes/mean', mean_successes, frame)
            self.writer.add_scalar('successes/iter', mean_successes, epoch_num)
            self.writer.add_scalar('successes/time', mean_successes, total_time)

        if self.mean_consecutive_successes.current_size > 0:
            mean_consecutive_successes = self.mean_consecutive_successes.get_mean()
            self.writer.add_scalar('consecutive_successes/mean', mean_consecutive_successes, frame)
            self.writer.add_scalar('consecutive_successes/iter', mean_consecutive_successes, epoch_num)
            self.writer.add_scalar('consecutive_successes/time', mean_consecutive_successes, total_time)

        if self.mean_abs_rot.current_size > 0:
            mean_abs_rot = self.mean_abs_rot.get_mean()
            self.writer.add_scalar('abs_rot/mean', mean_abs_rot, frame)
            self.writer.add_scalar('abs_rot/iter', mean_abs_rot, epoch_num)
            self.writer.add_scalar('abs_rot/time', mean_abs_rot, total_time)

        if self.mean_abs_pos.current_size > 0:
            mean_abs_pos = self.mean_abs_pos.get_mean()
            self.writer.add_scalar('abs_pos/mean', mean_abs_pos, frame)
            self.writer.add_scalar('abs_pos/iter', mean_abs_pos, epoch_num)
            self.writer.add_scalar('abs_pos/time', mean_abs_pos, total_time)

        if self.mean_rew_object_fallen.current_size > 0:
            mean_rew_object_fallen = self.mean_rew_object_fallen.get_mean()
            self.writer.add_scalar('object_fallen/mean', mean_rew_object_fallen, frame)
            self.writer.add_scalar('object_fallen/iter', mean_rew_object_fallen, epoch_num)
            self.writer.add_scalar('object_fallen/time', mean_rew_object_fallen, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        return  self.env.step(actions)

    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info
