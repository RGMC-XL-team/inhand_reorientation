"""
    This file provides pre-processing for the LfD data
"""

import os
import rospkg
import pickle
import numpy as np
from matplotlib import pyplot as plt

from leapsim.utils.utils import get_finger_indice_real


def load_and_split_dataset(seq_length, extra_args):
    """
        Load the dataset and split into training sequences
    """

    def split_data_fixed_length(data, seq_length=3):
        """ Split data into fixed-length sequences """
        data_length = len(data)
        data_in_deq = []
        for ptr in range(data_length - seq_length + 1):
            data_in_deq.append(data[ptr:ptr + seq_length])

        return np.array(data_in_deq)
    
    def unscale_np(x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)
    
    def debug_plot_hand_dof_data(pos, lower=None, upper=None, title=""):
        plt.figure()
        if lower is not None and upper is not None:
            plot_limits = True
        else:
            plot_limits = False
        for i in range(pos.shape[1]):
            plt.subplot(2, 4, i+1)
            plt.plot(pos[:, i])
            if plot_limits:
                plt.hlines(lower[i], 0, pos.shape[0], colors="b", linestyles="dashed")
                plt.hlines(upper[i], 0, pos.shape[0], colors="r", linestyles="dashed")
        plt.title(title)

    # parameters
    
    # _frozen_finger_indices = get_finger_indice_real(["finger1", "finger3"])
    # _free_finger_indices = get_finger_indice_real(["thumb", "finger2"])
    _frozen_finger_indices = get_finger_indice_real(extra_args["frozen_fingers"])
    _free_finger_indices = get_finger_indice_real(extra_args["free_fingers"])

    _action_scale = 1/6     # check this (this has been modified from 1/24 to 1/6)
    _action_norm_thresh = 1.5
    _rnn_seq_length = seq_length
    _num_obs = 4 * len(extra_args["free_fingers"]) + 7
    _num_action = 4 * len(extra_args["free_fingers"])
    _use_running_dof_limits = True
    _pos_range = {
        "x": [-0.08, 0.0],
        "y": [0.01, 0.07],
        "z": [0.06, 0.11]
    }

    rospack = rospkg.RosPack()
    data_path = os.path.join(rospack.get_path("leap_hardware"), "debug", "lfd_data.pkl")

    # Load the dataset
    origin_data = pickle.load(open(data_path, "rb"))
    leap_dof_lower = np.array(origin_data["meta_data"]["leap_dof_lower"])[_free_finger_indices]
    leap_dof_upper = np.array(origin_data["meta_data"]["leap_dof_upper"])[_free_finger_indices]

    # # Compute running min and max for the data
    # running_leap_dof_lower, running_leap_dof_upper = np.inf*np.ones(16,), -np.inf*np.ones(16,)
    leap_dof_all = np.empty((0, 16))
    for i in range(len(origin_data)-1):
        data_piece = origin_data[i+1]
        hand_data_piece = np.array(data_piece["leap_hand"])
        leap_dof_all = np.concatenate((leap_dof_all, hand_data_piece[:, :16]), axis=0)
        # running_leap_dof_lower = np.minimum(running_leap_dof_lower, np.min(hand_data_piece[100:, :16], axis=0))
        # running_leap_dof_upper = np.maximum(running_leap_dof_upper, np.max(hand_data_piece[100:, :16], axis=0))
    # running_leap_dof_lower = np.maximum(running_leap_dof_lower, origin_data["meta_data"]["leap_dof_lower"])[_free_finger_indices]
    # running_leap_dof_upper = np.minimum(running_leap_dof_upper, origin_data["meta_data"]["leap_dof_upper"])[_free_finger_indices]

    running_leap_dof_lower = (np.mean(leap_dof_all, axis=0) - 1*np.std(leap_dof_all, axis=0))[_free_finger_indices]
    running_leap_dof_upper = (np.mean(leap_dof_all, axis=0) + 1*np.std(leap_dof_all, axis=0))[_free_finger_indices]

    import pdb; pdb.set_trace()

    obj_pos_lower = np.array([_pos_range["x"][0], _pos_range["y"][0], _pos_range["z"][0]])
    obj_pos_upper = np.array([_pos_range["x"][1], _pos_range["y"][1], _pos_range["z"][1]])

    # Split dataset into sequences
    obs_dataset = np.empty((0, _rnn_seq_length, _num_obs))
    action_dataset = np.empty((0, _num_action))
    for i in range(len(origin_data)-1):
        data_piece = origin_data[i+1]
        hand_data_piece = np.array(data_piece["leap_hand"])
        obj_data_piece = np.array(data_piece["object"])

        action = (1/_action_scale) * np.diff(hand_data_piece, axis=0)
        action = np.clip(action, -1, 1)

        # Discard the static part
        start_idx = np.where(np.linalg.norm(action, axis=1) > _action_norm_thresh)[0][0]

        if _use_running_dof_limits:
            _leap_hand_dof_pos = unscale_np(hand_data_piece[start_idx:-1, :16][:, _free_finger_indices], running_leap_dof_lower, running_leap_dof_upper)
        else:
            _leap_hand_dof_pos = unscale_np(hand_data_piece[start_idx:-1, :16][:, _free_finger_indices], leap_dof_lower, leap_dof_upper)
        
        ## debug
        # --------------------
        # debug_plot_hand_dof_data(hand_data_piece[start_idx:-1, :16][:, _free_finger_indices], lower=running_leap_dof_lower, upper=running_leap_dof_upper, title="dof_scaled")
        # debug_plot_hand_dof_data(_leap_hand_dof_pos, title="dof_unscaled")
        # _debug_scaled_action = unscale_np(action[:, _free_finger_indices], lower=np.min(action), upper=np.max(action))
        # _debug_scaled_dof_vel = unscale_np(
        #     hand_data_piece[:, 16:][:, _free_finger_indices], \
        #     lower=np.min(hand_data_piece[:, 16:][:, _free_finger_indices], axis=0), \
        #     upper=np.max(hand_data_piece[:, 16:][:, _free_finger_indices], axis=0)
        # )
        # debug_plot_hand_dof_data(_debug_scaled_action, title="action_scaled")
        # debug_plot_hand_dof_data(_debug_scaled_dof_vel, title="dof_vel_scaled")
        # plt.show()
        # --------------------

        _object_pos = obj_data_piece[start_idx:-1]
        _object_pos[:, 0:3] = unscale_np(_object_pos[:, 0:3], obj_pos_lower, obj_pos_upper)
        obs_buf = np.concatenate((_leap_hand_dof_pos, _object_pos), axis=1)

        # Split data into sequences
        obs_seq = split_data_fixed_length(obs_buf, seq_length=_rnn_seq_length)
        action_seq = action[start_idx+_rnn_seq_length-1:, :16][:, _free_finger_indices]

        # Update dataset
        obs_dataset = np.concatenate((obs_dataset, obs_seq), axis=0)
        action_dataset = np.concatenate((action_dataset, action_seq), axis=0)

    return obs_dataset, action_dataset


if __name__ == "__main__":
    obs_dataset, action_dataset = load_and_split_dataset(
        seq_length=3,
        extra_args={"frozen_fingers": ["finger1", "finger3"], "free_fingers": ["thumb", "finger2"]}
    )
    import pdb; pdb.set_trace()
