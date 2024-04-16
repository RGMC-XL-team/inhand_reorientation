import numpy as np


def gen_linear_traj_qA_to_qB(qA, qB, N=200):
    traj_linear = np.linspace(qA, qB, N)
    return traj_linear
