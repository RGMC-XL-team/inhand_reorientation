import os

import numpy as np
from scipy.spatial.transform import Rotation as R

unit_x = np.array([1, 0, 0])
unit_y = np.array([0, 1, 0])
unit_z = np.array([0, 0, 1])

rmat_z1 = np.c_[unit_x, unit_y, unit_z]
rmat_z2 = np.c_[-unit_x, -unit_y, unit_z]
rmat_z3 = np.c_[unit_y, -unit_x, unit_z]
rmat_z4 = np.c_[-unit_y, unit_x, unit_z]

rmat_z5 = np.c_[unit_y, unit_x, -unit_z]
rmat_z6 = np.c_[-unit_y, -unit_x, -unit_z]
rmat_z7 = np.c_[unit_x, -unit_y, -unit_z]
rmat_z8 = np.c_[-unit_x, unit_y, -unit_z]

rmat_y1 = np.c_[unit_z, unit_x, unit_y]
rmat_y2 = np.c_[-unit_z, -unit_x, unit_y]
rmat_y3 = np.c_[unit_x, -unit_z, unit_y]
rmat_y4 = np.c_[-unit_x, unit_z, unit_y]

rmat_y5 = np.c_[unit_x, unit_z, -unit_y]
rmat_y6 = np.c_[-unit_x, -unit_z, -unit_y]
rmat_y7 = np.c_[unit_z, -unit_x, -unit_y]
rmat_y8 = np.c_[-unit_z, unit_x, -unit_y]

rmat_x1 = np.c_[unit_y, unit_z, unit_x]
rmat_x2 = np.c_[-unit_y, -unit_z, unit_x]
rmat_x3 = np.c_[unit_z, -unit_y, unit_x]
rmat_x4 = np.c_[-unit_z, unit_y, unit_x]

rmat_x5 = np.c_[unit_z, unit_y, -unit_x]
rmat_x6 = np.c_[-unit_z, -unit_y, -unit_x]
rmat_x7 = np.c_[unit_y, -unit_z, -unit_x]
rmat_x8 = np.c_[-unit_y, unit_z, -unit_x]

rmat_all = []
rmat_all += [rmat_x1, rmat_x2, rmat_x3, rmat_x4, rmat_x5, rmat_x6, rmat_x7, rmat_x8]
rmat_all += [rmat_y1, rmat_y2, rmat_y3, rmat_y4, rmat_y5, rmat_y6, rmat_y7, rmat_y8]
rmat_all += [rmat_z1, rmat_z2, rmat_z3, rmat_z4, rmat_z5, rmat_z6, rmat_z7, rmat_z8]

rmat_all_set = set([str(rmat) for rmat in rmat_all])
assert len(rmat_all) == len(rmat_all_set)

quat_all = []

for rmat in rmat_all:
    assert np.linalg.det(rmat) == 1
    quat = R.from_matrix(rmat).as_quat()
    quat_all.append(quat)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
np.save(os.path.join(CACHE_DIR, "reset_target_quat.npy"), np.array(quat_all))
print("quat all has shape: ", np.array(quat_all).shape)
