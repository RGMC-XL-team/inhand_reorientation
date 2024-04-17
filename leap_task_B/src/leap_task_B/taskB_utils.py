import numpy as np
from scipy.spatial.transform import Rotation as Rot


def get_4x4_transform_from_pos_quat(pos, quat):
    """
    quat: in scipy's xyzw format
    """
    transform = np.eye(4)
    transform[:3, 3] = pos
    transform[:3, :3] = Rot.from_quat(quat).as_matrix()
    return transform


def get_pinocchio_7x_pose_from_4x4_transform(transform):
    pose = np.zeros(
        7,
    )
    pose[:3] = transform[:3, 3]
    pose[3:] = Rot.from_matrix(transform[:3, :3]).as_quat()
    return pose
