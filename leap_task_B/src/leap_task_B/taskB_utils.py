from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class TaskBPose(object):
    def __init__(self) -> None:
        self._transform = np.eye(4)
        self._quat_xyzw = np.array([0, 0, 0, 1])
        self._euler_xyz = np.zeros(3,)
        self._pos_xyz = np.zeros(3,)

    def set_from_pos_quat(self, pos, quat):
        self._pos_xyz = pos.copy()
        self._quat_xyzw = quat.copy()
        self._euler_xyz = get_euler_from_quat(quat)
        self._transform = get_4x4_transform_from_pos_quat(pos, quat)

    def yaw(self):
        return restrict_angle_in_pi(self._euler_xyz[2])
    
    def xy(self):
        return self._pos_xyz[0:2].copy()


class TaskBInfo(object):
    def __init__(self) -> None:
        self.object_global_pose = TaskBPose()       # the object's body frame pose
        self.object_local_pose = TaskBPose()        # the object's upside face frame pose
        self.current_state = TaskBState(0)          # the current state of the task

        self.current_face = "A"                     # the current face
        self.target_face = "B"                      # the current target face
        self.desired_yaw = 0.0                      # the desired yaw angle for switching to FLIPPING
        self.tolerance_xy = 0.02                    # the tolerance for xy in meter
        self.tolerance_yaw = 0.25                   # the tolerance for yaw in rad

        self.time_budget = 10000.0                  # the time budget for the task (default: 30)
        self.time_spent = 0.0                       # the time spent on the current substep (face)
        self.time_current_start = 0.0               # the time when the current substep (face) starts

        self.object_center_pos = np.array([-0.0592509, 0.03504605, 0.08918902])

        self.running_policy = ""                    # running policy name, empty for none

    def load(self, data):
        self.tolerance_xy = data["tolerance"]["xy"]
        self.tolerance_yaw = data["tolerance"]["yaw"]
        self.time_budget = data["time_budget"]
        self.object_center_pos = np.array(data["object_center_pos"])


class TaskBState(Enum):
    WAIT = 0
    BEFORE_ROTATE = 1
    ROTATE_CW = 2
    ROTATE_CCW = 3
    BEFORE_FLIP = 4
    FLIP_OUT = 5
    FLIP_IN = 6
    DONE = 7
    TIMEOUT = 8


M_PI = np.pi
M_PI_2 = np.pi/2

DESIRED_YAW_DICT = {
    "A": {
        "A": None, "B": -M_PI_2, "C": M_PI, "D": M_PI_2, "E": None, "F": 0.0
    },
    "B": {
        "A": M_PI_2, "B": None, "C": M_PI, "D": None, "E": -M_PI_2, "F": 0.0
    },
    "C": {
        "A": 0.0, "B": -M_PI_2, "C": None, "D": M_PI_2, "E": M_PI, "F": None
    },
    "D": {
        "A": 0.0, "B": None, "C": -M_PI_2, "D": None, "E": M_PI, "F": M_PI_2
    },
    "E": {
        "A": None, "B": M_PI, "C": -M_PI_2, "D": 0.0, "E": None, "F": M_PI_2
    },
    "F": {
        "A": M_PI_2, "B": M_PI, "C": None, "D": 0.0, "E": -M_PI_2, "F": None
    }
}


def get_4x4_transform_from_pos_quat(pos, quat):
    """
    quat: in scipy's xyzw format
    """
    transform = np.eye(4)
    transform[:3, 3] = pos
    transform[:3, :3] = Rot.from_quat(quat).as_matrix()
    return transform


def get_3x3_rotation_from_quat(quat):
    rotation = Rot.from_quat(quat).as_matrix()
    return rotation


def get_z_up_rotation(rotation):
    """
    adjust the rotation matrix so that +z points upwards
    while the axis is unchanged
    """
    # equivalent to find the max{dot(axis, +z)}, axis=x, y, z
    up_axis = np.argmax(np.abs(rotation[2, :])) # 0, 1, 2
    rotation = np.roll(rotation, (2-up_axis), axis=1)

    if rotation[2, 2] < 0:
        x_axis, y_axis, z_axis = rotation[:, 0], rotation[:, 1], rotation[:, 2]
        rotation = np.c_[-y_axis, -x_axis, -z_axis]

    return rotation


def get_z_up_quat(quat):
    rotation = get_3x3_rotation_from_quat(quat)
    rotation = get_z_up_rotation(rotation)
    return Rot.from_matrix(rotation).as_quat()


def get_euler_from_quat(quat):
    euler_xyz = Rot.from_quat(quat).as_euler("xyz", degrees=False)
    return euler_xyz


def get_quat_from_euler(euler):
    quat = Rot.from_euler("xyz", euler, degrees=False).as_quat()
    return quat


def restrict_angle_in_pi(angle):
    """Restrict the angle in [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_angle_distance(angle1, angle2):
    """Return the abs distance between two angles in [-pi, pi]"""
    diff = np.abs(angle1 - angle2)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff


def get_z_axis_from_pos_quat(pos, quat):
    _transform = get_4x4_transform_from_pos_quat(pos, quat)
    return _transform[:3, 2]


def get_pinocchio_7x_pose_from_4x4_transform(transform):
    pose = np.zeros(
        7,
    )
    pose[:3] = transform[:3, 3]
    pose[3:] = Rot.from_matrix(transform[:3, :3]).as_quat()
    return pose


def normalize_array(array):
    return array / np.linalg.norm(array, ord=2, axis=-1, keepdims=True)
