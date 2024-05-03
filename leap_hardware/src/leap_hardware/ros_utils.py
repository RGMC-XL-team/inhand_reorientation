from typing import Union

import numpy as np
import rospy
import transforms3d as tf3d
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as Rot


def pos_quat_to_ros_transform(pos=[0.0, 0.0, 0.0], quat=[1, 0, 0, 0], parent_frame="world", child_frame="child_frame"):
    """
    pos: [x, y, z]
    quat: [w, x, y, z]
    """
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame

    x, y, z = pos
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.translation.z = z

    w, x, y, z = quat
    transform.transform.rotation.x = x
    transform.transform.rotation.y = y
    transform.transform.rotation.z = z
    transform.transform.rotation.w = w

    return transform


def ros_transform_to_ros_pose(tf: TransformStamped, stamped=False) -> Union[Pose, PoseStamped]:
    pose = Pose()
    pose.position = tf.transform.translation
    pose.orientation = tf.transform.rotation
    if stamped:
        pose_stamped = PoseStamped()
        pose_stamped.header = tf.header
        return pose_stamped
    else:
        return pose


def ros_transform_to_rigidtransform(
    tf: TransformStamped,
) -> Union[Pose, PoseStamped]:
    xyz = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
    quat = np.array(
        [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
    )
    return xyz_rpy_to_rigidtransform(xyz, Rot.from_quat(quat).as_euler("xyz"))


def ros_pose_to_rigid_transform(
    pose: Pose
):
    xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
    quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    return xyz_rpy_to_rigidtransform(xyz, Rot.from_quat(quat).as_euler("xyz"))


def rigidtransform_to_ros_transform(
    tf,
    parent_frame="world",
    child_frame="child_frame",
):
    xyz = tf[:3, 3]
    quat = Rot.from_matrix(tf[:3, :3]).as_quat()[[-1, 0, 1, 2]]
    ros_tf = pos_quat_to_ros_transform(pos=xyz, quat=quat, parent_frame=parent_frame, child_frame=child_frame)
    return ros_tf


def rpy_to_mat(rpy):
    assert len(rpy) == 3  # noqa: PLR2004
    # roll, pitch, yaw = rpy
    # return tf3d.euler.euler2mat(roll, pitch, yaw, "sxyz")
    return Rot.from_euler("xyz", rpy, degrees=False).as_matrix()


def xyz_rpy_to_rigidtransform(xyz, rpy):
    trans = xyz
    rot = rpy_to_mat(rpy)
    zoom = np.ones(
        3,
    )
    return tf3d.affines.compose(trans, rot, zoom)


def transform_to_posevec(tf):
    pos = tf[:3, 3]
    quat = Rot.from_matrix(tf[:3, :3]).as_quat()        # xyzw
    return np.concatenate([pos, quat])


def transform_to_pos_rvec(tf):
    pos = tf[:3, 3]
    rot = Rot.from_matrix(tf[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot])


def average_quaternions(quaternions):
    """
    quaternions: an (N, 4) array, wxyz format
    """
    nq = quaternions.shape[0]
    quaternions = [Quaternion(q) for q in quaternions]

    avg_q = quaternions[0]
    for iq in range(1, nq):
        q_i = quaternions[iq]
        avg_q = Quaternion.slerp(q0=avg_q, q1=q_i, amount=(1 / (iq + 1)))

    return avg_q.elements


def average_transforms(transforms):
    """
    transforms: array of Nx4x4, rigid transforms
    """
    avg_xyz = np.mean(transforms[:, :3, 3], axis=0)

    quats = np.zeros((transforms.shape[0], 4))
    for i in range(len(transforms)):
        rot = transforms[i, :3, :3]
        quats[i] = Rot.from_matrix(rot).as_quat()[[-1, 0, 1, 2]]  # xyzw -> wxyz

    avg_quat = average_quaternions(quats)

    avg_tf = xyz_rpy_to_rigidtransform(xyz=avg_xyz, rpy=Rot.from_quat(avg_quat[[1, 2, 3, 0]]).as_euler("xyz"))

    return avg_tf


def interpolate_two_transforms(tf0, tf1, amount=0.5):
    """
    amount=0 --> tf0
    amount=1 --> tf1
    """
    xyz0, xyz1 = tf0[:3, 3], tf1[:3, 3]
    int_xyz = (1 - amount) * xyz0 + amount * xyz1

    quat0 = Quaternion(Rot.from_matrix(tf0[:3, :3]).as_quat()[[-1, 0, 1, 2]])  # xyzw -> wxyz
    quat1 = Quaternion(Rot.from_matrix(tf1[:3, :3]).as_quat()[[-1, 0, 1, 2]])
    int_quat = Quaternion.slerp(quat0, quat1, amount).elements
    int_rpy = Rot.from_quat(int_quat[[1, 2, 3, 0]]).as_euler("xyz")  # wxyz -> xyzw

    int_tf = xyz_rpy_to_rigidtransform(int_xyz, int_rpy)

    return int_tf


def substract_two_transforms(tf0, tf1):
    """
    return tf1 - tf0, pos and angle axis
    """
    pos_diff = tf1[:3, 3] - tf0[:3, 3]

    quat0 = Quaternion(Rot.from_matrix(tf0[:3, :3]).as_quat()[[-1, 0, 1, 2]])  # xyzw -> wxyz
    quat1 = Quaternion(Rot.from_matrix(tf1[:3, :3]).as_quat()[[-1, 0, 1, 2]])
    quat_diff = quat1 * quat0.conjugate

    rot_diff = quat_diff.angle * quat_diff.axis

    return pos_diff, rot_diff


def compute_velocity_from_two_transforms(tf_old, tf_new, dt):
    pos_vel = (tf_new[:3, 3] - tf_old[:3, 3]) / dt

    quat_old = Quaternion(Rot.from_matrix(tf_old[:3, :3]).as_quat()[[-1, 0, 1, 2]])  # xyzw -> wxyz
    quat_new = Quaternion(Rot.from_matrix(tf_new[:3, :3]).as_quat()[[-1, 0, 1, 2]])
    quat_diff = quat_new * quat_old.conjugate

    ang_vel = (quat_diff.angle / dt) * quat_diff.axis

    return pos_vel, ang_vel
