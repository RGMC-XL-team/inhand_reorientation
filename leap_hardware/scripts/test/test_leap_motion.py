#! /usr/bin/env python3
import numpy as np
import rospy
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from sensor_msgs.msg import JointState, MultiDOFJointState
from test_gen_motion import gen_linear_traj_qA_to_qB
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

DEFAULT_STEP_SECS = 0.05


class LeapHandInfo:
    def __init__(self) -> None:
        self.joint_names = [
            "joint_1",
            "joint_0",
            "joint_2",
            "joint_3",
            "joint_12",
            "joint_13",
            "joint_14",
            "joint_15",
            "joint_5",
            "joint_4",
            "joint_6",
            "joint_7",
            "joint_9",
            "joint_8",
            "joint_10",
            "joint_11",
        ]  # TODO(yongpeng) fixed


leap_hand_info = LeapHandInfo()


def create_joint_state_from_pos(q):
    """
    :param q: np.array of shape (N,), N is dof
    """
    joint_state = JointState()
    joint_state.header.seq = 0
    joint_state.header.stamp = rospy.Time.now()
    joint_state.header.frame_id = "world"
    joint_state.name = leap_hand_info.joint_names
    joint_state.position = q.tolist()
    joint_state.velocity = [0.0 * len(q)]

    return joint_state


def display_motion_in_rviz(q_arr, publisher=None):
    """
    Publish the hand motion as DisplayTrajectory to RViz
    :param q_arr: np.array of shape (T, N), T is time steps
    """

    def create_robot_state(q0):
        robot_state = RobotState()
        robot_state.joint_state = create_joint_state_from_pos(q0)
        _empty_multi_dof_joint_state = MultiDOFJointState()
        _empty_multi_dof_joint_state.header = robot_state.joint_state.header
        robot_state.multi_dof_joint_state = _empty_multi_dof_joint_state
        return robot_state

    def create_robot_trajectory(q_arr):
        step_secs = DEFAULT_STEP_SECS
        robot_trajectory = JointTrajectory()
        robot_trajectory.joint_names = leap_hand_info.joint_names
        for idx in range(len(q_arr)):
            point = JointTrajectoryPoint()
            point.positions = q_arr[idx].tolist()
            point.velocities = [0.0 * 16]
            point.accelerations = [0.0 * 16]
            _time_from_start = idx * step_secs
            point.time_from_start.secs = int(_time_from_start)
            point.time_from_start.nsecs = int((_time_from_start - int(_time_from_start)) * 1e9)
            robot_trajectory.points.append(point)
        return robot_trajectory

    q_arr = np.atleast_2d(q_arr)

    if publisher is None:
        publisher = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectory, queue_size=20)

    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = create_robot_state(q_arr[0])
    robot_trajectory = RobotTrajectory()
    robot_trajectory.joint_trajectory = create_robot_trajectory(q_arr)
    display_trajectory.trajectory.append(robot_trajectory)

    publisher.publish(display_trajectory)


def test_main():
    qA = np.zeros(
        16,
    )
    qB = 0.2 * np.ones(
        16,
    )
    q_traj = gen_linear_traj_qA_to_qB(qA, qB, N=200)

    while not rospy.is_shutdown():
        display_motion_in_rviz(q_traj)
        rospy.sleep(1)


if __name__ == "__main__":
    rospy.init_node("test_leap_motion_node", anonymous=True)
    test_main()
