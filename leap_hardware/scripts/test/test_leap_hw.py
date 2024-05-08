#! /usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import rospkg
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from leap_hardware.hardware_controller import LeapHand
from leap_hardware.ros_utils import pos_quat_to_ros_transform


def display_trajectory(t_traj, y_traj, dim=0):
    plt.figure()
    plt.plot(t_traj, y_traj[:, dim], label=f"joint {dim}")
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("joint position (rad)")
    plt.show()


def generate_test_joint_trajectory(type="sine"):
    def generate_sine_trajectory(amp, mean, dt=100, period=5000, duration=5000, phase=0, dof=16):  # noqa: PLR0913
        """
        dt, period: integer micro-seconds
        size: (T, N), T is steps, N is dofs
        """
        period = int(period)
        dt = int(dt)
        assert period % dt == 0

        assert duration % period == 0

        t_knots = np.arange(0.0, duration + dt, dt)
        t_knots_sec = t_knots / 1e3

        y_knots = mean + amp * np.sin(2 * np.pi * t_knots / period + phase)
        y_knots_all_dof = np.tile(y_knots, (dof, 1)).T

        return t_knots_sec, y_knots_all_dof

    def generate_recorded_trajectory(debug_dir=""):
        """
        load and replay the recorded trajectory
        """
        rospack = rospkg.RosPack()
        debug_dir = os.path.join(rospack.get_path("leap_sim"), "leapsim/debug")
        joints_sim = np.load(os.path.join(debug_dir, "joints_sim_rotz_goal.npy")).reshape(-1, 16)
        targets_sim = np.load(os.path.join(debug_dir, "targets_sim_rotz_goal.npy"))

        return None, joints_sim

    if type == "sine":
        return generate_sine_trajectory(
            amp=0.15, mean=0.15, dt=50, period=1500, duration=4500, phase=3 / 2 * np.pi, dof=16
        )
    elif type == "recorded":
        return generate_recorded_trajectory()
    else:
        raise NotImplementedError


def test_trajectory_following(leap: LeapHand):
    def move_hand_to_pose(leap: LeapHand, start_position: np.ndarray, goal_position: np.ndarray):
        _rate = rospy.Rate(20)
        _iters = 4 * 20
        for i in range(_iters + 1):
            _progress = i / _iters
            _position = (1 - _progress) * start_position + _progress * goal_position
            leap.command_joint_position(_position)
            _rate.sleep()

    # Wait for connections.
    rospy.wait_for_service("/leap_position")

    hz = 20
    control_dt = 1 / hz
    run_time = 4.5  # sec
    ros_rate = rospy.Rate(hz)

    # modify this line to choose trajectory generation manner
    _, test_traj = generate_test_joint_trajectory(type="recorded")

    # collect data
    q_command = []
    q_reach = []

    # move hand to initial pose
    print("move to the initial position")
    current_position = leap.poll_joint_position()[0]
    move_hand_to_pose(leap, start_position=current_position, goal_position=test_traj[0])

    print("command to the initial position")
    for t_i in range(test_traj.shape[0]):
        leap.command_joint_position(test_traj[t_i, :])
        q_command.append(test_traj[t_i, :].tolist())
        q_reach.append(leap.poll_joint_position()[0].tolist())
        ros_rate.sleep()
    print("done")

    q_command = np.array(q_command)
    q_reach = np.array(q_reach)

    import pdb

    pdb.set_trace()


def test_fingertip_state(leap: LeapHand):
    def create_arrow_marker(id=0, start=[0, 0, 0], end=[0, 0, 0]):
        p_start = Point()
        p_start.x = start[0]
        p_start.y = start[1]
        p_start.z = start[2]
        p_end = Point()
        p_end.x = end[0]
        p_end.y = end[1]
        p_end.z = end[2]

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points.append(p_start)
        marker.points.append(p_end)
        marker.scale.x = 0.005
        marker.scale.y = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        return marker

    hz = 10
    ros_rate = rospy.Rate(hz)

    tf_brodcaster = tf2_ros.TransformBroadcaster()
    marker_pub = rospy.Publisher("fingertip_velocity", MarkerArray, queue_size=1)

    while not rospy.is_shutdown():
        _, _ = leap.poll_joint_state()
        fingertip_state, fingertip_link_names = leap.poll_fingertip_state()
        ftipvel_arrow_marker_array = MarkerArray()
        for i in range(4):
            ftip_state = fingertip_state[i]
            ftip_name = fingertip_link_names[i]

            # visualize fingertip transform
            ros_transform = pos_quat_to_ros_transform(
                ftip_state[:3], ftip_state[3:7][[-1, 0, 1, 2]], child_frame=f"{ftip_name}_pinocchio"
            )
            tf_brodcaster.sendTransform(ros_transform)

            # visualize fingertip (linear) velocity
            ftipvel_arrow_marker = create_arrow_marker(
                id=i, start=ftip_state[:3], end=ftip_state[:3] + ftip_state[7:10] * 3
            )
            ftipvel_arrow_marker_array.markers.append(ftipvel_arrow_marker)
        marker_pub.publish(ftipvel_arrow_marker_array)

        ros_rate.sleep()

    print("done")


def test_main(task_name=""):
    rospy.init_node("leaphand_test_node")

    # try to set up rospy
    leap = LeapHand()
    leap.leap_dof_lower = np.array(
        [
            -0.3140,
            -1.0470,
            -0.5060,
            -0.3660,
            -0.3490,
            -0.4700,
            -1.2000,
            -1.3400,
            -0.3140,
            -1.0470,
            -0.5060,
            -0.3660,
            -0.3140,
            -1.0470,
            -0.5060,
            -0.3660,
        ]
    )
    leap.leap_dof_upper = np.array(
        [
            2.2300,
            1.0470,
            1.8850,
            2.0420,
            2.0940,
            2.4430,
            1.9000,
            1.8800,
            2.2300,
            1.0470,
            1.8850,
            2.0420,
            2.2300,
            1.0470,
            1.8850,
            2.0420,
        ]
    )
    leap.sim_to_real_indices = [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
    leap.real_to_sim_indices = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

    # Wait for connections.
    rospy.wait_for_service("/leap_position")

    # test
    eval(f"test_{task_name}(leap)")


if __name__ == "__main__":
    test_main(task_name="trajectory_following")
    # t_traj, y_traj = generate_test_joint_trajectory(type="sine")
    # print("generated trajectory shape: {}".format(y_traj.shape))
    # display_trajectory(t_traj, y_traj, dim=0)
