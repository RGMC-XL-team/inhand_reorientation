# --------------------------------------------------------
# This script is modified from leap_sim/leapsim/hardware_controller.py
# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/HaozhiQi/hora/blob/main/hora/algo/deploy/robots/leap.py
# --------------------------------------------------------

import os

import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import JointState

from leap_hardware.srv import *
from leap_model_based.leaphand_pinocchio import LeapHandPinocchio


class LeapHand:
    def __init__(self, use_default=False):
        """Simple python interface to the leap Hand.

        The leapClient is a simple python interface to an leap
        robot hand.  It enables you to command the hand directly through
        python library calls (joint positions, joint torques, or 'named'
        grasps).

        The constructors sets up publishers and subscribes to the joint states
        topic for the hand.
        """

        # Topics (that can be remapped) for named graps
        # (ready/envelop/grasp/etc.), joint commands (position and
        # velocity), joint state (subscribing), and envelop torque. Note that
        # we can change the hand topic prefix (for example, to leapHand_0)
        # instead of remapping it at the command line.
        topic_joint_command = "/leaphand_node/cmd_ones"

        # Publishers for above topics.
        self.pub_joint = rospy.Publisher(topic_joint_command, JointState, queue_size=10)
        # rospy.Subscriber(topic_joint_state, JointState, self._joint_state_callback)
        self._joint_state = None
        self.leap_position = rospy.ServiceProxy("/leap_position", leap_position)
        self.leap_velocity = rospy.ServiceProxy("/leap_velocity", leap_velocity)
        self.leap_effort = rospy.ServiceProxy("/leap_effort", leap_effort)

        self.leap_dof_lower = -np.inf * np.ones(16)
        self.leap_dof_upper = np.inf * np.ones(16)
        self.sim_to_real_indices = np.arange(16).tolist()
        self.real_to_sim_indices = np.arange(16).tolist()

        if use_default:
            self.set_default()

        # joint state publisher
        self.leap_jstate_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)

        # pinocchio FK model
        rospack = rospkg.RosPack()
        LEAP_HAND_URDF = os.path.join(rospack.get_path("my_robot_description"), "urdf/leaphand.urdf")
        self.leap_kin = LeapHandPinocchio(LEAP_HAND_URDF)

        self.current_joint_positions = np.zeros(
            16,
        )
        self.current_joint_velocities = np.zeros(
            16,
        )
        self.fingertip_states = np.zeros((4, 13))

    # def _joint_state_callback(self, data):
    # self._joint_state = data

    def set_default(self):
        self.leap_dof_lower = np.array(
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
        self.leap_dof_upper = np.array(
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
        self.sim_to_real_indices = [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
        self.real_to_sim_indices = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

    def sim_to_real(self, values):
        return values[self.sim_to_real_indices]

    def command_joint_position(self, desired_pose):
        """
        Takes as input self.cur_targets (raw joints) in numpy array
        """

        # Check that the desired pose can have len() applied to it, and that
        # the number of dimensions is the same as the number of hand joints.
        if not hasattr(desired_pose, "__len__") or len(desired_pose) != 16:
            rospy.logwarn(f"Desired pose must be a {16}-d array: got {desired_pose}.")
            return False

        msg = JointState()  # Create and publish

        # convert desired_pose to ros_targets
        desired_pose = (2 * desired_pose - self.leap_dof_lower - self.leap_dof_upper) / (
            self.leap_dof_upper - self.leap_dof_lower
        )
        desired_pose = self.sim_to_real(desired_pose)
        desired_pose = np.clip(desired_pose, -1.0, 1.0)

        try:
            msg.position = desired_pose
            self.pub_joint.publish(msg)
            rospy.logdebug("Published desired pose.")
            return True
        except rospy.exceptions.ROSSerializationException:
            rospy.logwarn(f"Incorrect type for desired pose: {desired_pose}.")
            return False

    def real_to_sim(self, values):
        return values[self.real_to_sim_indices]

    def poll_joint_position(self):
        """Get the current joint positions of the hand.

        :param wait: If true, waits for a 'fresh' state reading.
        :return: Joint positions, or None if none have been received.
        """
        joint_position = np.array(self.leap_position().position)
        # joint_effort = np.array(self.leap_effort().effort)

        joint_position = self.LEAPhand_to_sim_ones(joint_position)
        joint_position = self.real_to_sim(joint_position)
        joint_position = (self.leap_dof_upper - self.leap_dof_lower) * (joint_position + 1) / 2 + self.leap_dof_lower

        self.current_joint_positions = joint_position.copy()
        self.publish_leap_joint_states()

        return (joint_position, None)

    def poll_joint_velocity(self):
        """Get the current joint velocities of the hand.

        :return: Joint velocities, or None if none have been received.
        """
        joint_velocity = np.array(self.leap_velocity().velocity)
        joint_velocity = self.real_to_sim(joint_velocity)

        self.current_joint_velocities = joint_velocity.copy()

        return (joint_velocity, None)

    def poll_joint_state(self):
        """Get the current joint states of the hand.

        :return: (Joint positions, Joint velocities), or None if none have been received.
        """
        joint_positions, _ = self.poll_joint_position()
        joint_velocities, _ = self.poll_joint_velocity()

        return joint_positions, joint_velocities

    def poll_fingertip_state(self, ordered_link_names=None):
        # convert to pinocchio joint order
        pin_q = self.current_joint_positions[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]]
        pin_v = self.current_joint_velocities[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]]
        self.leap_kin.updateFKvel(part_name="hand", part_joint_pos=pin_q, part_joint_vel=pin_v)

        fingertip_states = np.zeros((4, 13))
        # fingertip_link_names =
        if ordered_link_names is not None:
            fingertip_link_names = ordered_link_names
        else:
            fingertip_link_names = [
                "finger1_tip_center",
                "finger2_tip_center",
                "finger3_tip_center",
                "thumb_tip_center",
            ]
        for i in range(len(fingertip_link_names)):
            _ftip_name = fingertip_link_names[i]
            pos, quat = self.leap_kin.getFrameGlobalPose(_ftip_name)
            fingertip_states[i, :3] = pos
            fingertip_states[i, 3:7] = quat  # xyzw, IsaacGym format

            linear, angular = self.leap_kin.getFrameGlobalVelocity(_ftip_name)
            fingertip_states[i, 7:10] = linear
            fingertip_states[i, 10:] = angular

        return fingertip_states, fingertip_link_names

    def publish_leap_joint_states(self):
        leap_pos_new = self.current_joint_positions

        leap_joint_state = JointState()
        leap_joint_state.header.stamp = rospy.Time.now()
        leap_joint_state.name = [f"joint_{i}" for i in self.real_to_sim_indices]
        leap_joint_state.position = leap_pos_new
        leap_joint_state.velocity = [0.0] * 16
        leap_joint_state.effort = [0.0] * 16

        self.leap_jstate_pub.publish(leap_joint_state)

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)

        return sim_min, sim_max

    def LEAPhand_to_LEAPsim(self, joints):
        joints = np.array(joints)
        ret_joints = joints - 3.14
        return ret_joints

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)

        return joints


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
