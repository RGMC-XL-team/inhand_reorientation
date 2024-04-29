"""
    This script enables adjusting the joint positions
    of LeapHand with sliders (joint_state_publisher_gui)
"""

import rospy
import numpy as np
from sensor_msgs.msg import JointState

from leap_hardware.hardware_controller import LeapHand


class LeapHandSliders(object):
    def __init__(self) -> None:
        rospy.wait_for_service("/leap_position")

        self.leap_hw = LeapHand(use_default=True, enable_publisher=False)
        self.initialize_sliders()

    def initialize_sliders(self):
        """Record current leap positions and set rosparam"""
        leap_init_positions = self.leap_hw.poll_joint_position()[0]
        # rospy.loginfo("init joint positions: ", leap_init_positions)
        real_to_sim_indices = self.leap_hw.real_to_sim_indices.copy()
        for i in range(16):
            motor_id = real_to_sim_indices[i]
            motor_pos = leap_init_positions[i]
            rospy.set_param(f"/zeros/joint_{motor_id}", float(motor_pos))

        self.current_leap_position = np.array(leap_init_positions).copy()
        rospy.loginfo("Initialized sliders with current leap positions.")

    def main_loop(self):
        _rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            leap_joint_state = rospy.wait_for_message("/joint_states", JointState)
            leap_joint_position = np.array(leap_joint_state.position)
            leap_joint_position = self.leap_hw.real_to_sim(leap_joint_position)
            # rospy.loginfo("hand will move {} in joint space".format(np.linalg.norm(leap_joint_position-self.current_leap_position)))
            self.leap_hw.command_joint_position(leap_joint_position)
            _rate.sleep()


if __name__ == "__main__":
    rospy.init_node("leap_hand_sliders", log_level=rospy.INFO)
    leap_hand_sliders = LeapHandSliders()
    leap_hand_sliders.main_loop()
