#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from leap_hardware.srv import *


class LeapHandReal:
    def __init__(self):
        # self.P = robot_model
        control_rate = 20
        self.rate = rospy.Rate(control_rate)  # 20hz rate
        self.timestep = 1.0 / control_rate

        rospy.wait_for_service("/leap_position")
        self.leap_position = rospy.ServiceProxy("/leap_position", leap_position)
        self.leap_velocity = rospy.ServiceProxy("/leap_velocity", leap_velocity)
        self.leap_effort = rospy.ServiceProxy("/leap_effort", leap_effort)

        self.pub_target_joint = rospy.Publisher("/leaphand_node/cmd_leap", JointState, queue_size=1)

        rospy.sleep(0.2)

    def step(self):
        self.rate.sleep()

    def getHandJointPos(self):
        positions = np.array(self.leap_position().position)
        positions -= np.pi
        return positions

    def ctrlHandJointPos(self, target_joint_pos):
        target_joint_pos = target_joint_pos.copy() + np.pi

        target_state = JointState()
        target_state.position = target_joint_pos.tolist()
        self.pub_target_joint.publish(target_state)


if __name__ == "__main__":
    rospy.init_node("leaphand_real")
    leaphand_real = LeapHandReal()

    print(leaphand_real.getHandJointPos())

    target_joint_pos = np.zeros((16,))
    target_joint_pos[1] = 0.1
    leaphand_real.ctrlHandJointPos(target_joint_pos)

    rospy.sleep(1.0)
