#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState

from leap_hardware.srv import leap_effort, leap_position, leap_velocity
from leap_utils.mingrui.utils_ros import rosPoseToPosQuat


class LeapHandReal:
    def __init__(self, control_rate):
        # self.private_brodcaster = tf2_ros.TransformBroadcaster()
        # self.tfBuffer = tf2_ros.Buffer()

        # self.P = robot_model
        self.rate = rospy.Rate(control_rate)  # 20hz rate
        self.timestep = 1.0 / control_rate

        rospy.wait_for_service("/leap_position")
        self.leap_position = rospy.ServiceProxy("/leap_position", leap_position)
        self.leap_velocity = rospy.ServiceProxy("/leap_velocity", leap_velocity)
        self.leap_effort = rospy.ServiceProxy("/leap_effort", leap_effort)

        self.tag_posestamped = None
        rospy.Subscriber("tag_pose", PoseStamped, self.tagPoseCb)

        self.pub_target_joint = rospy.Publisher("/leaphand_node/cmd_leap", JointState, queue_size=1)
        self.leap_jstate_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)
        self.pub_desired_pos = rospy.Publisher("tag_desired_pos", PointStamped, queue_size=1)
        rospy.sleep(0.2)

    def tagPoseCb(self, msg):
        self.tag_posestamped = msg

    def step(self):
        self.rate.sleep()

    def getHandJointPos(self):
        positions = np.array(self.leap_position().position)
        positions -= np.pi
        # self.publishHandJointStates(positions)
        return positions

    def ctrlHandJointPos(self, target_joint_pos):
        self.publishHandJointStates(target_joint_pos.tolist())  # for rviz visualization

        target_joint_pos = target_joint_pos.copy() + np.pi
        target_state = JointState()
        target_state.position = target_joint_pos.tolist()
        self.pub_target_joint.publish(target_state)

    def getQRCodePose(self):
        if self.tag_posestamped is None:
            raise NameError("Cannot receive msg of tag poses.")
        pos, quat = rosPoseToPosQuat(self.tag_posestamped.pose)
        return pos, quat

    def visDesiredPos(self, pos):
        pointstamped = PointStamped()
        pointstamped.header.frame_id = "world"
        pointstamped.point.x = pos[0]
        pointstamped.point.y = pos[1]
        pointstamped.point.z = pos[2]
        self.pub_desired_pos.publish(pointstamped)

    def publishHandJointStates(self, joint_pos):
        leap_joint_state = JointState()
        leap_joint_state.header.stamp = rospy.Time.now()
        leap_joint_state.name = [f"joint_{i}" for i in range(16)]
        leap_joint_state.position = joint_pos
        leap_joint_state.velocity = [0.0] * 16
        leap_joint_state.effort = [0.0] * 16

        self.leap_jstate_pub.publish(leap_joint_state)


if __name__ == "__main__":
    rospy.init_node("leaphand_real")
    leaphand_real = LeapHandReal(control_rate=20)

    print(leaphand_real.getHandJointPos())

    # target_joint_pos = np.zeros((16,))
    # target_joint_pos[1] = 0.1
    # leaphand_real.ctrlHandJointPos(target_joint_pos)

    rospy.sleep(1.0)
