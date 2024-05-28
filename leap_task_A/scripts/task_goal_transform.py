#!/usr/bin/python

import rospy
import tf2_ros
from geometry_msgs.msg import Point
from tf2_geometry_msgs import PointStamped


class TaskGoalTransform:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        self.goal_sub = rospy.Subscriber("/rgcm_eval/task1/goal", Point, self.taskGoalCb)
        self.goal_pub = rospy.Publisher("/goal_in_world", PointStamped, queue_size=1)

        rospy.sleep(0.2)

    def taskGoalCb(self, msg):
        pointstamped = PointStamped()
        pointstamped.header.stamp = rospy.Time.now()
        pointstamped.header.frame_id = "camera_color_optical_frame"
        pointstamped.point = msg

        pointstamped_world = self.tfBuffer.transform(pointstamped, "world")
        self.goal_pub.publish(pointstamped_world)


if __name__ == "__main__":
    rospy.init_node("task_goal_transform")
    tag_process = TaskGoalTransform()
    rospy.spin()
