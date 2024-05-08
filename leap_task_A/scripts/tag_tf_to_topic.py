#!/usr/bin/python

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped


class TagProcess:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        self.tag_frame_id = "AprilTag"

        self.tag_pose_pub = rospy.Publisher("tag_pose", PoseStamped, queue_size=1)
        rospy.sleep(0.2)

    def publishTagPose(self):
        try:
            transform = self.tfBuffer.lookup_transform(
                "world",
                self.tag_frame_id,
                rospy.Time(0),  # latest
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2, f"Cannot do lookup transform. {e}")
            return

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = transform.transform.translation.x
        msg.pose.position.y = transform.transform.translation.y
        msg.pose.position.z = transform.transform.translation.z
        msg.pose.orientation.w = transform.transform.rotation.w
        msg.pose.orientation.x = transform.transform.rotation.x
        msg.pose.orientation.y = transform.transform.rotation.y
        msg.pose.orientation.z = transform.transform.rotation.z

        self.tag_pose_pub.publish(msg)

    def main(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.publishTagPose()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("tag_tf_to_topic")
    tag_process = TagProcess()
    tag_process.main()
