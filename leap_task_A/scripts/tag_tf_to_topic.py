#!/usr/bin/python

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped


class TagProcess:
    def __init__(self):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
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

            posestamped = PoseStamped()
            posestamped.header.stamp = rospy.Time.now()
            posestamped.header.frame_id = "world"
            posestamped.pose.position.x = transform.transform.translation.x
            posestamped.pose.position.y = transform.transform.translation.y
            posestamped.pose.position.z = transform.transform.translation.z
            posestamped.pose.orientation.w = transform.transform.rotation.w
            posestamped.pose.orientation.x = transform.transform.rotation.x
            posestamped.pose.orientation.y = transform.transform.rotation.y
            posestamped.pose.orientation.z = transform.transform.rotation.z

            self.tag_pose_pub.publish(posestamped)

        except:
            rospy.logwarn("Cannot publish tag poses.")
            pass

    def main(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.publishTagPose()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("tag_tf_to_topic")
    tag_process = TagProcess()
    tag_process.main()
