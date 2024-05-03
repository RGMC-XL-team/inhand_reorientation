import pdb
import tf2_ros
import rospy
import numpy as np

from leap_hardware.srv import face_pose
from leap_task_B.taskB_utils import get_z_axis_from_pos_quat
from leap_hardware.ros_utils import pos_quat_to_ros_transform


def test_pose_estimator():
    rospy.wait_for_service("/face_pose")
    face_pose_client = rospy.ServiceProxy("/face_pose", face_pose)
    private_brodcaster = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        face_is_upwards = {}
        for request_face in ["A", "B", "C", "D", "E", "F"]:
            received_pose = face_pose_client(request_face)
            current_face_upwards = False
            if not received_pose.visible:
                current_face_upwards = False
            else:
                _pose = np.array(received_pose.pose)
                z_axis = get_z_axis_from_pos_quat(_pose[0:3], _pose[3:7])
                # check alignment (cos>0.985, deg<10)
                if np.dot(z_axis, np.array([0, 0, 1])) > 0.985:
                    current_face_upwards = True
                else:
                    current_face_upwards = False
                # publish to /tf is visible
                object_tf_ros = pos_quat_to_ros_transform(
                    _pose[0:3], _pose[[6, 3, 4, 5]], parent_frame="world", child_frame=f"face_{request_face}"
                )
                private_brodcaster.sendTransform(object_tf_ros)

            face_is_upwards[request_face] = current_face_upwards

        print(face_is_upwards)
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("test_pose_estimator")
    rospy.loginfo("test_action_client started")
    test_pose_estimator()
