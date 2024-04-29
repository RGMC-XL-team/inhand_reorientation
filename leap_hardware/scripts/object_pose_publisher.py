#! /usr/bin/env python3

"""
This script reads apriltag poses, do filtering
and publish the newest tag frames
"""

from enum import Enum

import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import Pose

from leap_hardware.ros_utils import *
from leap_hardware.srv import object_state, face_pose

DEFAULT_CUBE_FACE_OFFSET_FILE = (
    "/home/yongpeng/competition/RGMC_XL/leap_ws/src/leap_hardware/config/cube_face_offset.yaml"
)
DEFAULT_CUBE_TAG_FILE = (
    "/home/yongpeng/competition/RGMC_XL/leap_ws/src/RGMC_In-Hand_Manipulation_2024/config/tags_cube.yaml"
)
DEFAULT_APRILTAG_OFFSET = [0.01, 0.01, 0.0]


class PoseStatus(Enum):
    UPDATED = 0
    OUTDATED = 1
    NOTFOUND = 2


class ObjectPosePublisher:
    def __init__(self) -> None:
        self.private_brodcaster = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.private_listener = tf2_ros.TransformListener(self.tfBuffer)

        publish_rate = rospy.get_param("/cube_pose_publisher/publish_rate", 30)
        self.rate = rospy.Rate(publish_rate)
        self.object_tf_name = rospy.get_param("/cube_pose_publisher/object_name", "cube")
        self.object_tf_smooth_factor = rospy.get_param("/cube_pose_publisher/smooth_factor", 0.1)

        self.last_object_tf = np.eye(4)
        self.current_object_tf = self.last_object_tf.copy()
        self.last_tf_time = rospy.Time.now()
        self.current_tf_time = rospy.Time.now()

        self.object_pose = transform_to_posevec(self.current_object_tf)  # wxyz
        self.object_velocity = np.zeros(
            6,
        )

        # useful dics and lists
        self.tag_to_object_offset = {}
        self.tag_to_face_map = {}
        self.face_to_tag_map = {}
        self.tag_pose = {}
        self.tag_status = {}
        self.tag_receive_dt = {}
        self.all_tag_ids = []

        self.load_cube_tag_config()
        self.load_cube_face_offset()

        # ROS service
        rospy.Service("face_pose", face_pose, self.get_face_pose_handler)
        rospy.Service("object_state", object_state, self.get_object_state_handler)

    def load_cube_tag_config(self):
        cube_tag_file = rospy.get_param("/cube_pose_publisher/cube_tag_file", DEFAULT_CUBE_TAG_FILE)
        with open(cube_tag_file) as f:
            try:
                tags = yaml.safe_load(f)["faces"]
                for i, face in enumerate(tags):
                    self.face_to_tag_map[face] = tags[face]["tag"]
                    for tag in tags[face]["tag"]:
                        self.tag_to_face_map[tag] = face

            except yaml.YAMLError as exc:
                rospy.logerr("Invalid Cube File")
                print(exc)
                return

        for face in self.face_to_tag_map:
            self.all_tag_ids += self.face_to_tag_map[face]
        self.all_tag_ids = list(set(self.all_tag_ids))
        for tag in self.all_tag_ids:
            self.tag_status[tag] = PoseStatus.NOTFOUND
            self.tag_pose[tag] = np.eye(4)

    def load_cube_face_offset(self):
        cube_face_offset_file = rospy.get_param(
            "/cube_pose_publisher/cube_face_offset_file", DEFAULT_CUBE_FACE_OFFSET_FILE
        )
        T_april2face = xyz_rpy_to_rigidtransform(DEFAULT_APRILTAG_OFFSET, [0, 0, np.pi / 2])
        with open(cube_face_offset_file) as f:
            faces = yaml.safe_load(f)["faces"]
            for face in faces:
                _xyz = faces[face]["xyz"]
                _rpy = faces[face]["rpy"]
                T_face2object = xyz_rpy_to_rigidtransform(_xyz, _rpy)
                T_object2april = np.linalg.inv(np.matmul(T_face2object, T_april2face))

                tags = self.face_to_tag_map[face]
                for tag in tags:
                    self.tag_to_object_offset[tag] = T_object2april

        T_face2april = xyz_rpy_to_rigidtransform(xyz=[-0.01, 0.01, 0.0], rpy=[0.0, 0.0, -np.pi/2])
        self.face_to_tag_transform = T_face2april

    def lookup_tag_poses_from_tf(self):
        for tag in self.all_tag_ids:
            tag_tf_time = None
            try:
                transform = self.tfBuffer.lookup_transform(
                    "world",
                    "tag_" + str(tag),
                    rospy.Time(0),  # latest
                )
                self.tag_status[tag] = PoseStatus.UPDATED
                self.tag_pose[tag] = ros_transform_to_rigidtransform(transform)
                self.tag_receive_dt[tag] = (rospy.Time.now() - transform.header.stamp).to_sec()
                tag_tf_time = transform.header.stamp
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.tag_status[tag] = PoseStatus.OUTDATED
                self.tag_receive_dt[tag] = -np.inf
            if self.tag_receive_dt[tag] > 0.1:
                self.tag_status[tag] = PoseStatus.OUTDATED
            if self.tag_status[tag] == PoseStatus.UPDATED:
                self.current_tf_time = tag_tf_time

    def get_object_pose_from_tag_pose(self):
        # there might be more than one tag pose available
        # each one can be used to calculate the object pose
        # we use the averaged result
        valid_transforms = []
        for tag in self.all_tag_ids:
            if self.tag_status[tag] == PoseStatus.UPDATED:
                _tag_pose = self.tag_pose[tag]
                _object_pose = np.matmul(_tag_pose, self.tag_to_object_offset[tag])
                valid_transforms.append(_object_pose)

        valid_transforms = np.array(valid_transforms).reshape(-1, 4, 4)
        if len(valid_transforms) > 1:
            avg_object_tf = average_transforms(valid_transforms)
        elif len(valid_transforms) > 0:
            avg_object_tf = valid_transforms[0]
        else:
            avg_object_tf = None

        # smoothing
        if avg_object_tf is not None:
            avg_object_tf = interpolate_two_transforms(
                tf0=self.last_object_tf, tf1=avg_object_tf, amount=self.object_tf_smooth_factor
            )

        return avg_object_tf

    def publish_object_pose(self, object_tf):
        if object_tf is None:
            return False

        self.current_object_tf = object_tf.copy()
        self.object_pose = transform_to_posevec(self.current_object_tf)

        object_tf_ros = rigidtransform_to_ros_transform(
            object_tf, parent_frame="world", child_frame=self.object_tf_name
        )
        self.private_brodcaster.sendTransform(object_tf_ros)
        return True

    def compute_other_states(self):
        dt = max((self.current_tf_time - self.last_tf_time).to_sec(), 1e-5)
        _pos_vel, _ang_vel = compute_velocity_from_two_transforms(self.last_object_tf, self.current_object_tf, dt)
        self.object_velocity = np.concatenate([_pos_vel, _ang_vel])

        # update history
        self.last_object_tf = self.current_object_tf.copy()
        self.last_tf_time = self.current_tf_time

    def get_object_state_handler(self, req):
        pos, quat = self.object_pose[:3], self.object_pose[3:]
        return {
            "pose": np.concatenate((pos, quat)),  # xyzw
            "velocity": self.object_velocity,
        }
    
    def get_face_pose_handler(self, req):
        tag = self.face_to_tag_map[req.face][0]
        if self.tag_status[tag] == PoseStatus.UPDATED:
            T_tag2world = self.tag_pose[tag]
            T_face2world = np.matmul(T_tag2world, self.face_to_tag_transform)
            return {
                "pose": transform_to_posevec(T_face2world),
                "visible": True
            }
        else:
            return {
                "pose": np.zeros(7),
                "visible": False
            }

    def main_loop(self):
        while not rospy.is_shutdown():
            # update tag pose and status
            self.lookup_tag_poses_from_tf()

            # update object pose
            object_pose = self.get_object_pose_from_tag_pose()

            # publish object pose
            pose_is_updated = self.publish_object_pose(object_pose)
            # rospy.loginfo("pose updated: {}".format(pose_is_updated))

            # update object velocity
            if pose_is_updated:
                self.compute_other_states()

            self.rate.sleep()


if __name__ == "__main__":
    nh = rospy.init_node("object_pose_publisher")
    publisher = ObjectPosePublisher()
    publisher.main_loop()
