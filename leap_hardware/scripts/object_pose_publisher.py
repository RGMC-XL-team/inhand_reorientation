#! /usr/bin/env python3

"""
This script reads apriltag poses, do filtering
and publish the newest tag frames
"""

from enum import Enum

import numpy as np
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import Pose, TransformStamped
from visualization_msgs.msg import Marker

from leap_hardware import ros_utils as rutil 
from leap_hardware.mean_std import RunningMeanCovariance
from leap_hardware.srv import object_state, face_pose
from apriltag_ros.msg import AprilTagDetectionArray

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
    FOUND_IN_CAM1 = 3


def get_visualization_marker(transform):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "cube"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x = transform.translation.x
    marker.pose.position.y = transform.translation.y
    marker.pose.position.z = transform.translation.z
    marker.pose.orientation = transform.rotation
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    # 设置颜色
    marker.color.r = 0.1
    marker.color.g = 0.5
    marker.color.b = 0.5
    marker.color.a = 0.8

    return marker


class ObjectPosePublisher:
    def __init__(self) -> None:
        self.private_brodcaster = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.private_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pose_pub = rospy.Publisher("/object_transform", TransformStamped, queue_size=10)
        self.vis_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)

        publish_rate = rospy.get_param("/cube_pose_publisher/publish_rate", 30)
        self.rate = rospy.Rate(publish_rate)

        self.enable_second_camera = rospy.get_param("/cube_pose_publisher/enable_second_camera", True)
        
        if not self.enable_second_camera:
            rospy.logwarn("Second camera is disabled!")
        
        if self.enable_second_camera:
            self.second_camera_name = rospy.get_param("/cube_pose_publisher/second_camera_name", "camera_d435")

        self.object_tf_name = rospy.get_param("/cube_pose_publisher/object_name", "cube")
        self.object_tf_smooth_factor = rospy.get_param("/cube_pose_publisher/smooth_factor", 0.1)
        self.object_pos_diff_thresh = rospy.get_param("/cube_pose_publisher/pos_diff_thresh", 0.02)
        self.object_rot_diff_thresh = rospy.get_param("/cube_pose_publisher/rot_diff_thresh", 0.25)
        self.cov_estimate_queue_size = rospy.get_param("/cube_pose_publisher/cov_estimate_queue_size", 5)
        self.cov_estimate_time_window = rospy.get_param("/cube_pose_publisher/cov_estimate_time_window", 0.5)
        self.cov_ratio_threshold = rospy.get_param("/cube_pose_publisher/cov_ratio_threshold", 5)

        self.last_object_tf = np.eye(4)
        self.current_object_tf = self.last_object_tf.copy()
        self.last_tf_time = rospy.Time.now()
        self.current_tf_time = rospy.Time.now()

        self.object_pose = rutil.transform_to_posevec(self.current_object_tf)  # wxyz
        self.object_velocity = np.zeros(
            6,
        )

        # useful dics and lists
        self.tag_to_object_offset = {}
        self.tag_to_face_map = {}
        self.face_to_tag_map = {}
        self.tag_pose = {}
        self.tag_status = {}
        self.obj_status = PoseStatus.NOTFOUND
        self.tag_receive_dt = {}
        self.all_tag_ids = []
        self.latest_object_tf_ros = None

        self.load_cube_tag_config()
        self.load_cube_face_offset()
        self.initialize_pose_covariance_estimate()

        if self.enable_second_camera:
            self.load_second_camera_transform()

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
        T_april2face = rutil.xyz_rpy_to_rigidtransform(DEFAULT_APRILTAG_OFFSET, [0, 0, np.pi / 2])
        with open(cube_face_offset_file) as f:
            faces = yaml.safe_load(f)["faces"]
            for face in faces:
                _xyz = faces[face]["xyz"]
                _rpy = faces[face]["rpy"]
                T_face2object = rutil.xyz_rpy_to_rigidtransform(_xyz, _rpy)
                T_object2april = np.linalg.inv(np.matmul(T_face2object, T_april2face))

                tags = self.face_to_tag_map[face]
                for tag in tags:
                    self.tag_to_object_offset[tag] = T_object2april

        T_face2april = rutil.xyz_rpy_to_rigidtransform(xyz=[-0.01, 0.01, 0.0], rpy=[0.0, 0.0, -np.pi/2])
        self.face_to_tag_transform = T_face2april

    def initialize_pose_covariance_estimate(self):
        self.pose_covariance_estimator = {}
        self.tag_pose_covariance = {}
        self.tag_pose_covariance_array = np.zeros(6,)

        for face in ["A", "B", "C", "D", "E", "F"]:
            # consider the first tag for small cube only
            tag = self.face_to_tag_map[face][0]
            self.pose_covariance_estimator[tag] = RunningMeanCovariance(
                queue_size=self.cov_estimate_queue_size,
                time_window=self.cov_estimate_time_window
            )

    def load_second_camera_transform(self):
        rospy.loginfo("Wait for second camera link to be published!")
        while not rospy.is_shutdown():
            try:
                transform = self.tfBuffer.lookup_transform(
                    "world",
                    f"{self.second_camera_name}_color_optical_frame",
                    rospy.Time(0),  # latest
                )
                self.T_camera1_to_world = rutil.ros_transform_to_rigidtransform(transform)
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass
        rospy.loginfo("Second camera link is published!")

    def estimate_covariance(self):
        for tag_idx, tag in enumerate(self.pose_covariance_estimator.keys()):
            cov = self.pose_covariance_estimator[tag].get_covariance()
            if cov is not None:
                self.tag_pose_covariance[tag] = cov
            else:
                self.tag_pose_covariance[tag] = np.inf
            self.tag_pose_covariance_array[tag_idx] = self.tag_pose_covariance[tag]

    def lookup_tag_poses_from_tf(self):
        # get tag detections published by camera1
        if self.enable_second_camera:
            tag_detections = rospy.wait_for_message("/camera_1/tag_detections", AprilTagDetectionArray)

        # look up /tf published by camera0
        for tag in self.all_tag_ids:
            tag_tf_time = None
            try:
                transform = self.tfBuffer.lookup_transform(
                    "world",
                    "tag_" + str(tag),
                    rospy.Time(0),  # latest
                )
                self.tag_pose[tag] = rutil.ros_transform_to_rigidtransform(transform)
                self.tag_receive_dt[tag] = (rospy.Time.now() - transform.header.stamp).to_sec()
                tag_tf_time = transform.header.stamp
                if self.tag_receive_dt[tag] > 0.1:
                    self.tag_status[tag] = PoseStatus.OUTDATED
                else:
                    self.tag_status[tag] = PoseStatus.UPDATED
                if self.tag_status[tag] == PoseStatus.UPDATED:
                    self.current_tf_time = tag_tf_time
                    self.pose_covariance_estimator[tag].append_new(transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.tag_status[tag] = PoseStatus.OUTDATED
                self.tag_receive_dt[tag] = -np.inf

            # check if visible in the second camera
            if self.tag_status[tag] == PoseStatus.OUTDATED and self.enable_second_camera:
                for detection in tag_detections.detections:
                    if detection.id[0] == tag:
                        self.tag_status[tag] = PoseStatus.FOUND_IN_CAM1
                        T_tag2cam = rutil.ros_pose_to_rigid_transform(detection.pose.pose.pose)
                        self.tag_pose[tag] = np.matmul(self.T_camera1_to_world, T_tag2cam)

        # # test covariance estimator
        # tag_pose_covariance = {}
        # for tag in self.pose_covariance_estimator.keys():
        #     cov = self.pose_covariance_estimator[tag].get_covariance()
        #     if cov is not None:
        #         tag_pose_covariance[tag] = cov
        #     else:
        #         tag_pose_covariance[tag] = np.inf
        # rospy.loginfo_throttle(0.5, f"tag pos cov: A({tag_pose_covariance[34]}) | B({tag_pose_covariance[44]}) | C({tag_pose_covariance[36]}) | D({tag_pose_covariance[32]}) | E({tag_pose_covariance[42]}) | F({tag_pose_covariance[40]})")

    def get_object_pose_from_tag_pose(self):
        # there might be more than one tag pose available
        # each one can be used to calculate the object pose
        # we use the averaged result

        # estimate tag pose covariance
        self.estimate_covariance()

        valid_transforms = []
        # for tag in self.all_tag_ids:
        for tag_id, tag in enumerate(self.pose_covariance_estimator.keys()):
            # select tag pose based on covariance
            if self.tag_status[tag] == PoseStatus.UPDATED:
                if self.tag_pose_covariance_array[tag_id] / np.min(self.tag_pose_covariance_array) > self.cov_ratio_threshold:
                    continue
                _tag_pose = self.tag_pose[tag]
                _object_pose = np.matmul(_tag_pose, self.tag_to_object_offset[tag])
                valid_transforms.append(_object_pose)

        do_pose_smoothing = False

        valid_transforms = np.array(valid_transforms).reshape(-1, 4, 4)
        
        if len(valid_transforms) <= 0:
            avg_object_tf = None
            do_pose_smoothing = False
            self.obj_status = PoseStatus.OUTDATED
        else:
            if len(valid_transforms) > 1:
                avg_object_tf = rutil.average_transforms(valid_transforms)
            elif len(valid_transforms) > 0:
                avg_object_tf = valid_transforms[0]

            # do smoothing when large deviation from previous pose is detected
            pos_deviation, rot_deviation = rutil.substract_two_transforms(
                tf0=self.last_object_tf, tf1=avg_object_tf
            )
            if np.linalg.norm(pos_deviation) > self.object_pos_diff_thresh or \
                np.linalg.norm(rot_deviation) > self.object_rot_diff_thresh:
                do_pose_smoothing = True

            # do smoothing when consecutive updates are received
            if self.obj_status in [PoseStatus.NOTFOUND, PoseStatus.OUTDATED]:
                do_pose_smoothing = False
            else:
                do_pose_smoothing = True

            self.obj_status = PoseStatus.UPDATED

        # smoothing (when: 1) large deviation from previous pose is detected and 2) )
        # if avg_object_tf is not None:
        if do_pose_smoothing:
            avg_object_tf = rutil.interpolate_two_transforms(
                tf0=self.last_object_tf, tf1=avg_object_tf, amount=self.object_tf_smooth_factor
            )

        return avg_object_tf

    def publish_object_pose(self, object_tf):
        if object_tf is None:
            return False

        self.current_object_tf = object_tf.copy()
        self.object_pose = rutil.transform_to_posevec(self.current_object_tf)

        object_tf_ros = rutil.rigidtransform_to_ros_transform(
            object_tf, parent_frame="world", child_frame=self.object_tf_name
        )
        self.latest_object_tf_ros = object_tf_ros
        self.pose_pub.publish(object_tf_ros)
        self.private_brodcaster.sendTransform(object_tf_ros)
        return True
    
    def publish_visualization_msg(self):
        if self.latest_object_tf_ros is None:
            return
        marker = get_visualization_marker(self.latest_object_tf_ros.transform)
        self.vis_pub.publish(marker)

    def compute_other_states(self):
        dt = max((self.current_tf_time - self.last_tf_time).to_sec(), 1e-5)
        _pos_vel, _ang_vel = rutil.compute_velocity_from_two_transforms(self.last_object_tf, self.current_object_tf, dt)
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
        if self.tag_status[tag] == PoseStatus.UPDATED or \
            self.tag_status[tag] == PoseStatus.FOUND_IN_CAM1:
            T_tag2world = self.tag_pose[tag]
            T_face2world = np.matmul(T_tag2world, self.face_to_tag_transform)
            return {
                "pose": rutil.transform_to_posevec(T_face2world),
                "visible": True
            }
        else:
            return {
                "pose": np.zeros(7),
                "visible": False
            }

    def main_loop(self):
        while not rospy.is_shutdown():
            # rospy.logwarn_throttle(1, "main loop")
            
            # update tag pose and status
            self.lookup_tag_poses_from_tf()

            # update object pose
            object_pose = self.get_object_pose_from_tag_pose()

            # publish object pose
            pose_is_updated = self.publish_object_pose(object_pose)
            # rospy.loginfo("pose updated: {}".format(pose_is_updated))

            # update object velocity
            if pose_is_updated:
                self.publish_visualization_msg()
                self.compute_other_states()

            self.rate.sleep()


if __name__ == "__main__":
    nh = rospy.init_node("object_pose_publisher")
    publisher = ObjectPosePublisher()
    publisher.main_loop()
