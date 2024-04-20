import os
import pdb
import time

import numpy as np
import rospkg
from leap_task_B.taskB_utils import *

from leap_hardware.leaphand_pinocchio import LeapHandPinocchio
from leap_hardware.leaphand_pinocchio_visualizer import LeapHandPinocchioVisualizer

dict_finger_grasp_point = {
    "finger0": np.array([0.0, -0.035, 0.0]),
    "finger1": np.array([0.035, 0.0, 0.0]),
    "finger2": np.array([0.0, 0.035, 0.0]),
    "thumb": np.array([-0.035, 0.0, 0.0]),
}

# canonical pose
hand_canonical_pose = np.array(
    [
        0.74250548,
        -1.12588324,
        1.27479686,
        0.93732102,
        0.33139858,
        1.7610687,
        1.24565105,
        1.21650539,
        0.32986467,
        0.10283528,
        1.20423354,
        1.03856357,
        0.81613652,
        1.34689365,
        0.8636898,
        1.17355378,
    ]
)

# feature points (surface point + [bias] silicon radius)
cube_feature_points = np.array(
    [
        [0.025, 0.0, 0.0],
        [0.025, 0.025, 0.0],
        [0.0, 0.025, 0.0],
        [-0.025, 0.025, 0.0],
        [-0.025, 0.0, 0.0],
        [-0.025, -0.025, 0.0],
        [0.0, -0.025, 0.0],
        [0.025, -0.025, 0.0],
    ]
)
cube_feature_points += 0.01 * normalize_array(cube_feature_points)
cube_feature_points[:, 2] += 0.02

expand_cube_feature_points = np.zeros((0, 3))
for i in range(len(cube_feature_points)):
    point_start = cube_feature_points[i % len(cube_feature_points)]
    point_end = cube_feature_points[(i + 1) % len(cube_feature_points)]
    point_expand = np.linspace(point_start, point_end, 20)
    if i != (len(cube_feature_points) - 1):
        expand_cube_feature_points = np.concatenate((expand_cube_feature_points, point_expand[::-1]), axis=0)
    else:
        expand_cube_feature_points = np.concatenate((expand_cube_feature_points, point_expand), axis=0)


class LeapGraspCommander:
    def __init__(self, visualize=False) -> None:
        self.cube_length = 0.05
        self._cube_transform = np.eye(4)
        self.enable_viewer = False

        self.setup_kinematics()
        if visualize:
            self.setup_meshcat_viewer()

    def setup_kinematics(self):
        rospack = rospkg.RosPack()
        urdf_path = os.path.join(rospack.get_path("my_robot_description"), "urdf/leaphand.urdf")
        self.leap_kin = LeapHandPinocchio(urdf_path=urdf_path)
        print("[SUCCESS] finished setting up leap hand kinematics")

    def setup_meshcat_viewer(self):
        self.leap_viewer = LeapHandPinocchioVisualizer(with_obj=True)
        viewer_ready = self.leap_viewer._setup_visualizer()
        if viewer_ready:
            print("[SUCCESS] finished initializing meshcat viewer")
            self.enable_viewer = True
        else:
            print("[ERROR] failed to initialize meshcat viewer")
            self.enable_viewer = False

    def view_pinocchio_model(self, hand_positions):
        if self.enable_viewer:
            _hand_dofs = hand_positions.copy()
            _obj_dofs = get_pinocchio_7x_pose_from_4x4_transform(self._cube_transform)
            _pin_dofs_all = np.concatenate((_hand_dofs, _obj_dofs))

            self.leap_viewer.display(_pin_dofs_all)

    def set_cube_transform(self, transform):
        self._cube_transform = transform

    def set_cube_transform_from_pos_quat(self, pos, quat):
        self._cube_transform = get_4x4_transform_from_pos_quat(pos, quat)

    def get_grasp_point(self, finger_name, reference="world"):
        """
        This function is deprecated
        """
        # TODO(yongpeng): remove this fixed point
        # _point = np.array([0.0, -self.cube_length / 2, 0.0])
        _point = dict_finger_grasp_point[finger_name]

        if reference == "world":
            _point_aug = np.append(_point, 1)
            _point_world = np.dot(self._cube_transform, _point_aug)
            print(f"{finger_name}'s desired tip point in world: {_point_world}")
            return _point_world[:3]
        else:
            return _point

    def get_hand_pos_from_finger_pos(self, finger_pos):
        """
        finger_pos: dict
        """
        hand_pos = np.zeros(
            16,
        )
        for finger_name, pos in finger_pos.items():
            hand_pos += self.leap_kin.jointOrderPartUserToAllPin(part_name=finger_name, q_part_normal=pos)

        return hand_pos

    def get_nearest_grasp_point_all_fingers(self, hand_pos):
        """
        hand_pos: in IsaacGym orders
        """
        # get fingertip pos
        fingertip_pos_dict = {}
        for finger_name in dict_finger_grasp_point.keys():
            finger_pos = hand_pos[self.leap_kin.part_joints_id[finger_name]]
            fingertip_pos_dict[finger_name] = self.leap_kin.getTcpGlobalPose(finger_name, finger_pos)[0].copy()
            print(f"{finger_name} FK result: {self.leap_kin.getTcpGlobalPose(finger_name, finger_pos)[0]}")

        # get nearest grasp point
        key_points_transformed = np.dot(
            self._cube_transform,
            np.append(expand_cube_feature_points, np.ones((len(expand_cube_feature_points), 1)), axis=-1).T,
        ).T[:, :3]

        key_point_free_list = np.array([True for _ in range(len(expand_cube_feature_points))], dtype=bool)
        fingertip_target_dict = {}
        for finger_name, finger_tip_pos in fingertip_pos_dict.items():
            dist_to_features = np.linalg.norm(key_points_transformed[:, :2] - finger_tip_pos[:2], axis=-1)
            dist_to_features[~key_point_free_list] = np.inf
            nearest_idx = np.argmin(dist_to_features)
            key_point_free_list[nearest_idx] = False
            fingertip_target_dict[finger_name] = key_points_transformed[nearest_idx]

        return fingertip_target_dict

    def solve_single_finger_grasp_IK(self, finger_name):
        finger_tip_point = self.get_grasp_point(finger_name)

        finger_tip_pose = get_4x4_transform_from_pos_quat(pos=finger_tip_point, quat=[0, 0, 0, 1])

        ik_weights = np.diag([1, 1, 1, 0, 0, 0]) * 10
        finger_init_pos = hand_canonical_pose[self.leap_kin.part_joints_id[finger_name]]

        sol = self.leap_kin.fingerIKSQP(
            finger_name=finger_name,
            finger_target_pose=finger_tip_pose,
            weights=ik_weights,
            finger_joint_pos_init=finger_init_pos,
        )

        hand_pos = self.leap_kin.jointOrderPartUserToAllPin(part_name=finger_name, q_part_normal=sol)
        self.view_pinocchio_model(hand_pos)

    def solve_hand_grasp_IK(self):
        ik_weights = np.diag([1, 1, 1, 0, 0, 0]) * 10

        finger_target_dict = self.get_nearest_grasp_point_all_fingers(hand_canonical_pose)

        finger_pos_sol_dict = {}

        for finger_name in dict_finger_grasp_point.keys():
            finger_tip_point = finger_target_dict[finger_name]
            finger_tip_pose = get_4x4_transform_from_pos_quat(pos=finger_tip_point, quat=[0, 0, 0, 1])
            finger_init_pos = hand_canonical_pose[self.leap_kin.part_joints_id[finger_name]]

            sol = self.leap_kin.fingerIKSQP(
                finger_name=finger_name,
                finger_target_pose=finger_tip_pose,
                weights=ik_weights,
                finger_joint_pos_init=finger_init_pos,
            )
            finger_pos_sol_dict[finger_name] = sol.copy()

        hand_pos = self.get_hand_pos_from_finger_pos(finger_pos_sol_dict)
        self.view_pinocchio_model(hand_pos)
        # time.sleep(0.5)

        return hand_pos


if __name__ == "__main__":
    grasp_commander = LeapGraspCommander(visualize=True)
    # grasp_commander.solve_single_finger_grasp_IK("thumb")

    for yaw in np.linspace(-45, 45, 30):
        obj_quat = Rot.from_euler("z", yaw, degrees=True).as_quat()
        grasp_commander.set_cube_transform_from_pos_quat(pos=[-0.05, 0.04, 0.087], quat=obj_quat)
        grasp_commander.solve_hand_grasp_IK()
