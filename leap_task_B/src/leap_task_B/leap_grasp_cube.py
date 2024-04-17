import os
import pdb
import time

import numpy as np
import rospkg
from taskB_utils import *

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

# feature points
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


class LeapGraspCommander:
    def __init__(self, visualize=False) -> None:
        self.cube_length = 0.05
        self._cube_transform = np.eye(4)

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
        pdb.set_trace()

    def solve_hand_grasp_IK(self):
        ik_weights = np.diag([1, 1, 1, 0, 0, 0]) * 10

        hand_pos = np.zeros(
            16,
        )

        for finger_name in dict_finger_grasp_point.keys():
            finger_tip_point = self.get_grasp_point(finger_name)
            finger_tip_pose = get_4x4_transform_from_pos_quat(pos=finger_tip_point, quat=[0, 0, 0, 1])
            finger_init_pos = hand_canonical_pose[self.leap_kin.part_joints_id[finger_name]]

            sol = self.leap_kin.fingerIKSQP(
                finger_name=finger_name,
                finger_target_pose=finger_tip_pose,
                weights=ik_weights,
                finger_joint_pos_init=finger_init_pos,
            )
            print("hand pos: ", hand_pos)
            hand_pos += self.leap_kin.jointOrderPartUserToAllPin(part_name=finger_name, q_part_normal=sol)

        self.view_pinocchio_model(hand_pos)
        time.sleep(0.5)


if __name__ == "__main__":
    grasp_commander = LeapGraspCommander(visualize=True)
    # grasp_commander.solve_single_finger_grasp_IK("thumb")

    for yaw in np.linspace(-45, 45, 30):
        obj_quat = Rot.from_euler("z", yaw, degrees=True).as_quat()
        grasp_commander.set_cube_transform_from_pos_quat(pos=[-0.05, 0.04, 0.087], quat=obj_quat)
        grasp_commander.solve_hand_grasp_IK()
