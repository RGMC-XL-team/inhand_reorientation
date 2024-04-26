import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import rospkg
import yaml
from scipy.spatial.transform import Rotation as sciR

# check if rospy is installed
if importlib.util.find_spec("rospy") is None:
    # manually add the path
    current_file = Path(__file__).resolve()
    repo_root_dir = current_file.parent.parent.parent
    sys.path.append(str(repo_root_dir / "leap_model_based" / "src"))
    sys.path.append(str(repo_root_dir / "leap_utils" / "src"))
    os.environ["ROS_PACKAGE_PATH"] = str(repo_root_dir)
else:
    import rospy
    from leaphand_real import LeapHandReal

from leaphand_mujoco import Simulation

from leap_model_based.leaphand_pinocchio import LeapHandPinocchio
from leap_utils.mingrui.utils import saveDict
from leap_utils.mingrui.utils_calc import posQuat2Isometry3d


# -----------------------------------------------------------
class LeapHandControl:
    def __init__(self, robot_model, b_use_ros=False):
        self.b_use_ros = b_use_ros
        self.robot_model = robot_model

        self.control_rate = 50

        if b_use_ros:
            self.env = LeapHandReal(control_rate=self.control_rate)
        else:
            self.env = Simulation(robot_model=robot_model)

        # for leaphand
        self.hand_target_joint_pos = self.env.getHandJointPos().copy()

        if int(1.0 / self.env.timestep) % self.control_rate != 0:
            raise NameError("Simulation rate % control rate != 0")

    # --------------------------------------
    def updateCurrentHandJointPos(self):
        self.hand_curr_joint_pos = self.env.getHandJointPos().copy()

    # --------------------------------------
    def getArmJointPos(self):
        return np.asarray([])

    # --------------------------------------
    def getFingerJointPos(self, finger_name, option="target"):
        if option == "real":
            finger_joint_pos = self.hand_curr_joint_pos[self.robot_model.finger_joints_id_in_hand[finger_name]]
        elif option == "target":
            finger_joint_pos = self.hand_target_joint_pos[self.robot_model.finger_joints_id_in_hand[finger_name]]
        else:
            raise NameError("Invalid option.")

        return finger_joint_pos

    # --------------------------------------
    def getFingerLocalPose(self, finger_name, local_position=None, option="target"):
        finger_joint_pos = self.getFingerJointPos(finger_name, option)
        finger_tcp_pos, finger_tcp_quat = self.robot_model.getFingerTcpLocalPose(
            finger_name, finger_joint_pos=finger_joint_pos, local_position=local_position
        )
        return finger_tcp_pos, finger_tcp_quat

    # --------------------------------------
    def getFingerGlobalPose(self, finger_name, local_position=None, option="target"):
        finger_tcp_pos, finger_tcp_quat = self.robot_model.getTcpGlobalPose(
            finger_name, self.getFingerJointPos(finger_name, option), local_position=local_position
        )
        # finger_tcp_pos, finger_tcp_quat = self.robot_model.getTcpGlobalPose(finger_name,
        #                                                         finger_joint_pos=self.getFingerJointPos(finger_name, option),
        #                                                         local_position=local_position)
        return finger_tcp_pos, finger_tcp_quat

    # --------------------------------------
    def clipHandJointPosInBound(self, joint_pos):
        joints_lb = np.array(self.robot_model.joint_id_to_lower_limits)[self.robot_model.part_joints_id["hand"]]
        joints_ub = np.array(self.robot_model.joint_id_to_upper_limits)[self.robot_model.part_joints_id["hand"]]
        joint_pos = np.clip(joint_pos, joints_lb, joints_ub)
        return joint_pos

    # --------------------------------------
    """
        modify the self.hand_target_joint_pos
    """

    def moveHandToJointPos(self, target_joint_pos, max_speed=np.deg2rad(45), option="target"):
        if option == "real":
            self.updateCurrentHandJointPos()
            current_joint_pos = self.hand_curr_joint_pos.copy()
        elif option == "target":
            current_joint_pos = self.hand_target_joint_pos.copy()
        else:
            raise NameError("Invalid option.")

        target_joint_pos = np.array(target_joint_pos)
        max_step_size = max_speed * (1.0 / self.control_rate)

        n_steps = np.max(np.abs(target_joint_pos - current_joint_pos) / max_step_size)
        n_steps = np.ceil(n_steps)

        for i in range(1, int(n_steps) + 1):
            t = float(i) / n_steps
            temp_target_joint_pos = current_joint_pos * (1 - t) + target_joint_pos * t
            self.env.ctrlHandJointPos(temp_target_joint_pos)

            for i in range(int(1.0 / self.control_rate / self.env.timestep)):
                self.env.step()

        self.hand_target_joint_pos = target_joint_pos

    # --------------------------------------
    def moveHandToInitialConfig(self):
        target_hand_joint_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0])
        self.moveHandToJointPos(target_hand_joint_pos, max_speed=np.deg2rad(90), option="real")

    # --------------------------------------
    def initialGrasping(self):
        self.updateCurrentHandJointPos()

        object_pos, object_quat = self.env.getObjectPose()
        object_rot_mat = sciR.from_quat(object_quat).as_matrix()

        finger0_pos, finger0_quat = self.getFingerGlobalPose("finger0", local_position=[0, 0, 0], option="real")
        finger1_pos, finger1_quat = self.getFingerGlobalPose("finger1", local_position=[0, 0, 0], option="real")
        thumb_pos, thumb_quat = self.getFingerGlobalPose("thumb", local_position=[0, 0, 0], option="real")

        fingertip_radius = 0.008

        finger0_target_quat = finger0_quat.copy()
        finger1_target_quat = finger1_quat.copy()
        thumb_target_quat = thumb_quat.copy()

        # for horizontal cylinder
        # finger0_grasp_offset = np.array([0.03 + fingertip_radius, -0.025, -0.0])
        # finger1_grasp_offset = np.array([0.03 + fingertip_radius, 0.025, 0.0])
        # thumb_grasp_offset = np.array([-0.03 - fingertip_radius, 0, 0])

        # for vertical cylinder
        # First, move the hand to a config near the grasping config, but no contact
        radius = 0.03 + fingertip_radius + 0.005
        finger0_grasp_offset = np.array([radius / np.sqrt(2), -0.02, radius / np.sqrt(2)])
        finger1_grasp_offset = np.array([radius / np.sqrt(2), -0.02, -radius / np.sqrt(2)])
        thumb_grasp_offset = np.array([-radius, -0.02, 0])

        finger0_target_pos = object_rot_mat @ finger0_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)
        finger1_target_pos = object_rot_mat @ finger1_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)
        thumb_target_pos = object_rot_mat @ thumb_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)

        finger0_res_joint_pos = self.robot_model.fingerIKSQP(
            "finger0",
            finger_target_pose=posQuat2Isometry3d(finger0_target_pos, finger0_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("finger0", option="real"),
            local_position=[0, 0, 0],
        )
        finger1_res_joint_pos = self.robot_model.fingerIKSQP(
            "finger1",
            finger_target_pose=posQuat2Isometry3d(finger1_target_pos, finger1_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("finger1", option="real"),
            local_position=[0, 0, 0],
        )
        thumb_res_joint_pos = self.robot_model.fingerIKSQP(
            "thumb",
            finger_target_pose=posQuat2Isometry3d(thumb_target_pos, thumb_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("thumb", option="real"),
            local_position=[0, 0, 0],
        )

        hand_target_joint_pos = np.zeros((16,))
        hand_target_joint_pos[0:4] = finger0_res_joint_pos
        hand_target_joint_pos[4:8] = finger1_res_joint_pos
        hand_target_joint_pos[12:16] = thumb_res_joint_pos
        self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=np.deg2rad(90))

        for i in range(int(1.0 / self.env.timestep)):
            self.env.step()

        # Second, move the hand to the grasping config and make the contact
        radius = 0.03 + fingertip_radius
        finger0_grasp_offset = np.array([radius / np.sqrt(2), -0.02, radius / np.sqrt(2)])
        finger1_grasp_offset = np.array([radius / np.sqrt(2), -0.02, -radius / np.sqrt(2)])
        thumb_grasp_offset = np.array([-radius, -0.02, 0])

        finger0_target_pos = object_rot_mat @ finger0_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)
        finger1_target_pos = object_rot_mat @ finger1_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)
        thumb_target_pos = object_rot_mat @ thumb_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)

        finger0_res_joint_pos = self.robot_model.fingerIKSQP(
            "finger0",
            finger_target_pose=posQuat2Isometry3d(finger0_target_pos, finger0_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("finger0", option="real"),
            local_position=[0, 0, 0],
        )
        finger1_res_joint_pos = self.robot_model.fingerIKSQP(
            "finger1",
            finger_target_pose=posQuat2Isometry3d(finger1_target_pos, finger1_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("finger1", option="real"),
            local_position=[0, 0, 0],
        )
        thumb_res_joint_pos = self.robot_model.fingerIKSQP(
            "thumb",
            finger_target_pose=posQuat2Isometry3d(thumb_target_pos, thumb_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("thumb", option="real"),
            local_position=[0, 0, 0],
        )

        hand_target_joint_pos = np.zeros((16,))
        hand_target_joint_pos[0:4] = finger0_res_joint_pos
        hand_target_joint_pos[4:8] = finger1_res_joint_pos
        hand_target_joint_pos[12:16] = thumb_res_joint_pos
        self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=np.deg2rad(90))

        for i in range(int(1.0 / self.env.timestep)):
            self.env.step()

    # --------------------------------------
    def initialGraspingFixed(self):
        self.updateCurrentHandJointPos()

        object_pos = np.array([-0.02, 0.015, 0.12])

        _, finger0_quat = self.getFingerGlobalPose("finger0", local_position=[0, 0, 0], option="real")
        _, finger1_quat = self.getFingerGlobalPose("finger1", local_position=[0, 0, 0], option="real")
        _, thumb_quat = self.getFingerGlobalPose("thumb", local_position=[0, 0, 0], option="real")

        fingertip_radius = 0
        finger0_grasp_offset = np.array([0.03 + fingertip_radius, -0.025, -0.0])
        finger1_grasp_offset = np.array([0.03 + fingertip_radius, 0.025, 0.0])
        thumb_grasp_offset = np.array([-0.03 - fingertip_radius, 0, 0])

        finger0_target_pos = finger0_grasp_offset + object_pos
        finger1_target_pos = finger1_grasp_offset + object_pos
        thumb_target_pos = thumb_grasp_offset + object_pos

        finger0_target_quat = finger0_quat.copy()
        finger1_target_quat = finger1_quat.copy()
        thumb_target_quat = thumb_quat.copy()

        # 每个手指分别求的效率更高
        finger0_res_joint_pos = self.robot_model.fingerIKSQP(
            "finger0",
            finger_target_pose=posQuat2Isometry3d(finger0_target_pos, finger0_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("finger0", option="real"),
            local_position=[0, 0, 0],
        )
        finger1_res_joint_pos = self.robot_model.fingerIKSQP(
            "finger1",
            finger_target_pose=posQuat2Isometry3d(finger1_target_pos, finger1_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("finger1", option="real"),
            local_position=[0, 0, 0],
        )
        thumb_res_joint_pos = self.robot_model.fingerIKSQP(
            "thumb",
            finger_target_pose=posQuat2Isometry3d(thumb_target_pos, thumb_target_quat),
            weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
            finger_joint_pos_init=self.getFingerJointPos("thumb", option="real"),
            local_position=[0, 0, 0],
        )

        hand_target_joint_pos = np.zeros((16,))
        hand_target_joint_pos[0:4] = finger0_res_joint_pos
        hand_target_joint_pos[4:8] = finger1_res_joint_pos
        hand_target_joint_pos[12:16] = thumb_res_joint_pos
        self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=np.deg2rad(90))

        for i in range(int(1.0 / self.env.timestep)):
            self.env.step()

    # --------------------------------------
    def moveObject(self, target_object_pos=None, target_object_quat=None, target_rel_movement=None):
        if target_rel_movement is not None:
            object_pos, object_quat = self.env.getQRCodePose()
            target_object_pos = object_pos + target_rel_movement
            target_object_quat = object_quat.copy()
        if target_object_pos is not None:
            target_object_pos = target_object_pos.copy()
            target_object_quat = target_object_quat.copy()

        self.env.visDesiredPos(target_object_pos)
        self.env.step(refresh=True)

        # repeat the trajectory optimization, like an MPC (very poor performance currently)
        for control_iter in range(1):
            self.updateCurrentHandJointPos()

            object_pos, object_quat = self.env.getQRCodePose()
            object_pose = posQuat2Isometry3d(object_pos, object_quat)

            finger0_pos, finger0_quat = self.getFingerGlobalPose("finger0", local_position=[0, 0, 0], option="target")
            finger0_pose = posQuat2Isometry3d(finger0_pos, finger0_quat)
            finger1_pos, finger1_quat = self.getFingerGlobalPose("finger1", local_position=[0, 0, 0], option="target")
            finger1_pose = posQuat2Isometry3d(finger1_pos, finger1_quat)
            thumb_pos, thumb_quat = self.getFingerGlobalPose("thumb", local_position=[0, 0, 0], option="target")
            thumb_pose = posQuat2Isometry3d(thumb_pos, thumb_quat)

            thumb_pose_in_object = np.linalg.inv(object_pose) @ thumb_pose
            finger0_pose_in_object = np.linalg.inv(object_pose) @ finger0_pose
            finger1_pose_in_object = np.linalg.inv(object_pose) @ finger1_pose

            print("Start trajectory optimization ...")
            traj_hand_joint_pos, planned_object_err = self.robot_model.relaxedTrajectoryOptimization2(
                T=3,
                delta_t=0.5,
                object_target_pose=posQuat2Isometry3d(target_object_pos, target_object_quat),
                thumb_target_rel_pose=thumb_pose_in_object,
                finger0_target_rel_pose=finger0_pose_in_object,
                finger1_target_rel_pose=finger1_pose_in_object,
                weights_object_pose=[10, 10, 10, 0.01, 0.01, 0.0],
                weights_rel_pose=[10, 10, 10, 0, 0, 0],
                weights_joint_vel=1e-4,
                object_pose_init=object_pose,
                hand_joint_pos_init=self.hand_curr_joint_pos.copy(),
            )

            for i, hand_joint_pos in enumerate(traj_hand_joint_pos):
                self.moveHandToJointPos(target_joint_pos=hand_joint_pos, max_speed=np.deg2rad(10))
                # print(f"Reached waypoint {i}.")

            # wait for one second
            for i in range(int(1.0 / self.env.timestep)):
                self.env.step()

            object_pos, _ = self.env.getQRCodePose()
            control_err = np.linalg.norm(object_pos - target_object_pos)
            print("planned_err: ", planned_object_err)
            print("actual_err: ", control_err)
            if control_err < planned_object_err:  # the criterion for switching to the next waypoint
                break

        b_back_to_initial_config = False
        # back to the intial configuration (currently, sliding easily happens in simulation and leads to failure)
        if b_back_to_initial_config:
            for i, hand_joint_pos in enumerate(np.flip(traj_hand_joint_pos, axis=0)):
                self.moveHandToJointPos(target_joint_pos=hand_joint_pos, max_speed=np.deg2rad(10))
            for i in range(int(1.0 / self.env.timestep)):  # wait for one second
                self.env.step()

        return control_err


# --------------------------------------
def test1():
    rospack = rospkg.RosPack()
    urdf_path = os.path.join(rospack.get_path("my_robot_description"), "urdf/leaphand.urdf")
    robot_model = LeapHandPinocchio(urdf_path=urdf_path)

    task_cfg_path = os.path.join(rospack.get_path("leap_task_A"), "config/taskA_corners.yaml")
    with open(task_cfg_path) as stream:
        task_cfg = yaml.safe_load(stream)
        print(task_cfg)
    target_waypoints = task_cfg["waypoints"]

    ctrl = LeapHandControl(robot_model=robot_model, b_use_ros=False)

    for i in range(1):  # repeat the whole task
        ctrl.moveHandToInitialConfig()
        ctrl.initialGrasping()

        object_pos, object_quat = ctrl.env.getQRCodePose()
        all_control_err = []

        for waypoint_idx, target_waypoint in enumerate(target_waypoints):
            print(f"waypoint_idx: {waypoint_idx}")

            target_pos = np.array([target_waypoint["x"], target_waypoint["y"], target_waypoint["z"]])
            target_object_pos = object_pos + target_pos

            control_err = ctrl.moveObject(target_object_pos=target_object_pos, target_object_quat=object_quat)
            all_control_err.append(control_err)

        print("all_control_err: ", all_control_err)
        print("ave_control_err: ", np.mean(all_control_err))

        ctrl.env.physics.reset()


# --------------------------------------
def test2():
    rospy.init_node("leaphand_real")

    rospack = rospkg.RosPack()
    urdf_path = os.path.join(rospack.get_path("my_robot_description"), "urdf/leaphand.urdf")
    robot_model = LeapHandPinocchio(urdf_path=urdf_path)

    task_cfg_path = os.path.join(rospack.get_path("leap_task_A"), "config/taskA.yaml")
    with open(task_cfg_path) as stream:
        task_cfg = yaml.safe_load(stream)
        print(task_cfg)
    target_waypoints = task_cfg["waypoints"]

    ctrl = LeapHandControl(robot_model=robot_model, b_use_ros=True)

    ctrl.moveHandToInitialConfig()
    ctrl.initialGraspingFixed()

    print("Please press 'Enter' to continue ...")
    input()

    object_pos, object_quat = ctrl.env.getQRCodePose()
    for waypoint_idx, target_waypoint in enumerate(target_waypoints):
        target_pos = np.array([target_waypoint["x"], target_waypoint["y"], target_waypoint["z"]])
        target_object_pos = object_pos + target_pos

        print(f"waypoint_idx: {waypoint_idx}")

        control_err = ctrl.moveObject(target_object_pos=target_object_pos, target_object_quat=object_quat)

        # print("Please press 'Enter' to continue ...")
        # input()


# --------------------------------------
def searchBestObjectPosition():
    rospack = rospkg.RosPack()
    urdf_path = os.path.join(rospack.get_path("my_robot_description"), "urdf/leaphand.urdf")
    robot_model = LeapHandPinocchio(urdf_path=urdf_path)

    task_cfg_path = os.path.join(rospack.get_path("leap_task_A"), "config/taskA_corners.yaml")
    with open(task_cfg_path) as stream:
        task_cfg = yaml.safe_load(stream)
        print(task_cfg)
    target_waypoints = task_cfg["waypoints"]

    ctrl = LeapHandControl(robot_model=robot_model, b_use_ros=False)

    res = {"object_pos": [], "all_control_err": [], "ave_control_err": []}

    for object_x in np.arange(-0.02, 0.02, 0.005):
        for object_y in np.arange(-0.02, 0.02, 0.005):
            for object_z in [0]:
                # for object_x in [-0.005]:
                #     for object_y in [0.01]:
                #         for object_z in [0]:
                # change the position of the target object
                ctrl.env.physics.bind(ctrl.env.model.find("body", "cylinder_mujoco/object")).pos = [
                    object_x,
                    object_y,
                    object_z,
                ]

                ctrl.moveHandToInitialConfig()
                ctrl.initialGrasping()

                object_pos, object_quat = ctrl.env.getQRCodePose()
                all_control_err = []
                for waypoint_idx, target_waypoint in enumerate(target_waypoints):
                    print(f"waypoint_idx: {waypoint_idx}")
                    target_pos = np.array([target_waypoint["x"], target_waypoint["y"], target_waypoint["z"]])
                    target_object_pos = object_pos + target_pos

                    control_err = ctrl.moveObjectMPC(
                        target_object_pos=target_object_pos, target_object_quat=object_quat
                    )

                    all_control_err.append(control_err)
                    if control_err > 0.1:
                        break

                print("object pos: ", [object_x, object_y, object_z])
                print("ave_control_err: ", np.mean(all_control_err))
                res["object_pos"].append([object_x, object_y, object_z])
                res["all_control_err"].append(all_control_err)
                res["ave_control_err"].append(np.mean(all_control_err))

                ctrl.env.physics.reset()

                saveDict(path=os.path.join(rospack.get_path("leap_task_A"), "results"), dict=res)


# ----------------------------------------------
if __name__ == "__main__":
    test1()  # simulation
    # test2() # real-world

    # searchBestObjectPosition()


# ---------------------- backup ---------------------------


# # --------------------------------------
#     def moveObject(self, target_object_pos=None, target_object_quat=None, target_rel_movement=None):
#         if target_rel_movement is not None:
#             object_pos, object_quat = self.env.getQRCodePose()
#             target_object_pos = object_pos + target_rel_movement
#             target_object_quat = object_quat.copy()
#         if target_object_pos is not None:
#             target_object_pos = target_object_pos.copy()
#             target_object_quat = target_object_quat.copy()

#         self.env.visDesiredPos(target_object_pos)
#         self.env.step()

#         self.updateCurrentHandJointPos()

#         object_pos, object_quat = self.env.getQRCodePose()
#         object_pose = posQuat2Isometry3d(object_pos, object_quat)

#         finger0_pos, finger0_quat = self.getFingerGlobalPose("finger0", local_position=[0, 0, 0], option="target")
#         finger0_pose = posQuat2Isometry3d(finger0_pos, finger0_quat)
#         finger1_pos, finger1_quat = self.getFingerGlobalPose("finger1", local_position=[0, 0, 0], option="target")
#         finger1_pose = posQuat2Isometry3d(finger1_pos, finger1_quat)
#         thumb_pos, thumb_quat = self.getFingerGlobalPose("thumb", local_position=[0, 0, 0], option="target")
#         thumb_pose = posQuat2Isometry3d(thumb_pos, thumb_quat)

#         thumb_pose_in_object = np.linalg.inv(object_pose) @ thumb_pose
#         finger0_pose_in_object = np.linalg.inv(object_pose) @ finger0_pose
#         finger1_pose_in_object = np.linalg.inv(object_pose) @ finger1_pose

#         print("Start trajectory optimization ...")
#         traj_hand_joint_pos = self.robot_model.relaxedTrajectoryOptimization2(
#             T=5,
#             delta_t=0.5,
#             object_target_pose=posQuat2Isometry3d(target_object_pos, target_object_quat),
#             thumb_target_rel_pose=thumb_pose_in_object,
#             finger0_target_rel_pose=finger0_pose_in_object,
#             finger1_target_rel_pose=finger1_pose_in_object,
#             weights_object_pose=[10, 10, 10, 0.01, 0.01, 0.0],
#             weights_rel_pose=[10, 10, 10, 0.0, 0.0, 0.0],
#             weights_joint_vel=1e-4,
#             object_pose_init=object_pose,
#             hand_joint_pos_init=self.hand_curr_joint_pos.copy(),
#         )

#         for i, hand_joint_pos in enumerate(traj_hand_joint_pos):
#             self.moveHandToJointPos(target_joint_pos=hand_joint_pos, max_speed=np.deg2rad(10))
#             print(f"Reached waypoint {i}.")

#         # wait for one second
#         for i in range(int(1.0 / self.env.timestep)):
#             self.env.step()

#         object_pos, _ = self.env.getQRCodePose()
#         control_err = np.linalg.norm(object_pos - target_object_pos)
#         print("control_err: ", control_err)

#         return control_err

#     # --------------------------------------
#     def moveObjectMPC(self, target_object_pos=None, target_object_quat=None, target_rel_movement=None):
#         if target_rel_movement is not None:
#             object_pos, object_quat = self.env.getQRCodePose()
#             target_object_pos = object_pos + target_rel_movement
#             target_object_quat = object_quat.copy()
#         if target_object_pos is not None:
#             target_object_pos = target_object_pos.copy()
#             target_object_quat = target_object_quat.copy()

#         for control_iter in range(1):
#             self.env.visDesiredPos(target_object_pos)
#             self.env.step(refresh=True)

#             self.updateCurrentHandJointPos()

#             object_pos, object_quat = self.env.getQRCodePose()
#             object_pose = posQuat2Isometry3d(object_pos, object_quat)

#             finger0_pos, finger0_quat = self.getFingerGlobalPose("finger0", local_position=[0, 0, 0], option="target")
#             finger0_pose = posQuat2Isometry3d(finger0_pos, finger0_quat)
#             finger1_pos, finger1_quat = self.getFingerGlobalPose("finger1", local_position=[0, 0, 0], option="target")
#             finger1_pose = posQuat2Isometry3d(finger1_pos, finger1_quat)
#             thumb_pos, thumb_quat = self.getFingerGlobalPose("thumb", local_position=[0, 0, 0], option="target")
#             thumb_pose = posQuat2Isometry3d(thumb_pos, thumb_quat)

#             thumb_pose_in_object = np.linalg.inv(object_pose) @ thumb_pose
#             finger0_pose_in_object = np.linalg.inv(object_pose) @ finger0_pose
#             finger1_pose_in_object = np.linalg.inv(object_pose) @ finger1_pose

#             print("Start trajectory optimization ...")
#             traj_hand_joint_pos = self.robot_model.relaxedTrajectoryOptimization2(
#                 T=1,
#                 delta_t=0.5,
#                 object_target_pose=posQuat2Isometry3d(target_object_pos, target_object_quat),
#                 thumb_target_rel_pose=thumb_pose_in_object,
#                 finger0_target_rel_pose=finger0_pose_in_object,
#                 finger1_target_rel_pose=finger1_pose_in_object,
#                 weights_object_pose=[10, 10, 10, 0.01, 0.01, 0.0],
#                 weights_rel_pose=[10, 10, 10, 0, 0, 0],
#                 weights_joint_vel=1e-4,
#                 object_pose_init=object_pose,
#                 hand_joint_pos_init=self.hand_curr_joint_pos.copy(),
#             )

#             for i, hand_joint_pos in enumerate(traj_hand_joint_pos):
#                 self.moveHandToJointPos(target_joint_pos=hand_joint_pos, max_speed=np.deg2rad(10))
#                 print(f"Reached waypoint {i}.")

#             # wait
#             for i in range(int(0.2 / self.env.timestep)):
#                 self.env.step()

#             object_pos, _ = self.env.getQRCodePose()
#             control_err = np.linalg.norm(object_pos - target_object_pos)
#             print("control_err: ", control_err)

#             if control_err < 1e-3:
#                 break

#         # wait for one second
#         for i in range(int(1.0 / self.env.timestep)):
#             self.env.step()

#         object_pos, _ = self.env.getQRCodePose()
#         control_err = np.linalg.norm(object_pos - target_object_pos)
#         print("control_err: ", control_err)

#         return control_err
