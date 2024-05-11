from __future__ import annotations

import os
import sys
import time
from math import radians
from pathlib import Path
from select import select
from typing import Literal

import numpy as np
import rospkg
import yaml
from scipy.spatial.transform import Rotation as sciR

if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import tty

# Manually add path if not in ROS
if "ROS_MASTER_URI" not in os.environ:
    current_file = Path(__file__).resolve()
    repo_root_dir = current_file.parent.parent.parent
    sys.path.append(str(repo_root_dir / "leap_model_based" / "src"))
    sys.path.append(str(repo_root_dir / "leap_utils" / "src"))
    os.environ["ROS_PACKAGE_PATH"] = str(repo_root_dir)
else:
    import rospy
    from apriltag_ros.msg import AprilTagDetectionArray
    from geometry_msgs.msg import PointStamped
    from leaphand_real import LeapHandReal
    from std_srvs.srv import Trigger

from leap_model_based.leaphand_pinocchio import LeapHandPinocchio
from leap_utils.mingrui.utils import saveDict
from leap_utils.mingrui.utils_calc import posQuat2Isometry3d

move_option = Literal["from_real", "from_last_target"]


def saveTerminalSettings():
    if sys.platform == "win32":
        return None
    return termios.tcgetattr(sys.stdin)


def getKey(settings, timeout):
    if sys.platform == "win32":
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vec_normalize(vec):
    return vec / np.linalg.norm(vec)


class LeapHandControl:
    def __init__(self, robot_model: LeapHandPinocchio, use_real_hardware: bool = False) -> None:
        self.use_real_hardware = use_real_hardware
        self.use_evaluator = False
        self.robot_model = robot_model

        self.control_rate = 50
        self.back_to_initial_config = True
        self.last_start_time = None

        # options
        self.back_to_initial_config = True
        self.second_finger_id = "finger2"  # "finger1" or "finger2"

        self.env: LeapHandReal | Simulation
        if use_real_hardware:
            self.env = LeapHandReal(control_rate=self.control_rate)
            self.rgmc_start_service = rospy.ServiceProxy("/rgcm_eval/start", Trigger)
            self.rgmc_record_service = rospy.ServiceProxy("/rgcm_eval/record", Trigger)
            self.rgmc_stop_service = rospy.ServiceProxy("/rgcm_eval/stop", Trigger)
            self.task_goal = None
            self.task_goal_sub = rospy.Subscriber("/goal_in_world", PointStamped, self.taskGoalCb)
        else:
            self.env = Simulation(robot_model=robot_model)

        # for leaphand
        self.hand_target_joint_pos = self.env.getHandJointPos().copy()

        if int(1.0 / self.env.timestep) % self.control_rate != 0:
            raise NameError("Simulation rate % control rate != 0")

    def taskGoalCb(self, msg):
        self.task_goal = msg

    def updateCurrentHandJointPos(self):
        self.hand_curr_joint_pos = self.env.getHandJointPos().copy()

    def getFingerJointPos(self, finger_name, option: move_option = "from_last_target"):
        if option == "from_real":
            finger_joint_pos = self.hand_curr_joint_pos[self.robot_model.finger_joints_id_in_hand[finger_name]]
        elif option == "from_last_target":
            finger_joint_pos = self.hand_target_joint_pos[self.robot_model.finger_joints_id_in_hand[finger_name]]
        else:
            raise NameError("Invalid option.")

        return finger_joint_pos

    def getFingerGlobalPose(self, finger_name, local_position=None, option: move_option = "from_last_target"):
        # print(option)
        # print(self.getFingerJointPos(finger_name, option))
        return self.robot_model.getTcpGlobalPose(
            finger_name, self.getFingerJointPos(finger_name, option), local_position=local_position
        )  # finger_tcp_pos, finger_tcp_quat

    def clipHandJointPosInBound(self, joint_pos):
        joints_lb = np.array(self.robot_model.joint_id_to_lower_limits)[self.robot_model.part_joints_id["hand"]]
        joints_ub = np.array(self.robot_model.joint_id_to_upper_limits)[self.robot_model.part_joints_id["hand"]]
        return np.clip(joint_pos, joints_lb, joints_ub)

    def moveHandToJointPos(self, target_joint_pos, max_speed=radians(45), option: move_option = "from_last_target"):
        """
        modify the self.hand_target_joint_pos
        """
        if option == "from_real":
            self.updateCurrentHandJointPos()
            current_joint_pos = self.hand_curr_joint_pos.copy()
        elif option == "from_last_target":
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

            for _ in range(int(1.0 / self.control_rate / self.env.timestep)):
                self.env.step()

        self.hand_target_joint_pos = target_joint_pos

    def moveHandToJointPosWithForces(
        self, target_joint_pos, max_speed=radians(45), option: move_option = "from_last_target"
    ):
        """
        modify the self.hand_target_joint_pos
        """
        if option == "from_real":
            self.updateCurrentHandJointPos()
            current_joint_pos = self.hand_curr_joint_pos.copy()
        elif option == "from_last_target":
            current_joint_pos = self.hand_target_joint_pos.copy()
        else:
            raise NameError("Invalid option.")

        target_joint_pos = np.array(target_joint_pos)
        max_step_size = max_speed * (1.0 / self.control_rate)

        n_steps = np.max(np.abs(target_joint_pos - current_joint_pos) / max_step_size)
        n_steps = np.max([1, np.ceil(n_steps)])

        for i in range(1, int(n_steps) + 1):
            t = float(i) / n_steps
            temp_target_joint_pos = current_joint_pos * (1 - t) + target_joint_pos * t
            temp_target_joint_pos = self.applyForces(temp_target_joint_pos, force_scale=0)
            self.env.ctrlHandJointPos(temp_target_joint_pos)

            for _ in range(int(1.0 / self.control_rate / self.env.timestep)):
                self.env.step()

        self.hand_target_joint_pos = target_joint_pos

    def moveHandToInitialConfig(self):
        target_hand_joint_pos = np.array(
            # [0, 0, 0, 0, -0.05368939, -0.34361163, -0.42798057, -0.35588351, 0, 0, 0, 0, np.pi / 2, 0, 0, 0]
            [0, 0, 0, 0, 0, -0.3, 0, 0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0]
        )
        self.moveHandToJointPos(target_hand_joint_pos, max_speed=radians(90), option="from_last_target")

    def applyForces(self, target_joint_pos, force_scale=100):
        self.robot_model.updateJacobians("hand", target_joint_pos)  # will call FK internally
        finger0_pos, _ = self.robot_model.getTcpGlobalPose("finger0")
        finger1_pos, _ = self.robot_model.getTcpGlobalPose(self.second_finger_id)
        thumb_pos, _ = self.robot_model.getTcpGlobalPose("thumb")
        center = (finger0_pos + finger1_pos + thumb_pos) / 3.0

        scale_factor = np.linalg.norm(center - thumb_pos)
        finger0_force = (center - finger0_pos) / scale_factor * force_scale
        finger1_force = (center - finger1_pos) / scale_factor * force_scale
        thumb_force = (center - thumb_pos) / scale_factor * force_scale

        finger0_joint_torque = self.robot_model.getGlobalJacobian("finger0", joint_part_name="finger0").T @ np.vstack(
            [finger0_force.reshape(-1, 1), np.zeros((3, 1))]
        )
        finger1_joint_torque = self.robot_model.getGlobalJacobian(
            self.second_finger_id, joint_part_name=self.second_finger_id
        ).T @ np.vstack([finger1_force.reshape(-1, 1), np.zeros((3, 1))])
        thumb_joint_torque = self.robot_model.getGlobalJacobian("thumb", joint_part_name="thumb").T @ np.vstack(
            [thumb_force.reshape(-1, 1), np.zeros((3, 1))]
        )

        kp = 800
        target_joint_pos_with_force = target_joint_pos.copy()
        target_joint_pos_with_force[self.robot_model.finger_joints_id_in_hand["finger0"]] += (
            1.0 / kp * finger0_joint_torque.reshape(-1)
        )
        target_joint_pos_with_force[self.robot_model.finger_joints_id_in_hand[self.second_finger_id]] += (
            1.0 / kp * finger1_joint_torque.reshape(-1)
        )
        target_joint_pos_with_force[self.robot_model.finger_joints_id_in_hand["thumb"]] += (
            1.0 / kp * thumb_joint_torque.reshape(-1)
        )
        return target_joint_pos_with_force

    """
        initial grasping strategy for cylinder object
    """

    def initialGrasping(self):
        self.updateCurrentHandJointPos()

        if self.use_real_hardware:
            if self.second_finger_id == "finger1":
                object_pos = np.array([-0.025, 0.016, 0.15])
            elif self.second_finger_id == "finger2":
                object_pos = np.array([-0.025, 0.035, 0.145])
            object_quat = np.array([0.70710678, 0.0, 0.0, 0.70710678])
        else:
            object_pos, object_quat = self.env.getObjectPose()

        _, finger0_quat = self.getFingerGlobalPose("finger0", local_position=[0, 0, 0], option="from_last_target")
        _, finger1_quat = self.getFingerGlobalPose(
            self.second_finger_id, local_position=[0, 0, 0], option="from_last_target"
        )
        _, thumb_quat = self.getFingerGlobalPose("thumb", local_position=[0, 0, 0], option="from_last_target")

        fingertip_radius = -0.005 if self.use_real_hardware else 0.01

        # First, move the hand to a config near the grasping config, but no contact
        # Second, move the hand to the grasping config and make the contact
        for radius in [0.03 + 0.012, 0.03 + fingertip_radius]:
            finger0_grasp_offset = np.array(
                [radius / np.sqrt(2) - 0.004, -0.02, radius / np.sqrt(2)]
            )  # x-axis offset is a BU DING
            finger1_grasp_offset = np.array([radius / np.sqrt(2), -0.02, -radius / np.sqrt(2)])
            thumb_grasp_offset = np.array([-radius, -0.02, 0.01])  # z-axis offset is a BU DING

            traj_hand_joint_pos, _ = self.robot_model.relaxedTrajectoryOptimization2(
                T=1,
                delta_t=1.0,
                object_target_pose=posQuat2Isometry3d(object_pos, object_quat),
                thumb_target_rel_pose=posQuat2Isometry3d(thumb_grasp_offset, [0, 0, 0, 1]),
                finger0_target_rel_pose=posQuat2Isometry3d(finger0_grasp_offset, [0, 0, 0, 1]),
                finger1_target_rel_pose=posQuat2Isometry3d(finger1_grasp_offset, [0, 0, 0, 1])
                if self.second_finger_id == "finger1"
                else None,
                finger2_target_rel_pose=posQuat2Isometry3d(finger1_grasp_offset, [0, 0, 0, 1])
                if self.second_finger_id == "finger2"
                else None,
                weights_object_pose=[100, 100, 100, 10, 10, 10],
                weights_rel_pose=[10, 10, 10, 0, 0, 0],
                weights_joint_vel=1e-4,
                object_pose_init=posQuat2Isometry3d(object_pos, object_quat),
                hand_joint_pos_init=self.hand_target_joint_pos.copy(),
            )

            hand_target_joint_pos = traj_hand_joint_pos[-1, :]
            self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=radians(90))

            for i in range(int(1.0 / self.env.timestep)):
                self.env.step()

        self.moveHandToJointPosWithForces(target_joint_pos=hand_target_joint_pos, max_speed=radians(90))
        for i in range(int(1.0 / self.env.timestep)):
            self.env.step()

    """
        move to task goal
    """

    def moveObject(self, target_object_pos=None, target_object_quat=None, target_rel_movement=None):
        if target_rel_movement is not None:
            object_pos, object_quat = self.env.getQRCodePose()
            target_object_pos = object_pos + target_rel_movement
            target_object_quat = object_quat.copy()
        if target_object_pos is not None:
            target_object_pos = target_object_pos.copy()
            target_object_quat = target_object_quat.copy()

        self.env.visDesiredPos(target_object_pos)
        if self.use_real_hardware:
            self.env.step()
        else:
            self.env.step(refresh=True)

        last_err = 1e10
        all_traj_hand_joint_pos = []
        # repeat the trajectory optimization, like an MPC
        # for control_iter in range(5):
        while True:
            self.updateCurrentHandJointPos()

            object_pos, object_quat = self.env.getQRCodePose()
            object_pose = posQuat2Isometry3d(object_pos, object_quat)
            finger0_pos, finger0_quat = self.getFingerGlobalPose(
                "finger0", local_position=[0, 0, 0], option="from_last_target"
            )
            finger0_pose = posQuat2Isometry3d(finger0_pos, finger0_quat)
            finger1_pos, finger1_quat = self.getFingerGlobalPose(
                self.second_finger_id, local_position=[0, 0, 0], option="from_last_target"
            )
            finger1_pose = posQuat2Isometry3d(finger1_pos, finger1_quat)
            thumb_pos, thumb_quat = self.getFingerGlobalPose(
                "thumb", local_position=[0, 0, 0], option="from_last_target"
            )
            thumb_pose = posQuat2Isometry3d(thumb_pos, thumb_quat)

            thumb_pose_in_object = np.linalg.inv(object_pose) @ thumb_pose
            finger0_pose_in_object = np.linalg.inv(object_pose) @ finger0_pose
            finger1_pose_in_object = np.linalg.inv(object_pose) @ finger1_pose

            print("Start trajectory optimization ...")
            traj_hand_joint_pos, planned_object_err = self.robot_model.relaxedTrajectoryOptimization2(
                T=5 if control_iter == 0 else 1,
                delta_t=0.5 if control_iter == 0 else 0.2,
                object_target_pose=posQuat2Isometry3d(target_object_pos, target_object_quat),
                thumb_target_rel_pose=thumb_pose_in_object,
                finger0_target_rel_pose=finger0_pose_in_object,
                finger1_target_rel_pose=finger1_pose_in_object if self.second_finger_id == "finger1" else None,
                finger2_target_rel_pose=finger1_pose_in_object if self.second_finger_id == "finger2" else None,
                weights_object_pose=[10, 10, 10, 0.01, 0.01, 0.0],
                weights_rel_pose=[10, 10, 10, 0.001, 0.001, 0.001],
                weights_joint_vel=1e-4,
                object_pose_init=object_pose,
                hand_joint_pos_init=self.hand_target_joint_pos.copy(),
            )

            for i, hand_joint_pos in enumerate(traj_hand_joint_pos):
                self.moveHandToJointPosWithForces(target_joint_pos=hand_joint_pos, max_speed=radians(20))
            all_traj_hand_joint_pos.extend(traj_hand_joint_pos)

            # wait for one second
            for _ in range(int(0.5 / self.env.timestep)):
                self.env.step()

            object_pos, _ = self.env.getQRCodePose()
            control_err = np.linalg.norm(object_pos - target_object_pos)
            print("planned_err: ", planned_object_err)
            print("actual_err: ", control_err)
            if control_err < planned_object_err:  # the criterion for switching to the next waypoint
                break
            if control_err >= last_err:
                break
            last_err = control_err

        # print("Please press 'Enter' to continue.")
        # input()

        if self.use_real_hardware and self.use_evaluator:
            self.rgmc_record_service()

        # back to the intial configuration (currently, sliding easily happens in simulation and leads to failure)
        if self.back_to_initial_config:
            for i, hand_joint_pos in enumerate(np.flip(all_traj_hand_joint_pos, axis=0)):
                self.moveHandToJointPosWithForces(target_joint_pos=hand_joint_pos, max_speed=radians(30))
            for i in range(int(1.0 / self.env.timestep)):  # wait for one second
                self.env.step()

        # print("self.hand_target_joint_pos: ", self.hand_target_joint_pos)

        return control_err

    """
        for grasping of arbitrary object
    """

    def humanGuidedGrasp(self):
        settings = saveTerminalSettings()

        # First, human guided movement to contact configuration
        while True:
            rospy.loginfo_once("Press 'r' to record the current configuration.")

            self.updateCurrentHandJointPos()
            finger0_target_pos, _ = self.getFingerGlobalPose("finger0", option="from_real")
            finger1_target_pos, _ = self.getFingerGlobalPose(self.second_finger_id, option="from_real")
            thumb_target_pos, _ = self.getFingerGlobalPose("thumb", option="from_real")
            finger0_target_pos[2] = 0.125
            finger1_target_pos[2] = 0.125
            thumb_target_pos[2] = 0.125

            finger0_res_joint_pos = self.robot_model.fingerIKSQP(
                "finger0",
                finger_target_pose=posQuat2Isometry3d(finger0_target_pos, [0, 0, 0, 1]),
                weights=np.diag([10, 10, 10, 0, 0, 0]),
                finger_joint_pos_init=self.getFingerJointPos("finger0", option="from_real"),
                local_position=[0, 0, 0],
            )
            finger1_res_joint_pos = self.robot_model.fingerIKSQP(
                self.second_finger_id,
                finger_target_pose=posQuat2Isometry3d(finger1_target_pos, [0, 0, 0, 1]),
                weights=np.diag([10, 10, 10, 0, 0, 0]),
                finger_joint_pos_init=self.getFingerJointPos(self.second_finger_id, option="from_real"),
                local_position=[0, 0, 0],
            )
            thumb_res_joint_pos = self.robot_model.fingerIKSQP(
                "thumb",
                finger_target_pose=posQuat2Isometry3d(thumb_target_pos, [0, 0, 0, 1]),
                weights=np.diag([10, 10, 10, 0, 0, 0]),
                finger_joint_pos_init=self.getFingerJointPos("thumb", option="from_real"),
                local_position=[0, 0, 0],
            )

            hand_target_joint_pos = self.hand_target_joint_pos.copy()
            hand_target_joint_pos[0:4] = finger0_res_joint_pos
            if self.second_finger_id == "finger1":
                hand_target_joint_pos[4:8] = finger1_res_joint_pos
            elif self.second_finger_id == "finger2":
                hand_target_joint_pos[8:12] = finger1_res_joint_pos
            hand_target_joint_pos[12:16] = thumb_res_joint_pos
            self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=radians(90))

            key = getKey(settings, 0.05)
            if key == "r":
                print("Record the current hand joint positions.")
                break

        # Second, close the fingers towards the center to grasp the object
        finger0_pos, _ = self.getFingerGlobalPose("finger0", option="from_last_target")
        finger1_pos, _ = self.getFingerGlobalPose(self.second_finger_id, option="from_last_target")
        thumb_pos, _ = self.getFingerGlobalPose("thumb", option="from_last_target")

        center = (finger0_pos + finger1_pos + thumb_pos) / 3.0
        finger0_target_pos = finger0_pos + 0.02 * vec_normalize(center - finger0_pos)
        finger1_target_pos = finger1_pos + 0.02 * vec_normalize(center - finger1_pos)
        thumb_target_pos = thumb_pos + 0.02 * vec_normalize(center - thumb_pos)

        traj_hand_joint_pos, _ = self.robot_model.relaxedTrajectoryOptimization2(
            T=1,
            delta_t=1.0,
            object_target_pose=posQuat2Isometry3d([0, 0, 0], [0, 0, 0, 1]),
            thumb_target_rel_pose=posQuat2Isometry3d(thumb_target_pos, [0, 0, 0, 1]),
            finger0_target_rel_pose=posQuat2Isometry3d(finger0_target_pos, [0, 0, 0, 1]),
            finger1_target_rel_pose=posQuat2Isometry3d(finger1_target_pos, [0, 0, 0, 1])
            if self.second_finger_id == "finger1"
            else None,
            finger2_target_rel_pose=posQuat2Isometry3d(finger1_target_pos, [0, 0, 0, 1])
            if self.second_finger_id == "finger2"
            else None,
            weights_object_pose=[100, 100, 100, 10, 10, 10],
            weights_rel_pose=[10, 10, 10, 0, 0, 0],
            weights_joint_vel=1e-4,
            object_pose_init=posQuat2Isometry3d([0, 0, 0], [0, 0, 0, 1]),
            hand_joint_pos_init=self.hand_target_joint_pos.copy(),
        )

        hand_target_joint_pos = traj_hand_joint_pos[-1, :]
        self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=radians(30))

        for i in range(int(1.0 / self.env.timestep)):  # wait for one second
            self.env.step()


def test1():
    rospack = rospkg.RosPack()
    urdf_path = os.path.join(rospack.get_path("my_robot_description"), "urdf/leaphand_taskA.urdf")
    robot_model = LeapHandPinocchio(urdf_path=urdf_path)

    task_cfg_path = os.path.join(rospack.get_path("leap_task_A"), "config/taskA_corners.yaml")
    with open(task_cfg_path) as stream:
        task_cfg = yaml.safe_load(stream)
        print(task_cfg)
    target_waypoints = task_cfg["waypoints"]

    ctrl = LeapHandControl(robot_model=robot_model, use_real_hardware=False)

    for i in range(1):  # repeat the whole task
        ctrl.moveHandToInitialConfig()
        ctrl.initialGrasping()

        object_pos, object_quat = ctrl.env.getQRCodePose()
        all_control_err = []

        for waypoint_idx, target_waypoint in enumerate(target_waypoints):
            if waypoint_idx != 2:
                continue

            print(f"waypoint_idx: {waypoint_idx}")

            target_pos = np.array([target_waypoint["x"], target_waypoint["y"], target_waypoint["z"]])
            target_object_pos = object_pos + target_pos

            control_err = ctrl.moveObject(target_object_pos=target_object_pos, target_object_quat=object_quat)
            all_control_err.append(control_err)

            print("Please press 'Enter' to continue ...")
            input()

        print("all_control_err: ", all_control_err)
        print("ave_control_err: ", np.mean(all_control_err))

        ctrl.env.physics.reset()


def read_configs():
    rospack = rospkg.RosPack()
    urdf_path = Path(rospack.get_path("my_robot_description")) / "urdf" / "leaphand_taskA.urdf"
    robot_model = LeapHandPinocchio(urdf_path=str(urdf_path))

    task_cfg_path = Path(rospack.get_path("leap_task_A")) / "config" / "taskA_corners.yaml"
    with open(task_cfg_path) as stream:
        task_cfg = yaml.safe_load(stream)
        print(task_cfg)
    target_waypoints = task_cfg["waypoints"]
    return robot_model, target_waypoints


def real_test():
    rospy.init_node("leaphand_real")
    robot_model, target_waypoints = read_configs()

    ctrl = LeapHandControl(robot_model=robot_model, use_real_hardware=True)
    all_control_err = []

    # print(ctrl.hand_target_joint_pos)

    ctrl.moveHandToInitialConfig()
    ctrl.initialGrasping()

    print("Please press 'Enter' to continue ...")
    input()

    ctrl.last_start_time = time.perf_counter()
    object_pos, object_quat = ctrl.env.getQRCodePose()

    for i in range(10):
        for waypoint_idx, target_waypoint in enumerate(target_waypoints):
            if waypoint_idx != 2:
                continue

            target_pos = np.array([target_waypoint["x"], target_waypoint["y"], target_waypoint["z"]])
            target_object_pos = object_pos + target_pos

            print(f"waypoint_idx: {waypoint_idx}")

            control_err = ctrl.moveObject(target_object_pos=target_object_pos, target_object_quat=object_quat)
            all_control_err.append(control_err)

            # print("Please press 'Enter' to continue ...")
            # input()

        print("all_control_err: ", all_control_err)
        print("ave_control_err: ", np.mean(all_control_err))


def real_test_with_evaluator():
    rospy.init_node("leaphand_real")
    robot_model, _ = read_configs()

    ctrl = LeapHandControl(robot_model=robot_model, use_real_hardware=True)
    ctrl.use_evaluator = True

    all_control_err = []

    ctrl.moveHandToInitialConfig()
    ctrl.initialGrasping()

    print("Please press 'Enter' to continue ...")
    input()

    ctrl.rgmc_start_service()  # call the evaluator to start
    ctrl.last_start_time = time.perf_counter()
    object_pos, object_quat = ctrl.env.getQRCodePose()

    while ctrl.task_goal is None:
        rospy.loginfo_once("Waiting for task goal ...")

    last_target_tag_pos_in_world = None
    while True:
        target_tag_pos_in_world = np.array([ctrl.task_goal.point.x, ctrl.task_goal.point.y, ctrl.task_goal.point.z])

        if last_target_tag_pos_in_world is not None and np.all(target_tag_pos_in_world == last_target_tag_pos_in_world):
            break
        last_target_tag_pos_in_world = target_tag_pos_in_world.copy()

        target_object_pos = target_tag_pos_in_world
        control_err = ctrl.moveObject(target_object_pos=target_object_pos, target_object_quat=object_quat)
        all_control_err.append(control_err)

        # print("Please press 'Enter' to continue ...")
        # input()

    ctrl.rgmc_stop_service()

    print("all_control_err: ", all_control_err)
    print("ave_control_err: ", np.mean(all_control_err))


def real_test_new_object():
    rospy.init_node("leaphand_real")
    robot_model, _ = read_configs()

    ctrl = LeapHandControl(robot_model=robot_model, use_real_hardware=True)

    ctrl.moveHandToInitialConfig()
    ctrl.initialGrasping()
    ctrl.humanGuidedGrasp()


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

    ctrl = LeapHandControl(robot_model=robot_model, use_real_hardware=False)

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
    # test1()  # simulation
    real_test()  # real-world
    # real_test_with_evaluator()
    # real_test_new_object()

    # searchBestObjectPosition()


# -------------------- backup -----------------------------


# def initialGrasping(self):
#     self.updateCurrentHandJointPos()

#     if self.use_real_hardware:
#         if self.second_finger_id == "finger1":
#             object_pos = np.array([-0.025, 0.016, 0.15])
#         elif self.second_finger_id == "finger2":
#             object_pos = np.array([-0.025, 0.035, 0.145])
#         object_quat = np.array([0.70710678, 0.0, 0.0, 0.70710678])
#     else:
#         object_pos, object_quat = self.env.getObjectPose()
#     object_rot_mat = sciR.from_quat(object_quat).as_matrix()

#     _, finger0_quat = self.getFingerGlobalPose("finger0", local_position=[0, 0, 0], option="from_last_target")
#     _, finger1_quat = self.getFingerGlobalPose(
#         self.second_finger_id, local_position=[0, 0, 0], option="from_last_target"
#     )
#     _, thumb_quat = self.getFingerGlobalPose("thumb", local_position=[0, 0, 0], option="from_last_target")

#     fingertip_radius = 0.00 if self.use_real_hardware else 0.01

#     finger0_target_quat = finger0_quat.copy()
#     finger1_target_quat = finger1_quat.copy()
#     thumb_target_quat = thumb_quat.copy()

#     # First, move the hand to a config near the grasping config, but no contact
#     # Second, move the hand to the grasping config and make the contact
#     for radius in [0.03 + 0.012, 0.03 + fingertip_radius]:
#         finger0_grasp_offset = np.array([radius / np.sqrt(2), -0.02, radius / np.sqrt(2)])
#         finger1_grasp_offset = np.array([radius / np.sqrt(2), -0.02, -radius / np.sqrt(2)])
#         thumb_grasp_offset = np.array([-radius, -0.02, 0.01])  # z-axis offset is a buding

#         finger0_target_pos = object_rot_mat @ finger0_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)
#         finger1_target_pos = object_rot_mat @ finger1_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)
#         thumb_target_pos = object_rot_mat @ thumb_grasp_offset.reshape(-1, 1) + object_pos.reshape(-1, 1)

#         finger0_res_joint_pos = self.robot_model.fingerIKSQP(
#             "finger0",
#             finger_target_pose=posQuat2Isometry3d(finger0_target_pos, finger0_target_quat),
#             weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
#             finger_joint_pos_init=self.getFingerJointPos("finger0", option="from_last_target"),
#             local_position=[0, 0, 0],
#         )
#         finger1_res_joint_pos = self.robot_model.fingerIKSQP(
#             self.second_finger_id,
#             finger_target_pose=posQuat2Isometry3d(finger1_target_pos, finger1_target_quat),
#             weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
#             finger_joint_pos_init=self.getFingerJointPos(self.second_finger_id, option="from_last_target"),
#             local_position=[0, 0, 0],
#         )
#         thumb_res_joint_pos = self.robot_model.fingerIKSQP(
#             "thumb",
#             finger_target_pose=posQuat2Isometry3d(thumb_target_pos, thumb_target_quat),
#             weights=np.diag([10, 10, 5, 0.001, 0, 0.001]),
#             finger_joint_pos_init=self.getFingerJointPos("thumb", option="from_last_target"),
#             local_position=[0, 0, 0],
#         )

#         hand_target_joint_pos = self.hand_target_joint_pos.copy()
#         hand_target_joint_pos[0:4] = finger0_res_joint_pos
#         if self.second_finger_id == "finger1":
#             hand_target_joint_pos[4:8] = finger1_res_joint_pos
#         elif self.second_finger_id == "finger2":
#             hand_target_joint_pos[8:12] = finger1_res_joint_pos
#         hand_target_joint_pos[12:16] = thumb_res_joint_pos
#         self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=radians(90))

#         for i in range(int(1.0 / self.env.timestep)):
#             self.env.step()

#     # print("Please press 'Enter' to continue.")
#     # input()

#     self.moveHandToJointPosWithForces(target_joint_pos=hand_target_joint_pos, max_speed=radians(90))
#     # hand_target_joint_pos_with_force = self.applyForces(hand_target_joint_pos, force_scale=1000)
#     # self.env.ctrlHandJointPos(target_joint_pos=hand_target_joint_pos_with_force)
#     for i in range(int(1.0 / self.env.timestep)):
#         self.env.step()
