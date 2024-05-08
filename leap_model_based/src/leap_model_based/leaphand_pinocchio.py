import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pinocchio as pin
import rospkg
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as sciR

# Manually add path if not in ROS
if "ROS_MASTER_URI" not in os.environ:
    current_file = Path(__file__).resolve()
    repo_root_dir = current_file.parents[3]
    sys.path.append(str(repo_root_dir / "leap_utils" / "src"))
    os.environ["ROS_PACKAGE_PATH"] = str(repo_root_dir)

import leap_utils.mingrui.utils_calc as ucalc


class LeapHandPinocchio:
    def __init__(self, urdf_path: str) -> None:
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # q = pin.randomConfiguration(self.model)
        # print(q.shape)

        # print([name for name in self.model.names])

        self.part_joints_name = {}
        self.part_joints_name["finger0"] = ["joint_0", "joint_1", "joint_2", "joint_3"]
        self.part_joints_name["finger1"] = ["joint_4", "joint_5", "joint_6", "joint_7"]
        self.part_joints_name["finger2"] = ["joint_8", "joint_9", "joint_10", "joint_11"]
        self.part_joints_name["thumb"] = ["joint_12", "joint_13", "joint_14", "joint_15"]
        self.part_joints_name["hand"] = (
            self.part_joints_name["finger0"]
            + self.part_joints_name["finger1"]
            + self.part_joints_name["finger2"]
            + self.part_joints_name["thumb"]
        )

        self.tcp_links_name = {}
        self.tcp_links_name["finger0"] = "finger1_tip_center"
        self.tcp_links_name["finger1"] = "finger2_tip_center"
        self.tcp_links_name["finger2"] = "finger3_tip_center"
        self.tcp_links_name["thumb"] = "thumb_tip_center"

        self.finger_joints_id_in_hand = {
            "finger0": [0, 1, 2, 3],
            "finger1": [4, 5, 6, 7],
            "finger2": [8, 9, 10, 11],
            "thumb": [12, 13, 14, 15],
        }

        # id: 0 ~ 15, used for specifying input joint pos
        self.joints_id_to_lower_limit = self.model.lowerPositionLimit
        self.joints_id_to_upper_limit = self.model.upperPositionLimit

        self.joints_name_to_id = {}
        for id, name in enumerate(self.model.names):
            if name != "universe":
                self.joints_name_to_id[name] = id - 1  # 只有和 pinocchio 交互的时候才用到这个顺序

        self.part_joints_id = {}
        for part_name, joints_name in self.part_joints_name.items():
            joints_id = [self.joints_name_to_id[name] for name in joints_name]
            self.part_joints_id[part_name] = joints_id

        self.tcp_links_id = {}
        for part_name, link_name in self.tcp_links_name.items():
            self.tcp_links_id[part_name] = self.model.getFrameId(link_name)

    def checkJointDim(self, part_name, joints):
        if isinstance(joints, list):
            return len(joints) == len(self.part_joints_id[part_name])
        else:
            return joints.size == len(self.part_joints_id[part_name])

    def jointOrderPartUserToAllPin(self, part_name, q_part_normal):
        q_all_pin = np.zeros((len(self.joints_name_to_id),))
        q_all_pin[self.part_joints_id[part_name]] = q_part_normal
        return q_all_pin

    def updateFK(self, part_name, part_joint_pos):
        """
        input:
            hand_joint_pos: the joint order is 0, 1, 2, ... 15, the same as the joint name
        """
        if not self.checkJointDim(part_name, part_joint_pos):
            raise NameError("The dim of input joint is wrong !")

        q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
        pin.framesForwardKinematics(self.model, self.data, q_pin)

    def updateFKvel(self, part_name, part_joint_pos, part_joint_vel):
        """
        input:
            hand_joint_pos: the joint order is 0, 1, 2, ... 15, the same as the joint name
            hand_joint_vel: the joint order is 0, 1, 2, ... 15, the same as the joint name
        """
        if not self.checkJointDim(part_name, part_joint_pos):
            raise NameError("The dim of input joint is wrong !")

        if not self.checkJointDim(part_name, part_joint_vel):
            raise NameError("The dim of input joint is wrong !")

        q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
        v_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_vel)
        # This only updates joint data
        pin.forwardKinematics(self.model, self.data, q_pin, v_pin)
        # Update frame data
        pin.updateFramePlacements(self.model, self.data)

    def getTcpGlobalPose(self, part_name, part_joint_pos=None, local_position=None):
        if part_joint_pos is not None:
            self.updateFK(part_name, part_joint_pos)
        return self.getFrameGlobalPose(self.tcp_links_name[part_name])

    def getFrameGlobalPose(self, frame_name):
        data = self.data.oMf[self.model.getFrameId(frame_name)]
        pos = data.translation
        rot_mat = data.rotation
        return pos, sciR.from_matrix(rot_mat).as_quat()

    def getFrameGlobalVelocity(self, frame_name):
        """
        This should be called after updateFKvel
        """
        data = pin.getFrameVelocity(
            self.model, self.data, self.model.getFrameId(frame_name), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        linear = data.linear
        angular = data.angular
        return linear, angular

    def updateJacobians(self, part_name, part_joint_pos):
        if not self.checkJointDim(part_name, part_joint_pos):
            raise NameError("The dim of input joint is wrong !")
        if part_name != "hand":
            raise NameError("The part_name must be 'hand'.")

        q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
        pin.computeJointJacobians(self.model, self.data, q_pin)  # call FK internally
        pin.updateFramePlacements(self.model, self.data)

    def getGlobalJacobian(self, part_name, part_joint_pos=None, local_position=None, joint_part_name=None):
        """
        input:
            part_name: which part the target tcp belongs to.
            joint_part_name: which part the joints belong to.
        """
        if part_joint_pos is not None:
            q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
            jaco_pin = pin.computeFrameJacobian(
                self.model,
                self.data,
                q=q_pin,
                frame_id=self.tcp_links_id[part_name],
                reference_frame=pin.LOCAL_WORLD_ALIGNED,
            )
        else:
            jaco_pin = pin.getFrameJacobian(
                self.model, self.data, frame_id=self.tcp_links_id[part_name], reference_frame=pin.LOCAL_WORLD_ALIGNED
            )

        return jaco_pin[:, self.part_joints_id[joint_part_name]]

    def getFrameGlobalJacobian(self, frame_name, joint_part_name=None):
        jaco_pin = pin.getFrameJacobian(
            self.model, self.data, frame_id=self.model.getFrameId(frame_name), reference_frame=pin.LOCAL_WORLD_ALIGNED
        )
        return jaco_pin[:, self.part_joints_id[joint_part_name]]

    def calcFingerPoseError(self, finger_name, finger_target_pose, finger_joint_pos, local_position):
        err = np.zeros((6,))
        finger_target_pos = finger_target_pose[0:3]
        finger_target_ori = sciR.from_quat(finger_target_pose[3:7])

        self.updateFK(finger_name, finger_joint_pos)
        finger_pos, finger_quat = self.getTcpGlobalPose(finger_name, local_position=local_position)
        err[0:3] = finger_pos - finger_target_pos
        err[3:6] = (sciR.from_quat(finger_quat) * finger_target_ori.inv()).as_rotvec()

        err = err.reshape(-1, 1)
        self.curr_err = err

        return err

    def fingerDiffIK(  # noqa: PLR0913
        self,
        finger_name,
        finger_target_pose,
        q_init,
        q_step: float,
        pos_tol: float,
        rot_tol: float,
        q_tol: float,
        local_position,
    ):
        finger_joint_pos_init = q_init[self.finger_joints_id_in_hand[finger_name]]
        finger_lower_limit = self.joints_id_to_lower_limit[self.part_joints_id[finger_name]]
        finger_upper_limit = self.joints_id_to_upper_limit[self.part_joints_id[finger_name]]

        curr_err = self.curr_err
        curr_joint_pos = finger_joint_pos_init.copy()

        while abs(np.linalg.norm(curr_err[0:3])) > pos_tol or abs(np.linalg.norm(curr_err[3:6])):
            jaco = self.getGlobalJacobian(finger_name, curr_joint_pos, local_position, finger_name)
            q_err = np.linalg.pinv(jaco) @ curr_err
            curr_joint_pos += q_err / np.linalg.norm(q_err) * q_step

            if np.any(np.abs(curr_joint_pos - finger_lower_limit) < q_tol) or np.any(
                np.abs(curr_joint_pos - finger_upper_limit) < q_tol
            ):
                return

    def fingerIKSQP(self, finger_name, finger_target_pose, weights, finger_joint_pos_init, local_position=None):  # noqa: PLR0913
        t_ik = perf_counter()

        finger_target_pos, finger_target_ori = ucalc.isometry3dToPosOri(finger_target_pose)
        self.curr_err = None

        def calcTargetError(finger_joint_pos):
            t1 = perf_counter()
            err = np.zeros((6,))

            self.updateFK(finger_name, finger_joint_pos)
            finger_pos, finger_quat = self.getTcpGlobalPose(finger_name, local_position=local_position)
            err[0:3] = finger_pos - finger_target_pos
            err[3:6] = (sciR.from_quat(finger_quat) * finger_target_ori.inv()).as_rotvec()
            # print("err: ", err)

            err = err.reshape(-1, 1)
            self.curr_err = err

            # print(f"Time cost of calc err: {perf_counter() - t1}")
            return err

        def objectFunction(finger_joint_pos):
            err = calcTargetError(finger_joint_pos)
            cost = 1.0 / 2.0 * err.T @ weights @ err
            return cost[0, 0]

        def objectJacobian(finger_joint_pos):
            t1 = perf_counter()
            err = self.curr_err.copy()

            # get all jacobians
            jaco = self.getGlobalJacobian(
                finger_name, finger_joint_pos, local_position=local_position, joint_part_name=finger_name
            )

            object_jaco = err.T @ weights @ jaco

            # print("Jacobian calculation, Use current error.")
            return object_jaco.reshape(-1)

        joint_pos_init = finger_joint_pos_init.copy()
        # bounds
        joint_pos_lb = np.array(self.joints_id_to_lower_limit)[self.part_joints_id[finger_name]]
        joint_pos_ub = np.array(self.joints_id_to_upper_limit)[self.part_joints_id[finger_name]]
        joint_pos_bounds = [(joint_pos_lb[i], joint_pos_ub[i]) for i in range(joint_pos_lb.shape[0])]

        res = minimize(
            fun=objectFunction,
            jac=objectJacobian,
            x0=joint_pos_init,
            bounds=joint_pos_bounds,
            method="SLSQP",
            options={"ftol": 1e-8, "disp": True},
        )
        res_joint_pos = res.x.reshape(-1)

        print(f"Time cost of SQP_IK: {perf_counter() - t_ik}")
        return res_joint_pos

    def relaxedTrajectoryOptimization(  # noqa: PLR0913, PLR0915
        self,
        T,
        delta_t,
        object_target_pose,
        object_pose_in_thumb,
        finger0_target_rel_pose=None,
        finger1_target_rel_pose=None,
        finger2_target_rel_pose=None,  # relative to thumb tip frame
        weights_object_pose=None,
        weights_rel_pose=None,
        weights_joint_vel=None,
        hand_joint_pos_init=[0] * 16,
        finger0_local_position=None,
        finger1_local_position=None,
        finger2_local_position=None,
        thumb_local_position=None,
    ):
        """
        input:
            finger_local_position: currently not supported
        """
        t_ik = perf_counter()
        hand_joint_pos_init = np.asarray(hand_joint_pos_init)
        object_target_pos, object_target_ori = ucalc.isometry3dToPosOri(object_target_pose)

        # add object frame
        thumb_tcp_parent_joint = self.model.frames[self.tcp_links_id["thumb"]].parent
        thumb_tcp_placement = self.model.frames[self.tcp_links_id["thumb"]].placement
        object_placement_in_thumb = pin.SE3(object_pose_in_thumb[0:3, 0:3], object_pose_in_thumb[0:3, 3])
        self.model.addFrame(
            pin.Frame(
                name="object",
                parent_joint=thumb_tcp_parent_joint,
                parent_frame=self.tcp_links_id["thumb"],
                placement=thumb_tcp_placement * object_placement_in_thumb,  # relative to parent joint
                type=pin.OP_FRAME,
            )
        )
        self.data = self.model.createData()

        fingers_target_rel_pose = {}
        if finger0_target_rel_pose is not None:
            fingers_target_rel_pose["finger0"] = finger0_target_rel_pose
        if finger1_target_rel_pose is not None:
            fingers_target_rel_pose["finger1"] = finger1_target_rel_pose
        if finger2_target_rel_pose is not None:
            fingers_target_rel_pose["finger2"] = finger2_target_rel_pose
        fingers_target_rel_pos_ori = {}
        for finger_name, target_rel_pose in fingers_target_rel_pose.items():
            fingers_target_rel_pos_ori[finger_name] = ucalc.isometry3dToPosOri(target_rel_pose)

        fingers_local_position = {}
        fingers_local_position["finger0"] = finger0_local_position
        fingers_local_position["finger1"] = finger1_local_position
        fingers_local_position["finger2"] = finger2_local_position
        fingers_local_position["thumb"] = thumb_local_position

        # init value
        traj_joint_pos_init = np.tile(hand_joint_pos_init, T + 1)
        # bounds
        joint_pos_lb = np.array(self.joints_id_to_lower_limit)[self.part_joints_id["hand"]]
        joint_pos_ub = np.array(self.joints_id_to_upper_limit)[self.part_joints_id["hand"]]
        traj_joint_pos_bounds = [
            (joint_pos_lb[i], joint_pos_ub[i]) for _ in range(T + 1) for i in range(joint_pos_lb.shape[0])
        ]

        weights = np.diag(
            weights_object_pose
            + weights_rel_pose * (T * len(fingers_target_rel_pose))
            + [weights_joint_vel] * (T * len(self.part_joints_id["hand"]))
        )
        self.curr_err = None

        def calcTargetError(traj_hand_joint_pos):
            t1 = perf_counter()
            traj_hand_joint_pos = traj_hand_joint_pos.reshape(T + 1, -1)
            err_list = []

            # ------------ object pose in world frame at {T} -------------
            self.updateFK("hand", traj_hand_joint_pos[-1, :])
            object_pos, object_quat = self.getFrameGlobalPose("object")
            err_list.extend(object_pos - object_target_pos)
            err_list.extend((sciR.from_quat(object_quat) * object_target_ori.inv()).as_rotvec())

            # ------------ finger pose in thumb frame at {t = 1,...T} ------------
            fingers_pos = {}
            fingers_quat = {}
            for finger_name in list(fingers_target_rel_pose.keys()) + ["thumb"]:
                fingers_pos[finger_name] = np.zeros((T, 3))
                fingers_quat[finger_name] = np.zeros((T, 4))

            for t in range(1, T + 1):
                self.updateFK("hand", traj_hand_joint_pos[t, :])
                for finger_name in list(fingers_target_rel_pose.keys()) + ["thumb"]:
                    fingers_pos[finger_name][t - 1, :], fingers_quat[finger_name][t - 1, :] = self.getTcpGlobalPose(
                        finger_name, local_position=fingers_local_position[finger_name]
                    )

            thumb_local_pose_inv = ucalc.batchIsometry3dInverse(
                ucalc.batchPosQuat2Isometry3d(fingers_pos["thumb"], fingers_quat["thumb"])
            )

            for finger_name, (finger_target_rel_pos, finger_target_rel_ori) in fingers_target_rel_pos_ori.items():
                finger_pos, finger_quat = fingers_pos[finger_name], fingers_quat[finger_name]
                finger_pose = ucalc.batchPosQuat2Isometry3d(finger_pos, finger_quat)
                finger_rel_pose = np.matmul(thumb_local_pose_inv, finger_pose)
                finger_rel_pos_err = finger_rel_pose[:, 0:3, 3] - np.tile(finger_target_rel_pos, (T, 1))
                finger_rel_ori_err = (
                    sciR.from_matrix(finger_rel_pose[:, 0:3, 0:3]) * finger_target_rel_ori.inv()
                ).as_rotvec()
                finger_rel_pose_err = np.hstack([finger_rel_pos_err, finger_rel_ori_err])
                err_list.extend(finger_rel_pose_err.reshape(-1))

            # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
            traj_hand_joint_vel = (traj_hand_joint_pos[1:, :] - traj_hand_joint_pos[0:-1, :]) / delta_t
            err_list.extend(traj_hand_joint_vel.reshape(-1))

            err = np.asarray(err_list).reshape(-1, 1)
            self.curr_err = err
            return err

        def objectFunction(traj_hand_joint_pos):
            err = calcTargetError(traj_hand_joint_pos)
            cost = 1.0 / 2.0 * err.T @ weights @ err
            return cost[0, 0]

        def objectJacobian(traj_hand_joint_pos):
            t1 = perf_cmeounter()
            traj_hand_joint_pos = traj_hand_joint_pos.reshape(T + 1, -1)
            err = self.curr_err.copy()
            jaco_list = []

            # ------------ object pose in world frame at {T} -------------
            self.updateJacobians("hand", traj_hand_joint_pos[-1, :])
            object_global_jaco = self.getFrameGlobalJacobian("object", joint_part_name="hand")
            whole_jaco = np.zeros((6, T + 1, 16))
            whole_jaco[:, -1, :] = object_global_jaco
            jaco_list.append(whole_jaco.reshape(6, -1))

            # ------------ finger pose in thumb frame at {t = 1,...T} ------------
            fingers_pos = {}
            fingers_quat = {}
            fingers_jaco = {}
            for finger_name in list(fingers_target_rel_pose.keys()) + ["thumb"]:
                fingers_pos[finger_name] = np.zeros((T, 3))
                fingers_quat[finger_name] = np.zeros((T, 4))
                fingers_jaco[finger_name] = np.zeros((T, 6, 4))

            for t in range(1, T + 1):
                self.updateJacobians("hand", traj_hand_joint_pos[t, :])  # will call FK internally
                for finger_name in list(fingers_target_rel_pose.keys()) + ["thumb"]:
                    fingers_pos[finger_name][t - 1, :], fingers_quat[finger_name][t - 1, :] = self.getTcpGlobalPose(
                        finger_name, local_position=fingers_local_position[finger_name]
                    )
                    fingers_jaco[finger_name][t - 1, :, :] = self.getGlobalJacobian(
                        finger_name, local_position=fingers_local_position[finger_name], joint_part_name=finger_name
                    )

            thumb_local_pose_inv = ucalc.batchIsometry3dInverse(
                ucalc.batchPosQuat2Isometry3d(fingers_pos["thumb"], fingers_quat["thumb"])
            )
            transformed_thumb_local_jaco = np.matmul(
                ucalc.batchDiagRotMat(thumb_local_pose_inv[:, 0:3, 0:3]), fingers_jaco["thumb"]
            )

            for finger_name, _ in fingers_target_rel_pos_ori.items():
                finger_pose = ucalc.batchPosQuat2Isometry3d(fingers_pos[finger_name], fingers_quat[finger_name])
                finger_rel_pose = np.matmul(thumb_local_pose_inv, finger_pose)
                finger_rel_pos = finger_rel_pose[:, 0:3, 3]

                temp_jaco = np.zeros((T, 6, 16))  # 自变量只包括t时刻的关节
                temp_jaco[:, :, self.finger_joints_id_in_hand[finger_name]] = np.matmul(
                    ucalc.batchDiagRotMat(thumb_local_pose_inv[:, 0:3, 0:3]), fingers_jaco[finger_name]
                )
                temp_jaco[:, :, self.finger_joints_id_in_hand["thumb"]] = -np.matmul(
                    ucalc.wrenchTransformationMatrix(finger_rel_pos), transformed_thumb_local_jaco
                )

                whole_jaco = np.zeros((T, 6, T + 1, 16))
                for t in range(T):
                    whole_jaco[t, :, t + 1, :] = temp_jaco[t]

                jaco_list.append(whole_jaco.reshape(T * 6, -1))

            # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
            whole_jaco = np.zeros((T, 16, T + 1, 16))
            for t in range(1, T + 1):
                whole_jaco[t - 1, :, t, :] = 1.0 / delta_t * np.eye(16)
                whole_jaco[t - 1, :, t - 1, :] = -1.0 / delta_t * np.eye(16)
            jaco_list.append(whole_jaco.reshape(T * 16, -1))

            jaco = np.vstack(jaco_list)
            object_jaco = err.T @ weights @ jaco

            # print(f"Time cost of calc jacobian: {time.time() - t1}")
            return object_jaco.reshape(-1)

        def x0EqConstraint(hand_joint_pos):
            hand_joint_pos = hand_joint_pos.reshape(T + 1, -1)
            eq = hand_joint_pos[0, :] - hand_joint_pos_init  # = 0
            return eq

        constraints_list = [dict(type="eq", fun=x0EqConstraint)]

        res = minimize(
            fun=objectFunction,
            jac=objectJacobian,
            constraints=constraints_list,
            x0=traj_joint_pos_init,
            bounds=traj_joint_pos_bounds,
            method="SLSQP",
            options={"ftol": 1e-10, "disp": True, "maxiter": 1000},
        )
        res_joint_pos = res.x.reshape(-1)

        print(f"Time cost of SQP_IK: {perf_counter() - t_ik}")

        traj_hand_joint_pos = res_joint_pos.reshape(T + 1, -1)
        return traj_hand_joint_pos

    def relaxedTrajectoryOptimization2(  # noqa: PLR0913, PLR0915
        self,
        T,
        delta_t,
        object_target_pose,
        thumb_target_rel_pose,
        finger0_target_rel_pose=None,
        finger1_target_rel_pose=None,
        finger2_target_rel_pose=None,
        weights_object_pose=None,
        weights_rel_pose=None,
        weights_joint_vel=None,
        object_pose_init=None,
        hand_joint_pos_init=[0] * 16,
        finger0_local_position=None,
        finger1_local_position=None,
        finger2_local_position=None,
        thumb_local_position=None,
    ):
        t_ik = perf_counter()
        hand_joint_pos_init = np.asarray(hand_joint_pos_init)
        object_target_pos, object_target_ori = ucalc.isometry3dToPosOri(object_target_pose)

        fingers_target_rel_pose = {}
        if thumb_target_rel_pose is not None:
            fingers_target_rel_pose["thumb"] = thumb_target_rel_pose
        if finger0_target_rel_pose is not None:
            fingers_target_rel_pose["finger0"] = finger0_target_rel_pose
        if finger1_target_rel_pose is not None:
            fingers_target_rel_pose["finger1"] = finger1_target_rel_pose
        if finger2_target_rel_pose is not None:
            fingers_target_rel_pose["finger2"] = finger2_target_rel_pose
        fingers_target_rel_pos_ori = {}
        for finger_name, target_rel_pose in fingers_target_rel_pose.items():
            fingers_target_rel_pos_ori[finger_name] = ucalc.isometry3dToPosOri(target_rel_pose)

        fingers_local_position = {}
        fingers_local_position["finger0"] = finger0_local_position
        fingers_local_position["finger1"] = finger1_local_position
        fingers_local_position["finger2"] = finger2_local_position
        fingers_local_position["thumb"] = thumb_local_position

        # init value
        object_pos_init, object_rotvec_init = ucalc.isometry3dToPosRotVec(object_pose_init)
        val_init = np.concatenate([object_pos_init, object_rotvec_init, hand_joint_pos_init], axis=0)
        traj_val_init = np.tile(val_init, T + 1)
        # bounds
        joint_pos_lb = np.array(self.joints_id_to_lower_limit)[self.part_joints_id["hand"]]
        joint_pos_ub = np.array(self.joints_id_to_upper_limit)[self.part_joints_id["hand"]]
        val_bounds = [(-100, 100) for _ in range(6)] + [
            (joint_pos_lb[i], joint_pos_ub[i]) for i in range(joint_pos_lb.shape[0])
        ]
        traj_val_bounds = [val_bound for _ in range(T + 1) for val_bound in val_bounds]

        weights = np.diag(
            weights_object_pose
            + weights_rel_pose * (T * len(fingers_target_rel_pose))
            + [weights_joint_vel] * (T * len(self.part_joints_id["hand"]))
        )
        self.curr_err = None

        def calcTargetError(val):
            t1 = perf_counter()
            val = val.reshape(T + 1, -1)
            traj_object_pos, traj_object_rotvec, traj_hand_joint_pos = val[:, 0:3], val[:, 3:6], val[:, 6 : 6 + 16]

            err_list = []
            # ------------ object pose in world frame at {T} -------------
            object_pos, object_rotvec = traj_object_pos[-1, :], traj_object_rotvec[-1, :]
            err_list.extend(object_pos - object_target_pos)
            err_list.extend((sciR.from_rotvec(object_rotvec) * object_target_ori.inv()).as_rotvec())

            # print("object_pose_err: ", err_list)

            # ------------ finger pose in object frame at {t = 1,...T} ------------
            fingers_pos = {}
            fingers_quat = {}
            for finger_name in fingers_target_rel_pose.keys():
                fingers_pos[finger_name] = np.zeros((T, 3))
                fingers_quat[finger_name] = np.zeros((T, 4))

            for t in range(1, T + 1):
                self.updateFK("hand", traj_hand_joint_pos[t, :])
                for finger_name in fingers_target_rel_pose.keys():
                    fingers_pos[finger_name][t - 1, :], fingers_quat[finger_name][t - 1, :] = self.getTcpGlobalPose(
                        finger_name, local_position=fingers_local_position[finger_name]
                    )

            object_pose_inv = ucalc.batchIsometry3dInverse(
                ucalc.batchPosRotVec2Isometry3d(traj_object_pos[1:, :], traj_object_rotvec[1:, :])
            )

            for finger_name, (finger_target_rel_pos, finger_target_rel_ori) in fingers_target_rel_pos_ori.items():
                finger_pos, finger_quat = fingers_pos[finger_name], fingers_quat[finger_name]
                finger_pose = ucalc.batchPosQuat2Isometry3d(finger_pos, finger_quat)
                finger_rel_pose = np.matmul(object_pose_inv, finger_pose)
                finger_rel_pos_err = finger_rel_pose[:, 0:3, 3] - np.tile(finger_target_rel_pos, (T, 1))
                finger_rel_ori_err = (
                    sciR.from_matrix(finger_rel_pose[:, 0:3, 0:3]) * finger_target_rel_ori.inv()
                ).as_rotvec()
                finger_rel_pose_err = np.hstack([finger_rel_pos_err, finger_rel_ori_err])
                err_list.extend(finger_rel_pose_err.reshape(-1))

            # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
            traj_hand_joint_vel = (traj_hand_joint_pos[1:, :] - traj_hand_joint_pos[0:-1, :]) / delta_t
            err_list.extend(traj_hand_joint_vel.reshape(-1))

            err = np.asarray(err_list).reshape(-1, 1)
            self.curr_err = err
            return err

        def objectFunction(val):
            err = calcTargetError(val)
            cost = 1.0 / 2.0 * err.T @ weights @ err
            return cost[0, 0]

        def objectJacobian(val):
            val = val.reshape(T + 1, -1)
            traj_object_pos, traj_object_rotvec, traj_hand_joint_pos = val[:, 0:3], val[:, 3:6], val[:, 6 : 6 + 16]

            err = self.curr_err.copy()
            jaco_list = []

            # ------------ object pose in world frame at {T} -------------
            whole_jaco = np.zeros((6, T + 1, 6 + 16))
            whole_jaco[0:3, -1, 0:3] = np.eye(3)
            whole_jaco[3:6, -1, 3:6] = ucalc.jacoDeRotVecToAngularVel(traj_object_rotvec[-1, :])
            jaco_list.append(whole_jaco.reshape(6, -1))

            # ------------ finger pose in thumb frame at {t = 1,...T} ------------
            fingers_pos = {}
            fingers_quat = {}
            fingers_jaco = {}
            for finger_name in fingers_target_rel_pose.keys():
                fingers_pos[finger_name] = np.zeros((T, 3))
                fingers_quat[finger_name] = np.zeros((T, 4))
                fingers_jaco[finger_name] = np.zeros((T, 6, 4))

            for t in range(1, T + 1):
                self.updateJacobians("hand", traj_hand_joint_pos[t, :])  # will call FK internally
                for finger_name in fingers_target_rel_pose.keys():
                    fingers_pos[finger_name][t - 1, :], fingers_quat[finger_name][t - 1, :] = self.getTcpGlobalPose(
                        finger_name, local_position=fingers_local_position[finger_name]
                    )
                    fingers_jaco[finger_name][t - 1, :, :] = self.getGlobalJacobian(
                        finger_name, local_position=fingers_local_position[finger_name], joint_part_name=finger_name
                    )

            object_pose_inv = ucalc.batchIsometry3dInverse(
                ucalc.batchPosRotVec2Isometry3d(traj_object_pos[1:, :], traj_object_rotvec[1:, :])
            )
            object_jaco = np.zeros((T, 6, 6))  # T = 1, ..., T
            object_jaco[:, 0:3, 0:3] = np.tile(np.eye(3), (T, 1, 1))
            object_jaco[:, 3:6, 3:6] = ucalc.jacoDeRotVecToAngularVel(traj_object_rotvec[1:, :])
            transform_mat = ucalc.batchDiagRotMat(object_pose_inv[:, 0:3, 0:3])
            transformed_object_jaco = np.matmul(transform_mat, object_jaco)

            for finger_name, _ in fingers_target_rel_pos_ori.items():
                finger_pose = ucalc.batchPosQuat2Isometry3d(fingers_pos[finger_name], fingers_quat[finger_name])
                finger_rel_pose = np.matmul(object_pose_inv, finger_pose)
                finger_rel_pos = finger_rel_pose[:, 0:3, 3]

                temp_jaco = np.zeros((T, 6, 6 + 16))  # 自变量只包括t时刻的关节
                temp_jaco[:, :, 6 + np.asarray(self.finger_joints_id_in_hand[finger_name])] = np.matmul(
                    transform_mat, fingers_jaco[finger_name]
                )
                temp_jaco[:, :, 0:6] = -np.matmul(
                    ucalc.wrenchTransformationMatrix(finger_rel_pos), transformed_object_jaco
                )

                whole_jaco = np.zeros((T, 6, T + 1, 6 + 16))
                for t in range(T):
                    whole_jaco[t, :, t + 1, :] = temp_jaco[t]

                jaco_list.append(whole_jaco.reshape(T * 6, -1))

            # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
            whole_jaco = np.zeros((T, 16, T + 1, 6 + 16))
            for t in range(1, T + 1):
                whole_jaco[t - 1, :, t, 6:] = 1.0 / delta_t * np.eye(16)
                whole_jaco[t - 1, :, t - 1, 6:] = -1.0 / delta_t * np.eye(16)
            jaco_list.append(whole_jaco.reshape(T * 16, -1))

            jaco = np.vstack(jaco_list)
            object_jaco = err.T @ weights @ jaco

            # print(f"Time cost of calc jacobian: {time.perf_counter() - t1}")
            return object_jaco.reshape(-1)

        def x0EqConstraint(val):
            val = val.reshape(T + 1, -1)
            eq = val[0, :] - val_init  # = 0
            return eq

        def x0EqConstraintJaco(val):
            jaco = np.zeros((22, val.size))
            jaco[:, 0:22] = np.eye(22)
            return jaco

        # def jointVelIneqConstraint(val):
        #     val = val.reshape(T + 1, -1)
        #     _, _, traj_hand_joint_pos = val[:, 0:3], val[:, 3:6], val[:, 6 : 6 + 16]
        #     traj_hand_joint_vel = (traj_hand_joint_pos[1:, :] - traj_hand_joint_pos[0:-1, :]) / delta_t
        #     joint_vel_ub = 100
        #     ineq = np.abs(traj_hand_joint_vel) - joint_vel_ub  # <= 0
        #     return -ineq.reshape(-1)

        # constraints_list = [dict(type="eq", fun=x0EqConstraint), dict(type="ineq", fun=jointVelIneqConstraint)]
        constraints_list = [dict(type="eq", fun=x0EqConstraint, jac=x0EqConstraintJaco)]
        # constraints_list = []

        res = minimize(
            fun=objectFunction,
            jac=objectJacobian,
            constraints=constraints_list,
            x0=traj_val_init,
            bounds=traj_val_bounds,
            method="SLSQP",
            options={"ftol": 1e-10, "disp": True, "maxiter": 1000},
        )
        res_val = res.x.reshape(-1)

        print(f"Time cost of SQP_IK: {perf_counter() - t_ik}")

        traj_object_pos = res_val.reshape(T + 1, -1)[:, 0:3]
        planned_object_err = np.linalg.norm(traj_object_pos[-1] - object_target_pos)
        traj_hand_joint_pos = res_val.reshape(T + 1, -1)[:, 6:]
        return traj_hand_joint_pos, planned_object_err

    # ------------------------------------------
    def test(self):
        thumb_tcp_parent_joint = self.model.frames[self.tcp_links_id["thumb"]].parent
        thumb_tcp_placement = self.model.frames[self.tcp_links_id["thumb"]].placement

        self.model.addFrame(
            pin.Frame(
                "object",
                parent_joint=thumb_tcp_parent_joint,
                parent_frame=self.tcp_links_id["thumb"],
                placement=thumb_tcp_placement,
                type=pin.OP_FRAME,
            )
        )


if __name__ == "__main__":
    rospack = rospkg.RosPack()
    urdf_path = Path(rospack.get_path("my_robot_description")) / "urdf" / "leaphand.urdf"

    robot_model = LeapHandPinocchio(urdf_path=str(urdf_path))

    # hand_joint_pos = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]) / 100.0
    hand_joint_pos = np.zeros((16,))
    robot_model.updateFK("hand", hand_joint_pos)

    pos, quat = robot_model.getTcpGlobalPose("finger0")
    pos += np.array([-0.02, 0, -0.05])

    res_joint_pos = robot_model.fingerIKSQP(
        "finger0",
        finger_target_pose=ucalc.posQuat2Isometry3d(pos, quat),
        weights=np.diag([10, 10, 10, 0.1, 0, 0.1]),
        finger_joint_pos_init=hand_joint_pos[0:4],
        local_position=[0, 0, 0],
    )

    print("res_joint_pos: ", res_joint_pos)

    # robot_model.test()
