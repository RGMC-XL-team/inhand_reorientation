import os
import sys
import time

import numpy as np
import pinocchio as pin
from scipy.optimize import minimize

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../scripts"))
from leap_hardware.my_utils.utils_calc import *


class LeapHandPinocchio:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # q        = pin.randomConfiguration(self.model)
        # print(q.shape)

        # print([name for name in self.model.names])

        self.part_joints_name = {}
        self.part_joints_name["finger0"] = ["1", "0", "2", "3"]
        self.part_joints_name["finger1"] = ["5", "4", "6", "7"]
        self.part_joints_name["finger2"] = ["9", "8", "10", "11"]
        self.part_joints_name["thumb"] = ["12", "13", "14", "15"]
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
        # use _fake_model_names instead of self.model.names to avoid the strange error
        _fake_model_names = [
            "universe",
            "1",
            "0",
            "2",
            "3",
            "12",
            "13",
            "14",
            "15",
            "5",
            "4",
            "6",
            "7",
            "9",
            "8",
            "10",
            "11",
        ]
        # for id, name in enumerate(self.model.names):
        for id, name in enumerate(_fake_model_names):
            if name != "universe":
                self.joints_name_to_id[name] = id - 1  # 只有和 pinocchio 交互的时候才用到这个顺序

        self.part_joints_id = {}
        for part_name, joints_name in self.part_joints_name.items():
            joints_id = [self.joints_name_to_id[name] for name in joints_name]
            self.part_joints_id[part_name] = joints_id

        self.tcp_links_id = {}
        for part_name, link_name in self.tcp_links_name.items():
            self.tcp_links_id[part_name] = self.model.getFrameId(link_name)

    # ------------------------------------------------
    def checkJointDim(self, part_name, joints):
        if isinstance(joints, list):
            return len(joints) == len(self.part_joints_id[part_name])
        else:
            return joints.size == len(self.part_joints_id[part_name])

    # ------------------------------------------------
    def jointOrderPartUserToAllPin(self, part_name, q_part_normal):
        q_all_pin = np.zeros((len(self.joints_name_to_id),))
        q_all_pin[self.part_joints_id[part_name]] = q_part_normal
        return q_all_pin

    # ------------------------------------------------
    """
        input:
            hand_joint_pos: the joint order is 0, 1, 2, ... 15, the same as the joint name
    """

    def updateFK(self, part_name, part_joint_pos):
        if not self.checkJointDim(part_name, part_joint_pos):
            raise NameError("The dim of input joint is wrong !")

        q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
        pin.framesForwardKinematics(self.model, self.data, q_pin)

    # ------------------------------------------------
    """
        input:
            hand_joint_pos: the joint order is 0, 1, 2, ... 15, the same as the joint name
            hand_joint_vel: the joint order is 0, 1, 2, ... 15, the same as the joint name
    """

    def updateFKvel(self, part_name, part_joint_pos, part_joint_vel):
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

    # ------------------------------------------------
    def getTcpGlobalPose(self, part_name, part_joint_pos=None, local_position=None):
        if part_joint_pos is not None:
            self.updateFK(part_name, part_joint_pos)
        return self.getFrameGlobalPose(self.tcp_links_name[part_name])

    # ------------------------------------------------
    def getFrameGlobalPose(self, frame_name):
        data = self.data.oMf[self.model.getFrameId(frame_name)]
        pos = data.translation
        rot_mat = data.rotation
        return pos, sciR.from_matrix(rot_mat).as_quat()

    # ------------------------------------------------
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

    # ------------------------------------------------
    def updateJacobians(self, part_name, part_joint_pos):
        if not self.checkJointDim(part_name, part_joint_pos):
            raise NameError("The dim of input joint is wrong !")
        if part_name != "hand":
            raise NameError("The part_name must be 'hand'.")

        q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
        pin.computeJointJacobians(self.model, self.data, q_pin)  # call FK internally
        pin.updateFramePlacements(self.model, self.data)

    # ------------------------------------------------
    """
        input:
            part_name: which part the target tcp belongs to.
            joint_part_name: which part the joints belong to.
    """

    def getGlobalJacobian(self, part_name, part_joint_pos=None, local_position=None, joint_part_name=None):
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

    # ------------------------------------------------
    def getFrameGlobalJacobian(self, frame_name, joint_part_name=None):
        jaco_pin = pin.getFrameJacobian(
            self.model, self.data, frame_id=self.model.getFrameId(frame_name), reference_frame=pin.LOCAL_WORLD_ALIGNED
        )
        return jaco_pin[:, self.part_joints_id[joint_part_name]]

    # ------------------------------------------------
    def calcFingerPoseError(self, finger_name, finger_target_pose, finger_joint_pos, local_position):
        err = np.zeros((6,))
        finger_target_pos, finger_target_ori = finger_target_ori[0:3], sciR.from_quat(finger_target_pose[3:7])

        self.updateFK(finger_name, finger_joint_pos)
        finger_pos, finger_quat = self.getTcpGlobalPose(finger_name, local_position=local_position)
        err[0:3] = finger_pos - finger_target_pos
        err[3:6] = (sciR.from_quat(finger_quat) * finger_target_ori.inv()).as_rotvec()

        err = err.reshape(-1, 1)
        self.curr_err = err

        return err

    # ------------------------------------------------
    def fingerDiffIK(
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

        init_error = self.calcFingerPoseError(finger_name, finger_target_pose, finger_joint_pos_init, local_position)
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

    # ------------------------------------------------
    def fingerIKSQP(self, finger_name, finger_target_pose, weights, finger_joint_pos_init, local_position=None):
        t_ik = time.time()

        finger_target_pos, finger_target_ori = isometry3dToPosOri(finger_target_pose)
        self.curr_err = None

        # ------------------------
        def calcTargetError(finger_joint_pos):
            t1 = time.time()
            err = np.zeros((6,))

            self.updateFK(finger_name, finger_joint_pos)
            finger_pos, finger_quat = self.getTcpGlobalPose(finger_name, local_position=local_position)
            err[0:3] = finger_pos - finger_target_pos
            err[3:6] = (sciR.from_quat(finger_quat) * finger_target_ori.inv()).as_rotvec()
            # print("err: ", err)

            err = err.reshape(-1, 1)
            self.curr_err = err

            # print(f"Time cost of calc err: {time.time() - t1}")
            return err

        # ------------------------
        def objectFunction(finger_joint_pos):
            err = calcTargetError(finger_joint_pos)
            cost = 1.0 / 2.0 * err.T @ weights @ err
            return cost[0, 0]

        # ------------------------
        def objectJacobian(finger_joint_pos):
            t1 = time.time()
            err = self.curr_err.copy()

            # get all jacobians
            jaco = self.getGlobalJacobian(
                finger_name, finger_joint_pos, local_position=local_position, joint_part_name=finger_name
            )

            object_jaco = err.T @ weights @ jaco

            # print("Jacobian calculation, Use current error.")
            return object_jaco.reshape(
                -1,
            )

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
        res_joint_pos = res.x.reshape(
            -1,
        )

        print(f"Time cost of SQP_IK: {time.time() - t_ik}")
        return res_joint_pos

    # ----------------------------------
    """
        input:
            finger_local_postion: currently not supported
    """

    def relaxedTrajectoryOptimization(
        self,
        T,
        delta_t,
        object_target_pose,
        object_pose_in_thumb,
        finger0_target_rel_pose=None,
        finger1_target_rel_pose=None,
        finger2_target_rel_pose=None,  # relative to thumbtip frame
        weights_object_pose=None,
        weights_rel_pose=None,
        weights_joint_vel=None,
        hand_joint_pos_init=[0] * 16,
        finger0_local_position=None,
        finger1_local_position=None,
        finger2_local_position=None,
        thumb_local_position=None,
    ):
        t_ik = time.time()
        hand_joint_pos_init = np.asarray(hand_joint_pos_init)
        object_target_pos, object_target_ori = isometry3dToPosOri(object_target_pose)

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
            fingers_target_rel_pos_ori[finger_name] = isometry3dToPosOri(target_rel_pose)

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

        # ------------------------
        def calcTargetError(traj_hand_joint_pos):
            t1 = time.time()
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

            thumb_local_pose_inv = batchIsometry3dInverse(
                batchPosQuat2Isometry3d(fingers_pos["thumb"], fingers_quat["thumb"])
            )

            for finger_name, (finger_target_rel_pos, finger_target_rel_ori) in fingers_target_rel_pos_ori.items():
                finger_pos, finger_quat = fingers_pos[finger_name], fingers_quat[finger_name]
                finger_pose = batchPosQuat2Isometry3d(finger_pos, finger_quat)
                finger_rel_pose = np.matmul(thumb_local_pose_inv, finger_pose)
                finger_rel_pos_err = finger_rel_pose[:, 0:3, 3] - np.tile(finger_target_rel_pos, (T, 1))
                finger_rel_ori_err = (
                    sciR.from_matrix(finger_rel_pose[:, 0:3, 0:3]) * finger_target_rel_ori.inv()
                ).as_rotvec()
                finger_rel_pose_err = np.hstack([finger_rel_pos_err, finger_rel_ori_err])
                err_list.extend(
                    finger_rel_pose_err.reshape(
                        -1,
                    )
                )

            # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
            traj_hand_joint_vel = (traj_hand_joint_pos[1:, :] - traj_hand_joint_pos[0:-1, :]) / delta_t
            err_list.extend(
                traj_hand_joint_vel.reshape(
                    -1,
                )
            )

            # -------------------------------------------------------------------
            err = np.asarray(err_list).reshape(-1, 1)
            self.curr_err = err
            return err

        # ------------------------
        def objectFunction(traj_hand_joint_pos):
            err = calcTargetError(traj_hand_joint_pos)
            cost = 1.0 / 2.0 * err.T @ weights @ err
            return cost[0, 0]

        # ------------------------
        def objectJacobian(traj_hand_joint_pos):
            t1 = time.time()
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

            thumb_local_pose_inv = batchIsometry3dInverse(
                batchPosQuat2Isometry3d(fingers_pos["thumb"], fingers_quat["thumb"])
            )
            transformed_thumb_local_jaco = np.matmul(
                batchDiagRotMat(thumb_local_pose_inv[:, 0:3, 0:3]), fingers_jaco["thumb"]
            )

            for finger_name, _ in fingers_target_rel_pos_ori.items():
                finger_pose = batchPosQuat2Isometry3d(fingers_pos[finger_name], fingers_quat[finger_name])
                finger_rel_pose = np.matmul(thumb_local_pose_inv, finger_pose)
                finger_rel_pos = finger_rel_pose[:, 0:3, 3]

                temp_jaco = np.zeros((T, 6, 16))  # 自变量只包括t时刻的关节
                temp_jaco[:, :, self.finger_joints_id_in_hand[finger_name]] = np.matmul(
                    batchDiagRotMat(thumb_local_pose_inv[:, 0:3, 0:3]), fingers_jaco[finger_name]
                )
                temp_jaco[:, :, self.finger_joints_id_in_hand["thumb"]] = -np.matmul(
                    wrenchTransformationMatrix(finger_rel_pos), transformed_thumb_local_jaco
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
            return object_jaco.reshape(
                -1,
            )

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
        res_joint_pos = res.x.reshape(
            -1,
        )

        print(f"Time cost of SQP_IK: {time.time() - t_ik}")

        traj_hand_joint_pos = res_joint_pos.reshape(T + 1, -1)
        return traj_hand_joint_pos

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


# ----------------------------------------------------
if __name__ == "__main__":
    project_dir = "/home/mingrui/Mingrui/research/project_RGMC_24/RGMC_XL/"
    urdf_path = project_dir + "leap_ws/src/my_robot_description/urdf/leaphand.urdf"

    robot_model = LeapHandPinocchio(urdf_path=urdf_path)

    # hand_joint_pos = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]) / 100.0
    hand_joint_pos = np.zeros((16,))
    robot_model.updateFK("hand", hand_joint_pos)

    pos, quat = robot_model.getTcpGlobalPose("finger0")
    pos += np.array([-0.02, 0, -0.05])

    res_joint_pos = robot_model.fingerIKSQP(
        "finger0",
        finger_target_pose=posQuat2Isometry3d(pos, quat),
        weights=np.diag([10, 10, 10, 0.1, 0, 0.1]),
        finger_joint_pos_init=hand_joint_pos[0:4],
        local_position=[0, 0, 0],
    )

    print("res_joint_pos: ", res_joint_pos)

    # robot_model.test()


# -------------------------------------- backup ----------------------------------------------------
# # ----------------------------------
#     def relaxedTrajectoryOptimization2(self,
#                     T,
#                     delta_t,
#                     object_target_pose,
#                     object_pose_in_thumb,
#                     finger0_target_rel_pose=None, finger1_target_rel_pose=None, finger2_target_rel_pose=None, # relative to thumbtip frame
#                     weights_object_pose=None,
#                     weights_rel_pose=None,
#                     weights_joint_vel=None,
#                     hand_joint_pos_init=[0]*16,
#                     finger0_local_position=None, finger1_local_position=None, finger2_local_position=None, thumb_local_position=None):

#         t_ik = time.time()
#         hand_joint_pos_init = np.asarray(hand_joint_pos_init)
#         object_target_pos, object_target_ori = isometry3dToPosOri(object_target_pose)

#         # add object frame
#         thumb_tcp_parent_joint = self.model.frames[self.tcp_links_id["thumb"]].parent
#         thumb_tcp_placement = self.model.frames[self.tcp_links_id["thumb"]].placement
#         object_placement_in_thumb = pin.SE3(object_pose_in_thumb[0:3, 0:3],
#                                             object_pose_in_thumb[0:3, 3])
#         self.model.addFrame(pin.Frame(name="object",
#                                       parent_joint=thumb_tcp_parent_joint,
#                                       parent_frame=self.tcp_links_id["thumb"],
#                                       placement=thumb_tcp_placement * object_placement_in_thumb, # relative to parent joint
#                                       type=pin.OP_FRAME))
#         self.data = self.model.createData()

#         fingers_target_rel_pose = {}
#         if finger0_target_rel_pose is not None:
#             fingers_target_rel_pose["finger0"] = finger0_target_rel_pose
#         if finger1_target_rel_pose is not None:
#             fingers_target_rel_pose["finger1"] = finger1_target_rel_pose
#         if finger2_target_rel_pose is not None:
#             fingers_target_rel_pose["finger2"] = finger2_target_rel_pose
#         fingers_target_rel_pos_ori = {}
#         for finger_name, target_rel_pose in fingers_target_rel_pose.items():
#             fingers_target_rel_pos_ori[finger_name] = isometry3dToPosOri(target_rel_pose)

#         fingers_local_position = {}
#         fingers_local_position["finger0"] = finger0_local_position
#         fingers_local_position["finger1"] = finger1_local_position
#         fingers_local_position["finger2"] = finger2_local_position
#         fingers_local_position["thumb"] = thumb_local_position

#         # get the optimized_joint_id
#         optimized_joint_id = []
#         optimized_fingers_name = []
#         for finger_name in ["finger0", "finger1", "finger2"]:
#             if finger_name in fingers_target_rel_pose.keys():
#                 optimized_joint_id.extend(self.finger_joints_id_in_hand[finger_name])
#                 optimized_fingers_name.append(finger_name)
#         optimized_joint_id.extend(self.finger_joints_id_in_hand["thumb"])
#         optimized_fingers_name.append("thumb")

#         delta_t = 0.5
#         weights = np.diag(weights_object_pose
#                           + weights_rel_pose * T * len(fingers_target_rel_pos_ori)
#                           + [weights_joint_vel] * T * len(optimized_joint_id) )

#         # self.thumb_local_pose_inv = None
#         # self.fingers_rel_pose = {}
#         self.curr_err = None

#         """
#             optimized_joint_pos:
#                 (T+1) * n_joints
#                 t = 0, ..., T
#         """

#         # init value
#         optimized_joint_pos_init = hand_joint_pos_init[optimized_joint_id]
#         traj_joint_pos_init = np.tile(optimized_joint_pos_init, T + 1)

#         # bounds
#         joint_pos_lb = np.array(self.joints_id_to_lower_limit)[self.part_joints_id["hand"]][optimized_joint_id]
#         joint_pos_ub = np.array(self.joints_id_to_upper_limit)[self.part_joints_id["hand"]][optimized_joint_id]
#         # joint_pos_bounds = [(joint_pos_lb[i], joint_pos_ub[i]) for i in range(joint_pos_lb.shape[0])]
#         traj_joint_pos_bounds = [(joint_pos_lb[i], joint_pos_ub[i]) for _ in range(T+1) for i in range(joint_pos_lb.shape[0])]


#         # ------------------------
#         def calcTargetError(optimized_joint_pos):
#             t1 = time.time()

#             optimized_joint_pos = optimized_joint_pos.reshape(T+1, -1)
#             fingers_joint_pos = {}
#             for i, finger_name in enumerate(optimized_fingers_name):
#                 fingers_joint_pos[finger_name] = optimized_joint_pos[:, 4*i : 4*i+4]

#             err_list = []
#             # ------------ object pose in world frame at {T} -------------
#             thumb_joint_pos = fingers_joint_pos["thumb"][-1]
#             thumb_pos, thumb_quat = self.getTcpGlobalPose("thumb",
#                                                             part_joint_pos=thumb_joint_pos,
#                                                             local_position=fingers_local_position["thumb"])
#             object_pose = posQuat2Isometry3d(thumb_pos, thumb_quat) @ object_pose_in_thumb
#             object_pos_err = object_target_pos - object_pose[0:3, 3].reshape(-1,)
#             object_ori_err = (object_target_ori * sciR.from_matrix(object_pose[0:3, 0:3]).inv()).as_rotvec()
#             err_list.extend(object_pos_err)
#             err_list.extend(object_ori_err)

#             # ------------ finger pose in thumb frame at {t = 1,...T} ------------
#             for t in range(1, T+1):
#                 thumb_local_pos, thumb_local_quat = self.getTcpGlobalPose("thumb",
#                                                                 part_joint_pos=fingers_joint_pos["thumb"][t],
#                                                                 local_position=fingers_local_position["thumb"])
#                 thumb_local_pose_inv = np.linalg.inv(posQuat2Isometry3d(thumb_local_pos, thumb_local_quat))

#                 for finger_name, (finger_target_rel_pos, finger_target_rel_ori) in fingers_target_rel_pos_ori.items():
#                     finger_local_pos, finger_local_quat = self.getTcpGlobalPose(finger_name,
#                                                                     part_joint_pos=fingers_joint_pos[finger_name][t],
#                                                                     local_position=fingers_local_position[finger_name])
#                     finger_local_pose = posQuat2Isometry3d(finger_local_pos, finger_local_quat) # finger pose in hand frame
#                     finger_rel_pose = thumb_local_pose_inv @ finger_local_pose # finger pose in thumb frame
#                     finger_rel_pos_err = finger_target_rel_pos - finger_rel_pose[0:3, 3].reshape(-1, )
#                     finger_rel_ori_err = (finger_target_rel_ori * sciR.from_matrix(finger_rel_pose[0:3, 0:3]).inv()).as_rotvec()

#                     err_list.extend(finger_rel_pos_err)
#                     err_list.extend(finger_rel_ori_err)

#             # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
#             for t in range(1, T+1):
#                 joint_vel = (optimized_joint_pos[t, :] - optimized_joint_pos[t-1, :]) / delta_t
#                 err_list.extend(joint_vel)

#             # --------------------------------------------------------------------
#             err = np.vstack(err_list).reshape(-1, 1)
#             self.curr_err = err.copy()

#             # print(f"Time cost of calc err: {time.time() - t1}")
#             return err

#         # ------------------------
#         def objectFunction(optimized_joint_pos):
#             err = calcTargetError(optimized_joint_pos)
#             cost = 1.0/2.0 * err.T @ weights @ err

#             # print("cost: ", cost)
#             # time.sleep(1000)

#             return cost[0, 0]


#         # ------------------------
#         def objectJacobian(optimized_joint_pos):
#             t1 = time.time()

#             optimized_joint_pos = optimized_joint_pos.reshape(T+1, -1)
#             fingers_joint_pos = {}
#             for i, finger_name in enumerate(optimized_fingers_name):
#                 fingers_joint_pos[finger_name] = optimized_joint_pos[:, 4*i : 4*i+4]

#             err = self.curr_err.copy()
#             # print("Jacobian calculation, Use current error.")

#             jaco_list = []

#             # ------------ object pose in world frame at {T} -------------
#             hand_joint_pos = np.zeros((16,))
#             for finger_name in ["finger0", "finger1", "finger2", "thumb"]:
#                 if finger_name in optimized_fingers_name:
#                     hand_joint_pos[self.finger_joints_id_in_hand[finger_name]] = fingers_joint_pos[finger_name][-1]
#             self.updateJacobians("hand", hand_joint_pos)
#             object_global_jaco = self.getFrameGlobalJacobian("object", joint_part_name="thumb")
#             whole_jaco = np.zeros((6, T+1, 16))
#             whole_jaco[:, -1, self.finger_joints_id_in_hand["thumb"]] = object_global_jaco
#             whole_jaco = whole_jaco[:, :, optimized_joint_id]
#             whole_jaco = whole_jaco.reshape(6, -1)

#             jaco_list.append(whole_jaco)

#             # ------------ finger pose in thumb frame at {t = 1,...T} ------------
#             for t in range(1, T+1):
#                 thumb_local_pos, thumb_local_quat = self.getTcpGlobalPose("thumb",
#                                                                 part_joint_pos=fingers_joint_pos["thumb"][t],
#                                                                 local_position=fingers_local_position["thumb"])
#                 thumb_local_pose_inv = np.linalg.inv(posQuat2Isometry3d(thumb_local_pos, thumb_local_quat))
#                 thumb_local_jaco = self.getGlobalJacobian("thumb",
#                                                             part_joint_pos=fingers_joint_pos["thumb"][t],
#                                                             local_position=fingers_local_position["thumb"],
#                                                             joint_part_name="thumb")

#                 transformed_thumb_local_jaco = diagRotMat(thumb_local_pose_inv[0:3, 0:3]) @ thumb_local_jaco

#                 for finger_name in fingers_target_rel_pos_ori.keys():
#                     finger_local_pos, finger_local_quat = self.getTcpGlobalPose(finger_name,
#                                                                     part_joint_pos=fingers_joint_pos[finger_name][t],
#                                                                     local_position=fingers_local_position[finger_name])
#                     finger_local_pose = posQuat2Isometry3d(finger_local_pos, finger_local_quat) # finger pose in hand frame
#                     finger_rel_pose = thumb_local_pose_inv @ finger_local_pose # finger pose in thumb frame
#                     finger_rel_pos = finger_rel_pose[0:3, 3].reshape(-1,)

#                     finger_local_jaco = self.getGlobalJacobian(finger_name,
#                                                                     part_joint_pos=fingers_joint_pos[finger_name][t],
#                                                                     local_position=fingers_local_position[finger_name],
#                                                                     joint_part_name=finger_name)

#                     whole_jaco = np.zeros((6, T+1, 16))
#                     whole_jaco[:, t, self.finger_joints_id_in_hand[finger_name]] = diagRotMat(thumb_local_pose_inv[0:3, 0:3]) @ finger_local_jaco
#                     whole_jaco[:, t, self.finger_joints_id_in_hand["thumb"]] = -np.block([[np.eye(3), -skew(finger_rel_pos)],
#                                                                                         [np.zeros((3,3)), np.eye(3)]]) @ transformed_thumb_local_jaco
#                     whole_jaco = whole_jaco[:, :, optimized_joint_id]
#                     whole_jaco = whole_jaco.reshape(6, -1)

#                     jaco_list.append(whole_jaco)

#             # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
#             for t in range(1, T+1):
#                 whole_jaco = np.zeros((len(optimized_joint_id), T+1, len(optimized_joint_id)))
#                 whole_jaco[:, t, :] = 1.0 / delta_t * np.eye(len(optimized_joint_id))
#                 whole_jaco[:, t-1, :] = -1.0 / delta_t * np.eye(len(optimized_joint_id))
#                 whole_jaco = whole_jaco.reshape(len(optimized_joint_id), -1)

#                 jaco_list.append(-whole_jaco) # 注意这里得加个负号，与其他jacobian一致

#             # --------------------------------------------------------------------
#             jaco = np.vstack(jaco_list)

#             object_jaco = err.T @ weights @ (-jaco)
#             # print(f"Time cost of calc jacobian: {time.time() - t1}")
#             return object_jaco.reshape(-1, )


#         def x0EqConstraint(optimized_joint_pos):
#             optimized_joint_pos = optimized_joint_pos.reshape(T+1, -1)
#             eq = optimized_joint_pos[0, :] - optimized_joint_pos_init # = 0
#             return eq


#         constraints_list = [dict(type='eq', fun=x0EqConstraint)
#                             ]

#         res = minimize(fun=objectFunction,
#                        jac=objectJacobian,
#                        constraints=constraints_list,
#                        x0=traj_joint_pos_init,
#                        bounds=traj_joint_pos_bounds,
#                        method='SLSQP',
#                        options={'ftol':1e-10, 'disp': True, 'maxiter':1000}
#                        )
#         res_joint_pos = res.x.reshape(-1, )

#         print(f"Time cost of SQP_IK: {time.time() - t_ik}")

#         res_joint_pos = res_joint_pos.reshape(T+1, -1)
#         traj_hand_joint_pos = np.tile(hand_joint_pos_init.copy(), (T+1, 1))
#         traj_hand_joint_pos[:, optimized_joint_id] = res_joint_pos

#         return traj_hand_joint_pos
