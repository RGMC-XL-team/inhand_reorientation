"""
    This script contains the high level
    decision maker for taskB
"""

import os

from copy import deepcopy
import numpy as np
import yaml
import time

import rospy
import rospkg
import actionlib
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger

from leap_task_B.taskB_utils import TaskBInfo, TaskBState, TaskBOfflineMode, DESIRED_YAW_DICT
from leap_task_B.taskB_utils import (
    compute_angle_distance,
    get_z_axis_from_pos_quat,
    compute_angle_between_two_axis
)
# from leap_task_B.taskB_recorder import TaskBTimer
from leap_hardware.srv import face_pose, face_poseRequest, object_state
from leap_task_B.msg import RunPolicyAction, RunPolicyGoal


class TaskBHighLevel(object):
    def __init__(self, online=False) -> None:
        self.config = yaml.safe_load(open(os.path.join(rospkg.RosPack().get_path("leap_task_B"), "config/taskB_XL.yaml")))
        self._set_default(online)
        self._initialize_service_and_actions()
        self._initialize_anomaly_detection()

        # self.timer = TaskBTimer()

    def _set_default(self, online=False):
        self.info = TaskBInfo()
        # _config = yaml.safe_load(open(os.path.join(rospkg.RosPack().get_path("leap_task_B"), "config/taskB_XL.yaml")))
        self.info.load(self.config)

        self.fsm_hz = rospy.get_param("fsm_hz", 30)
        self.fsm_rate = rospy.Rate(self.fsm_hz)
        self.object_center_pos = self.info.object_center_pos
        self.info.current_face = "A"

        self.online_mode = online
        self.allow_reset = self.config["reset_options"]["allow_reset"]
        self.forced_reset = self.config["reset_options"]["forced_reset"]

    def _initialize_service_and_actions(self):
        self.action_name = rospy.get_param("action_name", "run_policy")
        rospy.wait_for_service("/object_state")
        rospy.wait_for_service("/face_pose")
        self.object_state_client = rospy.ServiceProxy("/object_state", object_state)
        self.face_pose_client = rospy.ServiceProxy("/face_pose", face_pose)
        self._action_client = actionlib.SimpleActionClient(self.action_name, RunPolicyAction)
        self._action_client.wait_for_server()

        if self.online_mode:
            self._rgmc_client = {}
            for _action in ["start", "record", "stop"]:
                service_name = f"/rgcm_eval/{_action}"
                rospy.wait_for_service(service_name)
                self._rgmc_client[_action] = rospy.ServiceProxy(service_name, Trigger)

    def _initialize_anomaly_detection(self):
        """Initialize anomaly detection as a timer"""
        anomaly_config = self.config["anomaly_detection"]
        self._anomaly_detector = rospy.Timer(
            period=rospy.Duration(anomaly_config["period"]),
            callback=self._check_anomaly
        )
        self.anomaly_config = anomaly_config

    def _check_anomaly(self, event=None):
        """Check if anomaly occurs in the manipulation"""
        if not self.allow_reset:
            rospy.logwarn("Anomaly detection is disabled when reset is disabled!")
            return
        
        anomaly_detected = False

        # for waiting, check if timeout
        if self.info.current_state == TaskBState.WAIT:
            if rospy.get_time() - self.info.time_current_wait_start > self.anomaly_config["max_wait_time"]:
                rospy.logerr("Anomaly: timeout for waiting!")
                anomaly_detected = True

        # for rotation, check if the object is tilt too much
        if self.info.current_state in [TaskBState.ROTATE_CW, TaskBState.ROTATE_CCW]:
            current_z_axis = self.info.object_local_pose.z_axis()
            try:
                control_fail = bool(rospy.wait_for_message("/hand_control_fail", Bool, timeout=0.2).data)
            except rospy.ROSException:
                control_fail = False
            # anomaly cond1: object's face is not roughly upwards
            if compute_angle_between_two_axis(current_z_axis, [0., 0., 1.]) > self.anomaly_config["rot_tilt_angle_thresh"]:
                rospy.logerr("Anomaly: the object tilts too much for rotation!")
                anomaly_detected = True
            # anomaly cond2: rotation timeout
            # elif rospy.get_time() - self.info.time_current_rot_start > self.anomaly_config["max_rot_time"]:
            #     rospy.logerr("Anomaly: timeout for rotation!")
            #     anomaly_detected = True
            # anomaly cond3: hand control failure
            elif control_fail:
                rospy.logerr("Anomaly: hand control failure!")
                anomaly_detected = True

        # TODO(yongpeng): debug this flipping anomaly detection
        # for flipping, check if the object is far from center
        if self.info.current_state in [TaskBState.FLIP_IN, TaskBState.FLIP_OUT]:
            current_z_axis = self.info.object_local_pose.z_axis()
            current_xy = self.info.object_local_pose.xy()
            current_yaw = self.info.object_local_pose.yaw()
            try:
                control_fail = bool(rospy.wait_for_message("/hand_control_fail", Bool, timeout=0.2).data)
            except rospy.ROSException:
                control_fail = False
            # anomaly cond1: object's center is too far from the palm center
            if current_xy[0] <= self.anomaly_config["flip_x_lower_bound"] and \
                current_xy[1] >= self.anomaly_config["flip_y_upper_bound"]:
                rospy.logerr("Anomaly: the object moves out of palm center!")
                anomaly_detected = True
            # # anomaly cond2: object's face (roughly) points upwards, but rotates around z too much
            # elif compute_angle_between_two_axis(current_z_axis, [0., 0., 1.]) < np.deg2rad(10*1.5) and \
            #     compute_angle_distance(current_yaw, self.info.desired_yaw) > np.deg2rad(15*1.5):
            #     anomaly_detected = True
            # anomaly cond3: flipping timeout
            elif rospy.get_time() - self.info.time_current_flip_start > self.anomaly_config["max_flip_time"]:
                rospy.logerr("Anomaly: timeout for flipping!")
                anomaly_detected = True
            # anomaly cond4: hand control failure
            elif control_fail:
                rospy.logerr("Anomaly: hand control failure!")
                anomaly_detected = True

        # execute reset hand if anomaly detected
        if anomaly_detected:
            rospy.logerr("Will reset hand!")
            self._action_client.cancel_goal()
            self.call_reset_hand()
            avoid_too_much_reset = False
            if self.info.current_state == TaskBState.WAIT:
                self.info.current_state = TaskBState.BEFORE_ROTATE
            elif self.info.current_state in [TaskBState.ROTATE_CW, TaskBState.ROTATE_CCW]:
                self.info.current_state = TaskBState.BEFORE_ROTATE
                avoid_too_much_reset = True
            elif self.info.current_state in [TaskBState.FLIP_IN, TaskBState.FLIP_OUT]:
                self.info.current_state = TaskBState.BEFORE_FLIP
                avoid_too_much_reset = True
            """Allow only one more reset motion, otherwise, run out of time"""
            if avoid_too_much_reset:
                self.info.consecutive_reset_times = self.config["reset_options"]["max_reset_times"] - 1


    def _check_timeout(self):
        self.info.time_spent = rospy.get_time() - self.info.time_current_start
        return self.info.time_spent >= self.info.time_budget
    
    def _check_current_visible(self):
        """Check if the current face is visible"""
        _request = face_poseRequest()
        _request.face = self.info.current_face
        face_pose = self.face_pose_client(_request)
        if face_pose.visible:
            _pose = np.array(face_pose.pose)
            self.info.object_local_pose.set_from_pos_quat(_pose[0:3], _pose[3:7])
            return True
        else:
            rospy.logerr_throttle(1, f"Face {self.info.current_face} is not visible!")
            return False
        
    def _check_face_upwards(self, face="A"):
        """Ckeck if the target face is upwards"""
        _request = face_poseRequest()
        _request.face = face
        face_pose = self.face_pose_client(_request)
        if not face_pose.visible:
            return False
        _pose = face_pose.pose
        z_axis = get_z_axis_from_pos_quat(_pose[0:3], _pose[3:7])
        # check alignment
        # (cos>0.985, deg<10) | (cos>0.965, deg<15)
        # if np.dot(z_axis, np.array([0, 0, 1])) > 0.985:
        if np.dot(z_axis, np.array([0, 0, 1])) > 0.965:
            rospy.loginfo(f"Face {face} is upwards!")
            self.info._debug_face_pose = _pose
            return True
        else:
            rospy.loginfo_throttle(1, f"Face {face} is not upwards!")
            return False
        
    def _check_target_reachable(self):
        """Get and check if the target face is reachable"""
        _desired_yaw = DESIRED_YAW_DICT[self.info.current_face][self.info.target_face]
        if _desired_yaw is not None:
            self.info.desired_yaw = _desired_yaw
            return True
        else:
            rospy.logerr_throttle(1, f"Face {self.info.target_face} is not reachable \
                                  from {self.info.current_face}!")
            return False
        
    def _check_rotate_terminate_condition(self):
        """Check if the desired yaw is reached"""
        if self._check_current_visible() == False:
            return False
        _current_yaw = self.info.object_local_pose.yaw()
        _angle_diff = compute_angle_distance(angle1=_current_yaw, angle2=self.info.desired_yaw)
        if _angle_diff < self.info.tolerance_yaw:
            rospy.loginfo(f"Current rotation primitive done!")
            return True
        else:
            return False
        
    def _check_flip_start_condition(self):
        """Check if the object is centered for flipping"""
        if self._check_current_visible() == False:
            return False
        object_at_palm_center = np.linalg.norm(self.info.object_local_pose.xy() - self.object_center_pos[0:2]) < \
                                self.info.tolerance_xy
        object_no_rot = compute_angle_distance(self.info.object_local_pose.yaw(), self.info.desired_yaw) < self.info.tolerance_yaw/2
        if object_at_palm_center and object_no_rot:
            rospy.loginfo(f"Object is centered for flipping!")
            return True
        else:
            return False
        
    def _check_rotate_start_condition(self):
        """Check if the object is centered for rotation"""
        if self._check_current_visible() == False:
            return False
        if np.linalg.norm(self.info.object_local_pose.xy() - self.object_center_pos[0:2]) < \
            self.info.tolerance_xy:
            rospy.loginfo(f"Object is centered for flipping!")
            return True
        else:
            return False
        
    def _check_should_rotate_cw(self):
        """Check if the object should rotate clockwise"""
        _current_yaw = self.info.object_local_pose.yaw()
        _desired_yaw = self.info.desired_yaw
        _diff = _desired_yaw - _current_yaw
        
        if _diff > 0:
            anticlockwise_angle = _diff
            clockwise_angle = 2 * np.pi - _diff
        else:
            anticlockwise_angle = 2 * np.pi + _diff
            clockwise_angle = -_diff
        
        if clockwise_angle < anticlockwise_angle:
            return True
        else:
            if anticlockwise_angle > np.deg2rad(100):
                return True
            else:
                return False
        
    def call_reset_hand(self):
        """Reset the hand to home position"""
        self.info.running_policy = "RESET_HAND"
        goal = RunPolicyGoal(policy_name="RESET_HAND")
        self._action_client.send_goal(goal)
        self._action_client.wait_for_result()
        
    def call_reset_object(self, current_policy="ROT"):
        """Reset the object to the center"""
        if not self.allow_reset:
            rospy.logwarn("Reset is disabled!")
            return
        
        t_reset_start = time.time()

        self.info.running_policy = f"RESET_OBJECT_{current_policy}"
        goal = RunPolicyGoal(policy_name=f"RESET_OBJECT_{current_policy}")
        self._action_client.send_goal(goal)
        self._action_client.wait_for_result()
        self.info.consecutive_reset_times += 1

        t_reset_end = time.time()
        # self.timer.add_time_reset_object(t_reset_end - t_reset_start)

    def get_object_pose(self):
        """ Return x, y, z, w, x, y, z """
        transform = rospy.wait_for_message("/object_transform", TransformStamped).transform
        pose = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
            transform.rotation.w,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z
        ]
        return pose

    def clear_reset_object_status(self):
        self.info.consecutive_reset_times = 0.0

    def _check_max_reset_object_times_reached(self):
        return self.info.consecutive_reset_times >= self.config["reset_options"]["max_reset_times"]

    def execute_current_face(self):
        # self.timer.reset()
        self.info.current_state = TaskBState.BEFORE_WAIT
        self.info.time_current_start = rospy.get_time()

        # reset hand
        self.call_reset_hand()
        self.info.current_state = TaskBState.WAIT
        self.info.time_current_wait_start = rospy.get_time()
        rospy.loginfo("Hand reset done!")
        
        while not self._check_timeout():

            rospy.logdebug_throttle(1, f"Current state: {self.info.current_state}")

            if not self._check_target_reachable():
                self.info.current_state = TaskBState.ERROR
                break
            
            if self.info.current_state == TaskBState.WAIT:
                if self._check_current_visible() and \
                    self._check_face_upwards(face=self.info.current_face):
                    self.info.current_state = TaskBState.BEFORE_ROTATE
            
            elif self.info.current_state == TaskBState.BEFORE_ROTATE:
                if self._check_rotate_terminate_condition():
                    """Just skip rotating the object"""
                    self.info.current_state = TaskBState.BEFORE_FLIP
                elif self._check_rotate_start_condition() or self._check_max_reset_object_times_reached():
                    """Forced reset"""
                    if self.forced_reset and not self._check_max_reset_object_times_reached():
                        self.call_reset_object(current_policy="ROT")
                    self.clear_reset_object_status()
                    """Configure the rotation policy here"""
                    self.info.time_current_rot_start = rospy.get_time()
                    if self._check_should_rotate_cw():
                        t_rot_cw_start = time.time()
                        self.info.running_policy = "ROT_CW"
                        goal = RunPolicyGoal(policy_name="ROT_CW")
                        self._action_client.send_goal(goal)
                        self.info.current_state = TaskBState.ROTATE_CW
                    else:
                        t_rot_ccw_start = time.time()
                        self.info.running_policy = "ROT_CCW"
                        goal = RunPolicyGoal(policy_name="ROT_CCW")
                        self._action_client.send_goal(goal)
                        self.info.current_state = TaskBState.ROTATE_CCW
                else:
                    """Reset the object to the center"""
                    self.call_reset_object(current_policy="ROT")

            elif self.info.current_state == TaskBState.ROTATE_CW or \
                self.info.current_state == TaskBState.ROTATE_CCW:
                """Continue rotating the object"""
                if self._check_rotate_terminate_condition():
                    """Stop rotating the object"""
                    t_rot_end = time.time()
                    if self.info.running_policy == "ROT_CW":
                        pass
                        # self.timer.add_time_rot_cw(t_rot_end - t_rot_cw_start)
                    elif self.info.running_policy == "ROT_CCW":
                        self._action_client.cancel_goal()
                        # self.timer.add_time_rot_ccw(t_rot_end - t_rot_ccw_start)
                    # self.timer.add_object_pose(self.get_object_pose())
                    self.info.current_state = TaskBState.BEFORE_FLIP

            elif self.info.current_state == TaskBState.BEFORE_FLIP:
                if self._check_flip_start_condition() or self._check_max_reset_object_times_reached():
                    """Forced reset"""
                    if self.forced_reset and not self._check_max_reset_object_times_reached():
                        self.call_reset_object(current_policy="FLIP")
                    self.clear_reset_object_status()
                    """Configure the flipping policy here"""
                    self.info.time_current_flip_start = rospy.get_time()
                    self.info.running_policy = "FLIP_OUT"
                    goal = RunPolicyGoal(policy_name="FLIP_OUT")
                    _debug_flip_start_time = time.time()
                    t_flip_start = time.time()
                    self._action_client.send_goal(goal)
                    self.info.current_state = TaskBState.FLIP_OUT
                else:
                    """Reset the object to the center"""
                    self.call_reset_object(current_policy="FLIP")

            elif self.info.current_state == TaskBState.FLIP_IN or \
                self.info.current_state == TaskBState.FLIP_OUT:
                """Continue flipping the object"""
                target_face_upwards = self._check_face_upwards(face=self.info.target_face)
                current_face_upwards = self._check_face_upwards(face=self.info.current_face)
                if target_face_upwards and not current_face_upwards:
                    """Stop flipping the object"""
                    if self.info.running_policy == "FLIP_OUT":
                        self._action_client.cancel_goal()
                        _debug_flip_cancelled_time = time.time() - _debug_flip_start_time
                        rospy.logwarn(f"FLIP cancelled after {_debug_flip_cancelled_time} seconds!")
                        if _debug_flip_cancelled_time < 1.0:
                            print("current face pose: ", self.info._debug_face_pose)
                    t_flip_end = time.time()
                    # self.timer.add_time_flip(t_flip_end - t_flip_start)
                    # self.timer.add_object_pose(self.get_object_pose())
                    self.info.current_state = TaskBState.DONE
                    break

            else:
                raise ValueError(f"Invalid FSM state: {self.info.current_state}")

            self.fsm_rate.sleep()

        if self.info.current_state not in [TaskBState.DONE, TaskBState.ERROR]:
            self.info.current_state = TaskBState.TIMEOUT

        rospy.loginfo(f"Switch to target face {self.info.target_face} done \
                      with state {self.info.current_state}")

    def execute_new_face(self, face):
        assert face in ["A", "B", "C", "D", "E", "F"]
        self.info.target_face = face
        self.execute_current_face()

    def get_random_next_face(self, curr_face):
        """
        get the next face reachable from current one (randomly)
        """
        candidate_face_list = []
        for next_face in ["A", "B", "C", "D", "E", "F"]:
            if DESIRED_YAW_DICT[curr_face][next_face] is not None:
                candidate_face_list.append(next_face)
        return np.random.choice(candidate_face_list)

    def main_offline_test(self, mode=None):
        """
        test offline without a judge machine
        """
        if mode == TaskBOfflineMode.USER:
            while True:
                new_face = input("Enter next face [A~F], or Q to quit: ")
                self.execute_new_face(new_face)
                if self.info.current_state == TaskBState.DONE:
                    self.info.current_face = new_face
        elif mode == TaskBOfflineMode.FIXED:
            example_sequence = ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"] * 10
            for new_face in example_sequence[1:]:
                self.execute_new_face(new_face)
                if self.info.current_state == TaskBState.DONE:
                    self.info.current_face = new_face
        elif mode == TaskBOfflineMode.RANDOM:
            self.info.current_face = "A"
            num_executed_faces = 0
            while True:
                new_face = self.get_random_next_face(self.info.current_face)
                rospy.loginfo(f"Next face: {new_face}")
                self.execute_new_face(new_face)
                if self.info.current_state == TaskBState.DONE:
                    self.info.current_face = new_face
                    num_executed_faces += 1
                if num_executed_faces >= self.info.face_seq_length:
                    break
        elif mode == TaskBOfflineMode.RANDOM_FROM_FILE:
            self.info.current_face = "A"
            sequence_from_file = yaml.safe_load(open(os.path.join(rospkg.RosPack().get_path("leap_task_B"), "config/taskB_face_sequence_100.yaml")))
            for new_face in sequence_from_file[1:]:
                t_exec_start = time.time()
                self.execute_new_face(new_face)
                t_exec_end = time.time()
                exec_time = t_exec_end - t_exec_start
                # self.timer.save_data(new_face, exec_time)
                if self.info.current_state == TaskBState.DONE:
                    self.info.current_face = new_face
        else:
            raise NotImplementedError

    def start_task(self):
        self._rgmc_client["start"]()

    def stop_task(self):
        self._rgmc_client["stop"]()

    def record_finish(self):
        self._rgmc_client["record"]()

    def main_online_test(self):
        """
        test online with a judge machine
        """
        self.start_task()

        num_executed_faces = 0
        while num_executed_faces < self.info.face_seq_length:
            new_face = deepcopy(rospy.wait_for_message("/rgcm_eval/task2/goal", String).data)
            rospy.logwarn("Received new face: %s", new_face)
            if new_face == self.info.current_face:
                self.info.current_state == TaskBState.DONE
                # print(f"Current state: {self.info.current_state}")
                rospy.logwarn("Current face is the same as the target face, jump this time!")
                self.record_finish()
            else:
                self.execute_new_face(new_face)
                if self.info.current_state == TaskBState.DONE:
                    self.info.current_face = new_face
                    self.record_finish()
            num_executed_faces += 1

        self.stop_task()
        

if __name__ == "__main__":
    rospy.init_node("taskB_highlevel", log_level=rospy.INFO)
    # MODIFY THIS TWO LINES TO RUN ONLINE
    taskB = TaskBHighLevel(online=False)
    # taskB.main_offline_test(mode=TaskBOfflineMode.FIXED)
    taskB.main_offline_test(mode=TaskBOfflineMode.FIXED)
    # taskB.main_online_test()