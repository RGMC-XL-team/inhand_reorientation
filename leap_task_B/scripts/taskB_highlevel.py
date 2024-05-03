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
from std_msgs.msg import String
from std_srvs.srv import Trigger

from leap_task_B.taskB_utils import TaskBInfo, TaskBState, TaskBOfflineMode, DESIRED_YAW_DICT
from leap_task_B.taskB_utils import compute_angle_distance, get_z_axis_from_pos_quat
from leap_hardware.srv import face_pose, face_poseRequest, object_state
from leap_task_B.msg import RunPolicyAction, RunPolicyGoal


class TaskBHighLevel(object):
    def __init__(self) -> None:
        self._set_default()
        self._initialize_service_and_actions()

    def _set_default(self):
        self.info = TaskBInfo()
        _config = yaml.safe_load(open(os.path.join(rospkg.RosPack().get_path("leap_task_B"), "config/taskB_XL.yaml")))
        self.info.load(_config)

        self.fsm_hz = rospy.get_param("fsm_hz", 30)
        self.fsm_rate = rospy.Rate(self.fsm_hz)
        self.object_center_pos = self.info.object_center_pos
        self.info.current_face = "A"

        self.online_mode = rospy.get_param("online_mode", False)
        self.forced_reset = _config["control_options"]["forced_reset"]

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
        # check alignment (cos>0.985, deg<10)
        if np.dot(z_axis, np.array([0, 0, 1])) > 0.985:
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
        if np.linalg.norm(self.info.object_local_pose.xy() - self.object_center_pos[0:2]) < \
            self.info.tolerance_xy:
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
            return False
        
    def call_reset_object(self, current_policy="ROT"):
        """Reset the object to the center"""
        self.info.running_policy = f"RESET_OBJECT_{current_policy}"
        goal = RunPolicyGoal(policy_name=f"RESET_OBJECT_{current_policy}")
        self._action_client.send_goal(goal)
        self._action_client.wait_for_result()

    def execute_current_face(self):
        self.info.current_state = TaskBState.WAIT
        self.info.time_current_start = rospy.get_time()

        # reset hand
        self.info.running_policy = "RESET_HAND"
        goal = RunPolicyGoal(policy_name="RESET_HAND")
        self._action_client.send_goal(goal)
        self._action_client.wait_for_result()
        rospy.loginfo("Hand reset done!")
        
        while not self._check_timeout():

            rospy.logdebug_throttle(1, f"Current state: {self.info.current_state}")

            if not self._check_target_reachable():
                self.info.current_state = TaskBState.ERROR
                break
            
            if self.info.current_state == TaskBState.WAIT:
                if self._check_current_visible()  and \
                    self._check_face_upwards(face=self.info.current_face):
                    self.info.current_state = TaskBState.BEFORE_ROTATE
            
            elif self.info.current_state == TaskBState.BEFORE_ROTATE:
                if self._check_rotate_terminate_condition():
                    """Just skip rotating the object"""
                    self.info.current_state = TaskBState.BEFORE_FLIP
                elif self._check_rotate_start_condition():
                    """Forced reset"""
                    if self.forced_reset:
                        self.call_reset_object(current_policy="ROT")
                    """Configure the rotation policy here"""
                    if self._check_should_rotate_cw():
                        self.info.running_policy = "ROT_CW"
                        goal = RunPolicyGoal(policy_name="ROT_CW")
                        self._action_client.send_goal(goal)
                        self.info.current_state = TaskBState.ROTATE_CW
                    else:
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
                    if self.info.running_policy == "ROT_CCW":
                        self._action_client.cancel_goal()
                    self.info.current_state = TaskBState.BEFORE_FLIP

            elif self.info.current_state == TaskBState.BEFORE_FLIP:
                if self._check_flip_start_condition():
                    """Forced reset"""
                    if self.forced_reset:
                        self.call_reset_object(current_policy="FLIP")
                    """Configure the rotation policy here"""
                    self.info.running_policy = "FLIP_OUT"
                    goal = RunPolicyGoal(policy_name="FLIP_OUT")
                    _debug_flip_start_time = time.time()
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
            example_sequence = ["A", "B", "C", "D", "E", "F", "B", "E", "C", "E", "D"]
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
            self.execute_new_face(new_face)
            if self.info.current_state == TaskBState.DONE:
                self.info.current_face = new_face
                self.record_finish()
            num_executed_faces += 1

        self.stop_task()
        

if __name__ == "__main__":
    rospy.init_node("taskB_highlevel", log_level=rospy.INFO)
    taskB = TaskBHighLevel()
    taskB.main_offline_test(mode=TaskBOfflineMode.RANDOM)
    # taskB.main_online_test()
