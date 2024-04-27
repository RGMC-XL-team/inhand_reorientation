"""
    This script contains the high level
    decision maker for taskB
"""

import os

import numpy as np
import yaml

import rospy
import rospkg
import actionlib

from leap_task_B.taskB_utils import TaskBInfo, TaskBState, DESIRED_YAW_DICT
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

    def _initialize_service_and_actions(self):
        self.action_name = rospy.get_param("action_name", "run_policy")
        rospy.wait_for_service("/object_state")
        rospy.wait_for_service("/face_pose")
        self.object_state_client = rospy.ServiceProxy("/object_state", object_state)
        self.face_pose_client = rospy.ServiceProxy("/face_pose", face_pose)
        self._action_client = actionlib.SimpleActionClient(self.action_name, RunPolicyAction)
        self._action_client.wait_for_server()

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

            rospy.loginfo_throttle(1, f"Current state: {self.info.current_state}")
            
            if self.info.current_state == TaskBState.WAIT:
                if self._check_current_visible() and \
                    self._check_target_reachable() and \
                    self._check_face_upwards(face=self.info.current_face):
                    self.info.current_state = TaskBState.BEFORE_ROTATE
            
            elif self.info.current_state == TaskBState.BEFORE_ROTATE:
                if self._check_rotate_terminate_condition():
                    """Just skip rotating the object"""
                    self.info.current_state = TaskBState.BEFORE_FLIP
                else:
                    """Configure the rotation policy here"""
                    self.info.running_policy = "ROT_CCW"
                    goal = RunPolicyGoal(policy_name="ROT_CCW")
                    self._action_client.send_goal(goal)
                    self.info.current_state = TaskBState.ROTATE_CCW

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
                    """Configure the rotation policy here"""
                    self.info.running_policy = "FLIP_OUT"
                    goal = RunPolicyGoal(policy_name="FLIP_OUT")
                    self._action_client.send_goal(goal)
                    self.info.current_state = TaskBState.FLIP_OUT
                else:
                    """Reset the object to the center"""
                    self.info.running_policy = "RESET_OBJECT"
                    goal = RunPolicyGoal(policy_name="RESET_OBJECT")
                    self._action_client.send_goal(goal)
                    self._action_client.wait_for_result()

            elif self.info.current_state == TaskBState.FLIP_IN or \
                self.info.current_state == TaskBState.FLIP_OUT:
                """Continue flipping the object"""
                if self._check_face_upwards(face=self.info.target_face):
                    """Stop flipping the object"""
                    if self.info.running_policy == "FLIP_OUT":
                        self._action_client.cancel_goal()
                    self.info.current_state = TaskBState.DONE
                    break

            else:
                raise ValueError(f"Invalid FSM state: {self.info.current_state}")

            self.fsm_rate.sleep()

        if self.info.current_state != TaskBState.DONE:
            self.info.current_state = TaskBState.TIMEOUT

        rospy.loginfo(f"Switch to target face {self.info.target_face} done \
                      with state {self.info.current_state}")

    def execute_new_face(self, face):
        assert face in ["A", "B", "C", "D", "E", "F"]
        self.info.target_face = face
        self.execute_current_face()

    def main_loop(self):
        self.execute_new_face("B")
        if self.info.current_state == TaskBState.DONE:
            self.info.current_face = "B"
        self.execute_new_face("C")
        if self.info.current_state == TaskBState.DONE:
            self.info.current_face = "C"
        self.execute_new_face("D")
        if self.info.current_state == TaskBState.DONE:
            self.info.current_face = "D"
        self.execute_new_face("E")
        if self.info.current_state == TaskBState.DONE:
            self.info.current_face = "E"
        self.execute_new_face("F")

if __name__ == "__main__":
    rospy.init_node("taskB_highlevel", log_level=rospy.INFO)
    taskB = TaskBHighLevel()
    taskB.main_loop()
