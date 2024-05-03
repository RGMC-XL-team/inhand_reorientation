import pdb
import rospy
import actionlib

from leap_task_B.msg import RunPolicyGoal, RunPolicyAction


def test_action_client():
    action_client = actionlib.SimpleActionClient("run_policy", RunPolicyAction)
    action_client.wait_for_server()
    while not rospy.is_shutdown():
        _name_from_user = input("Please input the policy name: ")
        if _name_from_user not in ["RESET_HAND", "RESET_OBJECT", "RESET_OBJECT_AGGRESSIVE", "ROT_CW", "ROT_CCW", "FLIP_IN", "FLIP_OUT", "STOP"]:
            _name_from_user = "RESET_HAND"
        print(f"send request for policy {_name_from_user}")
        policy_name = _name_from_user
        if _name_from_user != "STOP":
            goal = RunPolicyGoal(policy_name=policy_name)
            action_client.send_goal(goal)
        else:
            action_client.cancel_goal()
        # action_client.wait_for_result()


if __name__ == "__main__":
    rospy.init_node("test_action_client")
    rospy.loginfo("test_action_client started")
    test_action_client()
