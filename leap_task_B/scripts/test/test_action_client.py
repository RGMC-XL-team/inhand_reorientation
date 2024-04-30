import pdb
import rospy
import actionlib

from leap_task_B.msg import RunPolicyGoal, RunPolicyAction


def test_action_client():
    action_client = actionlib.SimpleActionClient("run_policy", RunPolicyAction)
    action_client.wait_for_server()
    while not rospy.is_shutdown():
        pdb.set_trace()
        policy_name = "RESET_OBJECT"
        goal = RunPolicyGoal(policy_name=policy_name)
        action_client.send_goal(goal)
        action_client.wait_for_result()


if __name__ == "__main__":
    rospy.init_node("test_action_client")
    rospy.loginfo("test_action_client started")
    test_action_client()
