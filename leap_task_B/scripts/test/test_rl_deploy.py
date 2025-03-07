import os
import rospy, rospkg
import actionlib
from geometry_msgs.msg import TransformStamped
from leap_task_B.msg import RunPolicyGoal, RunPolicyAction
from ruamel.yaml import YAML
import time
yaml = YAML()


def get_object_pose():
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

def test_rl_deploy(record_time=60.0, save_file=False):
    input("Press Enter to start the experiment")

    action_client = actionlib.SimpleActionClient("run_policy", RunPolicyAction)
    action_client.wait_for_server()
    
    goal = RunPolicyGoal(policy_name="RESET_HAND")
    action_client.send_goal(goal)

    test_policy_name = "FLIP_OUT"
    goal = RunPolicyGoal(policy_name=test_policy_name)
    action_client.send_goal(goal)

    start_time = time.time()
    exp_data = []

    rate = rospy.Rate(20)
    
    while time.time() - start_time <= record_time:
        t_now = time.time() - start_time
        object_pose = get_object_pose()
        exp_data.append(
            {
                'index': len(exp_data),
                'time': t_now,
                'object_pose': object_pose
            }
        )

        rate.sleep()

        if save_file:
            yaml.dump(
                exp_data,
                open(os.path.join(rospkg.RosPack().get_path("leap_task_B"), "data", f"taskB_obj_pose_{test_policy_name}.yaml"), "w")
            )

    goal = RunPolicyGoal(policy_name="RESET_HAND")
    action_client.send_goal(goal)

if __name__ == "__main__":
    record_time = 20.0
    save_file = False
    rospy.init_node("test_rl_deploy")
    rospy.loginfo("test_rl_deploy started")
    test_rl_deploy(record_time, save_file)
