#! /usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from leap_hardware.leap_hand_utils import LEAPhand_to_sim_ones
from leap_hardware.srv import leap_position, leap_positionRequest

DEFAULT_REAL_TO_SIM_INDICES = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]
DEFAULT_LEAP_DOF_LOWER = np.array(
    [
        -0.3140,
        -1.0470,
        -0.5060,
        -0.3660,
        -0.3490,
        -0.4700,
        -1.2000,
        -1.3400,
        -0.3140,
        -1.0470,
        -0.5060,
        -0.3660,
        -0.3140,
        -1.0470,
        -0.5060,
        -0.3660,
    ]
)
DEFAULT_LEAP_DOF_UPPER = np.array(
    [
        2.2300,
        1.0470,
        1.8850,
        2.0420,
        2.0940,
        2.4430,
        1.9000,
        1.8800,
        2.2300,
        1.0470,
        1.8850,
        2.0420,
        2.2300,
        1.0470,
        1.8850,
        2.0420,
    ]
)


def node_main():
    def real_to_sim(values):
        return values[DEFAULT_REAL_TO_SIM_INDICES]

    # TODO(yongpeng): use rosparam to sync
    real_to_sim_indices = DEFAULT_REAL_TO_SIM_INDICES
    leap_dof_lower = DEFAULT_LEAP_DOF_LOWER
    leap_dof_upper = DEFAULT_LEAP_DOF_UPPER

    pub_freq = rospy.get_param("joint_state_publish_freq", 10)
    ros_rate = rospy.Rate(pub_freq)

    rospy.wait_for_service("/leap_position")
    leap_pos_client = rospy.ServiceProxy("/leap_position", leap_position)

    leap_jstate_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)

    while not rospy.is_shutdown():
        leap_pos_resp = leap_pos_client(leap_positionRequest())
        leap_pos_new = np.array(leap_pos_resp.position)
        leap_pos_new = real_to_sim(LEAPhand_to_sim_ones(leap_pos_new))
        leap_pos_new = (leap_dof_upper - leap_dof_lower) * (leap_pos_new + 1) / 2 + leap_dof_lower

        leap_joint_state = JointState()
        leap_joint_state.header.stamp = rospy.Time.now()
        leap_joint_state.name = [f"joint_{i}" for i in real_to_sim_indices]
        leap_joint_state.position = leap_pos_new
        leap_joint_state.velocity = [0.0] * 16
        leap_joint_state.effort = [0.0] * 16

        leap_jstate_pub.publish(leap_joint_state)

        ros_rate.sleep()


if __name__ == "__main__":
    rospy.init_node("leap_joint_state_publisher")
    node_main()
