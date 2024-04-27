#!/usr/bin/env python3

import os
import numpy as np
import pickle
import rospy
import rospkg

from leap_hardware.hardware_controller import LeapHand
from leap_hardware.srv import object_state
from sensor_msgs.msg import JointState


"""
This node collects data for the LeapHand states and (possibly)
object states through demonstration.
We hope the data could be useful for downstream tasks such as
imitation learning.
"""

class LeapDataCollector(object):
    def __init__(self) -> None:
        # hyper-parameters
        self.record_freq = rospy.get_param("/record_freq", 10)
        self.record_duration = rospy.get_param("/record_duration", 70)
        self.data_save_path = rospy.get_param("/data_save_path", "")
        self.publish_joint_states = rospy.get_param("/publish_joint_states", True)

        # prepare data save
        if self.data_save_path == "":
            rospack = rospkg.RosPack()
            self.data_save_path = os.path.join(rospack.get_path("leap_hardware"), "debug", "lfd_data.pkl")
            rospy.loginfo("LeapDataCollector: demonstration data will be saved to %s" % self.data_save_path)
        if os.path.exists(self.data_save_path):
            self.collected_data = pickle.load(open(self.data_save_path, "rb"))
        else:
            self.collected_data = {}
        self.num_data = len(self.collected_data)

        # prepare joint state publish
        if self.publish_joint_states:
            self.real_to_sim_indices = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]
        self.leap_jstate_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)

        # preapre perception
        rospy.loginfo("LeapDataCollector: waiting for object_state service...")
        rospy.wait_for_service("/object_state")
        self.object_state_proxy = rospy.ServiceProxy("/object_state", object_state)

        # prepare leap hardware
        rospy.loginfo("LeapDataCollector: waiting for leaphand service...")
        rospy.wait_for_service("/leap_position")
        rospy.wait_for_service("/leap_velocity")
        self.leap_hardware = LeapHand(use_default=True)

        # log metadata
        if "meta_data" not in self.collected_data:
            self.collected_data["meta_data"] = {
                "leap_dof_lower": self.leap_hardware.leap_dof_lower.tolist(),
                "leap_dof_upper": self.leap_hardware.leap_dof_upper.tolist(),
            }

        rospy.loginfo("LeapDataCollector: everything initialized, please move the hand to start position")

    def publish_leap_joint_states(self, states):
        leap_pos_new = states.copy()

        leap_joint_state = JointState()
        leap_joint_state.header.stamp = rospy.Time.now()
        leap_joint_state.name = [f"joint_{i}" for i in self.real_to_sim_indices]
        leap_joint_state.position = leap_pos_new
        leap_joint_state.velocity = [0.0] * 16
        leap_joint_state.effort = [0.0] * 16

        self.leap_jstate_pub.publish(leap_joint_state)

    def main_loop(self):
        rospy.sleep(8)
        rospy.loginfo("LeapDataCollector: press [enter] to start data collection...")
        input()
        
        _rate = rospy.Rate(self.record_freq)
        _start_time = rospy.Time.now()
        _data = {
            "leap_hand": [],
            "object": [],
            "time": []
        }

        _poll_time_cost = []

        while not rospy.is_shutdown():
            # record leaphand data
            _poll_start = rospy.Time.now()
            _pos, _vel = self.leap_hardware.poll_joint_state()
            _poll_end = rospy.Time.now()
            _poll_time_cost.append((_poll_end - _poll_start).to_sec())

            _data["leap_hand"].append(np.concatenate((_pos, _vel)).tolist())
            if self.publish_joint_states:
                self.publish_leap_joint_states(_pos)

            # record object data
            _obj_state = self.object_state_proxy().pose
            _data["object"].append(np.array(_obj_state).tolist())

            # record time
            _data["time"].append((rospy.Time.now() - _start_time).to_sec())

            # check timeout
            if (rospy.Time.now() - _start_time).to_sec() > self.record_duration:
                break

            _rate.sleep()
        
        self.num_data += 1
        self.collected_data.update({self.num_data: _data})
        with open(self.data_save_path, "wb") as f:
            pickle.dump(self.collected_data, f)
        rospy.loginfo("LeapDataCollector: data collection finished, {} pieces in total.".format(self.num_data))
        rospy.loginfo("LeapDataCollector: average poll time cost: {:.3f}s".format(np.mean(_poll_time_cost)))


if __name__ == "__main__":
    nh = rospy.init_node("leap_data_collector")
    collector = LeapDataCollector()
    collector.main_loop()
    