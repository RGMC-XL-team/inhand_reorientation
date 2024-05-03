import numpy as np
import rospy

from geometry_msgs.msg import TransformStamped

from leap_hardware.ros_utils import ros_transform_to_rigidtransform, transform_to_pos_rvec


class RunningMeanCovariance(object):
    def __init__(self, queue_size, time_window) -> None:
        self.queue_size = 30
        self.time_window = 2.0

        self.time_stamp_list = []
        self.pose_6d_list = []

        self.running_mean = np.zeros(6)
        self.running_covariance = np.zeros((6, 6))

    def get_size(self):
        assert len(self.time_stamp_list) == len(self.pose_6d_list)
        return len(self.pose_6d_list)
    
    def drop_outdated(self):
        outdated_indices = rospy.Time.now().to_sec() - np.array(self.time_stamp_list) >= self.time_window
        if np.sum(outdated_indices) <= 0:
            return
        cutoff_index = int(np.where(outdated_indices)[0][-1]+1)
        self.time_stamp_list = self.time_stamp_list[cutoff_index:]
        self.pose_6d_list = self.pose_6d_list[cutoff_index:]

    def append_new(self, transform: TransformStamped):
        _size = self.get_size()
        if _size >= self.queue_size:
            self.time_stamp_list.pop(0)
            self.pose_6d_list.pop(0)
        _transform = ros_transform_to_rigidtransform(transform)
        pos_rvec = transform_to_pos_rvec(_transform)

        self.time_stamp_list.append(transform.header.stamp.to_sec())
        self.pose_6d_list.append(pos_rvec)

    def get_covariance(self):
        self.drop_outdated()
        _size = self.get_size()
        if _size < self.queue_size:
            return None
        else:
            # TODO(yongpeng): use running covariance
            return np.var(self.pose_6d_list, axis=0).sum()
