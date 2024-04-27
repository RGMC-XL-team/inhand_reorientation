import numpy as np
from scipy.spatial.transform import Rotation as Rot
from matplotlib import pyplot as plt
import pickle

OBS_IDX_DICT = {
    "dof_pos": (0, 16),
    "dof_vel": (16, 32),
    "obj_pose": (32, 39),
    "obj_linvel": (39, 42),
    "obj_angvel": (42, 45),
    "goal_rot": (45, 49),
    "quat_dist": (49, 53),
    "ftip_state": (53, 105),
    "action": (105, 121),
}

MOTOR_IDS = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]

def plot_joint_position():
    q_command = np.load("./q_command.npy")
    q_reach = np.load("./q_reach.npy")

    hz = 20
    t_end = len(q_command) / hz
    t_knots = np.linspace(0, t_end, len(q_command))

    plt.figure()
    for i in range(16):
        _id = MOTOR_IDS[i]
        plt.subplot(4, 4, i + 1)
        plt.plot(t_knots, q_command[:, i], linestyle=":", label=f"cmd{_id}")
        plt.plot(t_knots, q_reach[:, i], linestyle="-", label=f"q{_id}")
        plt.legend()

    plt.title("hardware execution (use motor IDs)")
    plt.show()


def plot_object_pose():
    obs_full = np.load("./obs_full_sim_rotz_goal.npy").squeeze(1)
    idx_start, idx_end = OBS_IDX_DICT["obj_pose"]
    obj_pose = obs_full[:, idx_start:idx_end]

    hz = 20
    t_end = len(obj_pose) / hz
    t_knots = np.linspace(0, t_end, len(obj_pose))

    plt.figure()
    plt.plot(t_knots, obj_pose[:, :3], label=["x", "y", "z"])
    plt.legend()
    plt.title("position")

    plt.figure()
    plt.plot(t_knots, obj_pose[:, 3:], label=["qx", "qy", "qz", "qw"])
    plt.legend()
    plt.title("orientation")

    plt.show()


def plot_data_collection_result():
    data_id = 11
    data = pickle.load(open("./lfd_data.pkl", "rb"))
    leap_data = np.array(data[data_id]["leap_hand"])
    obj_data = np.array(data[data_id]["object"])
    record_time = np.array(data[data_id]["time"])

    print("average interval: {}(s)".format(np.diff(record_time).mean()))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(record_time, obj_data[:, :3], label=["x", "y", "z"])
    obj_euler = Rot.from_quat(obj_data[:, 3:]).as_euler("xyz", degrees=False)
    plt.plot(record_time, obj_euler, label=["rx", "ry", "rz"])
    plt.legend()
    plt.title("object data")

    plt.figure()
    for i in range(16):
        _id = MOTOR_IDS[i]
        plt.subplot(4, 4, i + 1)
        plt.plot(record_time, leap_data[:, i], label=f"q_{_id}")
        plt.legend()
    plt.title("leap hand data")

    plt.show()


if __name__ == "__main__":
    # plot_joint_position()
    # plot_object_pose()
    plot_data_collection_result()
