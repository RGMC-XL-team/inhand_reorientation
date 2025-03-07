# InHand_Reorientation

This repository is for the paper "Robust In-Hand Reorientation with Hierarchical RL-Based Motion Primitives and Model-Based Regrasping" submitted to IEEE RA-P.

Our approach won the championship of the In-Hand Manipulation Track of the 9th Robotic Grasping and Manipulation Competition (RGMC) and the Most Elegant Solution Award among all tracks. Please visit their [website](https://hangkaiyu.github.io/RGMC_in_hand_manipulation_subtrack.html) to learn more about the competition.

## Environment Setup

### Hardware
- 1x RealSense D405 camera (any RGB camera is supported)
- 1x LEAP Hand ([url](https://v1.leaphand.com/)) and set the baud rate to 3000000
- Other parts (will be launched after acceptance)
    - 3D-printed parts
    - Silicone fingertip
    - The object

### Software
- Ubuntu 20.04 with ROS Noetic
- Make sure the following ROS packages are installed
    ```
    actionlib
    std_msgs, sensor_msgs, geometry_msgs, visualization_msgs
    joint_state_publisher_gui
    robot_state_publisher
    ```
- Install dependencies of LEAP Hand such as the Dynamixel SDK, please refer to this [repo](https://github.com/leap-hand/LEAP_Hand_API)
- Install Pinocchio ([reference](https://stack-of-tasks.github.io/pinocchio/download.html)) and NVIDIA Isaac Gym Preview 4 ([reference](https://developer.nvidia.com/isaac-gym))
- Install Python dependencies (we recommend create a conda venv first)
    ```
    pip install -r requirements.txt
    ```
- Create a new ROS workspace and clone the repo into `src` folder, then `catkin_make_isolated` the workspace

## Run the task
```
# start the system
roslaunch leap_hardware system.launch

# start the RL agents
cd leap_sim/leapsim/hardware
python ./agent_hw.py

# run the task
cd leap_task_B/scripts
python ./taskB_highlevel.py
```

## Train the policies
We refer to the LEAP Hand official [repo](https://github.com/leap-hand/LEAP_Hand_Sim) for RL training 

```
cd leap_sim/leapsim

# train the ROT policy
python3 train.py task=LeapHandRot max_iterations=5000 task.env.object.type=cube_50mm task.env.grasp_cache_name=<YOUR-GRASP-CACHE-NAME> wandb_activate=true

# train the FLIP policy
python3 train.py task=LeapHandFlip max_iterations=5000 task.env.object.type=cube_50mm task.env.grasp_cache_name=<YOUR-GRASP-CACHE-NAME> wandb_activate=true
```