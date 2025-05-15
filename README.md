# InHand_Reorientation

This repository is for the paper "Robust In-Hand Reorientation with Hierarchical RL-Based Motion Primitives and Model-Based Regrasping" submitted to IEEE RA-P.

Our approach won the championship of the In-Hand Manipulation Track of the 9th Robotic Grasping and Manipulation Competition (RGMC) and the Most Elegant Solution Award among all tracks. Please visit their [website](https://hangkaiyu.github.io/RGMC_in_hand_manipulation_subtrack.html) to learn more about the competition.

## Environment Setup

### Hardware
- 1x RealSense D405 camera (any RGB camera is supported)
- 1x LEAP Hand ([url](https://v1.leaphand.com/)) and set the baud rate to 3000000
- Other parts (see the supplementary [CAD files](https://drive.google.com/drive/folders/1wRGaV5bB4EGYVtHWL2-WbT_hgUvj4KIQ?usp=sharing) and the corresponding .txt)
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
- Install dependencies of LEAP Hand such as the Dynamixel SDK ([reference](#install-leap-hand-sdk))
- Install Pinocchio ([reference](#install-pinocchio)) and NVIDIA Isaac Gym Preview 4 ([reference](#install-isaac-gym))
- Install Python dependencies into the `rlgpu` virtual env (you can find `requirements.txt` in the root of this repo)
    ```
    pip install -r requirements.txt
    ```
- Create a new ROS workspace and clone these repositories into `src` folder, then `catkin_make_isolated` the workspace
    ```
    cd path-to-your-workspace/src

    # clone this repo
    git clone https://github.com/RGMC-XL-team/inhand_reorientation.git
    
    # clone the evaluator repo (there are some dependencies in use)
    git clone https://github.com/Rice-RobotPI-Lab/RGMC_In-Hand_Manipulation_2024.git
    git checkout real-eval

    cd path-to-your-workspace/
    catkin_make_isolated
    ```
- Prepare the datasets `runs/` an `cache/` (see the supplementary [Datasets](https://drive.google.com/drive/folders/1GN-KxQTjhRVIoPXXx9aKyV5DzTSBRsf9?usp=sharing) and the corresponding .txt)

## Examples
### Run the task
```
# In the first terminal, start the system
roslaunch leap_hardware system.launch

# In the second terminal, start the RL agents
cd leap_sim/leapsim/hardware
python ./agent_hw.py

# In the third terminal, run the task
cd leap_task_B/scripts
python ./taskB_highlevel.py
```

### Train the policies
We refer to the LEAP Hand official [repo](https://github.com/leap-hand/LEAP_Hand_Sim) for RL training 

```
cd leap_sim/leapsim

# train the ROT policy
python3 train.py task=LeapHandRot max_iterations=5000 task.env.object.type=cube_50mm task.env.grasp_cache_name=custom_grasp_cache_v6flip wandb_activate=true

# train the FLIP policy
python3 train.py task=LeapHandFlip max_iterations=5000 task.env.object.type=cube_50mm task.env.grasp_cache_name=custom_grasp_cache_v4cw wandb_activate=true
```

## Appendix

### <span id="install_leap_hand_sdk">Install LEAP Hand SDK</span>
1. ***Hardware Setup*** Visit [official repo](https://github.com/leap-hand/LEAP_Hand_API) and follow the instructions in `Hardware Setup`. Please install Dynamixel Wizard following the instructions [here](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/#install-linux). Note that we configure the baud rate of all servos to be `3000000` instead of `4000000`. Check baud rate setting if you encounter any communication problems
2. ***Software Setup*** Follow the README in `python/` to install the python dependencies


### <span id="install_isaac_gym">Install Isaac Gym</span>
1. ***Download Package*** Download Isaac Gym Preview 4 release for Ubuntu 18.04/20.04 from the [official website](https://developer.nvidia.com/isaac-gym), please check if your system meets the requirements. Extract the downloaded file to `IsaacGym_Preview_4_Package/`
2. ***Create Conda Virtual Env*** Please DO NOT execute `./create_conda_env_rlgpu.sh` as requested by the official installation instructions. Instead, create conda environment manually to install the correct version of Python, torch, and cuda
    ```
    # create env (python 3.8)
    conda create -n rlgpu python=3.8
    
    # activate env
    conda activate rlgpu

    # install pytorch
    pip3 install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
    
    # install isaac gym
    cd IsaacGym_Preview_4_Package/isaacgym/python
    pip install -e .

    # (optional) test installation
    python joint_monkey.py
    ```

### <span id="install_pinocchio">Install Pinocchio</span>
1. ***Installation*** Follow the instructions in `Add robotpkg apt repository` and `Install Pinocchio` in the [official website](https://stack-of-tasks.github.io/pinocchio/download.html)
2. ***Configure environment variables*** Follow the instructions in `Configure environment variables`, except for the `PYTHONPATH` variable, change `python3.10` to `python3.8`. Because we are using `python3.8` for this project
3. ***Check*** Pinocchio should now be imported by any python3.8 env, try `import pinocchio as pin` in `rlgpu` (installed with [isaacgym](#install-isaac-gym))


## <span id="trouble_shooting">Trouble Shooting</span>
