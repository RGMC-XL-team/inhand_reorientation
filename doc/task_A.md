# Task A


## Installation

### Simulation

In Python env:
```bash
pip install mujoco
pip install git+https://github.com/google-deepmind/dm_control.git # install the latest dm_control
pip install rospkg
pip install pandas
pip install scikit-learn
```

## Usage

### Simulation

Notice:
* The path of the xml file in `my_robot_description/urdf/leaphand_xml/leaphand_mujoco.xml` and `my_robot_description/urdf/objects/cylinder_mujoco.xml` is overridden by `leaphand_mujoco.py`, to expand relative path to absolute path dynamically. 
* The cylinder may be horizontally placed or vertical placed. To change the setup, the object initial config in `leaphand_mujoco.py` and the initialGrasping() in `leaphand_control.py` need to be modified.

Change the main function in `leap_task_A/scripts/leaphand_control.py` to test1().

Simulation can be run without ROS.
```bash
python leaphand_control.py
```

If you are using macOS, you need to use mjpython:
```bash
mjpython leaphand_control.py
```

Run with ROS:
```bash
# in conda env
cd leap_task_A/scripts
source ../../../../devel_isolated/setup.bash
python leaphand_control.py
```


### Real world

Change the main function in ```leap_task_A/scripts/leaphand_control.py``` to test2().

Terminal 1:
```bash
# in conda env
cd PATH_TO_WORKSPACE
source devel_isolated/setup.bash
roslaunch leap_task_A real_prepare.launch
```

Terminal 2:
```bash
# in conda env
cd leap_task_A/scripts
source ../../../../devel_isolated/setup.bash
python leaphand_control.py
```