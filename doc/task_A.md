# Task A


## Installation

### Simulation

In Python env:
```bash
pip install mujoco
pip install git+https://github.com/google-deepmind/dm_control.git # install the latest dm_control
pip install rospkg
```

## Usage

### Simulation

Notice:
* The path of the xml file in ```my_robot_description/urdf/leaphand_xml/leaphand_mujoco.xml``` and ```my_robot_description/urdf/objects/cylinder_mujoco.xml``` must be the absolute path. Mingrui has not find a way to specifiy it as a relative path.
* The cylinder may be horizontally placed or vertical placed. To change the setup, the object initial config in ```leaphand_mujoco.py``` and the initialGrapsing() in ```leaphand_control.py``` need to be modified.

Change the main function in ```leap_task_A/scripts/leaphand_control.py``` to test1().

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