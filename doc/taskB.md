## taskB

1. 涉及文件结构

- `leap_task_B`
    - `config/taskB_XL.yaml` (config) taskB的参数
    - `scripts/taskB_highlevel.py` (ROS node) 接收`RGMC`上位机的指令（下一个朝上的面），进行逻辑决策，调用所有的policy
    - `src/.../leap_grasp_cube.py` (Python class) 输入cube的位姿，输出finger的抓取关节角
    - `src/.../taskB_utils` (utils)
    - `action/RunPolicy.action` (ROS action) 定义`taskB_highlevel.py`和`agent_hw.py`间的通信

- `leap_sim`
    - `leapsim/hardware/agent_hw` (ROS node) 接收`taskB_highlevel.py`的指令（通过ROS action通信），分配并执行所有的model-based或rl-based policy
    - `leapsim/cfd/dict/*.yaml`（config）有关rl policy的配置
    - `leapsim/cache/leap_canonical_pose*.npy`（numpy）灵巧手的home pose
    - `leapsim/runs/xxx-xxx-xxx/*` (weights) rl policy的网络参数

- `leap_hardware`
    - `src/.../hardware_controller.py` (Python class) R/W灵巧手的关节角
    - `src/.../leaphand_pinocchio.py` (Python class) 正逆运动学求解
    - `src/.../leaphand_pinocchio.py` (Python class) 上述结果可视化

2. 细节

- 灵巧手控制
    - ROS action定义
        - *RESET_HAND* | 灵巧手复位
        - *RESET_OBJECT* | 物体移到手掌中心
        - *ROT_CW* | 物体顺时针旋转（NotImplemented）
        - *ROT_CCW* | 物体逆时针旋转
        - *FLIP_IN* | 物体向内侧（掌心）翻转（NotImplemented）
        - *FLIP_OUT* | 物体向外侧（手指）翻转
    <br />注意：actions能够被中断，以便在ROT/FLIP到位时终止，action中断后，默认执行*RESET_HAND*（不会再次中断）。该action机制通过action_lib实现。
    - FSM定义
        - *WAIT* | 接收到新的`face command`，在当前`face`可见且朝上、目标`face`可达情况下进入*BEFORE_ROTATE*
        - *BEFORE_ROTATE* | 旋转到位后直接进入*BEFORE_FLIP*，否则发送`ROTATE_CW/ROTATE_CCW`的`action`
        - *ROTATE_CW/ROTATE_CCW* | 旋转到位后`cancel_goal`，进入*BEFORE_FLIP*
        - *BEFORE_FLIP* | 当物体不在手掌中心时发送*RESET_OBJECT*的`action`，否则发送*FLIP_OUT/FLIP_IN*的`action`
        - *FLIP_OUT/FLIP_IN* | 目标`face`朝上后`cancel_goal`并进入*DONE*
        - *DONE* | 到达目标`face`
        - *TIMEOUT* | 超时退出
