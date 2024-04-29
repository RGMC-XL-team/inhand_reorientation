## taskB

### 涉及文件结构

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

### 细节

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

### 运行

```
# Terminal1: 启动硬件（接通LeapHand电源）
roslaunch leap_hardware system.launch

# Terminal2: 启动policy action server
conda activate rlgpu
cd <path-to-leap_sim-rospkg>/leapsim/hardware
python agent_hw.py

# Terminal3: 启动TaskB
conda activate rlgpu
cd <path-to-leap_task_B-rospkg>/scripts
python ./taskB_highlevel.py
```

***TODO: 将上述启动命令写到All-In-One脚本里***

### 迭代

#### RL policy

- **生成LeapHand抓取位姿**
    - 方法一: 在仿真中可视化调节
    ```
    roslaunch my_robot_description view_leaphand_for_gym.launch
    # 通过sliders调节、通过RViz可视化LeapHand关节角
    # Ctrl+C退出后，在my_robot_description/config/default_leap_cube_50mm_new.yaml中，将canonical_pose（含16个关节角）复制到leapsim/cfg/task/LeapHandGrasp.yaml，注意不要覆盖已有的canonical_pose，将其注释掉
    ```
    - 方法二: 在真机上可视化调节
    ```
    roslaunch leap_hardware leap_sliders.launch
    # 通过sliders调节
    # 在leapsim/cfg/task/LeapHandGrasp.yaml中自行填入关节角，从上到下分别是motor(1/0/2/3/12/13/14/15/5/4/6/7/9/8/10/11)
    ```

- **生成RL的hand&object reset state**
    - 步骤一: 修改参数（Optional）<br/>
        除特殊说明外，参数位于LeapHandGrasp.yaml
        - `canonical_pose`: 生成grasp state的均值
        - `grasp_dof_search_radius`: 随机grasp state的搜索范围
        - `finger_dist_threshold`: 指尖和物体中心的最近距离
        - `grasp_cache_len`: 在LeapHandRot.yaml，记录数据的条数
    - 步骤二: 修改记录数据的条件（Optional）<br/>
        在leap_hand_grasp.py的compute_reward函数中，修改或增加cond1~cond4，当这些条件不全部满足时，当前生成的grasp state丢弃，当环境稳定运行50 steps时，记录grasp state
    - 步骤三: 生成数据 <br/>
        注意替换`<YOUR-CACHE-NAME>`，该过程可能较慢
        ```
        cd <path-to-leap_sim-rospkg>/leapsim
        for cube_scale in 0.9 0.95 1.0 1.05 1.1 
        do
            bash scripts/gen_grasp.sh $cube_scale <YOUR-CACHE-NAME> num_envs=1024 wandb_activate=false
        done
        ```
    - 步骤四: 可视化数据 <br/>
        Web可视化，需要pip[安装pinocchio](https://stack-of-tasks.github.io/pinocchio/download.html)和meshcat，若数据不符合逾期，重做步骤一~四
        ```
        cd <path-to-leap_sim-rospkg>/leapsim/visualize
        python ./leapviz.py
        # 修改leapviz.py中CACHE_PREFIX为<YOUR-CACHE-NAME>
        # 在浏览器查看
        # 按c+Enter查看下一条记录，共30条
        ```

- **训练RL policy**
    - 步骤一: 修改环境 <br/>
        - 修改compute_observation
        - 修改compute_reward
        - 修改其他
    - 步骤二: 训练 <br/>
        查看训练曲线需安装配置[wandb](https://wandb.ai/site)，并修改`config.yaml`中相关配置
        ```
        cd <path-to-leap_sim-rospkg>/leapsim
        # 如在服务器上训练，需先拷贝leapsim/cache中包含<YOUR-CACHE-NAME>的相关npy文件
        python3 train.py task=<LeapHandRot或LeapHandFlip> max_iterations=5000 task.env.object.type=cube_50mm task.env.grasp_cache_name=<YOUR-CACHE-NAME> wandb_activate=true
        ```
    - 步骤三: 部署 <br/>
        ```
        # 先保存config，确保当前LeapHandRot/LeapHandFlip.yaml的内容与所训练的policy相适应
        cd <path-to-leap_sim-rospkg>/leapsim
        python3 dump_config.py wandb_activate=false num_envs=1 headless=true test=true task=LeapHandRot checkpoint=/home/yongpeng/competition/leap_ws/src/leap_XL/leap_sim/leapsim/runs/<WANDB-RECORD-NAME>/nn/LeapHand.pth task.env.grasp_cache_name=<YOUR-CACHE-NAME>
        ```

