## 启动操作

1. 手眼标定
- 进行标定
```
roslaunch leap_hardware leap_calibration.launch
# 将8cm的AprilTag固定在某根finger末端，用手拖动finger进行标定
# 根据AprilTag的ID和size，修改apriltag_ros/apriltag_ros/config/tags.yaml:standalone_tags参数
# 根据相机ROS驱动，修改相机驱动的launch，修改AprilTag检测的camera_name和image_topic参数
# 根据二维码固定的finger，修改手眼标定的robot_effector_frame
# 根据所贴二维码，修改手眼标定的tracking_marker_frame
```
- 发布标定的结果
```
roslaunch leap_hardware leap_calibration_publish.launch
# 在RViz中通过点云和mesh对齐的情况判断标定结果是否准确
# 修改标定结果路径calibration_file
```

2. 启动系统
- All-In-One
```
roslaunch leap_hardware system.launch
# 启动LeapHand、相机（realsense）和AprilTag识别
# 识别物体pose（ROS service）
# 打开RViz，但手指位置不会更新
```

3. 训练及部署
- 训练
```
cd <path-to-leap_sim-pkg>/leapsim

python3 train.py task=<环境，如LeapHandRot> max_iterations=1000 task.env.grasp_cache_name=<手的reset pose distribution，如custom_grasp_cache> wandb_activate=false
```
- 部署
```
# sequential, proprioception input
cd <path-to-leap_sim-pkg>/leapsim
python3 deploy.py wandb_activate=false num_envs=1 headless=false test=true task=LeapHandRot checkpoint=runs/pretrained/nn/LeapHand.pth
# single-frame input (under developing)
cd <path-to-leap_hardware-pkg>/src/leap_hardware
python3 ./leap_hardware_agent.py
```
- 可视化grasping pose
```
cd <path-to-leap_sim-pkg>/leapsim/visualize

python ./leapviz.py
# 需要在python环境安装meshcat
```