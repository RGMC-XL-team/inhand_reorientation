## 硬件驱动安装

### Realsense
1. 安装librealsense
```
sudo apt install ros-noetic-librealsense2
```
2. 安装realsense ROS wrapper，注意不要使用官方repo，使用[此fork](https://github.com/rjwb1/realsense-ros/tree/development)
```
git clone https://github.com/rjwb1/realsense-ros.git
```

### Azure Kinect DK
Task A is currently using the Azure Kinect DK for perception. To install the dependences of the Kinect, please refer to [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/lib/b1669a3c-08d7-4e7e-b7ef-a4cbc0a545cf/file/AzureKinectDK/README.md).


### AprilTag Detection
1. 安装apriltag及apriltag_ros，参考[Quickstart](https://github.com/AprilRobotics/apriltag_ros?tab=readme-ov-file#quickstart:~:text=Washington)
2. 添加需检测的marker ID：编辑`apriltag_ros/apriltag_ros/config/tags.yaml:standalone_tags`，检测结果以`tag_<ID>`形式发布到TF

### Easy Handeye
1. 安装easy_handeye
```
git clone https://github.com/IFL-CAMP/easy_handeye.git
cd ./easy_handeye
rosdep install -iyr --from-paths src
```



### Build the Workspace
1. 由于当前workspace下有的pkg需要`catkin_make`，有的需要`catkin build`，我们执行
```
cd <path-to-leap_ws>
catkin_make_isolated
```

### Conda
The name of the conda env on the Alienware Laptop is ```RGMC24```.