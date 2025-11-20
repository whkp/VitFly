# 基于视觉Transformer的端到端四旋翼无人机避障系统


<!-- GIFs -->

#### 在仿真环境中的泛化能力
<img src="media/trees-vitlstm.gif" width="400" height="200"> <img src="media/walls-vitlstm.gif" width="250" height="200">

#### 零样本迁移到多障碍物和高速真实世界避障（GIF未加速）

<img src="media/multi-obstacle-vitlstm.gif" width="300" height="200"> <img src="media/7ms-vitlstm.gif" width="380" height="200">

<img src="media/multi-onboard-min.gif" width="300" height="200"> <img src="media/7ms-onboard-min.gif" width="380" height="200">

## 项目架构与工作流程

### 系统概述

本项目实现了一个基于视觉Transformer（ViT）模型的端到端四旋翼无人机避障系统。该系统可在仿真环境（Flightmare/Unity）和真实世界环境中运行，通过处理深度图像生成速度命令以实现自主导航。

### 核心组件

#### 1. 模型架构 (`models/`)
项目提供了多种用于视觉控制的神经网络架构：

- **ConvNet** (23.5万参数): 基线CNN + 全连接层架构
- **LSTMNet** (290万参数): CNN + LSTM用于时序建模
- **UNetConvLSTMNet** (290万参数): U-Net编码器-解码器 + LSTM
- **ViT** (310万参数): 视觉Transformer + 全连接层
- **ViTLSTM** (350万参数): 视觉Transformer + LSTM（性能最佳模型）
- **增强模型**: SpatialAttentionViT、MultiResolutionViT、RobustViTLSTM、TemporalConsistencyLSTM

`models/model.py`中的所有模型遵循统一接口：
- 输入: `[深度图像, 期望速度, 四元数, 隐藏状态(可选)]`
- 输出: `[速度命令(x,y,z), 隐藏状态]`
- 深度图像被归一化并调整为60×90像素

#### 2. 仿真环境 (`flightmare/`)
基于Flightmare仿真器和Unity渲染：

- **环境配置**: `flightmare/flightpy/configs/vision/config.yaml`
  - 支持多种障碍物场景（球体、树木、自定义环境）
  - 环境索引：球体场景0-100，树木场景0-499
  - 可配置的场景参数和无人机起始位置

- **Unity二进制文件**: 提供逼真的渲染和物理仿真
- **ROS集成**: 通过ROS话题进行状态、图像和命令通信

#### 3. 评估系统 (`envtest/ros/`)

**evaluation_node.py** - 性能监控和指标收集：
- 跟踪四旋翼状态（位置、速度、姿态）
- 监控障碍物距离和碰撞检测
- 记录轨迹数据和时间信息
- 生成包含成功率、完成时间和碰撞次数的评估摘要
- 将结果保存到带时间戳的目录：`evaluate/results_<时间戳>_<模型类型>/`
- 生成可视化图表（轨迹XY/3D、速度、障碍物距离）

**run_competition.py** - 主控制循环和模型推理：
- 订阅来自Unity仿真器的深度图像
- 加载并运行训练好的模型进行速度预测
- 向四旋翼控制器发布速度命令
- 支持基于视觉和基于状态（特权）的控制模式
- 将训练数据（图像+遥测）记录到`train_set/`用于数据集生成
- 处理不同架构的模型加载

**user_code.py** (引用但未显示)：
- 包含`compute_command_vision_based()`和`compute_command_state_based()`函数
- 实现ROS消息和模型推理之间的接口

#### 4. 启动脚本

**launch_evaluation.bash** - 编排仿真运行：
- 使用指定参数启动Flightmare仿真器
- 启动评估和竞赛节点
- 管理多次试验迭代并自动恢复
- 处理仿真器重置和超时检测（每次试验5分钟限制）
- 汇总多次运行的结果

**launch.bash** (在EVALUATION_GUIDE.md中引用)：
- 用于系统模型比较的高级批量评估脚本
- 支持多个模型 × 多个速度 × 多轮次
- 实现环境分配策略以确保公平比较
- 生成全面的分析报告和可视化

### 数据流架构

```
┌─────────────────────────────────────────────────────────────┐
│                      仿真环境                                 │
│  (Flightmare + Unity: 障碍物、物理、渲染)                    │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
             │ 深度图像                        │ 真实值
             │ (60×90, 归一化)                │ (状态、障碍物)
             ▼                                ▼
┌────────────────────────┐         ┌─────────────────────────┐
│   run_competition.py   │         │   evaluation_node.py    │
│  - 图像预处理           │         │  - 状态监控              │
│  - 模型推理             │         │  - 碰撞检测              │
│  - 命令生成             │         │  - 指标收集              │
└────────────┬───────────┘         └─────────────────────────┘
             │                                │
             │ 速度命令                        │ 性能数据
             │ (vx, vy, vz)                  │
             ▼                                ▼
┌────────────────────────┐         ┌─────────────────────────┐
│    四旋翼控制器         │         │    结果与分析            │
│  (Flightmare Pilot)    │         │  - YAML摘要             │
└────────────────────────┘         │  - 轨迹图表              │
                                    │  - 统计数据              │
                                    └─────────────────────────┘
```

### 运行模式

#### 基于视觉模式（主要）
1. 仿真器通过ROS话题`/kingfisher/dodgeros_pilot/unity/depth`发布深度图像
2. `run_competition.py`接收图像并进行预处理
3. 训练好的模型从深度图像+元数据预测速度命令
4. 命令发布到`/kingfisher/dodgeros_pilot/velocity_command`
5. `evaluation_node.py`监控性能并检测目标/碰撞

#### 基于状态模式（数据收集）
1. 使用特权状态信息（障碍物位置、速度）
2. 运行具有前瞻能力的专家策略
3. 收集深度图像+遥测数据用于训练数据集
4. 保存到`envtest/ros/train_set/`，结构如下：
   ```
   train_set/<模式>_<模型>_<时间戳>/
   ├── data.csv          # 遥测数据（姿态、速度、命令）
   └── <时间戳>.png      # 深度图像
   ```

### 训练流程

1. **数据收集**: 在基于状态模式下运行仿真以收集专家演示
2. **数据集准备**: 在`training/datasets/data/`中组织轨迹
3. **模型训练**: 使用配置文件执行`training/train.py`
4. **监控**: 使用TensorBoard跟踪训练指标
5. **评估**: 在基于视觉的仿真模式下测试训练好的模型

### 关键配置文件

- `flightmare/flightpy/configs/vision/config.yaml`: 环境和仿真设置
- `flightmare/flightpy/configs/scene.yaml`: 可用场景和起始位置
- `training/config/train.txt`: 训练超参数
- `envtest/ros/evaluation_config.yaml`: 评估参数（超时、边界框、目标距离）

### ROS通信话题

**订阅话题**：
- `/kingfisher/dodgeros_pilot/state` - 四旋翼状态（QuadState）
- `/kingfisher/dodgeros_pilot/unity/depth` - 深度图像（Image）
- `/kingfisher/dodgeros_pilot/groundtruth/obstacles` - 障碍物位置（ObstacleArray）
- `/kingfisher/start_navigation` - 启动信号（Empty）

**发布话题**：
- `/kingfisher/dodgeros_pilot/velocity_command` - 速度命令（TwistStamped）
- `/kingfisher/dodgeros_pilot/feedthrough_command` - 直接命令（Command）
- `/debug_img1`, `/debug_img2` - 调试可视化（Image）
- `/kingfisher/finish` - 完成信号（Empty）

### 性能评估指标

系统跟踪多个性能指标：

1. **成功率**: 无碰撞到达目标的试验百分比
2. **完成时间**: 穿越60米航线的持续时间
3. **碰撞次数**: 每次试验的障碍物撞击次数
4. **轨迹稳定性**: Y/Z位置的方差
5. **速度曲线**: 每段的平均速度
6. **安全裕度**: 到障碍物的最小距离

结果保存时包含完整的元数据，包括模型类型、速度设置、环境配置和时间戳，以便进行可重现的分析。

### 部署架构

对于真实世界部署，`depthfly` ROS包提供：
- 与Intel RealSense D435深度相机集成
- 实时模型推理（CPU上ViT+LSTM约25ms）
- 安全功能（触发信号、速度渐变、高度控制）
- 可配置的速度命令和坐标系

## 安装

注意：如果您只想训练模型而不在仿真中测试，可以直接跳到训练部分。

#### （可选）设置catkin工作空间

如果您想创建一个新的catkin工作空间，典型的工作流程如下（注意：此代码仅在ROS Noetic和Ubuntu 20.04上测试过）：
```
cd
mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config --extend /opt/ros/$ROS_DISTRO
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color
```

#### 克隆此仓库并设置

进入您想要的工作空间后，克隆此仓库（注意，我们将其重命名为`vitfly`）：
```
cd ~/catkin_ws/src
git clone git@github.com:whkp/VitFly.git vitfly
cd vitfly
```

为了复制我们用于训练和测试的Unity环境，您需要从[Datashare](https://upenn.app.box.com/v/ViT-quad-datashare)下载`environments.tar`（1GB）并解压到正确位置（如下）。我们提供了中等难度的球体场景和树木场景。其他障碍物环境由[ICRA 2022 DodgeDrone竞赛](https://github.com/uzh-rpg/agile_flight)提供。
```
tar -xvf <path/to/environments.tar> -C flightmare/flightpy/configs/vision
```

您还需要下载我们的Unity资源和二进制文件。从[Datashare](https://upenn.app.box.com/v/ViT-quad-datashare)下载`flightrender.tar`（450MB），然后：
```
tar -xvf <path/to/flightrender.tar> -C flightmare/flightrender
```

然后，通过提供的脚本安装依赖项：
```
bash setup_ros.bash
cd ../..
catkin build
source devel/setup.bash
cd src/vitfly
```

## 测试（仿真）

#### 下载预训练权重

从[Datashare](https://upenn.app.box.com/v/ViT-quad-datashare)下载`pretrained_models.tar`（50MB）。此压缩包包含ConvNet、LSTMnet、UNet、ViT和ViT+LSTM（我们的最佳模型）的预训练模型。
```
tar -xvf <path/to/pretrained_models.tar> -C models
```

#### 编辑配置文件

要在中等难度的球体或树木环境中测试，请按如下方式编辑文件`flightmare/flightpy/configs/vision/config.yaml`的第2-3行：
```
level: "spheres_medium" # 球体
env_folder: "environment_<0-100之间的任意整数>"
```
```
level: "trees" # 树木
env_folder: "environment_<0-499之间的任意整数>"
```

运行仿真时（下一节），您可以设置任意数量`N`的试验运行。要在相同的指定环境索引上运行试验，设置`datagen: 1`和`rollout: 0`。要在每次试验时顺序使用不同的环境索引，设置`datagen: 0`和`rollout: 1`。对于后者，使用前缀为`custom_`的环境。这些将静态障碍物设置为动态，以便在同一Unity实例中将它们移动到新位置。

您还可以根据Unity二进制文件中提供的场景更改`unity: scene: 2`场景索引。可用环境及其无人机起始位置在`flightmare/flightpy/configs/scene.yaml`中找到。

#### 运行仿真

`launch_evaluation.bash`脚本在使用`vision`模式时启动Flightmare和训练好的模型进行基于深度的飞行。要运行一次试验，执行：
```
bash launch_evaluation.bash 1 vision
```

一些细节：将`1`更改为您想要运行的任意试验次数。如果您查看bash脚本，会看到运行了多个python脚本。`envtest/ros/evaluation_node.py`统计碰撞、启动和中止试验，并向控制台打印其他统计信息。`envtest/ros/run_competition.py`订阅输入深度图像并将它们传递给相应的函数（位于`envtest/ros/user_code.py`），这些函数运行模型并返回期望的速度命令。话题`/debug_img1`流式传输带有叠加速度矢量箭头的深度图像，该箭头指示模型的输出速度命令。

## 训练

#### 下载并设置我们的数据集

训练数据集可从[Datashare](https://upenn.app.box.com/v/ViT-quad-datashare)获取`data.zip`（2.5GB，解压后3.4GB）。创建必要的目录并解压此数据（这可能需要一些时间）：
```
mkdir -p training/datasets/data training/logs
unzip <path/to/data.zip> -d training/datasets/data
```

此数据集包含各种球体环境中的580条轨迹。`data`中的每个数字命名的文件夹包含图像和遥测数据的专家轨迹。`<时间戳>.png`文件是深度图像，`data.csv`存储速度命令、姿态和其他遥测信息。

#### 训练模型

我们提供了一个脚本`train.py`，它在给定数据集上训练模型，参数从配置文件解析。我们提供了`training/config/train.txt`，其中包含一些默认超参数。注意，我们仅确认了GPU的功能。要训练：
```
python training/train.py --config training/config/train.txt
```

您可以使用Tensorboard监控训练和验证统计信息：
```
tensorboard --logdir training/logs
```

#### 在仿真中收集您自己的数据集

要创建自己的数据集，在状态模式下启动仿真（在按照测试部分中描述的对所选环境、相机参数或环境切换行为进行任何所需编辑后）以运行我们简单的特权专家策略。注意，此包含的前瞻专家策略具有有限的视野，并非万无一失，偶尔会撞到物体；但是，它在本文中成功使用。
```
bash launch_evaluation.bash 10 state
```
保存的深度图像和遥测数据会自动存储在`envtest/ros/train_set`中，格式可直接用于训练。将相关轨迹文件夹移动到您的新数据集目录。如果您之前清空了`train_set`目录，可以执行`mv envtest/ros/train_set/* training/datasets/new_dataset/`。然后，简单编辑您的配置文件`dataset = new_dataset`并按照上一节中的方式运行训练命令。

## 真实世界部署

我们提供了一个简单的ROS1包，用于在传入的深度相机流上运行训练好的模型。这个名为`depthfly`的包可以轻松修改以适应您的用例。在12核、16GB RAM、仅CPU的机器上（类似于用于硬件实验的机器），最复杂的模型ViT+LSTM单次推理应该需要约25ms。

您应该修改ROS节点python脚本`depthfly/scripts/run.py`中的`DEPTHFLY_PATH`、`self.desired_velocity`、`self.model_type`和`self.model_path`。此外，您需要适当修改订阅者和发布者中的ROS话题名称。

使用以下命令运行Realsense D435相机和模型推理节点：
```
roslaunch depthfly depthfly.launch
```

我们包含了一个`/trigger`信号，当持续发布时，会将预测的速度命令路由到给定话题`/robot/cmd_vel`。我们通常从基站计算机发送以下终端命令来实现这一点。如果您按Ctl+C此命令，rosnode将发送速度0.0命令以就地停止。
```
rostopic pub -r 50 /trigger std_msgs/Empty "{}"
```

一些细节：
- 请记住，模型被训练为以期望速度持续飞行，需要手动飞行员接管才能停止。
- 节点的原始输出发布在`/output`话题上。
- 由于假设我们正在躲避垂直障碍物，我们忽略z速度模型命令，而是采用p增益控制器来达到1.0的期望高度（第159行）（如果您决定接受模型的z命令，则确保读取机器人里程计并保持最小和最大高度）。
- 我们在`run()`函数中使用渐变来在2秒内平滑加速无人机到期望速度。
- 速度命令相对于x方向向前、y方向向左和z方向向上发布。

请安全飞行！

## 引用

```
@article{bhattacharya2024vision,
  title={Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance},
  author={Bhattacharya, Anish and Rao, Nishanth and Parikh, Dhruv and Kunapuli, Pratik and Matni, Nikolai and Kumar, Vijay},
  journal={arXiv preprint arXiv:2405.10391},
  year={2024}
}
```

## 致谢

仿真启动代码以及`flightmare`和`dodgedrone_simulation`的版本来自[ICRA 2022 DodgeDrone竞赛代码](https://github.com/uzh-rpg/agile_flight)。

---

### 以下是一些调试技巧

#### 构建flightlib时出现现有`eigen`的`catkin build`错误
错误消息：
```
CMake Error: The current CMakeCache.txt directory vitfly/flightmare/flightlib/externals/eigen/CMakeCache.txt is different than the directory <some-other-package>/eigen where CMakeCache.txt was created. This may result in binaries being created in the wrong place. If you are not sure, reedit the CMakeCache.txt.
```
可能的解决方案：
```
cd flightmare/flightlib/externals/eigen
rm -rf CMakeCache.txt CMakeFiles
cd <your-workspace>
catkin clean
catkin build
```

#### `[Pilot]        Not in hover, won't switch to velocity reference!`（警告）
只要进一步的控制台打印显示发送启动导航命令和运行compute_command_vision_based模型，您可以忽略此警告。

#### `[readTrainingObs] Configuration file � does not exists.`（警告）
当您处于`datagen: 1, rollout: 0`模式时会出现此警告，场景管理器会查找`datagen: 0, rollout: 1`模式所需的`custom_`前缀场景。您可以忽略此警告。
