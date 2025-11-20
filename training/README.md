## 数据集格式

训练代码期望数据集结构如下：

```
datasets/data/
  ├── 1707xxxxxxxx/
  │   ├── data.csv
  │   ├── 1707xxxxxx.xxx.png
  │   ├── 1707xxxxxx.xxx.png
  │   └── ...
  ├── 1707xxxxxxxx/
  │   ├── data.csv
  │   ├── 1707xxxxxx.xxx.png
  │   └── ...
  └── ...
```

每个轨迹文件夹应包含：
- `data.csv`：包含时间戳、期望速度、四元数、位置、速度、速度命令和控制参数等
- PNG图像文件：深度图像，文件名对应CSV中的时间戳

CSV文件应包含以下列：
- `timestamp`：时间戳
- `desired_vel`：期望速度
- `quat_1`, `quat_2`, `quat_3`, `quat_4`：四元数
- `velcmd_x`, `velcmd_y`, `velcmd_z`：速度命令
- `ct_cmd`：控制命令
- `br_cmd_x`, `br_cmd_y`, `br_cmd_z`：操作命令
- `is_collide`：碰撞标志（可选）


## 快速开始

### 1. 从头开始训练light 模型（推荐）

```bash
cd /home/namy/catkin_ws/src/vitfly/training

# 使用默认配置
python light_train_att.py

# 使用自定义配置
python light_train_att.py --config config/light.yaml
```
