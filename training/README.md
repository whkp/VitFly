# 多任务学习训练代码

这个目录包含为UltraLightMultiTaskTransformer模型设计的训练代码。该模型不仅可以预测控制指令，还可以估计障碍物距离和碰撞风险。

## 主要特性

- 支持从CSV格式的数据中加载轨迹数据
- 多任务学习：同时优化控制、障碍物感知和碰撞风险评估
- TensorBoard支持：可视化训练过程中的损失和指标
- 检查点保存和加载：支持中断后继续训练
- 灵活的学习率调度：包括预热和余弦衰减

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

## 安装依赖

```bash
pip install torch torchvision numpy pandas matplotlib opencv-python tqdm configargparse
```

## 使用方法

### 基本用法

```bash
python my_train_multitask.py
```

### 主要参数

- `--datadir`：数据集根目录路径（默认：/home/{用户名}/vitfly_ws/src/vitfly/training/datasets）
- `--dataset`：数据集名称（默认：data）
- `--batch_size`：批次大小（默认：32）
- `--lr`：学习率（默认：1e-4）
- `--N_eps`：训练轮数（默认：100）
- `--device`：设备（默认：cuda）
- `--val_split`：验证集比例（默认：0.2）

### 多任务权重

- `--control_weight`：控制任务的损失权重（默认：1.0）
- `--distance_weight`：距离估计任务的损失权重（默认：0.5）
- `--collision_weight`：碰撞风险任务的损失权重（默认：0.5）

### 从检查点恢复训练

```bash
python my_train_multitask.py --load_checkpoint --checkpoint_path /path/to/model.pt
```

## 测试数据集

为了验证数据集是否正确加载，可以运行测试脚本：

```bash
python test_dataset.py
```

## 训练结果

训练过程中，模型会保存在以下位置：
- 定期检查点：`{logdir}/train_{日期时间}_multitask/checkpoints/model_epXXXXXX.pt`
- 最佳模型：`{logdir}/train_{日期时间}_multitask/model_best.pt`
- 最终模型：`{logdir}/train_{日期时间}_multitask/model_final.pt`

## 日志和可视化

训练日志保存在工作空间目录中：
- 文本日志：`{logdir}/train_{日期时间}_multitask/log.txt`
- TensorBoard日志：`{logdir}/train_{日期时间}_multitask/`

可以使用TensorBoard查看训练进度：

```bash
tensorboard --logdir={logdir}
``` 