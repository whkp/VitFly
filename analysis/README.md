# VitFly轨迹数据分析工具

本分析工具包提供了用于分析和可视化飞行器轨迹数据的完整解决方案，专门针对不同机器学习模型（Expert、ViT、ViT+LSTM、ConvNet、RobustViT+LSTM）的性能比较。

## 📁 数据结构

```
data/
├── expert/          # 专家系统数据
├── vit/             # ViT模型数据
├── vitlstm/         # ViT+LSTM模型数据
├── convnet/         # ConvNet模型数据
└── robustVitLstm/   # RobustViT+LSTM模型数据
    ├── 0/           # 试验0
    │   ├── data.csv # 轨迹数据文件
    │   └── *.png    # 图像文件
    ├── 1/           # 试验1
    │   ├── data.csv
    │   └── *.png
    └── ...
```

### CSV数据格式
每个`data.csv`文件包含以下列：
- `pos_x`, `pos_y`, `pos_z`: 位置坐标 (米)
- `vel_x`, `vel_y`, `vel_z`: 速度分量 (米/秒)
- `desired_vel`: 期望速度 (米/秒)
- `is_collide`: 碰撞标记 (0/1)
- 其他传感器和控制数据列

## 🛠️ 主要脚本功能

### 1. plot.py - 综合轨迹分析器

这是主要的分析脚本，提供全面的轨迹数据分析功能。

#### 核心类和功能：

**TrajectoryAnalyzer类**
- **数据加载**: 自动扫描子文件夹，加载所有CSV文件
- **碰撞分析**: 计算碰撞率、碰撞事件数量、平均碰撞持续时间
- **轨迹统计**: 按X轴分段计算位置统计信息（均值、标准差）
- **可视化**: 支持2D和3D轨迹绘制，包含障碍物显示

**主要方法**:
```python
# 初始化分析器
analyzer = TrajectoryAnalyzer(data_folder, model_name, obs_folder)

# 分析碰撞
collision_stats = analyzer.analyze_collisions()

# 计算轨迹统计
traj_stats = analyzer.compute_trajectory_statistics(x_bins=60)

# 绘制2D轨迹（XY或XZ平面）
analyzer.plot_2d_trajectory(ax, plane='xy', color='blue')

# 绘制3D轨迹
analyzer.plot_3d_trajectory(ax, color='blue', with_obstacles=True)
```

#### 生成的可视化图表：

1. **碰撞分析图表** (`collision_analysis.png`)
   - 碰撞率比较
   - 碰撞事件数量
   - 平均碰撞持续时间
   - 安全排名

2. **2D轨迹投影** 
   - XY平面轨迹 (`trajectory_xy_plane.png`)
   - XZ平面轨迹 (`trajectory_xz_plane.png`)
   - 包含置信区间（均值±标准差）

3. **3D轨迹可视化** (`trajectory_3d.png`)
   - 三维空间轨迹对比
   - 静态障碍物显示

4. **性能指标热力图** (`metrics_heatmap.png`)
   - 标准化指标比较
   - 统计摘要表 (`statistical_summary.csv`)

### 2. result_plot.py - 模型对比分析器

专门用于多模型性能对比的脚本，提供更详细的统计分析。

#### 核心类和功能：

**ModelConfig类**
- 定义每个模型的可视化配置（颜色、标记、线型等）

**TrajectoryDataLoader类**
- 按试验加载数据，支持按速度过滤
- 计算详细的碰撞统计（试验级别和时间步级别）
- 提供位置统计分析

**ComparisonPlotter类**
- 生成多模型对比图表
- 创建性能摘要表

#### 生成的对比图表：

1. **碰撞对比** (`collision_comparison.png`)
   - 试验碰撞率
   - 成功率对比

2. **轨迹对比** (`trajectory_comparison.png`)
   - XY和XZ平面的轨迹散点图对比

3. **3D轨迹对比** (`trajectory_3d_comparison.png`)
   - 三维空间的模型轨迹对比

4. **性能摘要表** (`summary_table.png`, `model_comparison_summary.csv`)
   - 详细的模型性能指标表格

## 🚀 使用方法

### 基本使用

1. **运行综合分析**:
```bash
cd analysis/
python plot.py
```

2. **运行模型对比分析**:
```bash
python result_plot.py
```

### 自定义分析

```python
# 导入必要的模块
from plot import analyze_all_models
from result_plot import analyze_model_comparison

# 运行综合分析
analyze_all_models(data_directory="./data", 
                  obs_folder="path/to/obstacles")

# 运行模型对比
analyze_model_comparison(data_directory="./data", 
                        output_dir="./custom_plots")
```

### 参数配置

**关键参数**:
- `data_directory`: 数据目录路径
- `obs_folder`: 障碍物数据文件夹（可选）
- `output_dir`: 输出图表目录
- `x_bins`: X轴分段数量（默认60）
- `speed_filter`: 速度过滤器（可选）

## 📊 输出文件说明

### 图像文件
- **PNG格式**: 高分辨率（300 DPI）用于展示
- **PDF格式**: 矢量图用于论文发表

### 数据文件
- **CSV格式**: 包含所有计算的统计指标
- **可直接导入Excel或其他分析工具**

### 生成的指标

**碰撞相关指标**:
- 碰撞率（按试验和时间步计算）
- 碰撞事件数量
- 平均碰撞持续时间
- 成功率

**轨迹相关指标**:
- 位置统计（均值、标准差）
- 轨迹平滑度
- 轨迹变异性

## 🔧 自定义和扩展

### 添加新模型

1. 在`model_configs`字典中添加新模型配置:
```python
model_configs = {
    'new_model': {'color': 'purple', 'name': 'New Model'},
    # ...existing models
}
```

2. 确保数据文件夹结构与现有模型一致

### 修改可视化样式

```python
# 修改颜色方案
plt.style.use('seaborn')  # 或其他样式

# 自定义颜色
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
```

### 添加新的分析指标

```python
def custom_analysis_function(data):
    # 实现自定义分析逻辑
    return custom_metrics

# 在主分析函数中调用
custom_results = custom_analysis_function(trajectory_data)
```

## 📈 分析结果解读

### 性能指标解释

1. **碰撞率**: 
   - 试验碰撞率：发生碰撞的试验比例
   - 时间步碰撞率：所有时间步中碰撞的比例

2. **轨迹变异性**:
   - Y/Z标准差：反映轨迹在横向/纵向的稳定性
   - 标准差越小表示轨迹越稳定

3. **安全评分**:
   - 基于碰撞率计算：100% - 碰撞率
   - 越高表示模型越安全

### 模型性能排名

通常按以下顺序评估模型性能：
1. **安全性**: 碰撞率（越低越好）
2. **稳定性**: 轨迹变异性（越小越好）
3. **效率**: 成功完成任务的比例（越高越好）

## 🐛 故障排除

### 常见问题

1. **数据加载失败**:
   - 检查CSV文件格式和列名
   - 确认文件路径正确
   - 验证数据文件夹结构

2. **绘图错误**:
   - 检查matplotlib版本兼容性
   - 确认有足够的内存处理大型数据集
   - 验证输出目录写权限

3. **内存不足**:
   - 减少数据采样数量
   - 分批处理大型数据集
   - 使用数据过滤器

### 依赖要求

```bash
pip install numpy pandas matplotlib seaborn
```

## 📝 更新日志

- **v2.0**: 完全重构，支持实际CSV数据格式
- **v1.1**: 添加3D可视化和障碍物支持
- **v1.0**: 初始版本，基础轨迹分析功能

## 📧 联系信息

如有问题或建议，请联系开发团队或提交issue到项目仓库。