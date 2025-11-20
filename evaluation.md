# 无人机模型全面评估系统

## 概述
这个系统提供了对多个深度学习模型在不同速度下进行无人机避障任务的全面评估。系统支持自动化测试、结果分析和可视化，采用科学的环境分配策略确保实验结果的公平性和可信度。

## 🚀 快速开始

### 1. 全面评估（所有模型 × 所有速度）
```bash
# 每个配置运行5次，使用vision模式
bash launch.bash 5 vision

# 自定义输出目录
bash launch.bash 3 vision ./my_evaluation_results
```

### 2. 快速测试（部分模型 × 部分速度）
```bash
# 快速测试：3个主要模型 × 3个速度，每配置3次
bash quick_evaluation.bash 3 vision

# 快速测试：每配置1次（最快）
bash quick_evaluation.bash 1 vision
```

## 🎯 环境分配策略（重要特性）

### 科学的实验设计
为确保模型比较的公平性和结果的可信度，系统采用以下环境分配策略：

#### 1. **同一轮次统一环境**
- 第1轮测试：所有模型和速度都使用 `environment_1`
- 第2轮测试：所有模型和速度都使用 `environment_2`
- 第3轮测试：所有模型和速度都使用 `environment_3`
- 以此类推...

#### 2. **公平比较保证**
- ✅ **同一速度下不同模型使用相同环境** - 消除环境偏差
- ✅ **同一模型在不同速度下使用相同基础环境** - 确保速度对比一致性
- ✅ **只有多轮测试时才改变环境** - 增加测试多样性和鲁棒性

#### 3. **测试执行顺序**
```
第1轮 (environment_1):
  ├── 速度3.0m/s: ViT → ViTLSTM → ConvNet → RobustViTLSTM → LSTMNet
  ├── 速度4.0m/s: ViT → ViTLSTM → ConvNet → RobustViTLSTM → LSTMNet
  ├── 速度5.0m/s: ViT → ViTLSTM → ConvNet → RobustViTLSTM → LSTMNet
  └── ...

第2轮 (environment_2):
  ├── 速度3.0m/s: ViT → ViTLSTM → ConvNet → RobustViTLSTM → LSTMNet
  └── ...
```

#### 4. **实验优势**
- 🔬 **科学严谨**：符合对照实验设计原则
- 📊 **统计有效**：支持方差分析(ANOVA)等统计测试
- 🎯 **结果可信**：模型性能差异真正反映模型能力
- 🔄 **鲁棒性验证**：多环境测试增强结果可靠性

## 📊 测试配置

### 全面评估配置
- **模型数量**: 5个
  - ViT, ViTLSTM, ConvNet, RobustViTLSTM, LSTMNet
- **测试速度**: 5个 (3.0, 4.0, 5.0, 6.0, 7.0 m/s)
- **总配置**: 25个 (5模型 × 5速度)
- **环境分配**: 按轮次统一分配 (environment_1, environment_2, ...)
- **预计时间**: 每轮次约2-5分钟，总时间取决于轮次数

### 快速测试配置
- **模型数量**: 3个 (ViTLSTM, ConvNet, RobustViTLSTM)
- **测试速度**: 3个 (4.0, 5.0, 6.0 m/s)
- **总配置**: 9个 (3模型 × 3速度)

## 📁 输出结果结构

```
evaluation_results/
├── ViT_vel3.0/
│   ├── evaluation_ViT_vel3.0.yaml          # 包含所有轮次的评估结果
│   └── used_environment.txt                # 记录使用的环境信息
├── ViT_vel4.0/
│   ├── evaluation_ViT_vel4.0.yaml
│   └── used_environment.txt
├── ...
├── environment_mapping_YYYYMMDD_HHMMSS.txt # 环境分配记录
├── evaluation_summary_YYYYMMDD_HHMMSS.txt  # 评估汇总报告
└── analysis_results/                       # 自动生成的分析结果
    ├── success_rates_comparison.png
    ├── collision_rates_comparison.png
    ├── performance_radar_chart.png
    └── ...
```

### 环境映射文件示例
```
环境文件夹映射:
=================
ViT_vel3.0_round1 -> environment_1 (第1轮统一环境)
ViTLSTM_vel3.0_round1 -> environment_1 (第1轮统一环境)
ConvNet_vel3.0_round1 -> environment_1 (第1轮统一环境)
...
ViT_vel3.0_round2 -> environment_2 (第2轮统一环境)
ViTLSTM_vel3.0_round2 -> environment_2 (第2轮统一环境)
```

## 🛠️ 参数说明

### launch.bash 参数
```bash
bash launch.bash [轮次数] [模式] [输出目录] [人工模式]
```

- `轮次数`: 每个配置的测试轮次 (默认: 5)
- `模式`: "vision" 或 "state" (默认: vision)
- `输出目录`: 结果保存目录 (默认: ./evaluation_results)
- `人工模式`: "human" (仅在state模式下使用)

### 环境分配逻辑
- **轮次驱动**: 环境编号 = 当前轮次编号
- **统一性保证**: 同一轮次内所有配置使用相同环境
- **多样性增强**: 不同轮次使用不同环境

## ⏱️ 时间估算

### 全面评估
- **配置数**: 25个 (5模型 × 5速度)
- **每轮次时间**: 2-5分钟
- **总时间估算**:
  - 轮次=1: ~1-2小时
  - 轮次=3: ~3-6小时
  - 轮次=5: ~5-10小时

### 快速测试
- **配置数**: 9个
- **总时间估算**:
  - 轮次=1: ~30-45分钟
  - 轮次=3: ~1.5-3小时

## 🔍 监控和调试

### 实时监控
```bash
# 查看当前进度
tail -f evaluation_results/evaluation_summary_*.txt

# 查看环境分配
cat evaluation_results/environment_mapping_*.txt

# 查看模拟器进程
ps aux | grep visionsim_node

# 查看ROS节点
rosnode list

# 监控配置进度
grep "配置.*完成" evaluation_results/evaluation_summary_*.txt
```

### 测试进度跟踪
脚本会显示详细的进度信息：
```
========================================
配置 15/125
轮次: 3/5
模型: ViTLSTM
速度: 5.0 m/s
环境编号: 3 (第3轮统一环境)
========================================
```

### 故障排除
1. **模拟器卡住**: 脚本会自动重启
2. **超时处理**: 每轮次最多5分钟
3. **进程清理**: 自动清理残留进程
4. **环境验证**: 检查环境文件夹是否存在

## 📈 结果分析

### 自动分析
脚本完成后会自动运行分析，生成：
- 成功率对比图
- 碰撞率对比图
- 能耗分析图
- 综合性能雷达图
- 性能热力图
- 统计表格
- 环境分配公平性报告

### 手动分析
```bash
cd analysis
python3 comprehensive_analysis.py ../evaluation_results
```

### 环境分配验证
```bash
# 验证环境分配是否公平
python3 validate_environment_assignment.py evaluation_results \
    --mapping_file evaluation_results/environment_mapping_*.txt
```

## 🎯 使用建议

### 首次使用
```bash
# 1. 环境检查
source setup_ros.bash

# 2. 快速验证（推荐）
bash quick_evaluation.bash 1 vision

# 3. 检查环境分配
cat evaluation_results/environment_mapping_*.txt

# 4. 验证结果格式
ls evaluation_results/*/evaluation_*.yaml
```

### 生产环境
```bash
# 1. 使用screen后台运行
screen -S evaluation
bash launch.bash 3 vision ./full_evaluation

# 2. 分离screen (Ctrl+A+D)

# 3. 检查进度
screen -r evaluation

# 4. 分析结果
cd analysis
python3 comprehensive_analysis.py ../full_evaluation
```

### 验证实验质量
```bash
# 检查环境分配公平性
grep "统一环境" evaluation_results/environment_mapping_*.txt

# 统计测试完成情况
find evaluation_results -name "evaluation_*.yaml" | wc -l

# 检查每个配置的轮次数
for dir in evaluation_results/*/; do
    echo "$(basename $dir): $(grep -c '^-' $dir/evaluation_*.yaml) 轮次"
done
```

## ⚠️ 注意事项

### 环境配置
1. **环境文件夹**: 确保 environment_1 到 environment_N 存在
2. **权限检查**: 确保config.yaml文件可写
3. **备份恢复**: 脚本会自动备份和恢复config.yaml

### 实验设计
1. **轮次选择**: 建议至少3轮次以获得统计意义
2. **环境数量**: 确保有足够的环境文件夹支持轮次数
3. **结果验证**: 检查environment_mapping文件确认分配正确

### 系统资源
1. **系统资源**: 确保有足够的CPU和内存
2. **磁盘空间**: 每个配置约10-50MB数据
3. **网络稳定**: ROS通信需要稳定网络
4. **模型文件**: 确保所有模型文件存在且可访问

## 🔧 自定义配置

### 修改测试模型
编辑`launch.bash`中的MODELS数组：
```bash
declare -a MODELS=(
    "YourModel:../../models/your_model.pth"
    # 添加更多模型...
)
```

### 修改测试速度
编辑VELOCITIES数组：
```bash
declare -a VELOCITIES=(2.0 3.0 4.0 5.0 6.0 7.0 8.0)
```

### 修改环境分配策略
当前策略在脚本中的关键部分：
```bash
# 环境分配策略：基础环境编号 = 当前轮次
env_number=$round
```

### 修改超时设置
```bash
if ((($(date +%s) - start_time) >= 300)); then  # 300秒 = 5分钟
```

## 📊 实验结果解读

### 环境分配验证
正确的环境分配应该显示：
- 同一轮次内所有配置使用相同环境编号
- 不同轮次使用不同环境编号
- 环境映射文件记录详细的分配信息

### 统计分析建议
1. **模型比较**: 使用同一轮次的结果进行公平比较
2. **速度分析**: 观察模型在不同速度下的性能变化
3. **鲁棒性评估**: 分析模型在不同环境(轮次)下的稳定性
4. **方差分析**: 利用多轮次数据进行统计显著性检验

## 📞 支持与故障排除

### 常见问题
1. **环境文件夹不存在**: 检查 flightmare/flightpy/configs/vision/ 下的环境文件夹
2. **config.yaml权限**: 确保文件可写，检查备份是否创建
3. **ROS通信问题**: 重启roscore或检查网络配置
4. **模型加载失败**: 验证模型文件路径和格式

### 调试技巧
```bash
# 检查当前使用的环境
grep "env_folder" flightmare/flightpy/configs/vision/config.yaml

# 查看最近的评估结果
ls -lt evaluation_results/*/evaluation_*.yaml | head -5

# 监控ROS话题
rostopic list | grep -E "(start|reset|enable)"
```

## 版本信息
- **版本**: 3.0
- **更新日期**: 2025年5月26日
- **主要特性**: 科学环境分配策略，公平模型比较
- **支持模型**: 5个深度学习模型
- **支持速度**: 3-7 m/s可配置
- **环境策略**: 轮次统一分配，确保实验公平性