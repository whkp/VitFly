#!/bin/bash
# 快速测试脚本 - 仅测试部分模型和速度
# 使用方法: bash quick_evaluation.bash [rollouts] [vision|state]

# 默认参数
N=${1:-3}  # 默认每配置3次测试
MODE=${2:-vision}  # 默认vision模式

# 快速测试配置 - 仅测试主要模型
declare -a QUICK_MODELS=(
    "ViTLSTM:../../models/ViTLSTM_model.pth"
    "ConvNet:../../models/ConvNet_model.pth"
    "RobustViTLSTM:../../models/RobustViTLSTM.pth"
)

# 快速测试速度 - 仅测试3个速度
declare -a QUICK_VELOCITIES=(4.0 5.0 6.0)

OUTPUT_DIR="./quick_evaluation_$(date '+%Y%m%d_%H%M')"

echo "=========================================="
echo "快速评估模式"
echo "模型数量: ${#QUICK_MODELS[@]}"
echo "速度配置: ${#QUICK_VELOCITIES[@]} (${QUICK_VELOCITIES[*]})"
echo "每配置轮次: $N"
echo "总配置数: $((${#QUICK_MODELS[@]} * ${#QUICK_VELOCITIES[@]}))"
echo "预计总轮次: $((${#QUICK_MODELS[@]} * ${#QUICK_VELOCITIES[@]} * N))"
echo "=========================================="

# 设置模式参数
if [ "$MODE" = "vision" ]; then
    run_competition_args="--vision_based"
    realtimefactor=""
else
    run_competition_args="--state_based"
    realtimefactor="real_time_factor:=10.0"
fi

# 设置Flightmare路径
export FLIGHTMARE_PATH=$PWD/flightmare

mkdir -p "$OUTPUT_DIR"

config_count=0
total_configs=$((${#QUICK_MODELS[@]} * ${#QUICK_VELOCITIES[@]}))

for model_config in "${QUICK_MODELS[@]}"; do
    IFS=':' read -r model_type model_path <<< "$model_config"
    
    for velocity in "${QUICK_VELOCITIES[@]}"; do
        config_count=$((config_count + 1))
        
        echo ""
        echo "配置 $config_count/$total_configs: $model_type @ ${velocity}m/s"
        
        config_dir="$OUTPUT_DIR/${model_type}_vel${velocity}"
        mkdir -p "$config_dir"
        
        # 启动模拟器
        if [ -z $(pgrep visionsim_node) ]; then
            echo "启动模拟器..."
            roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
            ROS_PID="$!"
            sleep 10
        fi
        
        SUMMARY_FILE="$config_dir/evaluation.yaml"
        echo "" > "$SUMMARY_FILE"
        
        # 运行测试
        for i in $(seq 1 $N); do
            echo "  轮次 $i/$N"
            
            # 重置模拟器
            rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
            rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
            rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
            rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once
            
            export ROLLOUT_NAME="${model_type}_vel${velocity}_quick_${i}"
            
            cd ./envtest/ros/
            python3 evaluation_node.py --model_type "$model_type" &
            PY_PID="$!"
            python3 run_competition.py $run_competition_args --desVel "$velocity" --model_type "$model_type" --model_path "$model_path" &
            COMP_PID="$!"
            cd -
            
            sleep 2
            start_time=$(date +%s)
            
            # 等待完成（最多2分钟）
            while ps -p $PY_PID > /dev/null; do
                rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
                sleep 2
                
                if ((($(date +%s) - start_time) >= 120)); then
                    echo "    超时，终止轮次"
                    kill -SIGINT $PY_PID 2>/dev/null
                    break
                fi
            done
            
            # 合并结果
            if [ -f "./envtest/ros/summary.yaml" ]; then
                cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml" && mv "tmp.yaml" "$SUMMARY_FILE"
            fi
            
            kill -SIGINT "$COMP_PID" 2>/dev/null
        done
        
        echo "  配置完成: $config_dir"
    done
done

# 清理
if [ $ROS_PID ]; then
    kill -SIGINT "$ROS_PID"
fi

echo ""
echo "快速评估完成！结果保存在: $OUTPUT_DIR"