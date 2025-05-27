#!/bin/bash
# bash launch_evaluation.bash [rollouts_per_config] [vision|state] [output_dir]
# 全面测试所有模型在不同速度下的性能
# 使用方法: bash launch_evaluation.bash 5 vision ./evaluation_results

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  N=5
fi

# 定义所有模型配置
declare -a MODELS=(
    #  "ViT:../../models/ViT_model.pth"
    #  "ViTLSTM:../../models/ViTLSTM_model.pth"
    #  "ConvNet:../../models/ConvNet_model.pth"
    #  "RobustViTLSTM:../../models/RobustViTLSTM.pth"
     #"LSTMNet:../../models/LSTMnet_model.pth"
)

# 定义测试速度
declare -a VELOCITIES=(3.0  5.0  7.0)

# 输出目录设置
if [ $3 ]
then
  OUTPUT_BASE_DIR="$3"
else
  OUTPUT_BASE_DIR="./evaluation_results"
fi

# 定义config.yaml的路径
CONFIG_FILE="./flightmare/flightpy/configs/vision/config.yaml"

# 备份原始config.yaml文件
if [ ! -f "${CONFIG_FILE}.backup" ]; then
    echo "备份原始config.yaml文件..."
    cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
fi

# 函数：修改config.yaml中的env_folder
update_env_folder() {
    local env_number=$1
    local env_folder="environment_${env_number}"
    
    echo "更新config.yaml中的env_folder为: $env_folder"
    
    # 使用sed替换env_folder行
    sed -i "s/env_folder: \"environment_[0-9]\+\"/env_folder: \"$env_folder\"/" "$CONFIG_FILE"
    
    # 验证修改是否成功
    if grep -q "env_folder: \"$env_folder\"" "$CONFIG_FILE"; then
        echo "✓ env_folder已成功更新为: $env_folder"
    else
        echo "✗ env_folder更新失败"
        return 1
    fi
}

# 函数：恢复原始config.yaml文件
restore_config() {
    echo "恢复原始config.yaml文件..."
    cp "${CONFIG_FILE}.backup" "$CONFIG_FILE"
    echo "✓ config.yaml已恢复"
}

# 设置退出时恢复配置文件的陷阱
trap restore_config EXIT

echo $2

if [ "$2" = "vision" ]
then
  echo
  echo "[LAUNCH SCRIPT] Vision based!"
  echo
  run_competition_args="--vision_based"
  realtimefactor=""
elif [ "$2" = "state" ]
then
  echo
  echo "[LAUNCH SCRIPT] State based!"
  echo
  run_competition_args="--state_based"
  if [ "$4" = "human" ]
  then
    run_competition_args="--keyboard"
    realtimefactor="real_time_factor:=1.0"
  else
    run_competition_args=""
    realtimefactor="real_time_factor:=10.0"
  fi
else
  echo
  echo "[LAUNCH SCRIPT] Unknown or empty second argument: $2, only 'vision' or 'state' allowed!"
  echo
  exit 1
fi

# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi
# set FLIGHTMARE_PATH force
export FLIGHTMARE_PATH=$PWD/flightmare
echo $FLIGHTMARE_PATH

# 创建主输出目录
mkdir -p "$OUTPUT_BASE_DIR"

# 生成时间戳
datetime=$(date '+%Y%m%d_%H%M%S')
echo "开始全面评估 - 时间戳: $datetime"

# 计算总配置数
total_configs=$((${#MODELS[@]} * ${#VELOCITIES[@]}))
current_config=0

echo "=========================================="
echo "开始全面模型评估"
echo "模型数量: ${#MODELS[@]}"
echo "速度配置: ${#VELOCITIES[@]} (${VELOCITIES[*]})"
echo "每配置轮次: $N"
echo "总配置数: $total_configs"
echo "预计总轮次: $((total_configs * N))"
echo "输出目录: $OUTPUT_BASE_DIR"
echo "=========================================="

# 创建环境映射文件
ENV_MAPPING_FILE="$OUTPUT_BASE_DIR/environment_mapping_${datetime}.txt"
echo "环境文件夹映射:" > "$ENV_MAPPING_FILE"
echo "=================" >> "$ENV_MAPPING_FILE"

# 主循环：先按轮次，再按速度，最后按模型
# 这样确保：1) 同一速度下所有模型使用相同环境 2) 同一模型在不同速度下使用相同基础环境 3) 只有多轮测试时才改变环境
for round in $(eval echo {1..$N}); do
    echo ""
    echo "=========================================="
    echo "开始第 $round/$N 轮测试"
    echo "本轮将为每个速度选择环境 environment_$round"
    echo "=========================================="
    
    for velocity_idx in "${!VELOCITIES[@]}"; do
        velocity="${VELOCITIES[$velocity_idx]}"
        
        # 环境分配策略：
        # - 基础环境编号 = 当前轮次 (1, 2, 3, 4, 5...)
        # - 这确保：同一轮次内，所有速度和模型使用相同的基础环境
        # - 不同轮次使用不同环境，增加测试多样性
        env_number=$round
        
        echo ""
        echo "测试速度: ${velocity}m/s (环境: environment_${env_number})"
        
        for model_config in "${MODELS[@]}"; do
            # 解析模型配置
            IFS=':' read -r model_type model_path <<< "$model_config"
            
            current_config=$((current_config + 1))
            
            echo ""
            echo "=========================================="
            echo "配置 $current_config/$((total_configs * N))"
            echo "轮次: $round/$N"
            echo "模型: $model_type"
            echo "速度: $velocity m/s"
            echo "环境编号: $env_number (第${round}轮统一环境)"
            echo "=========================================="
            
            # 记录环境映射
            echo "${model_type}_vel${velocity}_round${round} -> environment_${env_number} (第${round}轮统一环境)" >> "$ENV_MAPPING_FILE"
            
            # 更新config.yaml中的env_folder
            if ! update_env_folder "$env_number"; then
                echo "错误：无法更新config.yaml，跳过当前配置"
                continue
            fi
            
            # 创建特定配置的输出目录
            config_dir="$OUTPUT_BASE_DIR/${model_type}_vel${velocity}"
            mkdir -p "$config_dir"
            
            # 在配置目录中记录使用的环境信息
            if [ $round -eq 1 ]; then
                echo "environment_${env_number}" > "$config_dir/used_environment.txt"
                echo "round_${round}" >> "$config_dir/used_environment.txt"
            else
                echo "environment_${env_number}" >> "$config_dir/used_environment.txt"
                echo "round_${round}" >> "$config_dir/used_environment.txt"
            fi
            
            # 启动模拟器
            if [ -z $(pgrep visionsim_node) ]
            then
              echo "启动模拟器..."
              roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
              ROS_PID="$!"
              echo "模拟器PID: $ROS_PID"
              sleep 10
            else
              ROS_PID=""
            fi

            # 为当前配置创建汇总文件
            SUMMARY_FILE="$config_dir/evaluation_${model_type}_vel${velocity}.yaml"
            if [ $round -eq 1 ]; then
                echo "" > "$SUMMARY_FILE"
            fi
            
            relaunch_sim=0

            echo ""
            echo "运行 第${round}轮 - $model_type @ ${velocity}m/s (环境: environment_${env_number})"
            
            # Reset the simulator if needed
            if ((relaunch_sim)); then
                echo "重启模拟器..."
                relaunch_sim=0
                killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare.
                sleep 10

                if [ -z $(pgrep visionsim_node) ]; then
                  roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
                  ROS_PID="$!"
                  echo "重启模拟器PID: $ROS_PID"
                  sleep 10
                else
                  killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare.
                  sleep 10
                fi
            fi

            start_time=$(date +%s)

            # 发布模拟器重置命令
            rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
            rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
            rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
            rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

            # 设置轮次名称
            export ROLLOUT_NAME="${model_type}_vel${velocity}_env${env_number}_round_${round}"
            echo "轮次名称: $ROLLOUT_NAME"

            cd ./envtest/ros/
            
            # 启动评估节点
            python3 evaluation_node.py --model_type "$model_type" &
            PY_PID="$!"
            
            # 启动竞赛脚本
            python3 run_competition.py $run_competition_args --desVel "$velocity" --model_type "$model_type" --model_path "$model_path" &
            COMP_PID="$!"

            cd -
            sleep 2

            # 等待评估脚本完成
            while ps -p $PY_PID > /dev/null; do
                echo "[$(date '+%H:%M:%S')] 发送导航开始命令..."
                rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
                sleep 2

                # 检查是否超时（5分钟）
                if ((($(date +%s) - start_time) >= 300)); then
                    echo "超时！终止当前轮次..."
                    kill -SIGINT $PY_PID
                    relaunch_sim=1
                    break
                fi
            done

            # 合并汇总文件
            if [ -f "./envtest/ros/summary.yaml" ]; then
                cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
                mv "tmp.yaml" "$SUMMARY_FILE"
            fi

            # 清理进程
            kill -SIGINT "$COMP_PID" 2>/dev/null
            
            echo "第${round}轮 ${model_type} @ ${velocity}m/s 完成"
        done
        
        echo "速度 ${velocity}m/s 在第${round}轮中的所有模型测试完成"
    done
    
    echo "第 $round/$N 轮测试完成"
    echo ""
done

# 清理模拟器
if [ $ROS_PID ]; then
  echo "关闭模拟器..."
  kill -SIGINT "$ROS_PID"
fi

echo ""
echo "=========================================="
echo "全面评估完成！"
echo "总配置数: $total_configs"
echo "总轮次数: $((total_configs * N))"
echo "结果目录: $OUTPUT_BASE_DIR"
echo "环境映射文件: $ENV_MAPPING_FILE"
echo "完成时间: $(date)"
echo "=========================================="

# 生成评估汇总报告
echo "生成评估汇总报告..."
REPORT_FILE="$OUTPUT_BASE_DIR/evaluation_summary_${datetime}.txt"

{
    echo "无人机避障模型全面评估报告"
    echo "=============================="
    echo "评估时间: $(date)"
    echo "总配置数: $total_configs"
    echo "每配置轮次: $N"
    echo "总测试轮次: $((total_configs * N))"
    echo ""
    echo "测试模型:"
    for model_config in "${MODELS[@]}"; do
        IFS=':' read -r model_type model_path <<< "$model_config"
        echo "  - $model_type ($model_path)"
    done
    echo ""
    echo "测试速度: ${VELOCITIES[*]} m/s"
    echo ""
    echo "环境分配策略:"
    echo "  - 同一轮次内所有模型和速度使用相同环境 (environment_轮次号)"
    echo "  - 第1轮: environment_1, 第2轮: environment_2, 第3轮: environment_3, ..."
    echo "  - 确保了模型间和速度间的公平比较"
    echo "  - 多轮测试提供环境多样性，增强结果可靠性"
    echo "  - 详细映射请查看: $(basename "$ENV_MAPPING_FILE")"
    echo ""
    echo "结果目录结构:"
    for model_config in "${MODELS[@]}"; do
        IFS=':' read -r model_type model_path <<< "$model_config"
        echo "  模型 ${model_type}:"
        for velocity in "${VELOCITIES[@]}"; do
            config_dir="${model_type}_vel${velocity}"
            echo "    $config_dir/ (包含${N}轮测试结果)"
        done
        echo ""
    done
    echo ""
    echo "环境使用模式:"
    for round in $(eval echo {1..$N}); do
        echo "  第${round}轮测试: 所有模型和速度都使用 environment_${round}"
    done
} > "$REPORT_FILE"

echo "汇总报告保存至: $REPORT_FILE"

echo "全面评估流程完成！"
