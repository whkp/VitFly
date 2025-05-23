"""
可解释性四旋翼避障模型 - 完整架构设计

这个设计基于认知科学原理，将避障任务分解为四个可解释的认知层次：
1. 多模态感知层：不仅处理视觉，还融合运动状态
2. 环境理解层：构建3D空间的语义地图
3. 策略推理层：基于目标和约束进行决策推理  
4. 执行控制层：将抽象策略转化为具体控制指令

每一层都设计有可视化接口，使得整个决策过程透明可见
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from ViTsubmodules import *

class TemporalFeatureExtractor(nn.Module):
    """
    时序特征提取器：处理连续帧之间的时间关系
    
    对于你的数据集格式，这个模块特别重要，因为每个轨迹文件夹包含
    多个时间戳的图像，我们需要建模时间上的运动模式
    """
    def __init__(self, feature_dim=64, sequence_length=5):
        super().__init__()
        self.sequence_length = sequence_length
        
        # 单帧特征提取器（基于你现有的ViT结构）
        self.frame_encoder = MixTransformerEncoderLayer(
            1, feature_dim, patch_size=7, stride=4, padding=3,
            n_layers=2, reduction_ratio=8, num_heads=4, expansion_factor=8
        )
        
        # 时序建模：使用Transformer来捕获帧间关系
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim * 15 * 23,  # 特征图展平后的维度
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=3
        )
        
        # 运动估计器：估计当前的运动状态
        self.motion_estimator = nn.Sequential(
            nn.Linear(feature_dim * 15 * 23, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 3D速度 + 3D角速度
        )
        
    def forward(self, image_sequence):
        """
        处理图像序列，输出时序特征和运动估计
        
        Args:
            image_sequence: (B, T, 1, H, W) 其中T是序列长度
        """
        batch_size, seq_len, channels, height, width = image_sequence.shape
        
        # 逐帧提取特征
        frame_features = []
        for t in range(seq_len):
            frame_feat = self.frame_encoder(image_sequence[:, t])
            frame_features.append(frame_feat.flatten(1))
        
        # 堆叠为时序特征
        temporal_features = torch.stack(frame_features, dim=1)  # (B, T, D)
        
        # Transformer编码时序关系
        # 注意：Transformer期望输入为(T, B, D)
        temporal_features = temporal_features.transpose(0, 1)
        encoded_sequence = self.temporal_encoder(temporal_features)
        encoded_sequence = encoded_sequence.transpose(0, 1)  # 转回(B, T, D)
        
        # 使用最新帧进行运动估计
        current_motion = self.motion_estimator(encoded_sequence[:, -1])
        
        return {
            'temporal_features': encoded_sequence,
            'current_frame_features': frame_features[-1],
            'estimated_motion': current_motion
        }

class MultiModalPerceptionModule(nn.Module):
    """
    多模态感知模块：融合视觉、运动状态、目标方向等多种信息
    
    这个模块解决了单纯视觉感知的局限性，通过融合多种感官信息
    来构建更完整的环境感知
    """
    def __init__(self, visual_feature_dim=64):
        super().__init__()
        
        # 视觉感知分支：专注于障碍物检测
        self.visual_processor = nn.Sequential(
            nn.Conv2d(visual_feature_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 运动状态编码器：处理IMU等运动信息
        self.motion_encoder = nn.Sequential(
            nn.Linear(6, 32),  # 6D运动状态
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # 目标导向编码器：处理目标方向信息
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, 16),  # 3D目标方向
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        # 多模态融合网络：将不同模态信息融合
        self.multimodal_fusion = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 视觉特征
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 添加位置编码来整合空间信息
        self.spatial_encoder = nn.Parameter(torch.randn(1, 32, 15, 23))
        
        # 感知输出头：生成可解释的感知结果
        self.obstacle_detector = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),  # +16 for spatial context
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        self.depth_estimator = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 语义分割头：识别不同类型的障碍物
        # 针对你的球形障碍物数据集，可以扩展到识别不同大小/材质的球体
        self.semantic_segmentation = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 1),  # 4类：背景、小球、中球、大球
            nn.Softmax(dim=1)
        )
        
    def forward(self, temporal_output, motion_state, goal_direction):
        """
        多模态感知处理
        
        Args:
            temporal_output: 时序特征提取器的输出
            motion_state: 当前运动状态 (B, 6)
            goal_direction: 目标方向 (B, 3)
        """
        visual_features = temporal_output['current_frame_features']
        estimated_motion = temporal_output['estimated_motion']
        
        # 重塑视觉特征到空间维度
        batch_size = visual_features.shape[0]
        visual_spatial = visual_features.view(batch_size, -1, 15, 23)
        
        # 处理视觉信息
        processed_visual = self.visual_processor(visual_spatial)
        
        # 编码运动状态（融合估计运动和真实运动）
        combined_motion = torch.cat([estimated_motion, motion_state], dim=1)
        motion_context = self.motion_encoder(combined_motion)
        
        # 编码目标方向
        goal_context = self.goal_encoder(goal_direction)
        
        # 空间上下文：将运动和目标信息广播到空间维度
        motion_spatial = motion_context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 15, 23)
        goal_spatial = goal_context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 15, 23)
        
        # 融合多模态信息
        fused_visual = self.multimodal_fusion(processed_visual)
        
        # 添加空间编码
        spatial_context = torch.cat([
            motion_spatial[:, :8, :, :],  # 取前8维运动特征
            goal_spatial[:, :8, :, :]     # 取前8维目标特征
        ], dim=1)
        
        contextual_features = torch.cat([fused_visual, spatial_context], dim=1)
        
        # 生成可解释的感知输出
        obstacle_mask = self.obstacle_detector(contextual_features)
        depth_estimation = self.depth_estimator(contextual_features)
        confidence_map = self.confidence_estimator(contextual_features)
        semantic_segmentation = self.semantic_segmentation(contextual_features)
        
        return {
            'obstacle_mask': obstacle_mask,
            'depth_estimation': depth_estimation,
            'confidence_map': confidence_map,
            'semantic_segmentation': semantic_segmentation,
            'fused_features': contextual_features,
            'motion_context': motion_context,
            'goal_context': goal_context
        }

class SpatialReasoningModule(nn.Module):
    """
    空间推理模块：基于感知结果进行3D空间推理
    
    这个模块模拟飞行员的空间认知能力，不仅理解当前状态，
    还能预测未来的空间变化和风险分布
    """
    def __init__(self, feature_dim=48):
        super().__init__()
        
        # 3D空间建模网络：将2D感知结果扩展到3D理解
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 可导航性分析器：评估每个区域的可通过性
        self.navigability_analyzer = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 动态风险评估器：考虑动态因素的风险预测
        self.risk_assessor = nn.Sequential(
            nn.Conv2d(32 + 8, 16, 3, padding=1),  # +8 for motion context
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 未来状态预测器：预测t+1时刻的障碍物分布
        self.future_predictor = nn.Sequential(
            nn.Conv2d(32 + 8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 方向偏好分析器：分析在3D空间中的最优移动方向
        self.direction_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 6)),
            nn.Flatten(),
            nn.Linear(4 * 6 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # 6个主要方向：+X,-X,+Y,-Y,+Z,-Z
            nn.Softmax(dim=1)
        )
        
        # 局部路径规划器：在局部区域内规划可行路径
        self.local_path_planner = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 2, 1),  # 输出局部目标点(x,y)
            nn.Tanh()  # 限制在[-1,1]范围内
        )
        
    def forward(self, perception_output):
        """
        空间推理处理
        """
        # 获取感知输出
        fused_features = perception_output['fused_features']
        motion_context = perception_output['motion_context']
        
        # 空间编码
        spatial_features = self.spatial_encoder(fused_features)
        
        # 可导航性分析
        navigability_map = self.navigability_analyzer(spatial_features)
        
        # 动态风险评估（融合运动上下文）
        motion_broadcast = motion_context.unsqueeze(-1).unsqueeze(-1).expand(-1, 8, 15, 23)
        risk_input = torch.cat([spatial_features, motion_broadcast], dim=1)
        risk_map = self.risk_assessor(risk_input)
        
        # 未来状态预测
        future_obstacles = self.future_predictor(risk_input)
        
        # 方向偏好分析
        direction_preferences = self.direction_analyzer(spatial_features)
        
        # 局部路径规划
        local_target = self.local_path_planner(spatial_features)
        
        return {
            'navigability_map': navigability_map,
            'risk_map': risk_map,
            'future_obstacles': future_obstacles,
            'direction_preferences': direction_preferences,
            'local_target': local_target,
            'spatial_features': spatial_features
        }

class StrategicPlanningModule(nn.Module):
    """
    策略规划模块：基于高级推理进行运动策略选择
    
    这个模块模拟飞行员的战术思维，不仅考虑当前状态，
    还考虑任务目标、安全约束、能耗优化等多个因素
    """
    def __init__(self, spatial_feature_dim=32):
        super().__init__()
        
        # 定义扩展的运动原语库（针对3D环境优化）
        self.motion_primitives = [
            "forward_cruise",      # 正常前进巡航
            "careful_forward",     # 谨慎前进（减速）
            "left_avoid",          # 左转避障
            "right_avoid",         # 右转避障
            "climb_avoid",         # 爬升避障
            "descend_avoid",       # 下降避障
            "hover_wait",          # 悬停等待
            "emergency_stop",      # 紧急停止
            "retreat_replan",      # 后退重新规划
            "spiral_navigate"      # 螺旋导航（复杂环境）
        ]
        
        # 任务优先级编码器：处理不同任务需求
        self.task_encoder = nn.Sequential(
            nn.Linear(7, 32),  # 任务状态：[到目标距离, 剩余时间, 能耗限制, 安全等级, 精确度要求, 速度偏好, 路径效率]
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # 约束分析器：处理各种约束条件
        self.constraint_analyzer = nn.Sequential(
            nn.Linear(5, 32),  # 约束：[最大速度, 最大加速度, 最小安全距离, 禁飞区, 能耗限制]
            nn.ReLU(),
            nn.Linear(32, 48)
        )
        
        # 情景理解器：整合所有上下文信息
        self.context_integrator = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 9)),  # 压缩空间特征
            nn.Flatten(),
            nn.Linear(6 * 9 * spatial_feature_dim + 64 + 48 + 6, 256),  # +64任务特征 +48约束特征 +6方向偏好
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 策略评估器：评估每个运动原语的适用性
        self.strategy_evaluator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.motion_primitives)),
            nn.Softmax(dim=1)
        )
        
        # 参数生成器：为选择的策略生成具体参数
        self.parameter_generator = nn.Sequential(
            nn.Linear(128 + len(self.motion_primitives), 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 8)  # 扩展参数：[强度, 方向角, 俯仰角, 滚转角, 持续时间, 加速度, 角速度, 优先级]
        )
        
        # 安全检查器：确保策略满足安全要求
        self.safety_checker = nn.Sequential(
            nn.Linear(8 + 1, 32),  # +1 for minimum obstacle distance
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 安全等级：[可接受, 需调整, 危险]
            nn.Softmax(dim=1)
        )
        
    def forward(self, spatial_reasoning_output, task_state, constraints, min_obstacle_distance):
        """
        策略规划处理
        
        Args:
            spatial_reasoning_output: 空间推理结果
            task_state: 任务状态信息 (B, 7)
            constraints: 约束条件 (B, 5)
            min_obstacle_distance: 最小障碍距离 (B, 1)
        """
        spatial_features = spatial_reasoning_output['spatial_features']
        direction_preferences = spatial_reasoning_output['direction_preferences']
        
        # 编码任务和约束
        task_context = self.task_encoder(task_state)
        constraint_context = self.constraint_analyzer(constraints)
        
        # 整合所有上下文信息
        spatial_compressed = F.adaptive_avg_pool2d(spatial_features, (6, 9)).flatten(1)
        context_input = torch.cat([
            spatial_compressed,
            task_context,
            constraint_context,
            direction_preferences
        ], dim=1)
        
        # 生成策略理解
        strategic_understanding = self.context_integrator(context_input)
        
        # 评估运动原语
        primitive_scores = self.strategy_evaluator(strategic_understanding)
        
        # 生成策略参数
        param_input = torch.cat([strategic_understanding, primitive_scores], dim=1)
        motion_parameters = self.parameter_generator(param_input)
        
        # 安全检查
        safety_input = torch.cat([motion_parameters, min_obstacle_distance], dim=1)
        safety_assessment = self.safety_checker(safety_input)
        
        return {
            'primitive_probabilities': primitive_scores,
            'motion_parameters': motion_parameters,
            'safety_assessment': safety_assessment,
            'strategic_understanding': strategic_understanding,
            'task_context': task_context,
            'constraint_context': constraint_context
        }

class AdaptiveControlModule(nn.Module):
    """
    自适应控制模块：将策略决策转化为精确的控制指令
    
    这个模块不仅执行控制，还能根据执行效果进行在线调整，
    模拟经验丰富飞行员的精细操控技能
    """
    def __init__(self):
        super().__init__()
        
        # 控制指令生成器：主要的控制逻辑
        self.control_generator = nn.Sequential(
            nn.Linear(8 + 10 + 3 + 1, 128),  # 运动参数 + 原语概率（前10个） + 安全评估 + 期望速度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # xyz速度分量
        )
        
        # 自适应调节器：根据历史表现调整控制增益
        self.adaptive_controller = nn.Sequential(
            nn.Linear(6 + 3, 32),  # 历史误差(3) + 当前状态(3)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 输出调节系数
            nn.Sigmoid()
        )
        
        # 物理约束执行器：确保输出满足物理限制
        self.physics_enforcer = nn.Sequential(
            nn.Linear(3 + 6, 16),  # 速度指令 + 当前状态
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Tanh()  # 限制输出范围
        )
        
        # 安全限制器：紧急情况下的安全保护
        self.safety_limiter = nn.Sequential(
            nn.Linear(3 + 1, 8),  # 速度指令 + 最小距离
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Sigmoid()  # 安全衰减系数
        )
        
        # 预测控制器：预测未来状态并提前调整
        self.predictive_controller = nn.Sequential(
            nn.Linear(3 + 3, 16),  # 当前速度 + 预测误差
            nn.ReLU(),
            nn.Linear(16, 3)  # 预测性补偿
        )
        
        # 物理参数（可根据实际无人机调整）
        self.max_velocity = nn.Parameter(torch.tensor(3.0), requires_grad=False)
        self.max_acceleration = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.safety_margin = nn.Parameter(torch.tensor(0.8), requires_grad=False)
        
    def forward(self, planning_output, current_state, desired_velocity, 
                min_obstacle_distance, prev_velocity=None, prediction_error=None):
        """
        自适应控制处理
        
        Args:
            planning_output: 策略规划结果
            current_state: 当前状态 (位置+速度) (B, 6)
            desired_velocity: 期望速度 (B, 1)
            min_obstacle_distance: 最小障碍距离 (B, 1)
            prev_velocity: 前一时刻速度 (B, 3)
            prediction_error: 预测误差 (B, 3)
        """
        primitive_probs = planning_output['primitive_probabilities']
        motion_params = planning_output['motion_parameters']
        safety_assessment = planning_output['safety_assessment']
        
        # 基础控制指令生成
        control_input = torch.cat([
            motion_params,
            primitive_probs[:, :10],  # 取前10个原语概率
            safety_assessment,
            desired_velocity
        ], dim=1)
        
        base_velocity = self.control_generator(control_input)
        
        # 自适应调节（如果有历史误差信息）
        if prev_velocity is not None:
            # 计算简单的历史误差
            velocity_error = base_velocity - prev_velocity
            adaptive_input = torch.cat([velocity_error, current_state[:, 3:6]], dim=1)  # 当前速度状态
            adaptive_gains = self.adaptive_controller(adaptive_input)
            base_velocity = base_velocity * adaptive_gains
        
        # 物理约束执行
        physics_input = torch.cat([base_velocity, current_state], dim=1)
        constrained_velocity = self.physics_enforcer(physics_input)
        
        # 应用物理限制
        velocity_magnitude = torch.norm(constrained_velocity, dim=1, keepdim=True)
        constrained_velocity = torch.where(
            velocity_magnitude > self.max_velocity,
            constrained_velocity * self.max_velocity / velocity_magnitude,
            constrained_velocity
        )
        
        # 安全限制
        safety_input = torch.cat([constrained_velocity, min_obstacle_distance], dim=1)
        safety_factors = self.safety_limiter(safety_input)
        safe_velocity = constrained_velocity * safety_factors
        
        # 预测性控制（如果有预测误差）
        if prediction_error is not None:
            predictive_input = torch.cat([safe_velocity, prediction_error], dim=1)
            predictive_compensation = self.predictive_controller(predictive_input)
            final_velocity = safe_velocity + 0.1 * predictive_compensation  # 小权重的预测补偿
        else:
            final_velocity = safe_velocity
        
        # 最终安全检查：确保不会撞向障碍物
        obstacle_safety_factor = torch.clamp(min_obstacle_distance / self.safety_margin, 0.1, 1.0)
        final_velocity = final_velocity * obstacle_safety_factor.unsqueeze(1)
        
        return {
            'velocity_command': final_velocity,
            'base_velocity': base_velocity,
            'safety_factors': safety_factors,
            'adaptive_gains': adaptive_gains if prev_velocity is not None else None
        }

class CompleteInterpretableModel(nn.Module):
    """
    完整的可解释避障模型：整合所有模块
    
    这个模型提供完整的决策透明度，每个步骤都可以被理解和可视化
    """
    def __init__(self, sequence_length=5):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # 初始化所有模块
        self.temporal_extractor = TemporalFeatureExtractor(sequence_length=sequence_length)
        self.perception_module = MultiModalPerceptionModule()
        self.spatial_reasoning = SpatialReasoningModule()
        self.strategic_planning = StrategicPlanningModule()
        self.adaptive_control = AdaptiveControlModule()
        
        # 存储中间结果用于解释和可视化
        self.decision_trace = {}
        
    def forward(self, inputs, return_trace=False):
        """
        完整的前向传播，包含所有可解释的决策步骤
        
        Args:
            inputs: 包含所有必要输入的字典
            return_trace: 是否返回完整的决策跟踪
        """
        # 解包输入
        image_sequence = inputs['image_sequence']  # (B, T, 1, H, W)
        motion_state = inputs['motion_state']      # (B, 6)
        goal_direction = inputs['goal_direction']   # (B, 3)
        task_state = inputs['task_state']          # (B, 7)
        constraints = inputs['constraints']         # (B, 5)
        desired_velocity = inputs['desired_velocity'] # (B, 1)
        current_state = inputs['current_state']    # (B, 6)
        
        # 可选输入
        prev_velocity = inputs.get('prev_velocity', None)
        prediction_error = inputs.get('prediction_error', None)
        
        # 第一步：时序特征提取 - "我看到了什么时间模式？"
        temporal_output = self.temporal_extractor(image_sequence)
        
        # 第二步：多模态感知 - "我理解了什么？"
        perception_output = self.perception_module(
            temporal_output, motion_state, goal_direction
        )
        
        # 第三步：空间推理 - "环境的空间结构如何？"
        spatial_output = self.spatial_reasoning(perception_output)
        
        # 计算最小障碍距离用于后续决策
        min_obstacle_distance = torch.min(
            perception_output['depth_estimation'].view(image_sequence.shape[0], -1),
            dim=1
        )[0]
        
        # 第四步：策略规划 - "我应该采取什么策略？"
        planning_output = self.strategic_planning(
            spatial_output, task_state, constraints, min_obstacle_distance.unsqueeze(1)
        )
        
        # 第五步：自适应控制 - "具体怎么执行？"
        control_output = self.adaptive_control(
            planning_output, current_state, desired_velocity,
            min_obstacle_distance.unsqueeze(1), prev_velocity, prediction_error
        )
        
        # 存储决策跟踪
        if return_trace:
            self.decision_trace = {
                'temporal_analysis': temporal_output,
                'perception_analysis': perception_output,
                'spatial_analysis': spatial_output,
                'strategic_planning': planning_output,
                'control_execution': control_output,
                'min_obstacle_distance': min_obstacle_distance
            }
            
        return control_output['velocity_command'], self.decision_trace if return_trace else None
    
    def explain_decision(self, inputs):
        """
        生成详细的决策解释
        """
        velocity_cmd, trace = self.forward(inputs, return_trace=True)
        
        if trace is None:
            return "决策跟踪不可用"
        
        # 分析各个模块的输出
        perception = trace['perception_analysis']
        spatial = trace['spatial_analysis']
        planning = trace['strategic_planning']
        control = trace['control_execution']
        
        # 生成解释文本
        explanation = f"""
        【可解释避障决策分析】
        
        === 时序感知分析 ===
        - 运动状态估计: 速度({trace['temporal_analysis']['estimated_motion'][0, :3]:.2f})
        - 角速度估计: ({trace['temporal_analysis']['estimated_motion'][0, 3:6]:.2f})
        
        === 多模态感知结果 ===
        - 障碍物覆盖率: {torch.mean(perception['obstacle_mask']).item():.1%}
        - 平均深度: {torch.mean(perception['depth_estimation']).item():.2f}
        - 感知置信度: {torch.mean(perception['confidence_map']).item():.1%}
        - 语义分割: 背景({torch.mean(perception['semantic_segmentation'][:, 0]).item():.1%}) 
                    小球({torch.mean(perception['semantic_segmentation'][:, 1]).item():.1%})
                    中球({torch.mean(perception['semantic_segmentation'][:, 2]).item():.1%})
                    大球({torch.mean(perception['semantic_segmentation'][:, 3]).item():.1%})
        
        === 空间推理结果 ===
        - 可导航空间: {torch.mean(spatial['navigability_map']).item():.1%}
        - 风险等级: {torch.mean(spatial['risk_map']).item():.2f}/1.0
        - 未来障碍预测: {torch.mean(spatial['future_obstacles']).item():.2f}
        - 方向偏好: +X({spatial['direction_preferences'][0, 0]:.2f}) 
                    -X({spatial['direction_preferences'][0, 1]:.2f})
                    +Y({spatial['direction_preferences'][0, 2]:.2f}) 
                    -Y({spatial['direction_preferences'][0, 3]:.2f})
                    +Z({spatial['direction_preferences'][0, 4]:.2f}) 
                    -Z({spatial['direction_preferences'][0, 5]:.2f})
        
        === 策略规划决策 ===
        - 选择策略: {self.strategic_planning.motion_primitives[torch.argmax(planning['primitive_probabilities'][0]).item()]}
        - 策略置信度: {torch.max(planning['primitive_probabilities'][0]).item():.1%}
        - 安全评估: 可接受({planning['safety_assessment'][0, 0]:.2f}) 
                    需调整({planning['safety_assessment'][0, 1]:.2f}) 
                    危险({planning['safety_assessment'][0, 2]:.2f})
        
        === 控制执行结果 ===
        - 最终速度指令: X({velocity_cmd[0, 0]:.2f}) Y({velocity_cmd[0, 1]:.2f}) Z({velocity_cmd[0, 2]:.2f}) m/s
        - 安全衰减系数: {control['safety_factors'][0].mean().item():.2f}
        - 最小障碍距离: {trace['min_obstacle_distance'][0].item():.2f}
        
        === 决策推理链 ===
        1. 感知到{torch.mean(perception['obstacle_mask']).item():.1%}的区域存在障碍物
        2. 空间分析显示{torch.mean(spatial['navigability_map']).item():.1%}的区域可通行
        3. 风险评估为{torch.mean(spatial['risk_map']).item():.2f}，选择策略：{self.strategic_planning.motion_primitives[torch.argmax(planning['primitive_probabilities'][0]).item()]}
        4. 安全系统评估为{"安全" if planning['safety_assessment'][0, 0] > 0.5 else "需要调整" if planning['safety_assessment'][0, 1] > 0.5 else "危险"}
        5. 应用{control['safety_factors'][0].mean().item():.2f}的安全衰减系数生成最终指令
        """
        
        return explanation

# 测试代码
if __name__ == '__main__':
    # 创建模型实例
    model = CompleteInterpretableModel(sequence_length=5)
    
    # 模拟输入数据（基于你的数据集格式）
    batch_size = 2
    test_inputs = {
        'image_sequence': torch.randn(batch_size, 5, 1, 60, 90),  # 5帧序列
        'motion_state': torch.randn(batch_size, 6),
        'goal_direction': torch.tensor([[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]]),  # 归一化方向
        'task_state': torch.randn(batch_size, 7),
        'constraints': torch.randn(batch_size, 5),
        'desired_velocity': torch.tensor([[1.5], [2.0]]),
        'current_state': torch.randn(batch_size, 6)
    }
    
    # 测试模型
    print("=== 可解释模型测试 ===")
    with torch.no_grad():
        velocity_output, trace = model(test_inputs, return_trace=True)
        print(f"输出速度指令形状: {velocity_output.shape}")
        print(f"决策跟踪模块数量: {len(trace) if trace else 0}")
        
        # 生成决策解释
        explanation = model.explain_decision(test_inputs)
        print("\n=== 决策解释示例 ===")
        print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数数量: {total_params:,}")
    
    # 分模块参数统计
    print("\n各模块参数分布:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {module_params:,} ({module_params/total_params*100:.1f}%)")