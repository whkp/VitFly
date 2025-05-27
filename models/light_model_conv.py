"""
基于ConvNet成功设计的优化可解释模型

借鉴ConvNet的成功因素：
1. 简单直接的架构设计
2. 有效的特征提取和下采样
3. 直接的多模态信息融合
4. 渐进式特征压缩
5. 合适的正则化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

def refine_inputs(X):
    """输入数据预处理，与原始模型保持一致"""
    # 填充四元数旋转矩阵如果未提供
    if X[2] is None:
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1

    # 如果输入深度图像尺寸不正确，调整其尺寸
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

    return X

class InterpretableConvBackbone(nn.Module):
    """
    基于ConvNet设计的可解释特征提取器
    保持ConvNet的有效架构，但增加可解释性输出
    """
    def __init__(self):
        super().__init__()
        
        # 借鉴ConvNet的卷积层设计
        self.conv1 = nn.Conv2d(1, 4, 3, 3)  # (1,60,90) -> (4,20,30)
        self.conv2 = nn.Conv2d(4, 10, 3, 2)  # (4,20,30) -> (10,9,14)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)  # -> (10,7,12)
        self.maxpool = nn.MaxPool2d(2, 1)  # 用于第一层特征增强
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(10)
        
        # 可解释性特征分析器
        self.obstacle_analyzer = nn.Sequential(
            nn.Conv2d(4, 2, 1),  # 从第一层特征分析障碍物
            nn.Sigmoid()
        )
        
        self.depth_analyzer = nn.Sequential(
            nn.Conv2d(10, 1, 1),  # 从第二层特征估计深度
            nn.Sigmoid()
        )
        
        # 空间推理分析器（简化版）
        self.spatial_analyzer = nn.Sequential(
            nn.Conv2d(10, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, 1),  # navigability + risk
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        前向传播，提取可解释特征
        """
        # 第一层卷积 + 可解释分析
        x1 = self.bn1(F.relu(self.conv1(x)))
        x1_enhanced = -self.maxpool(-x1)  # ConvNet的特殊maxpool技巧
        
        # 从第一层特征中分析障碍物
        obstacle_features = self.obstacle_analyzer(x1_enhanced)  # (B, 2, 20, 30)
        
        # 第二层卷积
        x2 = self.bn2(F.relu(self.conv2(x1_enhanced)))
        x2_pooled = self.avgpool(x2)  # (B, 10, 7, 12)
        
        # 从第二层特征中分析深度和空间
        depth_features = self.depth_analyzer(x2)  # (B, 1, 9, 14)
        spatial_features = self.spatial_analyzer(x2_pooled)  # (B, 2, 7, 12)
        
        # 展平用于后续处理
        visual_features = torch.flatten(x2_pooled, 1)  # (B, 10*7*12 = 840)
        
        return {
            'visual_features': visual_features,
            'obstacle_analysis': {
                'obstacle_mask': obstacle_features[:, :1],  # (B, 1, 20, 30)
                'obstacle_confidence': obstacle_features[:, 1:2]  # (B, 1, 20, 30)
            },
            'depth_analysis': {
                'depth_estimation': depth_features  # (B, 1, 9, 14)
            },
            'spatial_analysis': {
                'navigable_map': spatial_features[:, :1],  # (B, 1, 7, 12)
                'risk_map': spatial_features[:, 1:2]  # (B, 1, 7, 12)
            }
        }

class MotionPrimitiveAnalyzer(nn.Module):
    """
    运动原语分析器 - 简化但保持可解释性
    """
    def __init__(self, input_dim=845):
        super().__init__()
        
        # 运动原语定义
        self.motion_primitives = [
            "forward", "backward", "left", "right", 
            "up", "down", "slow_down", "emergency_stop"
        ]
        
        # 基于ConvNet的FC架构
        self.primitive_analyzer = nn.Sequential(
            nn.Linear(input_dim, 256, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(),
            nn.Linear(64, len(self.motion_primitives)),
            nn.Softmax(dim=1)
        )
        
        # 运动参数生成器
        self.parameter_generator = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),
            nn.LeakyReLU(),
            nn.Linear(128, 32, bias=False),
            nn.Tanh(),  # 限制参数范围
            nn.Linear(32, 4)  # [强度, 方向角, 倾斜角, 持续时间]
        )
        
    def forward(self, fused_features):
        """
        分析运动策略和参数
        """
        primitive_probs = self.primitive_analyzer(fused_features)
        motion_parameters = self.parameter_generator(fused_features)
        
        return {
            'primitive_probabilities': primitive_probs,
            'motion_parameters': motion_parameters,
            'selected_primitive': torch.argmax(primitive_probs, dim=1)
        }

class OptimizedInterpretableModel(nn.Module):
    """
    优化的可解释避障模型
    基于ConvNet的成功设计，但保持可解释性
    """
    def __init__(self):
        super().__init__()
        
        # 主要特征提取器
        self.backbone = InterpretableConvBackbone()
        
        # 运动分析器
        self.motion_analyzer = MotionPrimitiveAnalyzer(input_dim=845)  # 840视觉+1速度+4四元数
        
        # 最终控制器（借鉴ConvNet设计）
        self.final_controller = nn.Sequential(
            nn.Linear(845, 256, bias=False),
            nn.LeakyReLU(),
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(),
            nn.Linear(64, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 3)  # 最终速度输出
        )
        
        # 安全约束层
        self.safety_filter = nn.Sequential(
            nn.Linear(3 + 1, 16),  # 速度指令 + 最小距离
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )
        
    def forward(self, X, return_intermediates=False):
        """
        前向传播 - 兼容训练器接口
        """
        # 处理输入格式兼容性
        if isinstance(X, list) and len(X) >= 4:
            # 训练器格式: [images, desired_vel, curr_quat, target_dir]
            images = X[0]
            desired_vel = X[1] 
            curr_quat = X[2]
            target_dir = X[3] if len(X) > 3 else None
        else:
            raise ValueError("Invalid input format")
        
        # 确保输入维度正确
        if len(images.shape) == 5:  # (B, T, C, H, W)
            images = images[:, -1]  # 取最后一帧
        
        if len(desired_vel.shape) == 1:
            desired_vel = desired_vel.unsqueeze(1)
            
        # 标准化输入
        refined_inputs = refine_inputs([images, desired_vel, curr_quat])
        
        # 1. 视觉特征提取和可解释分析
        backbone_output = self.backbone(refined_inputs[0])
        visual_features = backbone_output['visual_features']
        
        # 2. 多模态特征融合（借鉴ConvNet的直接拼接策略）
        metadata = torch.cat((refined_inputs[1] * 0.1, refined_inputs[2]), dim=1).float()
        fused_features = torch.cat((visual_features, metadata), dim=1).float()
        
        # 3. 运动原语分析
        motion_analysis = self.motion_analyzer(fused_features)
        
        # 4. 计算最小障碍距离（用于安全约束）
        depth_map = backbone_output['depth_analysis']['depth_estimation']
        min_obstacle_distance = torch.min(depth_map.view(depth_map.shape[0], -1), dim=1)[0]
        
        # 5. 生成控制指令
        base_velocity = self.final_controller(fused_features)
        
        # 6. 应用安全约束
        safety_input = torch.cat([base_velocity, min_obstacle_distance.unsqueeze(1)], dim=1)
        safety_factors = self.safety_filter(safety_input)
        final_velocity = base_velocity * safety_factors
        
        if return_intermediates:
            # 构建中间结果（兼容训练器）
            intermediates = {
                'perception': {
                    'obstacle_mask': backbone_output['obstacle_analysis']['obstacle_mask'],
                    'depth_estimation': backbone_output['depth_analysis']['depth_estimation'],
                    'confidence': backbone_output['obstacle_analysis']['obstacle_confidence']
                },
                'spatial_analysis': {
                    'navigable_map': backbone_output['spatial_analysis']['navigable_map'],
                    'risk_map': backbone_output['spatial_analysis']['risk_map'],
                    'direction_preferences': self._compute_direction_preferences(motion_analysis)
                },
                'motion_planning': motion_analysis,
                'min_obstacle_distance': min_obstacle_distance
            }
            return final_velocity, intermediates
        
        return final_velocity
    
    def _compute_direction_preferences(self, motion_analysis):
        """
        从运动原语概率计算方向偏好（兼容训练器）
        """
        probs = motion_analysis['primitive_probabilities']
        batch_size = probs.shape[0]
        
        # 将8个运动原语映射到6个方向
        direction_prefs = torch.zeros(batch_size, 6, device=probs.device)
        direction_prefs[:, 0] = probs[:, 0]  # forward
        direction_prefs[:, 1] = probs[:, 1]  # backward  
        direction_prefs[:, 2] = probs[:, 2]  # left
        direction_prefs[:, 3] = probs[:, 3]  # right
        direction_prefs[:, 4] = probs[:, 4]  # up
        direction_prefs[:, 5] = probs[:, 5]  # down
        
        return direction_prefs
    
    def explain_decision(self, inputs):
        """
        生成决策解释
        """
        with torch.no_grad():
            velocity_cmd, intermediates = self.forward(inputs, return_intermediates=True)
            
            # 分析各个组件
            perception = intermediates['perception']
            spatial = intermediates['spatial_analysis'] 
            planning = intermediates['motion_planning']
            
            # 生成详细解释
            obstacle_coverage = torch.mean(perception['obstacle_mask']).item()
            avg_depth = torch.mean(perception['depth_estimation']).item()
            confidence = torch.mean(perception['confidence']).item()
            
            navigable_ratio = torch.mean(spatial['navigable_map']).item()
            risk_level = torch.mean(spatial['risk_map']).item()
            
            selected_primitive_idx = planning['selected_primitive'][0].item()
            selected_primitive = self.motion_analyzer.motion_primitives[selected_primitive_idx]
            primitive_confidence = torch.max(planning['primitive_probabilities'][0]).item()
            
            min_distance = intermediates['min_obstacle_distance'][0].item()
            
            explanation = f"""
【基于ConvNet优化的可解释避障决策】

=== 视觉感知分析 ===
• 障碍物覆盖率: {obstacle_coverage:.1%}
• 平均深度距离: {avg_depth:.2f}m  
• 感知置信度: {confidence:.1%}
• 最近障碍距离: {min_distance:.2f}m

=== 空间理解分析 ===  
• 可导航空间比例: {navigable_ratio:.1%}
• 环境风险等级: {risk_level:.2f}/1.0
• 空间安全评估: {'安全' if risk_level < 0.3 else '中等风险' if risk_level < 0.7 else '高风险'}

=== 运动决策分析 ===
• 选择的运动策略: {selected_primitive}
• 策略选择置信度: {primitive_confidence:.1%}
• 运动参数: 强度{planning['motion_parameters'][0, 0]:.2f}, 角度{planning['motion_parameters'][0, 1]:.2f}

=== 控制输出 ===
• 速度指令: X={velocity_cmd[0, 0]:.2f}, Y={velocity_cmd[0, 1]:.2f}, Z={velocity_cmd[0, 2]:.2f} m/s
• 速度模长: {torch.norm(velocity_cmd[0]).item():.2f} m/s

=== 安全性评估 ===
• 距离安全性: {'安全' if min_distance > 0.5 else '警告' if min_distance > 0.2 else '危险'}
• 速度合理性: {'合理' if torch.norm(velocity_cmd[0]).item() < 2.0 else '过快'}
            """
            
            return explanation.strip()

# 轻量级模型（向后兼容）
class CompleteInterpretableModel(OptimizedInterpretableModel):
    """
    向后兼容的轻量级模型别名
    """
    def __init__(self, sequence_length=3):
        super().__init__()
        self.sequence_length = sequence_length

# 测试代码
if __name__ == '__main__':
    print("=== 优化可解释模型测试 ===")
    
    # 创建模型
    model = OptimizedInterpretableModel()
    
    # 测试输入
    batch_size = 2
    test_inputs = [
        torch.randn(batch_size, 1, 60, 90),  # 图像
        torch.tensor([[1.5], [2.0]]),        # 期望速度
        torch.randn(batch_size, 4),          # 四元数
        torch.tensor([[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]])  # 目标方向
    ]
    
    with torch.no_grad():
        # 基础测试
        velocity_output = model(test_inputs)
        print(f"✓ 基础前向传播成功，输出形状: {velocity_output.shape}")
        
        # 中间结果测试
        velocity_output, intermediates = model(test_inputs, return_intermediates=True)
        print(f"✓ 可解释模式成功，中间结果包含: {list(intermediates.keys())}")
        
        # 决策解释测试
        explanation = model.explain_decision(test_inputs)
        print(f"✓ 决策解释生成成功，长度: {len(explanation)} 字符")
        
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数: {total_params:,}")
    
    # 与ConvNet对比
    convnet_params = 235269
    print(f"ConvNet参数: {convnet_params:,}")
    print(f"参数比例: {total_params/convnet_params:.1f}x ConvNet")
    
    print(f"\n=== 优化特点 ===")
    print("• 基于ConvNet成功架构设计")
    print("• 保持简单直接的特征提取")
    print("• 直接多模态信息融合")
    print("• 渐进式特征压缩")
    print("• 增强的可解释性输出")
    print("• 兼容原始训练器接口")