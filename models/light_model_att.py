"""
基于ViT+LSTM设计的复杂注意力可解释模型

核心特点：
1. 借鉴LSTMNetVIT的成功架构 (3,563,663参数)
2. 增强的多头注意力机制
3. 分层可解释性输出
4. 多模态注意力融合
5. 时序注意力建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from ViTsubmodules import *

def refine_inputs(X):
    """输入数据预处理，与原始模型保持一致"""
    if X[2] is None:
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1

    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

    return X

class InterpretableMultiHeadAttention(nn.Module):
    """
    可解释的多头注意力机制
    提供注意力权重可视化和分析功能
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 注意力权重存储（用于可解释性）
        self.attention_weights = None
        self.attention_entropy = None
        
    def forward(self, query, key, value, mask=None, return_attention=True):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 多头投影
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 存储注意力权重用于可解释性
        if return_attention:
            self.attention_weights = attention_weights.detach()
            # 计算注意力熵（衡量注意力分散程度）
            self.attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8), 
                dim=-1
            ).mean(dim=1)  # (batch, heads)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        
        if return_attention:
            return output, attention_weights
        return output

class EnhancedViTEncoder(nn.Module):
    """
    增强的Vision Transformer编码器
    基于原始ViT设计但增加可解释性
    """
    def __init__(self):
        super().__init__()
        
        # 借鉴LSTMNetVIT的编码器设计
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, 
                                     n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, 
                                     n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])
        
        # 可解释性注意力模块
        self.interpretable_attention = InterpretableMultiHeadAttention(
            d_model=64, num_heads=8, dropout=0.1
        )
        
        # 空间注意力分析器
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # 上采样和处理（保持原始设计）
        self.up_sample = nn.Upsample(size=(16, 24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48, 12, 3, padding=1)
        
        # 特征解码器（扩展原始设计）
        self.feature_decoder = nn.Sequential(
            nn.Linear(4608, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )
        
        # 可解释性输出头
        self.obstacle_attention_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 15*23),  # 输出空间注意力图
            nn.Sigmoid()
        )
        
        self.depth_attention_head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 15*23),
            nn.Sigmoid()
        )
        
        self.semantic_attention_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, 8),  # 8个语义类别的注意力
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 多层特征提取（借鉴LSTMNetVIT）
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
            
        out = embeds[1:]
        
        # 上采样和融合（保持原始设计）
        fused = torch.cat([self.pxShuffle(out[1]), self.up_sample(out[0])], dim=1)
        compressed = self.down_sample(fused)  # (B, 12, 16, 24)
        
        # 展平并转换为序列用于注意力计算
        seq_features = compressed.view(batch_size, 12, -1).transpose(1, 2)  # (B, 384, 12)
        
        # 应用可解释注意力
        attended_features, attention_weights = self.interpretable_attention(
            seq_features, seq_features, seq_features
        )
        
        # 空间注意力
        spatial_attended, spatial_attention = self.spatial_attention(
            attended_features, attended_features, attended_features
        )
        
        # 重新整形并解码
        final_features = spatial_attended.mean(dim=1)  # 全局平均池化
        decoded_features = self.feature_decoder(
            compressed.flatten(1)  # 使用原始压缩特征进行解码
        )
        
        # 生成可解释性输出
        obstacle_attention = self.obstacle_attention_head(decoded_features)
        depth_attention = self.depth_attention_head(decoded_features)  
        semantic_attention = self.semantic_attention_head(decoded_features)
        
        return {
            'visual_features': decoded_features,
            'raw_features': compressed,
            'attention_maps': {
                'feature_attention': attention_weights,
                'spatial_attention': spatial_attention,
                'obstacle_attention': obstacle_attention.view(batch_size, 1, 15, 23),
                'depth_attention': depth_attention.view(batch_size, 1, 15, 23),
                'semantic_attention': semantic_attention
            },
            'attention_entropy': self.interpretable_attention.attention_entropy
        }

class MultiModalAttentionFusion(nn.Module):
    """
    多模态注意力融合模块
    """
    def __init__(self, visual_dim=512, motion_dim=4, goal_dim=3):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.motion_dim = motion_dim
        self.goal_dim = goal_dim
        
        # 模态编码器
        self.visual_encoder = nn.Linear(visual_dim, 256)
        self.motion_encoder = nn.Linear(motion_dim + 1, 64)  # +1 for desired velocity
        self.goal_encoder = nn.Linear(goal_dim, 64)
        
        # 交叉注意力模块
        self.visual_to_motion_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )
        self.motion_to_visual_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # 融合注意力
        self.fusion_attention = InterpretableMultiHeadAttention(
            d_model=384, num_heads=6, dropout=0.1  # 256 + 64 + 64
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
    def forward(self, visual_features, motion_state, goal_direction, desired_velocity):
        batch_size = visual_features.shape[0]
        
        # 编码各模态
        vis_encoded = self.visual_encoder(visual_features).unsqueeze(1)  # (B, 1, 256)
        
        motion_input = torch.cat([motion_state, desired_velocity], dim=1)
        motion_encoded = self.motion_encoder(motion_input).unsqueeze(1)  # (B, 1, 64)
        
        goal_encoded = self.goal_encoder(goal_direction).unsqueeze(1)  # (B, 1, 64)
        
        # 交叉注意力
        vis_attended, vis_motion_attn = self.visual_to_motion_attention(
            vis_encoded, motion_encoded, motion_encoded
        )
        motion_attended, motion_vis_attn = self.motion_to_visual_attention(
            motion_encoded, vis_encoded, vis_encoded
        )
        
        # 拼接所有特征
        fused_input = torch.cat([
            vis_attended, motion_attended, goal_encoded
        ], dim=2)  # (B, 1, 384)
        
        # 融合注意力
        final_features, fusion_attention = self.fusion_attention(
            fused_input, fused_input, fused_input
        )
        
        # 输出投影
        output = self.output_projection(final_features.squeeze(1))
        
        return {
            'fused_features': output,
            'attention_weights': {
                'visual_to_motion': vis_motion_attn,
                'motion_to_visual': motion_vis_attn,
                'fusion_attention': fusion_attention
            },
            'modality_contributions': {
                'visual': torch.norm(vis_attended, dim=2),
                'motion': torch.norm(motion_attended, dim=2),
                'goal': torch.norm(goal_encoded, dim=2)
            }
        }

class TemporalAttentionLSTM(nn.Module):
    """
    基于注意力机制的时序LSTM模块
    """
    def __init__(self, input_size=128, hidden_size=256, num_layers=3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 主要LSTM（借鉴LSTMNetVIT设计）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True
        )
        
        # 时序注意力机制
        self.temporal_attention = InterpretableMultiHeadAttention(
            d_model=hidden_size, num_heads=8, dropout=0.1
        )
        
        # 状态注意力（注意力到不同隐状态层）
        self.state_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_layers),
            nn.Softmax(dim=1)
        )
        
        # 输出处理
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, 64)
        
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.shape
        
        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # 时序注意力
        attended_out, temporal_attn = self.temporal_attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # 状态注意力（对不同层的隐状态加权）
        final_hidden = h_n.transpose(0, 1)  # (batch, layers, hidden)
        state_weights = self.state_attention(lstm_out[:, -1])  # 基于最后时刻
        weighted_hidden = torch.sum(
            final_hidden * state_weights.unsqueeze(2), dim=1
        )  # (batch, hidden)
        
        # 输出处理
        output = self.output_norm(weighted_hidden)
        output = self.output_projection(output)
        
        return {
            'output': output,
            'hidden': (h_n, c_n),
            'temporal_attention': temporal_attn,
            'state_attention': state_weights,
            'lstm_features': lstm_out
        }

class HierarchicalMotionPlanner(nn.Module):
    """
    分层运动规划器 - 基于注意力的决策
    """
    def __init__(self, input_dim=64):
        super().__init__()
        
        # 运动原语定义（扩展版）
        self.motion_primitives = [
            "forward_fast", "forward_slow", "backward_fast", "backward_slow",
            "left_turn", "right_turn", "left_strafe", "right_strafe",  
            "up_fast", "up_slow", "down_fast", "down_slow",
            "hover", "emergency_stop", "spiral_up", "spiral_down"
        ]
        
        # 分层规划网络
        self.high_level_planner = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4个高级策略类别
            nn.Softmax(dim=1)
        )
        
        # 低级运动原语选择器
        self.primitive_selectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 4),  # 每个高级策略对应4个原语
                nn.Softmax(dim=1)
            ) for _ in range(4)
        ])
        
        # 注意力权重生成器
        self.attention_generator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.motion_primitives)),
            nn.Softmax(dim=1)
        )
        
        # 参数生成器（基于选中的原语）
        self.parameter_generator = nn.Sequential(
            nn.Linear(input_dim + len(self.motion_primitives), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 6)  # [速度强度, 方向角, 俯仰角, 偏航角, 持续时间, 置信度]
        )
        
    def forward(self, features):
        batch_size = features.shape[0]
        
        # 高级策略选择
        high_level_probs = self.high_level_planner(features)
        
        # 低级原语选择
        primitive_probs = []
        for i, selector in enumerate(self.primitive_selectors):
            probs = selector(features)
            weighted_probs = probs * high_level_probs[:, i:i+1]
            primitive_probs.append(weighted_probs)
        
        primitive_probs = torch.cat(primitive_probs, dim=1)
        
        # 注意力权重
        attention_weights = self.attention_generator(features)
        
        # 结合注意力的最终原语概率
        final_probs = primitive_probs * attention_weights
        final_probs = F.softmax(final_probs, dim=1)
        
        # 参数生成
        param_input = torch.cat([features, final_probs], dim=1)
        motion_parameters = self.parameter_generator(param_input)
        
        return {
            'high_level_strategy': high_level_probs,
            'primitive_probabilities': final_probs,
            'attention_weights': attention_weights,
            'motion_parameters': motion_parameters,
            'selected_primitive': torch.argmax(final_probs, dim=1)
        }

class AdaptiveController(nn.Module):
    """
    自适应控制器 - 基于注意力的控制生成
    """
    def __init__(self, planning_dim=6, context_dim=64):
        super().__init__()
        
        # 控制注意力模块
        self.control_attention = InterpretableMultiHeadAttention(
            d_model=context_dim, num_heads=4, dropout=0.1
        )
        
        # 基础控制生成器
        self.base_controller = nn.Sequential(
            nn.Linear(planning_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # xyz速度
            nn.Tanh()
        )
        
        # 安全注意力模块
        self.safety_attention = nn.Sequential(
            nn.Linear(context_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 安全权重
            nn.Sigmoid()
        )
        
        # 自适应增益控制
        self.adaptive_gain = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, motion_params, context_features, safety_constraints):
        batch_size = motion_params.shape[0]
        
        # 上下文注意力
        context_input = context_features.unsqueeze(1)
        attended_context, context_attn = self.control_attention(
            context_input, context_input, context_input
        )
        attended_context = attended_context.squeeze(1)
        
        # 基础控制生成
        control_input = torch.cat([motion_params, attended_context], dim=1)
        base_velocity = self.base_controller(control_input)
        
        # 安全注意力调整
        safety_input = torch.cat([attended_context, base_velocity], dim=1)
        safety_weights = self.safety_attention(safety_input)
        
        # 自适应增益
        adaptive_gain = self.adaptive_gain(attended_context)
        
        # 最终控制输出
        final_velocity = base_velocity * safety_weights * adaptive_gain
        
        return {
            'velocity_cmd': final_velocity,
            'base_velocity': base_velocity,
            'safety_weights': safety_weights,
            'adaptive_gain': adaptive_gain,
            'control_attention': context_attn
        }

class ComplexAttentionInterpretableModel(nn.Module):
    """
    复杂的基于注意力机制的可解释模型
    基于LSTMNetVIT的成功架构，但大幅增强可解释性
    """
    def __init__(self, sequence_length=5):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # 核心模块
        self.vision_encoder = EnhancedViTEncoder()
        self.multimodal_fusion = MultiModalAttentionFusion()
        self.temporal_processor = TemporalAttentionLSTM()
        self.motion_planner = HierarchicalMotionPlanner()
        self.adaptive_controller = AdaptiveController()
        
        # 全局注意力协调器
        self.global_attention_coordinator = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),  # 各模块的特征维度
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, inputs, hidden_state=None, return_intermediates=False):
        """
        复杂的前向传播 - 完整的注意力流程
        """
        # 处理输入格式兼容性
        if isinstance(inputs, list) and len(inputs) >= 4:
            images = inputs[0]
            desired_vel = inputs[1]
            curr_quat = inputs[2]
            target_dir = inputs[3] if len(inputs) > 3 else torch.zeros_like(curr_quat[:, :3])
        else:
            raise ValueError("Invalid input format")
            
        # 确保维度正确
        if len(images.shape) == 5:  # 序列输入
            images = images[:, -1]  # 取最后一帧
        if len(desired_vel.shape) == 1:
            desired_vel = desired_vel.unsqueeze(1)
            
        batch_size = images.shape[0]
        
        # 标准化输入
        refined_inputs = refine_inputs([images, desired_vel, curr_quat])
        
        # 1. 视觉编码和注意力分析
        vision_output = self.vision_encoder(refined_inputs[0])
        
        # 2. 多模态注意力融合
        motion_state = curr_quat  # 简化的运动状态
        fusion_output = self.multimodal_fusion(
            vision_output['visual_features'],
            motion_state,
            target_dir,
            desired_vel
        )
        
        # 3. 时序注意力处理
        # 为LSTM创建序列（简化版时序建模）
        sequence_input = fusion_output['fused_features'].unsqueeze(1).repeat(1, 3, 1)
        temporal_output = self.temporal_processor(sequence_input, hidden_state)
        
        # 4. 分层运动规划
        planning_output = self.motion_planner(temporal_output['output'])
        
        # 5. 自适应控制生成
        # 计算安全约束
        obstacle_attention = vision_output['attention_maps']['obstacle_attention']
        min_obstacle_distance = torch.min(
            torch.mean(obstacle_attention, dim=[2, 3])
        ).unsqueeze(0).expand(batch_size)
        
        control_output = self.adaptive_controller(
            planning_output['motion_parameters'],
            temporal_output['output'],
            min_obstacle_distance.unsqueeze(1)
        )
        
        # 6. 全局注意力协调
        global_features = torch.cat([
            fusion_output['fused_features'],
            temporal_output['output'],
            vision_output['visual_features'][:, :64]  # 截取匹配维度
        ], dim=1)
        
        global_attention_weight = self.global_attention_coordinator(global_features)
        final_velocity = control_output['velocity_cmd'] * global_attention_weight
        
        if return_intermediates:
            # 构建详细的中间结果
            intermediates = {
                'vision_analysis': vision_output,
                'multimodal_fusion': fusion_output,
                'temporal_processing': temporal_output,
                'motion_planning': planning_output,
                'adaptive_control': control_output,
                'global_attention': global_attention_weight,
                # 兼容训练器的接口
                'perception': {
                    'obstacle_mask': vision_output['attention_maps']['obstacle_attention'],
                    'depth_estimation': vision_output['attention_maps']['depth_attention'],
                    'confidence': torch.ones_like(vision_output['attention_maps']['obstacle_attention'])
                },
                'spatial_analysis': {
                    'navigable_map': 1.0 - vision_output['attention_maps']['obstacle_attention'],
                    'risk_map': vision_output['attention_maps']['obstacle_attention'],
                    'direction_preferences': self._compute_direction_preferences(planning_output)
                },
                'motion_planning': {
                    'primitive_probabilities': planning_output['primitive_probabilities'],
                    'motion_parameters': planning_output['motion_parameters'][:, :4]  # 取前4个参数
                },
                'min_obstacle_distance': min_obstacle_distance
            }
            return final_velocity, intermediates
            
        return final_velocity, temporal_output['hidden']
    
    def _compute_direction_preferences(self, planning_output):
        """计算方向偏好（兼容训练器）"""
        probs = planning_output['primitive_probabilities']
        batch_size = probs.shape[0]
        
        # 将16个运动原语映射到6个方向
        direction_prefs = torch.zeros(batch_size, 6, device=probs.device)
        direction_prefs[:, 0] = probs[:, 0] + probs[:, 1]  # forward (fast + slow)
        direction_prefs[:, 1] = probs[:, 2] + probs[:, 3]  # backward
        direction_prefs[:, 2] = probs[:, 4] + probs[:, 6]  # left (turn + strafe)
        direction_prefs[:, 3] = probs[:, 5] + probs[:, 7]  # right
        direction_prefs[:, 4] = probs[:, 8] + probs[:, 9]  # up
        direction_prefs[:, 5] = probs[:, 10] + probs[:, 11]  # down
        
        return direction_prefs
    
    def get_attention_analysis(self, inputs):
        """
        获取完整的注意力分析
        """
        with torch.no_grad():
            _, intermediates = self.forward(inputs, return_intermediates=True)
            
            attention_analysis = {
                'vision_attention': {
                    'feature_attention': intermediates['vision_analysis']['attention_maps']['feature_attention'],
                    'spatial_attention': intermediates['vision_analysis']['attention_maps']['spatial_attention'],
                    'obstacle_attention': intermediates['vision_analysis']['attention_maps']['obstacle_attention'],
                    'depth_attention': intermediates['vision_analysis']['attention_maps']['depth_attention'],
                    'semantic_attention': intermediates['vision_analysis']['attention_maps']['semantic_attention'],
                    'attention_entropy': intermediates['vision_analysis']['attention_entropy']
                },
                'multimodal_attention': intermediates['multimodal_fusion']['attention_weights'],
                'temporal_attention': {
                    'temporal_weights': intermediates['temporal_processing']['temporal_attention'],
                    'state_weights': intermediates['temporal_processing']['state_attention']
                },
                'planning_attention': {
                    'high_level_strategy': intermediates['motion_planning']['high_level_strategy'],
                    'attention_weights': intermediates['motion_planning']['attention_weights']
                },
                'control_attention': intermediates['adaptive_control']['control_attention'],
                'global_attention': intermediates['global_attention']
            }
            
            return attention_analysis
    
    def explain_decision(self, inputs):
        """
        生成详细的决策解释
        """
        velocity_cmd, intermediates = self.forward(inputs, return_intermediates=True)
        attention_analysis = self.get_attention_analysis(inputs)
        
        # 分析各个模块的贡献
        vision_contrib = torch.mean(intermediates['vision_analysis']['attention_entropy']).item()
        planning_contrib = torch.max(intermediates['motion_planning']['primitive_probabilities'][0]).item()
        selected_primitive_idx = intermediates['motion_planning']['selected_primitive'][0].item()
        selected_primitive = self.motion_planner.motion_primitives[selected_primitive_idx]
        
        # 多模态贡献分析
        modal_contribs = intermediates['multimodal_fusion']['modality_contributions']
        visual_contrib = modal_contribs['visual'][0].item()
        motion_contrib = modal_contribs['motion'][0].item()
        goal_contrib = modal_contribs['goal'][0].item()
        
        explanation = f"""
【复杂注意力机制可解释避障决策】

=== 视觉注意力分析 ===
• 注意力熵: {vision_contrib:.3f} (越低越专注)
• 障碍物关注度: {torch.mean(intermediates['vision_analysis']['attention_maps']['obstacle_attention']).item():.3f}
• 深度关注度: {torch.mean(intermediates['vision_analysis']['attention_maps']['depth_attention']).item():.3f}
• 语义理解: {torch.argmax(intermediates['vision_analysis']['attention_maps']['semantic_attention'][0]).item()}类最突出

=== 多模态融合分析 ===
• 视觉贡献度: {visual_contrib:.3f}
• 运动状态贡献度: {motion_contrib:.3f}  
• 目标导向贡献度: {goal_contrib:.3f}
• 主导模态: {'视觉' if visual_contrib > max(motion_contrib, goal_contrib) else '运动' if motion_contrib > goal_contrib else '目标'}

=== 时序注意力分析 ===
• LSTM层权重分布: {intermediates['temporal_processing']['state_attention'][0].cpu().numpy()}
• 时序依赖强度: {torch.mean(intermediates['temporal_processing']['temporal_attention']).item():.3f}

=== 分层运动规划 ===
• 高级策略分布: {intermediates['motion_planning']['high_level_strategy'][0].cpu().numpy()}
• 选择的运动原语: {selected_primitive}
• 原语选择置信度: {planning_contrib:.1%}
• 注意力聚焦度: {torch.max(intermediates['motion_planning']['attention_weights'][0]).item():.3f}

=== 自适应控制分析 ===
• 基础速度指令: {intermediates['adaptive_control']['base_velocity'][0].cpu().numpy()}
• 安全调整因子: {intermediates['adaptive_control']['safety_weights'][0].cpu().numpy()}
• 自适应增益: {intermediates['adaptive_control']['adaptive_gain'][0].item():.3f}
• 最终输出: {velocity_cmd[0].cpu().numpy()}

=== 全局协调 ===
• 全局注意力权重: {intermediates['global_attention'][0].item():.3f}
• 系统整体置信度: {planning_contrib * intermediates['global_attention'][0].item():.1%}

=== 决策可信度评估 ===
• 视觉感知可信度: {'高' if vision_contrib < 0.5 else '中' if vision_contrib < 1.0 else '低'}
• 规划决策可信度: {'高' if planning_contrib > 0.7 else '中' if planning_contrib > 0.4 else '低'}
• 整体系统可信度: {'高' if planning_contrib > 0.6 and vision_contrib < 0.8 else '中等'}
        """
        
        return explanation.strip()

# 测试代码
if __name__ == '__main__':
    print("=== 复杂注意力可解释模型测试 ===")
    
    # 创建模型
    model = ComplexAttentionInterpretableModel(sequence_length=5)
    
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
        velocity_output, hidden = model(test_inputs)
        print(f"✓ 基础前向传播成功，输出形状: {velocity_output.shape}")
        
        # 中间结果测试
        velocity_output, intermediates = model(test_inputs, return_intermediates=True)
        print(f"✓ 可解释模式成功，中间结果包含: {len(intermediates)} 个模块")
        
        # 注意力分析测试
        attention_analysis = model.get_attention_analysis(test_inputs)
        print(f"✓ 注意力分析成功，包含 {len(attention_analysis)} 类注意力")
        
        # 决策解释测试
        explanation = model.explain_decision(test_inputs)
        print(f"✓ 决策解释生成成功，长度: {len(explanation)} 字符")
        
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n复杂注意力模型总参数: {total_params:,}")
    
    # 与其他模型对比
    lstmnet_vit_params = 3563663
    convnet_params = 235269
    print(f"LSTMNetVIT参数: {lstmnet_vit_params:,}")
    print(f"ConvNet参数: {convnet_params:,}")  
    print(f"参数比例: {total_params/lstmnet_vit_params:.1f}x LSTMNetVIT, {total_params/convnet_params:.1f}x ConvNet")
    
    print(f"\n=== 复杂模型特点 ===")
    print("• 基于LSTMNetVIT成功架构")
    print("• 多层次注意力机制")
    print("• 分层运动规划") 
    print("• 多模态注意力融合")
    print("• 全面的可解释性输出")
    print("• 兼容训练器接口")