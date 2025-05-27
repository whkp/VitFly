"""
复杂注意力可解释模型训练系统

特点：
1. 专为注意力机制设计的损失函数
2. 注意力权重的正则化和优化
3. 分层训练策略
4. 丰富的注意力可视化
5. 多阶段训练流程
"""

import os, sys
from os.path import join as opj
import numpy as np
import torch
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import traceback
import pickle
import configargparse

# 修复matplotlib Qt错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import getpass

from dataloading import *
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
# 导入复杂注意力模型
try:  
    from light_model_att import ComplexAttentionInterpretableModel
except ImportError:
    print("Warning: Using fallback model import")
    try:
        from attention_interpretable_model import ComplexAttentionInterpretableModel
    except ImportError:
        from light_model import CompleteInterpretableModel as ComplexAttentionInterpretableModel

class AttentionAwareLoss(nn.Module):
    """
    注意力感知损失函数
    专门为注意力机制设计的多任务损失
    """
    def __init__(self, task_weights=None):
        super().__init__()
        
        # 注意力专用任务权重
        if task_weights is None:
            self.task_weights = {
                'control': 3.0,              # 主要控制任务
                'perception': 0.8,           # 感知任务
                'attention_consistency': 0.5, # 注意力一致性
                'attention_sparsity': 0.3,   # 注意力稀疏性
                'temporal_stability': 0.4,    # 时序稳定性
                'multimodal_alignment': 0.6,  # 多模态对齐
                'safety': 1.2,               # 安全约束
            }
        else:
            self.task_weights = task_weights
            
    def perception_loss(self, predicted_perception, depth_images):
        """
        感知损失 - 基于注意力的感知评估
        """
        obstacle_mask = predicted_perception['obstacle_mask']
        depth_estimation = predicted_perception['depth_estimation'] 
        confidence = predicted_perception['confidence']
        
        # 调整尺寸
        _, _, h_out, w_out = obstacle_mask.shape
        depth_images_resized = F.interpolate(
            depth_images, 
            size=(h_out, w_out), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 障碍物检测损失（注意力引导）
        obstacle_gt = (depth_images_resized < 0.35).float()
        obstacle_loss = F.binary_cross_entropy(obstacle_mask, obstacle_gt)
        
        # 深度估计损失（注意力加权）
        attention_weights = confidence  # 使用置信度作为注意力权重
        weighted_depth_loss = F.mse_loss(
            depth_estimation * attention_weights, 
            depth_images_resized * attention_weights
        )
        
        # 置信度校准损失
        depth_error = torch.abs(depth_estimation - depth_images_resized)
        confidence_target = torch.exp(-2 * depth_error)  # 误差越小置信度越高
        confidence_loss = F.kl_div(
            torch.log(confidence + 1e-8), 
            confidence_target, 
            reduction='batchmean'
        )
        
        total_perception_loss = obstacle_loss + weighted_depth_loss + 0.2 * confidence_loss
        
        return total_perception_loss, {
            'obstacle_loss': obstacle_loss.item(),
            'weighted_depth_loss': weighted_depth_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
    
    def attention_consistency_loss(self, attention_maps, model_outputs):
        """
        注意力一致性损失 - 确保不同层的注意力相互支持
        """
        total_consistency_loss = 0
        consistency_details = {}
        
        # 1. 视觉注意力内部一致性
        if 'vision_analysis' in model_outputs['intermediates']:
            vision_attn = model_outputs['intermediates']['vision_analysis']['attention_maps']
            
            # 障碍物注意力与深度注意力应该相关
            obstacle_attn = vision_attn['obstacle_attention'].flatten(1)
            depth_attn = vision_attn['depth_attention'].flatten(1)
            
            obstacle_depth_consistency = F.mse_loss(
                F.normalize(obstacle_attn, dim=1),
                F.normalize(depth_attn, dim=1)
            )
            total_consistency_loss += obstacle_depth_consistency
            consistency_details['obstacle_depth_consistency'] = obstacle_depth_consistency.item()
        
        # 2. 多模态注意力对齐
        if 'multimodal_fusion' in model_outputs['intermediates']:
            multimodal_attn = model_outputs['intermediates']['multimodal_fusion']['attention_weights']
            
            # 视觉到运动和运动到视觉的注意力应该相互补充
            if 'visual_to_motion' in multimodal_attn and 'motion_to_visual' in multimodal_attn:
                v2m_attn = multimodal_attn['visual_to_motion'].flatten(1)
                m2v_attn = multimodal_attn['motion_to_visual'].flatten(1)
                
                # 对称性约束
                symmetry_loss = F.mse_loss(v2m_attn, m2v_attn)
                total_consistency_loss += 0.5 * symmetry_loss
                consistency_details['multimodal_symmetry'] = symmetry_loss.item()
        
        # 3. 规划注意力合理性
        if 'motion_planning' in model_outputs['intermediates']:
            planning = model_outputs['intermediates']['motion_planning']
            
            # 高级策略与具体原语应该一致
            if 'high_level_strategy' in planning and 'primitive_probabilities' in planning:
                high_level = planning['high_level_strategy']
                primitives = planning['primitive_probabilities']
                
                # 原语分布应该与高级策略分布相对应
                # 这里简化为检查分布的熵相似性
                high_entropy = -torch.sum(high_level * torch.log(high_level + 1e-8), dim=1)
                primitive_entropy = -torch.sum(primitives * torch.log(primitives + 1e-8), dim=1)
                
                entropy_consistency = F.mse_loss(high_entropy, primitive_entropy)
                total_consistency_loss += 0.3 * entropy_consistency
                consistency_details['planning_entropy_consistency'] = entropy_consistency.item()
        
        return total_consistency_loss, consistency_details
    
    def attention_sparsity_loss(self, model_outputs):
        """
        注意力稀疏性损失 - 鼓励注意力集中而非分散
        """
        total_sparsity_loss = 0
        sparsity_details = {}
        
        # 1. 视觉注意力稀疏性
        if 'vision_analysis' in model_outputs['intermediates']:
            vision_attn = model_outputs['intermediates']['vision_analysis']['attention_maps']
            
            for attn_name, attn_map in vision_attn.items():
                if 'attention' in attn_name and attn_map.dim() >= 3:
                    # 计算L1稀疏性（鼓励稀疏）
                    sparsity = torch.mean(torch.abs(attn_map))
                    total_sparsity_loss += sparsity
                    sparsity_details[f'{attn_name}_sparsity'] = sparsity.item()
        
        # 2. 注意力熵惩罚（熵越低越集中）
        if 'vision_analysis' in model_outputs['intermediates']:
            attention_entropy = model_outputs['intermediates']['vision_analysis'].get('attention_entropy')
            if attention_entropy is not None:
                # 鼓励低熵（高集中度）
                entropy_penalty = torch.mean(attention_entropy)
                total_sparsity_loss += 0.1 * entropy_penalty
                sparsity_details['entropy_penalty'] = entropy_penalty.item()
        
        return total_sparsity_loss, sparsity_details
    
    def temporal_stability_loss(self, model_outputs, previous_outputs=None):
        """
        时序稳定性损失 - 注意力权重的时序平滑性
        """
        if previous_outputs is None:
            return torch.tensor(0.0, device=next(iter(model_outputs.values())).device), {}
        
        total_stability_loss = 0
        stability_details = {}
        
        # 时序注意力稳定性
        if ('temporal_processing' in model_outputs['intermediates'] and 
            'temporal_processing' in previous_outputs['intermediates']):
            
            current_temporal = model_outputs['intermediates']['temporal_processing']
            previous_temporal = previous_outputs['intermediates']['temporal_processing']
            
            # 时序注意力权重的平滑性
            if ('temporal_attention' in current_temporal and 
                'temporal_attention' in previous_temporal):
                
                current_attn = current_temporal['temporal_attention']
                previous_attn = previous_temporal['temporal_attention']
                
                # 时序平滑性约束
                temporal_diff = F.mse_loss(current_attn, previous_attn)
                total_stability_loss += temporal_diff
                stability_details['temporal_attention_stability'] = temporal_diff.item()
        
        return total_stability_loss, stability_details
    
    def multimodal_alignment_loss(self, model_outputs):
        """
        多模态对齐损失 - 确保不同模态的特征相互补充
        """
        total_alignment_loss = 0
        alignment_details = {}
        
        if 'multimodal_fusion' in model_outputs['intermediates']:
            fusion_output = model_outputs['intermediates']['multimodal_fusion']
            
            # 模态贡献度应该均衡（避免某个模态完全主导）
            if 'modality_contributions' in fusion_output:
                contributions = fusion_output['modality_contributions']
                
                # 计算贡献度方差（鼓励均衡）
                contrib_values = torch.stack([
                    contributions['visual'].flatten(),
                    contributions['motion'].flatten(), 
                    contributions['goal'].flatten()
                ])
                
                contrib_variance = torch.var(contrib_values, dim=0).mean()
                total_alignment_loss += contrib_variance
                alignment_details['contribution_variance'] = contrib_variance.item()
                
                # 防止某个模态贡献度过低
                min_contrib = torch.min(contrib_values, dim=0)[0].mean()
                underutilization_penalty = F.relu(0.1 - min_contrib)  # 至少10%贡献
                total_alignment_loss += underutilization_penalty
                alignment_details['underutilization_penalty'] = underutilization_penalty.item()
        
        return total_alignment_loss, alignment_details
    
    def safety_constraint_loss(self, velocity_cmd, min_obstacle_distance, model_outputs):
        """
        安全约束损失 - 基于注意力的安全评估
        """
        velocity_magnitude = torch.norm(velocity_cmd, dim=1)
        
        # 基础安全约束
        safe_velocity = torch.clamp(min_obstacle_distance * 2.0, 0.1, 2.0)
        velocity_safety_loss = F.smooth_l1_loss(velocity_magnitude, safe_velocity)
        
        # 注意力引导的安全约束
        safety_attention_loss = 0
        if 'adaptive_control' in model_outputs['intermediates']:
            control_output = model_outputs['intermediates']['adaptive_control']
            
            # 安全权重应该与障碍物距离相关
            if 'safety_weights' in control_output:
                safety_weights = control_output['safety_weights']
                expected_safety = 1.0 - torch.exp(-min_obstacle_distance)
                
                safety_weight_loss = F.mse_loss(
                    torch.mean(safety_weights, dim=1), 
                    expected_safety
                )
                safety_attention_loss += safety_weight_loss
        
        # 速度平滑性
        velocity_smoothness = torch.mean(torch.abs(velocity_cmd))
        
        total_safety_loss = (velocity_safety_loss + 
                           0.3 * safety_attention_loss + 
                           0.1 * velocity_smoothness)
        
        return total_safety_loss, {
            'velocity_safety': velocity_safety_loss.item(),
            'safety_attention': safety_attention_loss,
            'velocity_smoothness': velocity_smoothness.item()
        }
    
    def forward(self, model_outputs, ground_truth, previous_outputs=None):
        """
        计算总的注意力感知损失
        """
        velocity_cmd = model_outputs['velocity_cmd']
        intermediates = model_outputs['intermediates']
        
        gt_velocity = ground_truth['velocity_cmd']
        depth_images = ground_truth['depth_images']
        
        # 1. 主要控制任务损失
        control_loss = F.smooth_l1_loss(velocity_cmd, gt_velocity)
        
        # 2. 感知损失
        perception_loss, perception_details = self.perception_loss(
            intermediates['perception'], depth_images
        )
        
        # 3. 注意力一致性损失
        consistency_loss, consistency_details = self.attention_consistency_loss(
            None, model_outputs
        )
        
        # 4. 注意力稀疏性损失
        sparsity_loss, sparsity_details = self.attention_sparsity_loss(model_outputs)
        
        # 5. 时序稳定性损失
        stability_loss, stability_details = self.temporal_stability_loss(
            model_outputs, previous_outputs
        )
        
        # 6. 多模态对齐损失
        alignment_loss, alignment_details = self.multimodal_alignment_loss(model_outputs)
        
        # 7. 安全约束损失
        safety_loss, safety_details = self.safety_constraint_loss(
            velocity_cmd, intermediates['min_obstacle_distance'], model_outputs
        )
        
        # 加权总损失
        total_loss = (
            self.task_weights['control'] * control_loss +
            self.task_weights['perception'] * perception_loss +
            self.task_weights['attention_consistency'] * consistency_loss +
            self.task_weights['attention_sparsity'] * sparsity_loss +
            self.task_weights['temporal_stability'] * stability_loss +
            self.task_weights['multimodal_alignment'] * alignment_loss +
            self.task_weights['safety'] * safety_loss
        )
        
        # 整合所有损失细节
        loss_details = {
            'total_loss': total_loss.item(),
            'control_loss': control_loss.item(),
            'perception_loss': perception_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'stability_loss': stability_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'safety_loss': safety_loss.item(),
        }
        
        # 合并详细信息
        loss_details.update(perception_details)
        loss_details.update(consistency_details)
        loss_details.update(sparsity_details)
        loss_details.update(stability_details)
        loss_details.update(alignment_details)
        loss_details.update(safety_details)
        
        return total_loss, loss_details

class AttentionVisualizationManager:
    """
    专门的注意力可视化管理器
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_attention_comprehensive(self, depth_image, model_outputs, attention_analysis, save_path=None):
        """
        全面的注意力机制可视化
        """
        intermediates = model_outputs['intermediates']
        
        # 创建大型图形
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
        
        # 原始输入
        ax1 = fig.add_subplot(gs[0, 0])
        depth_np = depth_image[0, 0].cpu().numpy()
        im1 = ax1.imshow(depth_np, cmap='viridis')
        ax1.set_title('原始深度图像', fontsize=12)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 视觉注意力maps
        if 'vision_analysis' in intermediates:
            vision_attn = intermediates['vision_analysis']['attention_maps']
            
            # 障碍物注意力
            ax2 = fig.add_subplot(gs[0, 1])
            obstacle_attn = vision_attn['obstacle_attention'][0, 0].cpu().numpy()
            im2 = ax2.imshow(obstacle_attn, cmap='Reds', vmin=0, vmax=1)
            ax2.set_title('障碍物注意力', fontsize=12)
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # 深度注意力
            ax3 = fig.add_subplot(gs[0, 2])
            depth_attn = vision_attn['depth_attention'][0, 0].cpu().numpy()
            im3 = ax3.imshow(depth_attn, cmap='Blues', vmin=0, vmax=1)
            ax3.set_title('深度注意力', fontsize=12)
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            
            # 语义注意力
            ax4 = fig.add_subplot(gs[0, 3])
            semantic_attn = vision_attn['semantic_attention'][0].cpu().numpy()
            bars = ax4.bar(range(len(semantic_attn)), semantic_attn)
            ax4.set_title('语义类别注意力', fontsize=12)
            ax4.set_xlabel('语义类别')
            ax4.set_ylabel('注意力权重')
            
            # 注意力熵分析
            ax5 = fig.add_subplot(gs[0, 4])
            if 'attention_entropy' in intermediates['vision_analysis']:
                entropy = intermediates['vision_analysis']['attention_entropy'][0].cpu().numpy()
                ax5.bar(range(len(entropy)), entropy)
                ax5.set_title('注意力熵分析', fontsize=12)
                ax5.set_xlabel('注意力头')
                ax5.set_ylabel('熵值')
        
        # 多模态注意力可视化
        if 'multimodal_fusion' in intermediates:
            fusion = intermediates['multimodal_fusion']
            
            # 模态贡献度
            ax6 = fig.add_subplot(gs[1, 0])
            if 'modality_contributions' in fusion:
                contribs = fusion['modality_contributions']
                modal_names = ['Visual', 'Motion', 'Goal']
                modal_values = [
                    contribs['visual'][0].item(),
                    contribs['motion'][0].item(),
                    contribs['goal'][0].item()
                ]
                bars = ax6.pie(modal_values, labels=modal_names, autopct='%1.1f%%')
                ax6.set_title('模态贡献度分布', fontsize=12)
            
            # 交叉注意力权重
            if 'attention_weights' in fusion:
                attn_weights = fusion['attention_weights']
                
                # 视觉-运动交叉注意力
                if 'visual_to_motion' in attn_weights:
                    ax7 = fig.add_subplot(gs[1, 1])
                    v2m_attn = attn_weights['visual_to_motion'][0, 0].cpu().numpy()
                    sns.heatmap(v2m_attn, ax=ax7, cmap='viridis')
                    ax7.set_title('视觉→运动注意力', fontsize=12)
                
                # 运动-视觉交叉注意力
                if 'motion_to_visual' in attn_weights:
                    ax8 = fig.add_subplot(gs[1, 2])
                    m2v_attn = attn_weights['motion_to_visual'][0, 0].cpu().numpy()
                    sns.heatmap(m2v_attn, ax=ax8, cmap='plasma')
                    ax8.set_title('运动→视觉注意力', fontsize=12)
        
        # 时序注意力分析
        if 'temporal_processing' in intermediates:
            temporal = intermediates['temporal_processing']
            
            # 时序注意力权重
            if 'temporal_attention' in temporal:
                ax9 = fig.add_subplot(gs[1, 3])
                temporal_attn = temporal['temporal_attention'][0, 0].cpu().numpy()
                sns.heatmap(temporal_attn, ax=ax9, cmap='coolwarm')
                ax9.set_title('时序注意力模式', fontsize=12)
            
            # LSTM状态注意力
            if 'state_attention' in temporal:
                ax10 = fig.add_subplot(gs[1, 4])
                state_attn = temporal['state_attention'][0].cpu().numpy()
                ax10.bar(range(len(state_attn)), state_attn)
                ax10.set_title('LSTM层注意力权重', fontsize=12)
                ax10.set_xlabel('LSTM层')
                ax10.set_ylabel('注意力权重')
        
        # 运动规划注意力
        if 'motion_planning' in intermediates:
            planning = intermediates['motion_planning']
            
            # 高级策略分布
            if 'high_level_strategy' in planning:
                ax11 = fig.add_subplot(gs[2, 0])
                high_level = planning['high_level_strategy'][0].cpu().numpy()
                strategy_names = ['前进策略', '机动策略', '避障策略', '悬停策略']
                ax11.pie(high_level, labels=strategy_names, autopct='%1.1f%%')
                ax11.set_title('高级策略分布', fontsize=12)
            
            # 运动原语选择
            if 'primitive_probabilities' in planning:
                ax12 = fig.add_subplot(gs[2, 1:3])
                primitive_probs = planning['primitive_probabilities'][0].cpu().numpy()
                # 假设有16个运动原语
                primitive_names = [f'P{i}' for i in range(len(primitive_probs))]
                bars = ax12.bar(range(len(primitive_probs)), primitive_probs)
                ax12.set_title('运动原语选择概率', fontsize=12)
                ax12.set_xlabel('运动原语')
                ax12.set_ylabel('选择概率')
                ax12.tick_params(axis='x', rotation=45)
                
                # 高亮最大概率
                max_idx = np.argmax(primitive_probs)
                bars[max_idx].set_color('red')
                ax12.text(max_idx, primitive_probs[max_idx] + 0.01, 
                         f'{primitive_probs[max_idx]:.3f}', 
                         ha='center', va='bottom', fontweight='bold')
            
            # 注意力权重分布
            if 'attention_weights' in planning:
                ax13 = fig.add_subplot(gs[2, 3])
                attn_weights = planning['attention_weights'][0].cpu().numpy()
                ax13.bar(range(len(attn_weights)), attn_weights)
                ax13.set_title('规划注意力权重', fontsize=12)
                ax13.set_xlabel('原语索引')
                ax13.set_ylabel('注意力权重')
        
        # 控制注意力分析
        if 'adaptive_control' in intermediates:
            control = intermediates['adaptive_control']
            
            # 安全权重
            if 'safety_weights' in control:
                ax14 = fig.add_subplot(gs[2, 4])
                safety_weights = control['safety_weights'][0].cpu().numpy()
                ax14.bar(['X', 'Y', 'Z'], safety_weights)
                ax14.set_title('安全调整权重', fontsize=12)
                ax14.set_ylabel('权重值')
            
            # 自适应增益
            if 'adaptive_gain' in control:
                ax15 = fig.add_subplot(gs[3, 0])
                gain = control['adaptive_gain'][0].item()
                ax15.bar(['自适应增益'], [gain])
                ax15.set_title(f'自适应增益: {gain:.3f}', fontsize=12)
                ax15.set_ylim([0, 1])
        
        # 全局注意力协调
        if 'global_attention' in intermediates:
            ax16 = fig.add_subplot(gs[3, 1])
            global_attn = intermediates['global_attention'][0].item()
            ax16.bar(['全局权重'], [global_attn])
            ax16.set_title(f'全局注意力: {global_attn:.3f}', fontsize=12)
            ax16.set_ylim([0, 1])
        
        # 最终控制输出对比
        ax17 = fig.add_subplot(gs[3, 2:5])
        if 'base_velocity' in intermediates.get('adaptive_control', {}):
            base_vel = intermediates['adaptive_control']['base_velocity'][0].cpu().numpy()
            final_vel = model_outputs['velocity_cmd'][0].cpu().numpy()
            
            x_pos = np.arange(3)
            width = 0.35
            
            ax17.bar(x_pos - width/2, base_vel, width, label='基础控制', alpha=0.7)
            ax17.bar(x_pos + width/2, final_vel, width, label='最终输出', alpha=0.7)
            
            ax17.set_xlabel('速度分量')
            ax17.set_ylabel('速度值')
            ax17.set_title('控制输出对比', fontsize=12)
            ax17.set_xticks(x_pos)
            ax17.set_xticklabels(['X', 'Y', 'Z'])
            ax17.legend()
            ax17.grid(True, alpha=0.3)
        
        plt.suptitle('复杂注意力机制全面分析', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_attention_evolution(self, attention_history, save_path=None):
        """
        可视化注意力演化过程
        """
        if len(attention_history) < 2:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('注意力演化分析', fontsize=16)
        
        # 提取不同类型的注意力权重随时间的变化
        vision_entropy = [h.get('vision_entropy', 0) for h in attention_history]
        planning_confidence = [h.get('planning_confidence', 0) for h in attention_history]
        safety_attention = [h.get('safety_attention', 0) for h in attention_history]
        
        time_steps = range(len(attention_history))
        
        # 视觉注意力熵演化
        axes[0, 0].plot(time_steps, vision_entropy, 'b-o')
        axes[0, 0].set_title('视觉注意力熵演化')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('注意力熵')
        axes[0, 0].grid(True)
        
        # 规划置信度演化
        axes[0, 1].plot(time_steps, planning_confidence, 'r-s')
        axes[0, 1].set_title('规划决策置信度演化')
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('置信度')
        axes[0, 1].grid(True)
        
        # 安全注意力演化
        axes[0, 2].plot(time_steps, safety_attention, 'g-^')
        axes[0, 2].set_title('安全注意力演化')
        axes[0, 2].set_xlabel('时间步')
        axes[0, 2].set_ylabel('安全权重')
        axes[0, 2].grid(True)
        
        # 多模态贡献度演化
        if len(attention_history) > 0 and 'modal_contributions' in attention_history[0]:
            visual_contrib = [h['modal_contributions'].get('visual', 0) for h in attention_history]
            motion_contrib = [h['modal_contributions'].get('motion', 0) for h in attention_history]
            goal_contrib = [h['modal_contributions'].get('goal', 0) for h in attention_history]
            
            axes[1, 0].plot(time_steps, visual_contrib, 'b-', label='视觉')
            axes[1, 0].plot(time_steps, motion_contrib, 'r-', label='运动')
            axes[1, 0].plot(time_steps, goal_contrib, 'g-', label='目标')
            axes[1, 0].set_title('多模态贡献度演化')
            axes[1, 0].set_xlabel('时间步')
            axes[1, 0].set_ylabel('贡献度')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 注意力稳定性分析
        if len(attention_history) > 1:
            stability_scores = []
            for i in range(1, len(attention_history)):
                # 计算相邻时间步注意力的相似性
                prev_attn = attention_history[i-1].get('overall_attention', 0)
                curr_attn = attention_history[i].get('overall_attention', 0)
                stability = 1.0 - abs(prev_attn - curr_attn)
                stability_scores.append(stability)
            
            axes[1, 1].plot(range(1, len(attention_history)), stability_scores, 'purple')
            axes[1, 1].set_title('注意力稳定性')
            axes[1, 1].set_xlabel('时间步')
            axes[1, 1].set_ylabel('稳定性分数')
            axes[1, 1].grid(True)
        
        # 整体注意力分布
        if len(attention_history) > 0:
            final_attention = attention_history[-1]
            attention_components = ['视觉', '多模态', '时序', '规划', '控制']
            attention_values = [
                final_attention.get('vision_attention', 0.2),
                final_attention.get('multimodal_attention', 0.2),
                final_attention.get('temporal_attention', 0.2),
                final_attention.get('planning_attention', 0.2),
                final_attention.get('control_attention', 0.2)
            ]
            
            axes[1, 2].pie(attention_values, labels=attention_components, autopct='%1.1f%%')
            axes[1, 2].set_title('最终注意力分布')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

class COMPLEX_ATTENTION_TRAINER:
    """
    复杂注意力模型训练器
    专门为注意力机制优化的训练流程
    """
    def __init__(self, args=None):
        self.args = args
        if self.args is not None:
            # 继承基础训练参数
            self.device = args.device
            self.basedir = args.basedir
            self.logdir = args.logdir
            self.datadir = args.datadir
            self.ws_suffix = args.ws_suffix
            self.dataset_name = args.dataset
            self.short = args.short
            self.val_split = args.val_split
            self.seed = args.seed
            self.load_checkpoint = args.load_checkpoint
            self.checkpoint_path = args.checkpoint_path
            self.lr = args.lr
            self.N_eps = args.N_eps
            self.lr_warmup_epochs = args.lr_warmup_epochs
            self.lr_decay = args.lr_decay
            self.save_model_freq = args.save_model_freq
            self.val_freq = args.val_freq
            
            # 注意力训练特定参数
            self.attention_visualization_freq = getattr(args, 'attention_visualization_freq', 25)
            self.staged_training = getattr(args, 'staged_training', True)
            self.attention_loss_scheduling = getattr(args, 'attention_loss_scheduling', True)
            self.early_stopping_patience = getattr(args, 'early_stopping_patience', 30)
            
        else:
            raise Exception("No arguments provided")

        # 创建工作空间
        expname = datetime.now().strftime('complex_attention_d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname)
        wkspc_ctr = 2
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, expname + f'_{str(wkspc_ctr)}')
            wkspc_ctr += 1
        self.workspace = self.workspace + self.ws_suffix
        os.makedirs(self.workspace)
        
        # 初始化记录器和可视化
        self.writer = SummaryWriter(self.workspace)
        self.visualizer = AttentionVisualizationManager(opj(self.workspace, 'attention_visualizations'))
        
        # 保存配置
        if self.args is not None:
            f = opj(self.workspace, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
        
        # 初始化日志
        f = opj(self.workspace, 'log.txt')
        self.logfile = open(f, 'w')
        self.mylogger(f'[复杂注意力训练器] 创建工作空间 {self.workspace}')

        # 数据加载
        self.dataset_dir = opj(self.datadir, self.dataset_name)
        self.load_data()
        
        # 初始化模型和训练组件
        self.setup_model_and_training()
        
        # 训练统计
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.attention_history = []
        self.previous_model_outputs = None
        
    def mylogger(self, msg):
        """日志记录"""
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        formatted_msg = f"{timestamp} {msg}"
        print(formatted_msg)
        self.logfile.write(formatted_msg + '\n')
        self.logfile.flush()
            
    def load_data(self):
        """数据加载 - 与原始版本保持一致"""
        if self.load_checkpoint:
            try:
                try:
                    train_val_dirs_obj = np.load(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.npy'), allow_pickle=True)
                    if isinstance(train_val_dirs_obj, np.ndarray) and train_val_dirs_obj.dtype == object:
                        train_val_dirs = tuple(train_val_dirs_obj)
                    else:
                        train_val_dirs = tuple(train_val_dirs_obj)
                except (ValueError, OSError):
                    import pickle
                    with open(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.pkl'), 'rb') as f:
                        train_val_dirs = pickle.load(f)
                    self.mylogger('[数据加载] 使用pickle格式加载train_val_dirs')
            except Exception as e:
                self.mylogger(f'[数据加载] 无法加载train_val_dirs: {str(e)}，将重新生成')
                train_val_dirs = None
        else:
            train_val_dirs = None

        train_data, val_data, is_png, (self.train_dirs, self.val_dirs) = dataloader(
            opj(self.basedir, self.dataset_dir),
            val_split=self.val_split,
            short=self.short,
            seed=self.seed,
            train_val_dirs=train_val_dirs
        )
        
        # 解包数据
        self.train_meta, self.train_ims, self.train_trajlength, self.train_desvel, self.train_currquat, self.train_currctbr = train_data
        self.val_meta, self.val_ims, self.val_trajlength, self.val_desvel, self.val_currquat, self.val_currctbr = val_data
        
        # 生成目标方向
        self.train_target_dir = self._generate_target_directions(self.train_ims.shape[0])
        self.val_target_dir = self._generate_target_directions(self.val_ims.shape[0])
        
        # 预加载到设备
        self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr, self.train_target_dir = preload(
            (self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr, self.train_target_dir),
            self.device
        )
        self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr, self.val_target_dir = preload(
            (self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr, self.val_target_dir),
            self.device
        )
        
        # 提取速度指令
        self.train_velcmd = self.train_meta[:, range(13, 16) if is_png else range(12, 15)]
        self.val_velcmd = self.val_meta[:, range(13, 16) if is_png else range(12, 15)]
        
        self.mylogger(f'[数据加载] 训练集: {self.train_ims.shape[0]} 图像, 验证集: {self.val_ims.shape[0]} 图像')
        
    def _generate_target_directions(self, num_samples):
        """生成目标方向"""
        target_directions = np.zeros((num_samples, 3))
        target_directions[:, 0] = 1.0  # 前进方向
        noise_scale = 0.15  # 稍微增加噪声以提高泛化
        target_directions += np.random.normal(0, noise_scale, target_directions.shape)
        norms = np.linalg.norm(target_directions, axis=1, keepdims=True)
        target_directions = target_directions / (norms + 1e-8)
        return target_directions.astype(np.float32)
        
    def setup_model_and_training(self):
        """设置模型和训练组件"""
        self.mylogger('[设置] 初始化复杂注意力可解释模型')
        
        # 初始化复杂注意力模型
        self.model = ComplexAttentionInterpretableModel(sequence_length=5).to(self.device).float()
        
        # 注意力感知损失函数
        self.attention_loss = AttentionAwareLoss().to(self.device)
        
        # 优化器 - 为注意力机制调整
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=5e-5,  # 更小的权重衰减
            betas=(0.9, 0.98)   # 稍微调整beta值
        )
        
        # 学习率调度器 - 更复杂的调度策略
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # 初始周期
            T_mult=2,  # 周期倍增因子
            eta_min=self.lr * 0.01
        )
        
        # 训练统计
        self.num_training_steps = self.train_trajlength.shape[0]
        self.num_val_steps = self.val_trajlength.shape[0]
        self.num_eps_trained = 0
        self.total_its = 0
        
        if self.load_checkpoint:
            self.load_from_checkpoint()
            
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.mylogger(f'[设置] 模型总参数数量: {total_params:,}')
        
        # 与其他模型对比
        lstmnet_vit_params = 3563663
        convnet_params = 235269
        self.mylogger(f'[设置] LSTMNetVIT参数数量: {lstmnet_vit_params:,}')
        self.mylogger(f'[设置] ConvNet参数数量: {convnet_params:,}')  
        self.mylogger(f'[设置] 参数比例: {total_params/lstmnet_vit_params:.1f}x LSTMNetVIT, {total_params/convnet_params:.1f}x ConvNet')
            
    def load_from_checkpoint(self):
        """从检查点加载模型"""
        try:
            self.num_eps_trained = int(self.checkpoint_path[-10:-4])
        except:
            self.num_eps_trained = 0
            
        self.mylogger(f'[设置] 从检查点加载: {self.checkpoint_path}')
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            if 'attention_history' in checkpoint:
                self.attention_history = checkpoint['attention_history']
        else:
            self.model.load_state_dict(checkpoint)
            
    def save_model(self, ep, is_best=False):
        """保存模型和训练状态"""
        suffix = '_best' if is_best else ''
        self.mylogger(f'[保存] 保存模型 (轮次 {ep}){" [最佳]" if is_best else ""}')
        
        checkpoint = {
            'epoch': ep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_weights': self.attention_loss.task_weights,
            'best_val_loss': self.best_val_loss,
            'attention_history': self.attention_history[-100:] if len(self.attention_history) > 100 else self.attention_history  # 保存最近100个
        }
        
        save_path = opj(self.workspace, f'complex_attention_model_{str(ep).zfill(6)}{suffix}.pth')
        torch.save(checkpoint, save_path)
        
        # 保存训练验证分割
        try:
            train_val_dirs_obj = np.array([self.train_dirs, self.val_dirs], dtype=object)
            np.save(opj(self.workspace, 'train_val_dirs.npy'), train_val_dirs_obj)
        except Exception as e:
            import pickle
            with open(opj(self.workspace, 'train_val_dirs.pkl'), 'wb') as f:
                pickle.dump((self.train_dirs, self.val_dirs), f)
            self.mylogger(f'[保存] 使用pickle保存train_val_dirs: {str(e)}')
    
    def adjust_attention_loss_weights(self, epoch):
        """动态调整注意力损失权重"""
        progress = epoch / self.N_eps
        
        if self.attention_loss_scheduling:
            # 分阶段训练策略
            if progress < 0.3:  # 前30%：重点训练基础功能
                self.attention_loss.task_weights['control'] = 4.0
                self.attention_loss.task_weights['perception'] = 1.0
                self.attention_loss.task_weights['attention_consistency'] = 0.2
                self.attention_loss.task_weights['attention_sparsity'] = 0.1
            elif progress < 0.7:  # 中40%：注重注意力机制
                self.attention_loss.task_weights['control'] = 3.0
                self.attention_loss.task_weights['perception'] = 0.8
                self.attention_loss.task_weights['attention_consistency'] = 0.6
                self.attention_loss.task_weights['attention_sparsity'] = 0.4
                self.attention_loss.task_weights['temporal_stability'] = 0.5
            else:  # 后30%：全面优化
                self.attention_loss.task_weights['control'] = 2.5
                self.attention_loss.task_weights['perception'] = 0.8
                self.attention_loss.task_weights['attention_consistency'] = 0.8
                self.attention_loss.task_weights['attention_sparsity'] = 0.5
                self.attention_loss.task_weights['temporal_stability'] = 0.6
                self.attention_loss.task_weights['multimodal_alignment'] = 0.8
        
    def train(self):
        """主训练循环"""
        self.mylogger(f'[训练] 开始复杂注意力模型训练，共 {self.N_eps} 轮')
        train_start = time.time()
        
        # 计算轨迹起始位置
        train_traj_starts = np.cumsum(self.train_trajlength) - self.train_trajlength
        
        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):
            epoch_start = time.time()
            
            # 动态调整损失权重
            self.adjust_attention_loss_weights(ep)
            
            # 定期保存和验证
            if ep % self.save_model_freq == 0 and ep > self.num_eps_trained:
                self.save_model(ep)
                
            # 验证和早停检查
            if ep % self.val_freq == 0:
                val_loss = self.validation(ep)
                
                # 早停逻辑
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_model(ep, is_best=True)
                    self.mylogger(f'[训练] 新的最佳验证损失: {val_loss:.6f}')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        self.mylogger(f'[训练] 早停触发，验证损失未改善 {self.patience_counter} 次')
                        break
                
            # 注意力可视化
            if ep % self.attention_visualization_freq == 0 and ep > 0:
                self.visualize_attention_progress(ep)
                
            # 初始化轮次统计
            epoch_metrics = {}
            
            # 随机打乱轨迹
            shuffled_indices = np.random.permutation(len(train_traj_starts))
            
            ### 训练循环 ###
            self.model.train()
            hidden_state = None  # 初始化隐状态
            
            for it in range(self.num_training_steps):
                self.optimizer.zero_grad()
                
                # 获取轨迹数据
                traj_idx = shuffled_indices[it]
                start_idx = train_traj_starts[traj_idx] + 1
                end_idx = train_traj_starts[traj_idx] + self.train_trajlength[traj_idx]
                
                # 限制批次大小以节省内存
                if end_idx - start_idx > 12:  # 更小的批次大小
                    end_idx = start_idx + 12
                
                # 准备输入
                inputs = [
                    self.train_ims[start_idx:end_idx, :, :].unsqueeze(1),
                    self.train_desvel[start_idx:end_idx].view(-1, 1),
                    self.train_currquat[start_idx:end_idx],
                    self.train_target_dir[start_idx:end_idx]
                ]
                
                # 前向传播
                velocity_cmd, intermediates = self.model(inputs, hidden_state, return_intermediates=True)
                
                # 准备真值数据
                gt_cmd = self.train_velcmd[start_idx:end_idx, :]  
                gt_cmd_norm = gt_cmd / (inputs[1] + 1e-8)
                
                ground_truth = {
                    'velocity_cmd': gt_cmd_norm,
                    'depth_images': inputs[0]
                }
                
                model_outputs = {
                    'velocity_cmd': velocity_cmd,
                    'intermediates': intermediates
                }
                
                # 计算注意力感知损失
                total_loss, loss_details = self.attention_loss(
                    model_outputs, ground_truth, self.previous_model_outputs
                )
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                # 存储当前输出用于时序稳定性
                self.previous_model_outputs = {
                    'velocity_cmd': velocity_cmd.detach(),
                    'intermediates': {k: v.detach() if torch.is_tensor(v) else v 
                                   for k, v in intermediates.items()}
                }
                
                # 更新统计
                for key, value in loss_details.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0
                    if isinstance(value, (int, float)):
                        epoch_metrics[key] += value
                    elif torch.is_tensor(value):
                        epoch_metrics[key] += value.item()
                    
                self.total_its += 1
                
                # 清理内存
                if it % 5 == 0:
                    torch.cuda.empty_cache()
                
            # 学习率调度
            self.lr_scheduler.step()
                
            # 计算轮次平均值
            for key in epoch_metrics:
                epoch_metrics[key] /= self.num_training_steps
                
            # 记录日志
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - train_start
            
            self.mylogger(
                f'[训练] 轮次 {ep + 1}/{self.num_eps_trained + self.N_eps} | '
                f'总损失: {epoch_metrics["total_loss"]:.6f} | '
                f'控制: {epoch_metrics["control_loss"]:.6f} | '
                f'感知: {epoch_metrics["perception_loss"]:.6f} | '
                f'注意力一致性: {epoch_metrics.get("consistency_loss", 0):.6f} | '
                f'注意力稀疏性: {epoch_metrics.get("sparsity_loss", 0):.6f} | '
                f'时序稳定性: {epoch_metrics.get("stability_loss", 0):.6f} | '
                f'多模态对齐: {epoch_metrics.get("alignment_loss", 0):.6f} | '
                f'安全: {epoch_metrics.get("safety_loss", 0):.6f} | '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.2e} | '
                f'轮次时间: {epoch_time:.1f}s | '
                f'总时间: {elapsed_time:.1f}s'
            )
            
            # TensorBoard记录
            for key, value in epoch_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, ep)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], ep)
            self.writer.add_scalar('train/epoch_time', epoch_time, ep)
            
            # 记录注意力历史
            if len(epoch_metrics) > 0:
                attention_record = {
                    'epoch': ep,
                    'vision_entropy': epoch_metrics.get('entropy_penalty', 0),
                    'planning_confidence': 1.0 - epoch_metrics.get('planning_entropy_consistency', 0),
                    'safety_attention': epoch_metrics.get('safety_attention', 0),
                    'overall_attention': epoch_metrics.get('sparsity_loss', 0)
                }
                self.attention_history.append(attention_record)
            
        # 训练完成
        final_time = time.time() - train_start
        self.mylogger(f'[训练] 完成训练，总用时: {final_time:.2f}s')
        self.mylogger(f'[训练] 最佳验证损失: {self.best_val_loss:.6f}')
        self.save_model(self.num_eps_trained + self.N_eps - 1)
        
        # 生成最终注意力演化分析
        if len(self.attention_history) > 1:
            self.visualizer.visualize_attention_evolution(
                self.attention_history,
                opj(self.workspace, 'final_attention_evolution.png')
            )
        
    def visualize_attention_progress(self, epoch):
        """可视化注意力训练进度"""
        self.mylogger(f'[可视化] 生成注意力分析可视化 (轮次 {epoch})')
        
        try:
            with torch.no_grad():
                self.model.eval()
                
                # 取验证集样本
                sample_input = [
                    self.val_ims[:1, :, :].unsqueeze(1),
                    self.val_desvel[:1].view(-1, 1),
                    self.val_currquat[:1],
                    self.val_target_dir[:1]
                ]
                
                velocity_cmd, intermediates = self.model(sample_input, return_intermediates=True)
                
                model_outputs = {
                    'velocity_cmd': velocity_cmd,
                    'intermediates': intermediates
                }
                
                # 获取注意力分析
                attention_analysis = self.model.get_attention_analysis(sample_input)
                
                # 保存可视化结果
                vis_dir = opj(self.workspace, 'attention_visualizations', f'epoch_{epoch:06d}')
                os.makedirs(vis_dir, exist_ok=True)
                
                # 全面的注意力可视化
                self.visualizer.visualize_attention_comprehensive(
                    sample_input[0], model_outputs, attention_analysis,
                    opj(vis_dir, 'comprehensive_attention_analysis.png')
                )
                
                # 生成详细的决策解释
                explanation = self.model.explain_decision(sample_input)
                with open(opj(vis_dir, 'detailed_explanation.txt'), 'w', encoding='utf-8') as f:
                    f.write(explanation)
                    
                # 保存注意力权重数据
                attention_data = {
                    'epoch': epoch,
                    'attention_analysis': attention_analysis,
                    'model_outputs': model_outputs
                }
                torch.save(attention_data, opj(vis_dir, 'attention_data.pth'))
                    
        except Exception as e:
            self.mylogger(f'[可视化] 注意力可视化过程中出现错误: {str(e)}')
            import traceback
            traceback.print_exc()
                
    def validation(self, ep):
        """验证函数"""
        self.mylogger(f'[验证] 开始验证 (轮次 {ep})')
        
        val_start = time.time()
        val_metrics = {}
        
        with torch.no_grad():
            self.model.eval()
            
            val_traj_starts = np.cumsum(self.val_trajlength) - self.val_trajlength
            hidden_state = None
            
            num_val_batches = min(self.num_val_steps, 25)  # 限制验证步数
            for it in range(num_val_batches):
                start_idx = val_traj_starts[it] + 1
                end_idx = val_traj_starts[it] + self.val_trajlength[it]
                
                # 限制验证批次大小
                if end_idx - start_idx > 6:
                    end_idx = start_idx + 6
                
                inputs = [
                    self.val_ims[start_idx:end_idx, :, :].unsqueeze(1),
                    self.val_desvel[start_idx:end_idx].view(-1, 1),
                    self.val_currquat[start_idx:end_idx],
                    self.val_target_dir[start_idx:end_idx]
                ]
                
                velocity_cmd, intermediates = self.model(inputs, hidden_state, return_intermediates=True)
                
                gt_cmd = self.val_velcmd[start_idx:end_idx, :]
                gt_cmd_norm = gt_cmd / (inputs[1] + 1e-8)
                
                ground_truth = {
                    'velocity_cmd': gt_cmd_norm,
                    'depth_images': inputs[0]
                }
                
                model_outputs = {
                    'velocity_cmd': velocity_cmd,
                    'intermediates': intermediates
                }
                
                total_loss, loss_details = self.attention_loss(
                    model_outputs, ground_truth, None  # 验证时不使用previous_outputs
                )
                
                for key, value in loss_details.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0
                    if isinstance(value, (int, float)):
                        val_metrics[key] += value
                    elif torch.is_tensor(value):
                        val_metrics[key] += value.item()
                    
            # 计算平均值
            for key in val_metrics:
                val_metrics[key] /= num_val_batches
                
            # 记录验证结果
            val_time = time.time() - val_start
            self.mylogger(
                f'[验证] 完成验证 | '
                f'总损失: {val_metrics["total_loss"]:.6f} | '
                f'控制: {val_metrics["control_loss"]:.6f} | '
                f'感知: {val_metrics["perception_loss"]:.6f} | '
                f'注意力一致性: {val_metrics.get("consistency_loss", 0):.6f} | '
                f'注意力稀疏性: {val_metrics.get("sparsity_loss", 0):.6f} | '
                f'时序稳定性: {val_metrics.get("stability_loss", 0):.6f} | '
                f'多模态对齐: {val_metrics.get("alignment_loss", 0):.6f} | '
                f'安全: {val_metrics.get("safety_loss", 0):.6f} | '
                f'用时: {val_time:.2f}s'
            )
            
            # TensorBoard记录
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, ep)
                
        self.writer.flush()
        return val_metrics["total_loss"]

def argparsing():
    """参数解析"""
    import configargparse
    parser = configargparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument('--config', is_config_file=True, default='config/complex_attention.yaml', help='配置文件路径')
    parser.add_argument('--basedir', type=str, default=f'/home/{getpass.getuser()}/ws/vitfly_ws/src/vitfly', help='基础目录')
    parser.add_argument('--logdir', type=str, default='training/logs', help='日志目录')
    parser.add_argument('--datadir', type=str, default=f'/home/{getpass.getuser()}/ws/vitfly_ws/src/vitfly/training/datasets', help='数据目录')
    parser.add_argument('--ws_suffix', type=str, default='', help='工作空间后缀')
    parser.add_argument('--dataset', type=str, default='data', help='数据集名称')
    parser.add_argument('--short', type=int, default=0, help='短数据集大小')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='加载检查点')
    parser.add_argument('--checkpoint_path', type=str, default='', help='检查点路径')
    
    # 训练参数 - 针对复杂注意力模型调整
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')  # 较低的学习率
    parser.add_argument('--N_eps', type=int, default=200, help='训练轮数')  # 更多训练轮数
    parser.add_argument('--lr_warmup_epochs', type=int, default=15, help='预热轮数')
    parser.add_argument('--lr_decay', action='store_true', default=True, help='学习率衰减')
    parser.add_argument('--save_model_freq', type=int, default=25, help='保存频率')
    parser.add_argument('--val_freq', type=int, default=5, help='验证频率')
    
    # 复杂注意力训练特有参数
    parser.add_argument('--attention_visualization_freq', type=int, default=25, help='注意力可视化频率')
    parser.add_argument('--staged_training', action='store_true', default=True, help='分阶段训练')
    parser.add_argument('--attention_loss_scheduling', action='store_true', default=True, help='注意力损失调度')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='早停耐心值')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # 设置CUDA内存管理 - 更保守的设置
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.6)  # 使用60%的GPU内存
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    args = argparsing()
    print("=== 复杂注意力可解释训练配置 ===")
    print(f"基于LSTMNetVIT成功设计: ✓")
    print(f"多层次注意力机制: ✓")
    print(f"分阶段训练策略: {args.staged_training}")
    print(f"注意力损失调度: {args.attention_loss_scheduling}")
    print(f"注意力可视化频率: {args.attention_visualization_freq}")
    print(f"早停机制: {args.early_stopping_patience} 轮")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.N_eps}")
    
    try:
        trainer = COMPLEX_ATTENTION_TRAINER(args)
        trainer.train()
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()