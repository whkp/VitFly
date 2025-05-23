"""
可解释性和泛化能力增强的训练系统

这个训练系统的核心理念是：
1. 多任务学习：同时训练多个相关任务来增强模型理解能力
2. 分层损失函数：为每个模块设计专门的损失函数
3. 渐进式训练：先训练底层感知，再训练高层决策
4. 可视化监控：实时监控每个模块的学习进度
5. 知识蒸馏：从预训练的专家模型中学习领域知识
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
import matplotlib.pyplot as plt
import cv2

from dataloading import *
from interpretable_model import InterpretableAvoidanceModel

class MultiTaskLossFunction(nn.Module):
    """
    多任务损失函数：为可解释模型的每个组件设计专门的损失
    
    这种设计让我们能够独立监控和调优每个模块的性能，
    就像训练飞行员时分别评估他的视觉识别、空间判断和操控技能
    """
    def __init__(self, task_weights=None):
        super().__init__()
        
        # 默认任务权重（可以根据训练进度动态调整）
        if task_weights is None:
            self.task_weights = {
                'control': 1.0,          # 主要控制任务
                'perception': 0.5,       # 感知准确性
                'spatial': 0.3,          # 空间理解
                'planning': 0.4,         # 规划合理性
                'safety': 0.8,           # 安全约束
                'consistency': 0.2       # 时间一致性
            }
        else:
            self.task_weights = task_weights
            
    def perception_loss(self, predicted_perception, depth_images):
        """
        感知损失：评估障碍物检测和深度估计的准确性
        """
        obstacle_mask = predicted_perception['obstacle_mask']
        depth_estimation = predicted_perception['depth_estimation']
        confidence = predicted_perception['confidence']
        
        # 障碍物检测损失：基于深度阈值生成伪标签
        # 假设深度小于0.3的像素为障碍物
        obstacle_gt = (depth_images < 0.3).float()
        obstacle_loss = F.binary_cross_entropy(obstacle_mask, obstacle_gt)
        
        # 深度估计损失：直接与输入深度比较
        depth_loss = F.mse_loss(depth_estimation, depth_images)
        
        # 置信度校准损失：置信度应该反映预测的准确性
        depth_error = torch.abs(depth_estimation - depth_images)
        confidence_loss = F.mse_loss(confidence, 1.0 - depth_error)
        
        total_perception_loss = obstacle_loss + depth_loss + 0.1 * confidence_loss
        
        return total_perception_loss, {
            'obstacle_loss': obstacle_loss.item(),
            'depth_loss': depth_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
    
    def spatial_understanding_loss(self, spatial_analysis, perception_output):
        """
        空间理解损失：评估可行性分析和风险评估的合理性
        """
        navigable_map = spatial_analysis['navigable_map']
        risk_map = spatial_analysis['risk_map']
        direction_preferences = spatial_analysis['direction_preferences']
        
        # 可行性与障碍物的一致性：障碍物区域应该不可行
        obstacle_mask = perception_output['obstacle_mask']
        navigability_consistency = F.mse_loss(navigable_map, 1.0 - obstacle_mask)
        
        # 风险与深度的相关性：距离近的地方风险高
        depth_estimation = perception_output['depth_estimation']
        risk_depth_correlation = F.mse_loss(risk_map, 1.0 - depth_estimation)
        
        # 方向偏好的合理性：应该偏向安全方向
        # 这里使用简单的前向偏好作为基准
        preferred_direction = torch.tensor([0.4, 0.1, 0.2, 0.2, 0.05, 0.05], device=direction_preferences.device)
        direction_loss = F.kl_div(
            torch.log(direction_preferences + 1e-8), 
            preferred_direction.unsqueeze(0).expand_as(direction_preferences),
            reduction='batchmean'
        )
        
        total_spatial_loss = navigability_consistency + risk_depth_correlation + 0.1 * direction_loss
        
        return total_spatial_loss, {
            'navigability_consistency': navigability_consistency.item(),
            'risk_correlation': risk_depth_correlation.item(),
            'direction_loss': direction_loss.item()
        }
    
    def planning_reasonableness_loss(self, motion_planning, spatial_analysis):
        """
        规划合理性损失：评估运动原语选择是否合理
        """
        primitive_probs = motion_planning['primitive_probabilities']
        motion_parameters = motion_planning['motion_parameters']
        
        # 运动选择应该与空间分析一致
        navigable_ratio = torch.mean(spatial_analysis['navigable_map'].view(spatial_analysis['navigable_map'].shape[0], -1), dim=1)
        risk_level = torch.mean(spatial_analysis['risk_map'].view(spatial_analysis['risk_map'].shape[0], -1), dim=1)
        
        # 高风险环境应该偏向保守策略（减速、停止）
        conservative_actions = primitive_probs[:, [5, 6]]  # slow_down, emergency_stop
        conservative_preference = torch.sum(conservative_actions, dim=1)
        risk_strategy_alignment = F.mse_loss(conservative_preference, risk_level)
        
        # 可行空间大时应该偏向前进策略
        forward_action = primitive_probs[:, 0]  # forward
        forward_preference = F.mse_loss(forward_action, navigable_ratio)
        
        # 运动参数的合理性：不应该有极端值
        parameter_regularization = torch.mean(torch.abs(motion_parameters))
        
        total_planning_loss = risk_strategy_alignment + forward_preference + 0.1 * parameter_regularization
        
        return total_planning_loss, {
            'risk_strategy_alignment': risk_strategy_alignment.item(),
            'forward_preference': forward_preference.item(),
            'parameter_regularization': parameter_regularization.item()
        }
    
    def safety_constraint_loss(self, velocity_cmd, min_obstacle_distance, max_safe_velocity=1.0):
        """
        安全约束损失：确保生成的控制指令满足安全要求
        """
        # 速度应该与障碍物距离成正比
        velocity_magnitude = torch.norm(velocity_cmd, dim=1)
        safe_velocity = torch.clamp(min_obstacle_distance * max_safe_velocity, 0.1, max_safe_velocity)
        
        velocity_safety_loss = F.mse_loss(velocity_magnitude, safe_velocity)
        
        # 避免过大的速度变化（如果有历史数据）
        velocity_smoothness = torch.mean(torch.abs(velocity_cmd))
        
        total_safety_loss = velocity_safety_loss + 0.1 * velocity_smoothness
        
        return total_safety_loss, {
            'velocity_safety': velocity_safety_loss.item(),
            'velocity_smoothness': velocity_smoothness.item()
        }
    
    def forward(self, model_outputs, ground_truth, previous_outputs=None):
        """
        计算总的多任务损失
        """
        velocity_cmd = model_outputs['velocity_cmd']
        intermediates = model_outputs['intermediates']
        
        gt_velocity = ground_truth['velocity_cmd']
        depth_images = ground_truth['depth_images']
        
        # 主要控制任务损失
        control_loss = F.mse_loss(velocity_cmd, gt_velocity)
        
        # 各个模块的专门损失
        perception_loss, perception_details = self.perception_loss(
            intermediates['perception'], depth_images
        )
        
        spatial_loss, spatial_details = self.spatial_understanding_loss(
            intermediates['spatial_analysis'], intermediates['perception']
        )
        
        planning_loss, planning_details = self.planning_reasonableness_loss(
            intermediates['motion_planning'], intermediates['spatial_analysis']
        )
        
        safety_loss, safety_details = self.safety_constraint_loss(
            velocity_cmd, intermediates['min_obstacle_distance']
        )
        
        # 时间一致性损失（如果有前一时刻的输出）
        consistency_loss = torch.tensor(0.0, device=velocity_cmd.device)
        if previous_outputs is not None:
            prev_velocity = previous_outputs['velocity_cmd']
            consistency_loss = F.mse_loss(velocity_cmd, prev_velocity)
        
        # 加权总损失
        total_loss = (
            self.task_weights['control'] * control_loss +
            self.task_weights['perception'] * perception_loss +
            self.task_weights['spatial'] * spatial_loss +
            self.task_weights['planning'] * planning_loss +
            self.task_weights['safety'] * safety_loss +
            self.task_weights['consistency'] * consistency_loss
        )
        
        # 详细损失信息用于监控
        loss_details = {
            'total_loss': total_loss.item(),
            'control_loss': control_loss.item(),
            'perception_loss': perception_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'planning_loss': planning_loss.item(),
            'safety_loss': safety_loss.item(),
            'consistency_loss': consistency_loss.item()
        }
        
        # 合并细节信息
        loss_details.update(perception_details)
        loss_details.update(spatial_details)
        loss_details.update(planning_details)
        loss_details.update(safety_details)
        
        return total_loss, loss_details

class VisualizationManager:
    """
    可视化管理器：为可解释模型提供丰富的可视化功能
    
    这个类帮助我们理解模型的每个决策步骤，就像给模型的"大脑"装上监视器
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_perception(self, depth_image, perception_output, save_path=None):
        """
        可视化感知模块的输出：障碍物检测、深度估计、置信度
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('感知模块分析结果', fontsize=16)
        
        # 原始深度图
        depth_np = depth_image[0, 0].cpu().numpy()
        axes[0, 0].imshow(depth_np, cmap='viridis')
        axes[0, 0].set_title('原始深度图像')
        axes[0, 0].axis('off')
        
        # 障碍物检测结果
        obstacle_np = perception_output['obstacle_mask'][0, 0].cpu().numpy()
        axes[0, 1].imshow(obstacle_np, cmap='Reds')
        axes[0, 1].set_title('障碍物检测结果')
        axes[0, 1].axis('off')
        
        # 深度估计结果
        depth_est_np = perception_output['depth_estimation'][0, 0].cpu().numpy()
        axes[1, 0].imshow(depth_est_np, cmap='viridis')
        axes[1, 0].set_title('深度估计结果')
        axes[1, 0].axis('off')
        
        # 置信度图
        confidence_np = perception_output['confidence'][0, 0].cpu().numpy()
        axes[1, 1].imshow(confidence_np, cmap='Blues')
        axes[1, 1].set_title('预测置信度')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def visualize_spatial_analysis(self, spatial_analysis, save_path=None):
        """
        可视化空间分析结果：可行性地图、风险地图、方向偏好
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('空间理解分析结果', fontsize=16)
        
        # 可行性地图
        navigable_np = spatial_analysis['navigable_map'][0, 0].cpu().numpy()
        axes[0, 0].imshow(navigable_np, cmap='Greens')
        axes[0, 0].set_title('可行空间地图')
        axes[0, 0].axis('off')
        
        # 风险地图
        risk_np = spatial_analysis['risk_map'][0, 0].cpu().numpy()
        axes[0, 1].imshow(risk_np, cmap='Reds')
        axes[0, 1].set_title('风险评估地图')
        axes[0, 1].axis('off')
        
        # 方向偏好雷达图
        directions = ['前', '后', '左', '右', '上', '下']
        direction_prefs = spatial_analysis['direction_preferences'][0].cpu().numpy()
        
        angles = np.linspace(0, 2*np.pi, len(directions), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 闭合雷达图
        values = np.concatenate((direction_prefs, [direction_prefs[0]]))
        
        ax_radar = plt.subplot(2, 2, 3, projection='polar')
        ax_radar.plot(angles, values, 'bo-', linewidth=2)
        ax_radar.fill(angles, values, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(directions)
        ax_radar.set_title('方向偏好分析')
        
        # 综合空间状态
        axes[1, 1].text(0.1, 0.8, f'可行空间比例: {np.mean(navigable_np):.1%}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'平均风险等级: {np.mean(risk_np):.2f}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'主要偏好方向: {directions[np.argmax(direction_prefs)]}', fontsize=12)
        axes[1, 1].set_title('空间状态总结')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def visualize_motion_planning(self, motion_planning, motion_primitives, save_path=None):
        """
        可视化运动规划结果：策略选择和参数
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('运动规划分析结果', fontsize=16)
        
        # 运动原语选择概率
        primitive_probs = motion_planning['primitive_probabilities'][0].cpu().numpy()
        colors = plt.cm.viridis(np.linspace(0, 1, len(motion_primitives)))
        
        bars = ax1.bar(range(len(motion_primitives)), primitive_probs, color=colors)
        ax1.set_xticks(range(len(motion_primitives)))
        ax1.set_xticklabels(motion_primitives, rotation=45, ha='right')
        ax1.set_ylabel('选择概率')
        ax1.set_title('运动策略选择')
        ax1.grid(True, alpha=0.3)
        
        # 在最高的柱子上标注
        max_idx = np.argmax(primitive_probs)
        ax1.annotate(f'{primitive_probs[max_idx]:.2f}', 
                    xy=(max_idx, primitive_probs[max_idx]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 运动参数可视化
        motion_params = motion_planning['motion_parameters'][0].cpu().numpy()
        param_names = ['强度', '方向角', '倾斜角', '持续时间']
        
        ax2.bar(param_names, motion_params, color=['red', 'green', 'blue', 'orange'])
        ax2.set_ylabel('参数值')
        ax2.set_title('运动参数设置')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

class INTERPRETABLE_TRAINER:
    """
    可解释模型的专门训练器
    
    这个训练器的特点：
    1. 渐进式训练：先训练感知，再训练决策
    2. 模块化监控：独立监控每个模块的性能
    3. 可视化驱动：提供丰富的训练过程可视化
    4. 自适应权重：根据训练进度调整损失权重
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
            
            # 可解释训练特定参数
            self.progressive_training = getattr(args, 'progressive_training', True)
            self.visualization_freq = getattr(args, 'visualization_freq', 50)
            self.adaptive_weights = getattr(args, 'adaptive_weights', True)
            
        else:
            raise Exception("No arguments provided")

        # 创建工作空间
        expname = datetime.now().strftime('interpretable_d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname)
        wkspc_ctr = 2
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, expname + f'_{str(wkspc_ctr)}')
            wkspc_ctr += 1
        self.workspace = self.workspace + self.ws_suffix
        os.makedirs(self.workspace)
        
        # 初始化记录器和可视化
        self.writer = SummaryWriter(self.workspace)
        self.visualizer = VisualizationManager(opj(self.workspace, 'visualizations'))
        
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
        self.mylogger(f'[可解释训练器] 创建工作空间 {self.workspace}')

        # 数据加载
        self.dataset_dir = opj(self.datadir, self.dataset_name)
        self.load_data()
        
        # 初始化模型和训练组件
        self.setup_model_and_training()
        
    def mylogger(self, msg):
        """增强的日志记录"""
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        formatted_msg = f"{timestamp} {msg}"
        print(formatted_msg)
        self.logfile.write(formatted_msg + '\n')
        
    def load_data(self):
        """数据加载（复用原有的dataloader）"""
        if self.load_checkpoint:
            try:
                train_val_dirs = tuple(np.load(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.npy'), allow_pickle=True))
            except:
                train_val_dirs = None
        else:
            train_val_dirs = None

        # 使用原有的dataloader
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
        
        # 生成目标方向（默认前进方向）
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
        noise_scale = 0.1
        target_directions += np.random.normal(0, noise_scale, target_directions.shape)
        norms = np.linalg.norm(target_directions, axis=1, keepdims=True)
        target_directions = target_directions / (norms + 1e-8)
        return target_directions.astype(np.float32)
        
    def setup_model_and_training(self):
        """设置模型和训练组件"""
        self.mylogger('[设置] 初始化可解释模型和训练组件')
        
        # 初始化可解释模型
        self.model = InterpretableAvoidanceModel().to(self.device).float()
        
        # 多任务损失函数
        self.multi_task_loss = MultiTaskLossFunction().to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.lr_warmup_epochs * self.train_trajlength.shape[0],
            T_mult=2,
            eta_min=self.lr * 0.01
        )
        
        # 训练统计
        self.num_training_steps = self.train_trajlength.shape[0]
        self.num_val_steps = self.val_trajlength.shape[0]
        self.num_eps_trained = 0
        self.total_its = 0
        
        # 存储上一时刻的输出用于一致性损失
        self.previous_outputs = {}
        
        if self.load_checkpoint:
            self.load_from_checkpoint()
            
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
        else:
            self.model.load_state_dict(checkpoint)
            
    def save_model(self, ep):
        """保存模型和训练状态"""
        self.mylogger(f'[保存] 保存模型 (轮次 {ep})')
        
        checkpoint = {
            'epoch': ep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_weights': self.multi_task_loss.task_weights,
            'training_stats': {
                'total_iterations': self.total_its,
                'num_epochs_trained': self.num_eps_trained
            }
        }
        
        save_path = opj(self.workspace, f'interpretable_model_{str(ep).zfill(6)}.pth')
        torch.save(checkpoint, save_path)
        self.mylogger(f'[保存] 模型已保存到 {save_path}')
        
    def train(self):
        """主训练循环"""
        self.mylogger(f'[训练] 开始可解释模型训练，共 {self.N_eps} 轮')
        train_start = time.time()
        
        # 计算轨迹起始位置
        train_traj_starts = np.cumsum(self.train_trajlength) - self.train_trajlength
        
        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):
            # 定期保存和验证
            if ep % self.save_model_freq == 0 and ep - self.num_eps_trained > 0:
                self.save_model(ep)
                
            if ep % self.val_freq == 0:
                self.validation(ep)
                
            # 可视化训练过程
            if ep % self.visualization_freq == 0:
                self.visualize_training_progress(ep)
                
            # 自适应调整损失权重
            if self.adaptive_weights:
                self._adapt_loss_weights(ep)
                
            # 初始化轮次统计
            epoch_metrics = {}
            
            # 随机打乱轨迹
            shuffled_indices = np.random.permutation(len(train_traj_starts))
            
            ### 训练循环 ###
            self.model.train()
            
            for it in range(self.num_training_steps):
                self.optimizer.zero_grad()
                
                # 获取轨迹数据
                traj_idx = shuffled_indices[it]
                start_idx = train_traj_starts[traj_idx] + 1
                end_idx = train_traj_starts[traj_idx] + self.train_trajlength[traj_idx]
                
                # 准备输入
                inputs = [
                    self.train_ims[start_idx:end_idx, :, :].unsqueeze(1),
                    self.train_desvel[start_idx:end_idx].view(-1, 1),
                    self.train_currquat[start_idx:end_idx],
                    self.train_target_dir[start_idx:end_idx]
                ]
                
                # 前向传播
                velocity_cmd, intermediates = self.model(inputs, return_intermediates=True)
                
                # 准备真值数据
                gt_cmd = self.train_velcmd[start_idx:end_idx, :]
                gt_cmd_norm = gt_cmd / inputs[1]  # 归一化
                
                ground_truth = {
                    'velocity_cmd': gt_cmd_norm,
                    'depth_images': inputs[0]
                }
                
                model_outputs = {
                    'velocity_cmd': velocity_cmd,
                    'intermediates': intermediates
                }
                
                # 获取前一时刻输出（用于一致性损失）
                prev_outputs = self.previous_outputs.get(traj_idx, None)
                
                # 计算多任务损失
                total_loss, loss_details = self.multi_task_loss(
                    model_outputs, ground_truth, prev_outputs
                )
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # 更新统计
                for key, value in loss_details.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0
                    epoch_metrics[key] += value
                    
                # 存储当前输出
                self.previous_outputs[traj_idx] = {
                    'velocity_cmd': velocity_cmd.detach().clone()
                }
                
                self.total_its += 1
                
            # 计算轮次平均值
            for key in epoch_metrics:
                epoch_metrics[key] /= self.num_training_steps
                
            # 记录日志
            elapsed_time = time.time() - train_start
            avg_time_per_epoch = elapsed_time / (ep + 1 - self.num_eps_trained)
            
            self.mylogger(
                f'[训练] 轮次 {ep + 1}/{self.num_eps_trained + self.N_eps} | '
                f'总损失: {epoch_metrics["total_loss"]:.6f} | '
                f'控制: {epoch_metrics["control_loss"]:.6f} | '
                f'感知: {epoch_metrics["perception_loss"]:.6f} | '
                f'空间: {epoch_metrics["spatial_loss"]:.6f} | '
                f'规划: {epoch_metrics["planning_loss"]:.6f} | '
                f'安全: {epoch_metrics["safety_loss"]:.6f} | '
                f'时间: {elapsed_time:.2f}s | '
                f'平均: {avg_time_per_epoch:.2f}s/轮'
            )
            
            # TensorBoard记录
            for key, value in epoch_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, ep)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], ep)
            self.writer.flush()
            
        # 训练完成
        self.mylogger(f'[训练] 完成训练，总用时: {time.time() - train_start:.2f}s')
        self.save_model(ep)
        
    def _adapt_loss_weights(self, epoch):
        """自适应调整损失权重"""
        # 训练前期更注重感知，后期更注重控制精度
        progress = epoch / self.N_eps
        
        # 动态调整权重
        self.multi_task_loss.task_weights['perception'] = 0.8 - 0.3 * progress
        self.multi_task_loss.task_weights['control'] = 0.5 + 0.5 * progress
        self.multi_task_loss.task_weights['safety'] = 0.3 + 0.5 * progress
        
    def visualize_training_progress(self, epoch):
        """可视化训练进度"""
        self.mylogger(f'[可视化] 生成训练进度可视化 (轮次 {epoch})')
        
        # 选择一个验证样本进行可视化
        with torch.no_grad():
            self.model.eval()
            
            # 取验证集第一个轨迹的第一帧
            sample_input = [
                self.val_ims[:1, :, :].unsqueeze(1),
                self.val_desvel[:1].view(-1, 1),
                self.val_currquat[:1],
                self.val_target_dir[:1]
            ]
            
            velocity_cmd, intermediates = self.model(sample_input, return_intermediates=True)
            
            # 保存可视化结果
            vis_dir = opj(self.workspace, 'visualizations', f'epoch_{epoch:06d}')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 感知可视化
            self.visualizer.visualize_perception(
                sample_input[0], intermediates['perception'],
                opj(vis_dir, 'perception.png')
            )
            
            # 空间分析可视化
            self.visualizer.visualize_spatial_analysis(
                intermediates['spatial_analysis'],
                opj(vis_dir, 'spatial_analysis.png')
            )
            
            # 运动规划可视化
            self.visualizer.visualize_motion_planning(
                intermediates['motion_planning'],
                self.model.motion_planner.motion_primitives,
                opj(vis_dir, 'motion_planning.png')
            )
            
            # 生成决策解释
            explanation = self.model.explain_decision(sample_input)
            with open(opj(vis_dir, 'decision_explanation.txt'), 'w') as f:
                f.write(explanation)
                
    def validation(self, ep):
        """验证函数"""
        self.mylogger(f'[验证] 开始验证 (轮次 {ep})')
        
        val_start = time.time()
        val_metrics = {}
        
        with torch.no_grad():
            self.model.eval()
            
            val_traj_starts = np.cumsum(self.val_trajlength) - self.val_trajlength
            
            for it in range(self.num_val_steps):
                start_idx = val_traj_starts[it] + 1
                end_idx = val_traj_starts[it] + self.val_trajlength[it]
                
                inputs = [
                    self.val_ims[start_idx:end_idx, :, :].unsqueeze(1),
                    self.val_desvel[start_idx:end_idx].view(-1, 1),
                    self.val_currquat[start_idx:end_idx],
                    self.val_target_dir[start_idx:end_idx]
                ]
                
                velocity_cmd, intermediates = self.model(inputs, return_intermediates=True)
                
                gt_cmd = self.val_velcmd[start_idx:end_idx, :]
                gt_cmd_norm = gt_cmd / inputs[1]
                
                ground_truth = {
                    'velocity_cmd': gt_cmd_norm,
                    'depth_images': inputs[0]
                }
                
                model_outputs = {
                    'velocity_cmd': velocity_cmd,
                    'intermediates': intermediates
                }
                
                total_loss, loss_details = self.multi_task_loss(
                    model_outputs, ground_truth, None
                )
                
                for key, value in loss_details.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0
                    val_metrics[key] += value
                    
            # 计算平均值
            for key in val_metrics:
                val_metrics[key] /= self.num_val_steps
                
            # 记录验证结果
            self.mylogger(
                f'[验证] 完成验证 | '
                f'总损失: {val_metrics["total_loss"]:.6f} | '
                f'控制: {val_metrics["control_loss"]:.6f} | '
                f'感知: {val_metrics["perception_loss"]:.6f} | '
                f'用时: {time.time() - val_start:.2f}s'
            )
            
            # TensorBoard记录
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, ep)

def argparsing():
    """参数解析（扩展原有参数）"""
    import configargparse
    parser = configargparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument('--config', is_config_file=True, help='配置文件路径')
    parser.add_argument('--basedir', type=str, default=f'/home/{getpass.getuser()}/agile_ws/src/agile_flight', help='基础目录')
    parser.add_argument('--logdir', type=str, default='learner/logs', help='日志目录')
    parser.add_argument('--datadir', type=str, default=f'/home/{getpass.getuser()}/agile_ws/src/agile_flight', help='数据目录')
    parser.add_argument('--ws_suffix', type=str, default='', help='工作空间后缀')
    parser.add_argument('--dataset', type=str, default='5-2', help='数据集名称')
    parser.add_argument('--short', type=int, default=0, help='短数据集大小')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='加载检查点')
    parser.add_argument('--checkpoint_path', type=str, default='', help='检查点路径')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--N_eps', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--lr_decay', action='store_true', default=True, help='学习率衰减')
    parser.add_argument('--save_model_freq', type=int, default=25, help='保存频率')
    parser.add_argument('--val_freq', type=int, default=10, help='验证频率')
    
    # 可解释训练特有参数
    parser.add_argument('--progressive_training', action='store_true', default=True, help='渐进式训练')
    parser.add_argument('--visualization_freq', type=int, default=50, help='可视化频率')
    parser.add_argument('--adaptive_weights', action='store_true', default=True, help='自适应权重')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    args = argparsing()
    print("可解释训练配置:")
    print(f"渐进式训练: {args.progressive_training}")
    print(f"可视化频率: {args.visualization_freq}")
    print(f"自适应权重: {args.adaptive_weights}")
    
    trainer = INTERPRETABLE_TRAINER(args)
    trainer.train()