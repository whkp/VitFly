"""
基于ConvNet成功设计的优化训练系统

改进要点：
1. 简化损失函数，聚焦核心任务
2. 借鉴ConvNet的训练策略
3. 优化数据流和内存使用
4. 增强可解释性监控
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

# 修复matplotlib Qt错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import getpass

from dataloading import *
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
# 导入优化后的模型
try:
    from optimized_light_model import OptimizedInterpretableModel
except ImportError:
    print("Warning: Using fallback model import")
    from light_model import CompleteInterpretableModel as OptimizedInterpretableModel

class SimpleEffectiveLoss(nn.Module):
    """
    基于ConvNet成功经验的简化损失函数
    聚焦核心任务，避免过度复杂化
    """
    def __init__(self, task_weights=None):
        super().__init__()
        
        # 简化的任务权重，借鉴ConvNet的直接设计
        if task_weights is None:
            self.task_weights = {
                'control': 2.0,          # 主要控制任务，权重最高
                'perception': 0.5,       # 感知辅助任务
                'consistency': 0.3,      # 内部一致性
                'safety': 1.0,           # 安全约束
            }
        else:
            self.task_weights = task_weights
            
    def perception_auxiliary_loss(self, predicted_perception, depth_images):
        """
        简化的感知辅助损失 - 不强求完美，只要有助于控制
        """
        obstacle_mask = predicted_perception['obstacle_mask']
        depth_estimation = predicted_perception['depth_estimation']
        confidence = predicted_perception['confidence']
        
        # 调整尺寸匹配
        _, _, h_out, w_out = obstacle_mask.shape
        depth_images_resized = F.interpolate(
            depth_images, 
            size=(h_out, w_out), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 障碍物检测损失（宽松标准）
        obstacle_gt = (depth_images_resized < 0.4).float()  # 更宽松的阈值
        obstacle_loss = F.binary_cross_entropy(obstacle_mask, obstacle_gt, reduction='mean')
        
        # 深度估计损失（使用平滑L1损失，对异常值更鲁棒）
        depth_loss = F.smooth_l1_loss(depth_estimation, depth_images_resized)
        
        # 置信度损失（简化）
        depth_error = torch.abs(depth_estimation - depth_images_resized)
        confidence_target = torch.exp(-depth_error)  # 误差越小，置信度越高
        confidence_loss = F.mse_loss(confidence, confidence_target)
        
        total_perception_loss = obstacle_loss + 0.5 * depth_loss + 0.1 * confidence_loss
        
        return total_perception_loss, {
            'obstacle_loss': obstacle_loss.item(),
            'depth_loss': depth_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
    
    def consistency_loss(self, model_outputs):
        """
        内部一致性损失 - 确保不同组件的输出相互支持
        """
        intermediates = model_outputs['intermediates']
        velocity_cmd = model_outputs['velocity_cmd']
        
        # 感知与规划一致性
        obstacle_density = torch.mean(intermediates['perception']['obstacle_mask'], dim=[2, 3])
        primitive_probs = intermediates['motion_planning']['primitive_probabilities']
        
        # 障碍物多时应该偏向保守策略（slow_down, emergency_stop）
        conservative_probs = primitive_probs[:, [6, 7]].sum(dim=1)  # 假设这些是保守策略的索引
        obstacle_conservative_consistency = F.mse_loss(
            conservative_probs, obstacle_density.squeeze()
        )
        
        # 速度与风险一致性
        velocity_magnitude = torch.norm(velocity_cmd, dim=1)
        risk_level = torch.mean(intermediates['spatial_analysis']['risk_map'], dim=[2, 3]).squeeze()
        velocity_risk_consistency = F.mse_loss(
            velocity_magnitude, (1.0 - risk_level) * 2.0  # 风险越高，速度应该越低
        )
        
        total_consistency_loss = obstacle_conservative_consistency + 0.5 * velocity_risk_consistency
        
        return total_consistency_loss, {
            'obstacle_conservative_consistency': obstacle_conservative_consistency.item(),
            'velocity_risk_consistency': velocity_risk_consistency.item()
        }
    
    def safety_constraint_loss(self, velocity_cmd, min_obstacle_distance, max_safe_velocity=2.0):
        """
        安全约束损失 - 借鉴ConvNet的简单有效设计
        """
        velocity_magnitude = torch.norm(velocity_cmd, dim=1)
        
        # 简单的距离-速度关系
        safe_velocity = torch.clamp(min_obstacle_distance * max_safe_velocity, 0.1, max_safe_velocity)
        velocity_safety_loss = F.smooth_l1_loss(velocity_magnitude, safe_velocity)
        
        # 速度平滑性（避免突变）
        velocity_smoothness = torch.mean(torch.abs(velocity_cmd))
        
        # 速度合理性（避免过大值）
        velocity_magnitude_penalty = F.relu(velocity_magnitude - max_safe_velocity).mean()
        
        total_safety_loss = (velocity_safety_loss + 
                           0.1 * velocity_smoothness + 
                           0.5 * velocity_magnitude_penalty)
        
        return total_safety_loss, {
            'velocity_safety': velocity_safety_loss.item(),
            'velocity_smoothness': velocity_smoothness.item(),
            'velocity_magnitude_penalty': velocity_magnitude_penalty.item()
        }
    
    def forward(self, model_outputs, ground_truth):
        """
        计算总的多任务损失 - 简化但有效
        """
        velocity_cmd = model_outputs['velocity_cmd']
        intermediates = model_outputs['intermediates']
        
        gt_velocity = ground_truth['velocity_cmd']
        depth_images = ground_truth['depth_images']
        
        # 1. 主要控制任务损失（使用平滑L1损失，更鲁棒）
        control_loss = F.smooth_l1_loss(velocity_cmd, gt_velocity)
        
        # 2. 感知辅助损失
        perception_loss, perception_details = self.perception_auxiliary_loss(
            intermediates['perception'], depth_images
        )
        
        # 3. 内部一致性损失
        consistency_loss, consistency_details = self.consistency_loss(model_outputs)
        
        # 4. 安全约束损失
        safety_loss, safety_details = self.safety_constraint_loss(
            velocity_cmd, intermediates['min_obstacle_distance']
        )
        
        # 5. 加权总损失（简化权重策略）
        total_loss = (
            self.task_weights['control'] * control_loss +
            self.task_weights['perception'] * perception_loss +
            self.task_weights['consistency'] * consistency_loss +
            self.task_weights['safety'] * safety_loss
        )
        
        # 详细损失信息
        loss_details = {
            'total_loss': total_loss.item(),
            'control_loss': control_loss.item(),
            'perception_loss': perception_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'safety_loss': safety_loss.item(),
        }
        
        # 合并细节信息
        loss_details.update(perception_details)
        loss_details.update(consistency_details)
        loss_details.update(safety_details)
        
        return total_loss, loss_details

class EnhancedVisualizationManager:
    """
    增强的可视化管理器 - 专门适配优化模型
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_model_analysis(self, depth_image, model_outputs, save_path=None):
        """
        全面的模型分析可视化
        """
        intermediates = model_outputs['intermediates']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('优化可解释模型分析结果', fontsize=16)
        
        # 原始深度图
        depth_np = depth_image[0, 0].cpu().numpy()
        im1 = axes[0, 0].imshow(depth_np, cmap='viridis')
        axes[0, 0].set_title('原始深度图像')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # 障碍物检测
        obstacle_np = intermediates['perception']['obstacle_mask'][0, 0].cpu().numpy()
        im2 = axes[0, 1].imshow(obstacle_np, cmap='Reds', vmin=0, vmax=1)
        axes[0, 1].set_title('障碍物检测')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # 深度估计
        depth_est_np = intermediates['perception']['depth_estimation'][0, 0].cpu().numpy()
        im3 = axes[0, 2].imshow(depth_est_np, cmap='viridis')
        axes[0, 2].set_title('深度估计')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # 置信度图
        confidence_np = intermediates['perception']['confidence'][0, 0].cpu().numpy()
        im4 = axes[0, 3].imshow(confidence_np, cmap='Blues', vmin=0, vmax=1)
        axes[0, 3].set_title('预测置信度')
        axes[0, 3].axis('off')
        plt.colorbar(im4, ax=axes[0, 3], fraction=0.046)
        
        # 可导航性地图
        navigable_np = intermediates['spatial_analysis']['navigable_map'][0, 0].cpu().numpy()
        im5 = axes[1, 0].imshow(navigable_np, cmap='Greens', vmin=0, vmax=1)
        axes[1, 0].set_title('可导航区域')
        axes[1, 0].axis('off')
        plt.colorbar(im5, ax=axes[1, 0], fraction=0.046)
        
        # 风险地图
        risk_np = intermediates['spatial_analysis']['risk_map'][0, 0].cpu().numpy()
        im6 = axes[1, 1].imshow(risk_np, cmap='Reds', vmin=0, vmax=1)
        axes[1, 1].set_title('风险评估')
        axes[1, 1].axis('off')
        plt.colorbar(im6, ax=axes[1, 1], fraction=0.046)
        
        # 运动原语选择
        if 'motion_planning' in intermediates:
            motion_probs = intermediates['motion_planning']['primitive_probabilities'][0].cpu().numpy()
            motion_primitives = ['前进', '后退', '左转', '右转', '上升', '下降', '减速', '急停']
            
            bars = axes[1, 2].bar(range(len(motion_primitives)), motion_probs)
            axes[1, 2].set_xticks(range(len(motion_primitives)))
            axes[1, 2].set_xticklabels(motion_primitives, rotation=45, ha='right')
            axes[1, 2].set_ylabel('选择概率')
            axes[1, 2].set_title('运动策略选择')
            axes[1, 2].grid(True, alpha=0.3)
            
            # 标注最高概率
            max_idx = np.argmax(motion_probs)
            axes[1, 2].text(max_idx, motion_probs[max_idx] + 0.01, 
                           f'{motion_probs[max_idx]:.2f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # 方向偏好分析
        if 'direction_preferences' in intermediates['spatial_analysis']:
            direction_prefs = intermediates['spatial_analysis']['direction_preferences'][0].cpu().numpy()
            directions = ['前', '后', '左', '右', '上', '下']
            
            axes[1, 3].bar(range(len(directions)), direction_prefs)
            axes[1, 3].set_xticks(range(len(directions)))
            axes[1, 3].set_xticklabels(directions)
            axes[1, 3].set_title('方向偏好')
            axes[1, 3].set_ylabel('偏好度')
            axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

class OPTIMIZED_INTERPRETABLE_TRAINER:
    """
    优化的可解释模型训练器
    基于ConvNet成功经验设计
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
            
            # 优化训练特定参数
            self.visualization_freq = getattr(args, 'visualization_freq', 50)
            self.adaptive_weights = getattr(args, 'adaptive_weights', True)
            self.early_stopping_patience = getattr(args, 'early_stopping_patience', 20)
            
        else:
            raise Exception("No arguments provided")

        # 创建工作空间
        expname = datetime.now().strftime('optimized_interpretable_d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname)
        wkspc_ctr = 2
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, expname + f'_{str(wkspc_ctr)}')
            wkspc_ctr += 1
        self.workspace = self.workspace + self.ws_suffix
        os.makedirs(self.workspace)
        
        # 初始化记录器和可视化
        self.writer = SummaryWriter(self.workspace)
        self.visualizer = EnhancedVisualizationManager(opj(self.workspace, 'visualizations'))
        
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
        self.mylogger(f'[优化训练器] 创建工作空间 {self.workspace}')

        # 数据加载
        self.dataset_dir = opj(self.datadir, self.dataset_name)
        self.load_data()
        
        # 初始化模型和训练组件
        self.setup_model_and_training()
        
        # 训练统计
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
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
        noise_scale = 0.1
        target_directions += np.random.normal(0, noise_scale, target_directions.shape)
        norms = np.linalg.norm(target_directions, axis=1, keepdims=True)
        target_directions = target_directions / (norms + 1e-8)
        return target_directions.astype(np.float32)
        
    def setup_model_and_training(self):
        """设置模型和训练组件"""
        self.mylogger('[设置] 初始化优化可解释模型')
        
        # 初始化优化模型
        self.model = OptimizedInterpretableModel().to(self.device).float()
        
        # 简化损失函数
        self.multi_task_loss = SimpleEffectiveLoss().to(self.device)
        
        # 优化器 - 借鉴ConvNet的设置但稍作调整
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,  # 适度的权重衰减
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器 - 使用更平滑的调度
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.N_eps,
            pct_start=0.1,  # 10%时间用于warmup
            anneal_strategy='cos'
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
        
        # 与ConvNet对比
        convnet_params = 235269
        self.mylogger(f'[设置] ConvNet参数数量: {convnet_params:,}')
        self.mylogger(f'[设置] 参数比例: {total_params/convnet_params:.1f}x ConvNet')
            
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
            'loss_weights': self.multi_task_loss.task_weights,
            'best_val_loss': self.best_val_loss,
        }
        
        save_path = opj(self.workspace, f'optimized_model_{str(ep).zfill(6)}{suffix}.pth')
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
        
    def train(self):
        """主训练循环"""
        self.mylogger(f'[训练] 开始优化可解释模型训练，共 {self.N_eps} 轮')
        train_start = time.time()
        
        # 计算轨迹起始位置
        train_traj_starts = np.cumsum(self.train_trajlength) - self.train_trajlength
        
        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):
            epoch_start = time.time()
            
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
                
            # 可视化训练过程
            if ep % self.visualization_freq == 0 and ep > 0:
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
                
                # 限制批次大小以节省内存
                if end_idx - start_idx > 16:  # 进一步减小批次大小
                    end_idx = start_idx + 16
                
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
                gt_cmd_norm = gt_cmd / (inputs[1] + 1e-8)
                
                ground_truth = {
                    'velocity_cmd': gt_cmd_norm,
                    'depth_images': inputs[0]
                }
                
                model_outputs = {
                    'velocity_cmd': velocity_cmd,
                    'intermediates': intermediates
                }
                
                # 计算损失
                total_loss, loss_details = self.multi_task_loss(
                    model_outputs, ground_truth
                )
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # 梯度裁剪
                self.optimizer.step()
                
                # 更新统计
                for key, value in loss_details.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0
                    epoch_metrics[key] += value
                    
                self.total_its += 1
                
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
                f'一致性: {epoch_metrics.get("consistency_loss", 0):.6f} | '
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
            
            # 内存清理
            if ep % 5 == 0:
                torch.cuda.empty_cache()
            
        # 训练完成
        final_time = time.time() - train_start
        self.mylogger(f'[训练] 完成训练，总用时: {final_time:.2f}s')
        self.mylogger(f'[训练] 最佳验证损失: {self.best_val_loss:.6f}')
        self.save_model(self.num_eps_trained + self.N_eps - 1)
        
    def _adapt_loss_weights(self, epoch):
        """自适应调整损失权重"""
        progress = epoch / self.N_eps
        
        # 训练前期更注重感知学习，后期更注重控制精度
        self.multi_task_loss.task_weights['perception'] = 0.8 - 0.3 * progress
        self.multi_task_loss.task_weights['control'] = 1.5 + 0.5 * progress
        self.multi_task_loss.task_weights['consistency'] = 0.2 + 0.1 * progress
        
    def visualize_training_progress(self, epoch):
        """可视化训练进度"""
        self.mylogger(f'[可视化] 生成训练进度可视化 (轮次 {epoch})')
        
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
                
                # 保存可视化结果
                vis_dir = opj(self.workspace, 'visualizations', f'epoch_{epoch:06d}')
                os.makedirs(vis_dir, exist_ok=True)
                
                # 全面的模型分析可视化
                self.visualizer.visualize_model_analysis(
                    sample_input[0], model_outputs,
                    opj(vis_dir, 'model_analysis.png')
                )
                
                # 生成决策解释
                explanation = self.model.explain_decision(sample_input)
                with open(opj(vis_dir, 'decision_explanation.txt'), 'w', encoding='utf-8') as f:
                    f.write(explanation)
                    
        except Exception as e:
            self.mylogger(f'[可视化] 可视化过程中出现错误: {str(e)}')
                
    def validation(self, ep):
        """验证函数"""
        self.mylogger(f'[验证] 开始验证 (轮次 {ep})')
        
        val_start = time.time()
        val_metrics = {}
        
        with torch.no_grad():
            self.model.eval()
            
            val_traj_starts = np.cumsum(self.val_trajlength) - self.val_trajlength
            
            num_val_batches = min(self.num_val_steps, 30)  # 限制验证步数
            for it in range(num_val_batches):
                start_idx = val_traj_starts[it] + 1
                end_idx = val_traj_starts[it] + self.val_trajlength[it]
                
                # 限制验证批次大小
                if end_idx - start_idx > 8:
                    end_idx = start_idx + 8
                
                inputs = [
                    self.val_ims[start_idx:end_idx, :, :].unsqueeze(1),
                    self.val_desvel[start_idx:end_idx].view(-1, 1),
                    self.val_currquat[start_idx:end_idx],
                    self.val_target_dir[start_idx:end_idx]
                ]
                
                velocity_cmd, intermediates = self.model(inputs, return_intermediates=True)
                
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
                
                total_loss, loss_details = self.multi_task_loss(
                    model_outputs, ground_truth
                )
                
                for key, value in loss_details.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0
                    val_metrics[key] += value
                    
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
                f'一致性: {val_metrics.get("consistency_loss", 0):.6f} | '
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
    parser.add_argument('--config', is_config_file=True, default='config/optimized_light.yaml', help='配置文件路径')
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
    
    # 训练参数 - 基于ConvNet成功经验调整
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')  # 稍微提高
    parser.add_argument('--N_eps', type=int, default=150, help='训练轮数')  # 增加训练轮数
    parser.add_argument('--lr_warmup_epochs', type=int, default=10, help='预热轮数')
    parser.add_argument('--lr_decay', action='store_true', default=True, help='学习率衰减')
    parser.add_argument('--save_model_freq', type=int, default=20, help='保存频率')
    parser.add_argument('--val_freq', type=int, default=5, help='验证频率')
    
    # 优化训练特有参数
    parser.add_argument('--visualization_freq', type=int, default=50, help='可视化频率')
    parser.add_argument('--adaptive_weights', action='store_true', default=True, help='自适应权重')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='早停耐心值')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # 设置更保守的CUDA内存管理
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)  # 使用70%的GPU内存
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    args = argparsing()
    print("=== 优化可解释训练配置 ===")
    print(f"基于ConvNet成功设计: ✓")
    print(f"简化损失函数策略: ✓")
    print(f"早停机制: {args.early_stopping_patience} 轮")
    print(f"可视化频率: {args.visualization_freq}")
    print(f"自适应权重: {args.adaptive_weights}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.N_eps}")
    
    try:
        trainer = OPTIMIZED_INTERPRETABLE_TRAINER(args)
        trainer.train()
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()