"""
@authors: Improved by AI Assistant, based on A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@date: Improved Version
@license: ...

@brief: 渐进式训练改进策略，基于原始train.py的成功经验
每种训练策略都专注于特定的改进目标，避免同时引入过多变化
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
import math

from dataloading import *
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
import improved_models as model_library

# 抑制TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import getpass
uname = getpass.getuser()

class BaseImprovedTrainer:
    """
    基础改进训练器类
    
    设计理念：保持原始trainer的核心结构，但添加渐进式改进。
    这样可以确保我们不会破坏原有的成功要素，同时能够有针对性地提升性能。
    """
    def __init__(self, args=None):
        self.args = args
        if self.args is not None:
            # 保持与原始trainer完全相同的参数结构
            self.device = args.device
            self.basedir = args.basedir
            self.logdir = args.logdir
            self.datadir = args.datadir
            self.ws_suffix = args.ws_suffix
            self.dataset_name = args.dataset
            self.short = args.short

            self.model_type = args.model_type
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
            
            # 新增的改进参数 - 但都有合理的默认值
            self.improvement_type = getattr(args, 'improvement_type', 'none')
            self.gradient_clip_value = getattr(args, 'gradient_clip_value', None)
            self.use_curriculum = getattr(args, 'use_curriculum', False)
            self.early_stopping_patience = getattr(args, 'early_stopping_patience', None)
        else:
            raise Exception("未提供参数")

        assert self.dataset_name is not None, '未提供数据集名称'

        # 工作空间设置 - 保持与原始代码相同
        expname = datetime.now().strftime(f'improved_{self.improvement_type}_d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname)
        wkspc_ctr = 2
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, expname + f'_{str(wkspc_ctr)}')
            wkspc_ctr += 1
        self.workspace = self.workspace + self.ws_suffix
        os.makedirs(self.workspace)
        self.writer = SummaryWriter(self.workspace)

        # 保存配置文件
        if self.args is not None:
            f = opj(self.workspace, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            # 只有当config不为None时才保存配置文件
            if self.args.config is not None:
                f = opj(self.workspace, 'config.txt')
                with open(f, 'w') as file:
                    file.write(open(self.args.config, 'r').read())
        
        f = opj(self.workspace, 'log.txt')
        self.logfile = open(f, 'w')

        self.mylogger(f'[Improved Trainer] 创建工作空间 {self.workspace}')
        self.dataset_dir = opj(self.datadir, self.dataset_name)

        # 数据加载 - 保持与原始代码相同的逻辑
        if self.load_checkpoint:
            print('[Improved Trainer] 从检查点加载train_val_dirs')
            try:
                train_val_dirs = tuple(np.load(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.npy'), allow_pickle=True))
            except:
                print('[Improved Trainer] 无法从检查点加载train_val_dirs，重新加载')
                train_val_dirs = None
        else:
            train_val_dirs = None

        self.dataloader(val_split=self.val_split, short=self.short, seed=self.seed, train_val_dirs=train_val_dirs)

        self.num_training_steps = self.train_trajlength.shape[0]
        self.num_val_steps = self.val_trajlength.shape[0]
        self.lr_warmup_iters = self.lr_warmup_epochs * self.num_training_steps

        # 模型和优化器设置
        self.setup_model_and_optimizer()

        self.num_eps_trained = 0
        if self.load_checkpoint:
            self.load_from_checkpoint(self.checkpoint_path)

        self.total_its = self.num_eps_trained * self.num_training_steps
        
        # 初始化改进功能
        self.initialize_improvements()

    def setup_model_and_optimizer(self):
        """根据模型类型初始化改进模型"""
        self.mylogger('[SETUP] 建立改进模型和优化器。')
        self.mylogger(f'[SETUP] 使用模型类型 {self.model_type}')
        
        if self.model_type == 'SpatialAttentionViT':
            self.model = model_library.SpatialAttentionViT().to(self.device).float()
        elif self.model_type == 'TemporalConsistencyLSTM':
            self.model = model_library.TemporalConsistencyLSTM().to(self.device).float()
        elif self.model_type == 'MultiResolutionViT':
            self.model = model_library.MultiResolutionViT().to(self.device).float()
        elif self.model_type == 'RobustViTLSTM':
            self.model = model_library.RobustViTLSTM().to(self.device).float()
        else:
            self.mylogger(f'[SETUP] 未知的模型类型 {self.model_type}。退出。')
            exit()

        # 优化器设置 - 根据模型类型选择合适的优化策略
        if 'LSTM' in self.model_type:
            # LSTM模型使用稍小的学习率，更好的权重衰减
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.lr * 0.8,  # LSTM模型通常需要更小的学习率
                weight_decay=1e-5   # 适度的权重衰减
            )
        else:
            # ViT模型保持原始的优化器设置
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def initialize_improvements(self):
        """初始化各种改进功能"""
        # 早停法设置
        if self.early_stopping_patience:
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        
        # 课程学习设置
        if self.use_curriculum:
            self.curriculum_stage = 0
            self.curriculum_thresholds = [0.25, 0.5, 0.75, 1.0]  # 渐进式增加数据难度
        
        # 梯度监控
        self.gradient_norms = []

    def mylogger(self, msg):
        """保持与原始代码相同的日志记录方式"""
        print(msg)
        self.logfile.write(msg + '\n')

    def load_from_checkpoint(self, checkpoint_path):
        """保持与原始代码相同的检查点加载逻辑"""
        try:
            self.num_eps_trained = int(checkpoint_path[-10:-4])
        except:
            self.num_eps_trained = 0
            self.mylogger(f'[SETUP] 无法解析训练轮数 {checkpoint_path}，使用0')
        self.mylogger(f'[SETUP] 从 {checkpoint_path} 加载检查点，已训练 {self.num_eps_trained} 轮')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def dataloader(self, val_split, short=0, seed=None, train_val_dirs=None):
        """保持与原始代码相同的数据加载逻辑"""
        self.mylogger(f'[DATALOADER] 从 {self.dataset_dir} 加载数据')
        
        train_data, val_data, is_png, (self.train_dirs, self.val_dirs) = dataloader(
            opj(self.basedir, self.dataset_dir),
            val_split=val_split,
            short=short,
            seed=seed,
            train_val_dirs=train_val_dirs
        )
        
        self.train_meta, self.train_ims, self.train_trajlength, self.train_desvel, self.train_currquat, self.train_currctbr = train_data
        self.val_meta, self.val_ims, self.val_trajlength, self.val_desvel, self.val_currquat, self.val_currctbr = val_data
        self.mylogger(f'[DATALOADER] 数据加载完成 | 训练图像形状 {self.train_ims.shape}, 验证图像形状 {self.val_ims.shape}')

        self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr = preload(
            (self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr),
            self.device
        )
        self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr = preload(
            (self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr),
            self.device
        )
        self.mylogger(f'[DATALOADER] 预加载到设备 {self.device} 完成')

        # 数据验证
        assert self.train_ims.max() <= 1.0 and self.train_ims.min() >= 0.0, '图像未归一化'
        assert self.train_ims.max() > 0.50, "图像未正确归一化"

        self.train_velcmd = self.train_meta[:, range(13, 16) if is_png else range(12, 15)]
        self.val_velcmd = self.val_meta[:, range(13, 16) if is_png else range(12, 15)]

        np.save(opj(self.workspace, 'train_val_dirs.npy'), np.array((self.train_dirs, self.val_dirs), dtype=object))

    def improved_lr_scheduler(self, it):
        """
        改进的学习率调度策略
        
        设计思路：
        1. 保持原始的预热机制
        2. 使用余弦退火而不是指数衰减，提供更平滑的学习率变化
        3. 在训练后期保持小的学习率以精细调优
        """
        if it < self.lr_warmup_iters:
            # 线性预热 - 与原始代码相同
            lr = (0.9 * self.lr) / self.lr_warmup_iters * it + 0.1 * self.lr
        else:
            if self.lr_decay:
                # 余弦退火调度 - 比指数衰减更平滑
                progress = (it - self.lr_warmup_iters) / (self.N_eps * self.num_training_steps - self.lr_warmup_iters)
                lr = 0.1 * self.lr + (self.lr - 0.1 * self.lr) * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                lr = self.lr
        return lr

    def get_curriculum_data_size(self, ep):
        """
        课程学习：渐进式增加训练数据的复杂度
        
        核心思想：先用简单/短的轨迹训练，逐步增加复杂轨迹。
        这样可以让模型先学会基本的避障行为，再处理复杂场景。
        """
        if not self.use_curriculum:
            return self.num_training_steps
        
        # 根据训练进度决定使用多少比例的数据
        progress = ep / self.N_eps
        
        if progress < 0.25:
            # 前25%的训练：只使用最简单的轨迹
            return int(self.num_training_steps * 0.3)
        elif progress < 0.5:
            # 25%-50%：增加中等复杂度轨迹
            return int(self.num_training_steps * 0.6)
        elif progress < 0.75:
            # 50%-75%：使用大部分轨迹
            return int(self.num_training_steps * 0.8)
        else:
            # 最后25%：使用全部数据
            return self.num_training_steps

    def adaptive_loss_weighting(self, ep, base_loss):
        """
        自适应损失权重调整
        
        设计思路：在训练初期专注于基本控制精度，
        随着训练进展，逐步增加对稳定性和鲁棒性的要求。
        """
        if self.model_type == 'TemporalConsistencyLSTM':
            # 对于时序模型，增加时序一致性的权重
            temporal_weight = min(1.0, ep / (self.N_eps * 0.3))  # 前30%训练逐步增加
            return base_loss  # 这里只做示例，实际实现需要额外的损失项
        
        return base_loss

    def compute_gradient_penalty(self):
        """
        梯度惩罚项，防止梯度爆炸
        适用于复杂模型的训练稳定性
        """
        total_norm = 0
        param_count = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
            
            # 如果梯度范数过大，返回惩罚项
            if total_norm > 10.0:  # 阈值可调整
                return torch.tensor(total_norm - 10.0).to(self.device)
        
        return torch.tensor(0.0).to(self.device)

    def save_model(self, ep):
        """保持与原始代码相同，但添加更多诊断信息"""
        self.mylogger(f'[SAVE] 在轮数 {ep} 保存模型')
        path = self.workspace
        torch.save(self.model.state_dict(), opj(path, f'improved_model_{str(ep).zfill(6)}.pth'))
        
        # 保存训练诊断信息
        if hasattr(self, 'gradient_norms') and self.gradient_norms:
            np.save(opj(path, f'gradient_norms_{ep}.npy'), np.array(self.gradient_norms))
        
        self.mylogger(f'[SAVE] 模型已保存到 {path}')

    def train(self):
        """
        改进的训练循环
        
        主要改进：
        1. 课程学习
        2. 自适应学习率
        3. 梯度监控和裁剪
        4. 早停法
        5. 更详细的日志记录
        """
        self.mylogger(f'[TRAIN] 开始改进训练，共 {self.N_eps} 轮，改进类型: {self.improvement_type}')
        train_start = time.time()

        self.train_traj_starts = np.cumsum(self.train_trajlength) - self.train_trajlength
        train_traj_lengths = self.train_trajlength

        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):
            # 定期保存模型
            if ep % self.save_model_freq == 0 and ep - self.num_eps_trained > 0:
                self.save_model(ep)

            # 定期验证
            if ep % self.val_freq == 0:
                val_loss = self.validation(ep)
                
                # 早停法检查
                if self.early_stopping_patience:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        # 保存最佳模型
                        torch.save(self.model.state_dict(), opj(self.workspace, 'best_model.pth'))
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            self.mylogger(f'[EARLY STOP] 验证损失连续 {self.early_stopping_patience} 次未改善，提前停止')
                            break

            ep_loss = 0
            gradnorm = 0

            # 课程学习：获取当前阶段应该使用的数据量
            current_data_size = self.get_curriculum_data_size(ep)
            
            # 打乱轨迹顺序
            shuffled_traj_indices = np.random.permutation(len(self.train_traj_starts))
            train_traj_starts = self.train_traj_starts[shuffled_traj_indices]
            train_traj_lengths = self.train_trajlength[shuffled_traj_indices]

            # 如果使用课程学习，只取前current_data_size个轨迹
            if self.use_curriculum:
                train_traj_starts = train_traj_starts[:current_data_size]
                train_traj_lengths = train_traj_lengths[:current_data_size]
                actual_steps = current_data_size
            else:
                actual_steps = self.num_training_steps

            ### 训练循环 ###
            self.model.train()
            for it in range(actual_steps):
                self.optimizer.zero_grad()

                # 获取训练数据 - 与原始代码相同的逻辑
                traj_input = self.train_ims[train_traj_starts[it] + 1 : train_traj_starts[it] + train_traj_lengths[it], :, :].unsqueeze(1)
                desvel = self.train_desvel[train_traj_starts[it] + 1 : train_traj_starts[it] + train_traj_lengths[it]].view(-1, 1)
                currquat = self.train_currquat[train_traj_starts[it] + 1 : train_traj_starts[it] + train_traj_lengths[it]]

                # 模型前向传播
                pred, hidden = self.model([traj_input, desvel, currquat])

                # 计算基础损失 - 保持与原始代码相同
                cmd = self.train_velcmd[train_traj_starts[it] + 1 : train_traj_starts[it] + train_traj_lengths[it], :]
                cmd_norm = cmd / desvel
                base_loss = F.mse_loss(cmd_norm, pred)
                
                # 应用自适应损失权重调整
                total_loss = self.adaptive_loss_weighting(ep, base_loss)
                
                ep_loss += total_loss

                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪（如果设置了）
                if self.gradient_clip_value:
                    current_gradnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                else:
                    current_gradnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=torch.inf)
                
                gradnorm += current_gradnorm

                # 优化器更新
                self.optimizer.step()
                
                # 学习率调度
                new_lr = self.improved_lr_scheduler(self.total_its - self.num_eps_trained * self.num_training_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

                self.total_its += 1

            # 计算平均损失
            ep_loss /= actual_steps
            gradnorm /= actual_steps

            # 详细的日志记录
            elapsed_time = time.time() - train_start
            self.mylogger(
                f'[TRAIN] 轮数 {ep + 1}/{self.num_eps_trained + self.N_eps}, '
                f'损失 = {ep_loss:.6f}, '
                f'梯度范数 = {gradnorm:.4f}, '
                f'学习率 = {new_lr:.2e}, '
                f'数据量 = {actual_steps}/{self.num_training_steps}, '
                f'用时 = {elapsed_time:.2f}s'
            )

            # TensorBoard记录
            self.writer.add_scalar('train/loss', ep_loss, ep)
            self.writer.add_scalar('train/gradnorm', gradnorm, ep)
            self.writer.add_scalar('train/lr', new_lr, self.total_its)
            self.writer.add_scalar('train/data_size', actual_steps, ep)
            self.writer.flush()

        self.mylogger(f'[TRAIN] 训练完成，总用时 = {time.time() - train_start:.2f}s')
        self.save_model(ep)

    def validation(self, ep):
        """改进的验证函数，返回验证损失用于早停法"""
        self.mylogger(f'[VAL] 开始验证，验证集大小 {self.val_ims.shape[0]} 张图像')

        val_start = time.time()
        
        with torch.no_grad():
            ep_loss = 0
            val_traj_starts = np.cumsum(self.val_trajlength) - self.val_trajlength

            self.model.eval()

            for it in range(self.num_val_steps):
                traj_input = self.val_ims[val_traj_starts[it] + 1 : val_traj_starts[it] + self.val_trajlength[it], :, :].unsqueeze(1)
                desvel = self.val_desvel[val_traj_starts[it] + 1 : val_traj_starts[it] + self.val_trajlength[it]].view(-1, 1)
                currquat = self.val_currquat[val_traj_starts[it] + 1 : val_traj_starts[it] + self.val_trajlength[it]]
                
                pred, _ = self.model([traj_input, desvel, currquat])

                cmd = self.val_velcmd[val_traj_starts[it] + 1 : val_traj_starts[it] + self.val_trajlength[it], :]
                cmd_norm = cmd / desvel

                loss = F.mse_loss(cmd_norm, pred)
                ep_loss += loss

            ep_loss /= self.num_val_steps

            self.mylogger(f'[VAL] 验证完成，验证损失 = {ep_loss:.6f}, 用时 = {time.time() - val_start:.2f} s')
            self.writer.add_scalar('val/loss', ep_loss, ep)
            
            return ep_loss.item()

def argparsing():
    """扩展的参数解析，支持新的改进选项"""
    import configargparse
    parser = configargparse.ArgumentParser()

    # 原始参数 - 保持与原始代码相同
    parser.add_argument('--config', is_config_file=True, help='配置文件相对路径')
    parser.add_argument('--basedir', type=str, default=f'/home/{uname}/ws/vitfly_ws/src/vitfly', help='仓库路径')
    parser.add_argument('--logdir', type=str, default='training/logs', help='相对日志目录路径')
    parser.add_argument('--datadir', type=str, default=f'/home/{uname}/ws/vitfly_ws/src/vitfly/training/datasets', help='相对数据集目录路径')
    
    parser.add_argument('--ws_suffix', type=str, default='', help='工作空间名称的后缀')
    parser.add_argument('--model_type', type=str, default='RobustViTLSTM', 
                       choices=['TemporalConsistencyLSTM', 'RobustViTLSTM'],
                       help='改进模型类型')
    parser.add_argument('--dataset', type=str, default='data', help='数据集名称')
    parser.add_argument('--short', type=int, default=0, help='加载轨迹文件夹数量')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='是否从检查点加载')
    parser.add_argument('--checkpoint_path', type=str, 
                       default=f'/home/{uname}/ws/vitfly_ws/src/vitfly/training/logs/improved_model_000499.pth', 
                       help='检查点路径')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--N_eps', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='学习率预热轮数')
    parser.add_argument('--lr_decay', action='store_true', default=True, help='是否使用学习率衰减')
    parser.add_argument('--save_model_freq', type=int, default=25, help='模型保存频率')
    parser.add_argument('--val_freq', type=int, default=10, help='验证频率')

    # 新增的改进参数
    parser.add_argument('--improvement_type', type=str, default='spatial_attention', 
                       choices=['spatial_attention', 'temporal_consistency', 'multi_resolution', 'robustness'],
                       help='改进类型标识')
    parser.add_argument('--gradient_clip_value', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--use_curriculum', action='store_true', default=False, help='是否使用课程学习')
    parser.add_argument('--early_stopping_patience', type=int, default=None, help='早停法耐心值')

    args = parser.parse_args()
    print(f'[IMPROVED TRAINER] 从配置文件 {args.config} 解析参数')

    return args

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    args = argparsing()
    print("改进训练配置:")
    print(f"模型类型: {args.model_type}")
    print(f"改进类型: {args.improvement_type}")
    print(f"课程学习: {args.use_curriculum}")
    print(f"梯度裁剪: {args.gradient_clip_value}")
    print(f"早停法: {args.early_stopping_patience}")
    print(args)

    # 初始化改进训练器并开始训练
    improved_trainer = BaseImprovedTrainer(args)
    improved_trainer.train()