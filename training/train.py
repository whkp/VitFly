"""
作者: A Bhattacharya
组织机构: 宾夕法尼亚大学GRASP实验室
日期: ...
许可证: ...

简介: 本模块包含在论文“Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance”中使用的训练例程，由Bhattacharya等人编写。
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

from dataloading import *
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
import model as model_library  # 导入自定义模型库

# NOTE: 这行代码用于抑制TensorFlow的警告和信息（尽管在此代码中未使用TensorFlow）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import getpass
uname = getpass.getuser()

# 定义一个训练类，用于训练网络以从深度图像中预测动作
# 训练器可以通过两种方式加载：
# 1. 仅用于数据加载，此时提供dataset_name，通常设置no_model=True
# 2. 用于模型训练，此时提供args参数
class TRAINER:
    def __init__(self, args=None):
        self.args = args
        if self.args is not None:
            # 从参数中获取各种配置
            self.device = args.device  # 设备（如'cuda'）
            self.basedir = args.basedir  # 基础目录路径
            self.logdir = args.logdir  # 日志目录路径
            self.datadir = args.datadir  # 数据目录路径
            self.ws_suffix = args.ws_suffix  # 工作空间后缀
            self.dataset_name = args.dataset  # 数据集名称
            self.short = args.short  # 数据集子集大小

            self.model_type = args.model_type  # 模型类型
            self.val_split = args.val_split  # 验证集比例
            self.seed = args.seed  # 随机种子
            self.load_checkpoint = args.load_checkpoint  # 是否加载检查点
            self.checkpoint_path = args.checkpoint_path  # 检查点路径
            self.lr = args.lr  # 学习率
            self.N_eps = args.N_eps  # 训练轮数
            self.lr_warmup_epochs = args.lr_warmup_epochs  # 学习率预热轮数
            self.lr_decay = args.lr_decay  # 是否使用学习率衰减
            self.save_model_freq = args.save_model_freq  # 模型保存频率
            self.val_freq = args.val_freq  # 验证频率
        else:
            raise Exception("未提供参数")

        # 确保数据集名称已提供
        assert self.dataset_name is not None, '未提供数据集名称，既未通过args也未通过dataset_name参数提供'

        ###############
        ## 工作空间 ##
        ###############

        # 生成一个唯一的实验名称，基于当前日期和时间
        expname = datetime.now().strftime('d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname)
        wkspc_ctr = 2
        # 确保工作空间目录唯一，如果已存在则添加计数器后缀
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, expname + f'_{str(wkspc_ctr)}')
            wkspc_ctr += 1
        self.workspace = self.workspace + self.ws_suffix  # 添加后缀
        os.makedirs(self.workspace)  # 创建工作空间目录
        self.writer = SummaryWriter(self.workspace)  # 初始化TensorBoard记录器

        # 保存有序的参数、配置和日志文件
        if self.args is not None:
            # 保存参数到args.txt
            f = opj(self.workspace, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            # 保存配置文件到config.txt
            f = opj(self.workspace, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())
        # 创建日志文件log.txt
        f = opj(self.workspace, 'log.txt')
        self.logfile = open(f, 'w')

        self.mylogger(f'[LearnerLSTM init] 创建工作空间 {self.workspace}')

        # 数据集目录路径
        self.dataset_dir = opj(self.datadir, self.dataset_name)

        #################
        ## 数据加载 ##
        #################

        if self.load_checkpoint:
            print('[LearnerLSTM init] 从检查点加载train_val_dirs')
            try:
                # 尝试从检查点目录加载训练和验证文件夹列表
                train_val_dirs = tuple(np.load(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.npy'), allow_pickle=True))
            except:
                print('[LearnerLSTM init] 无法从检查点加载train_val_dirs，重新从头加载数据')
                train_val_dirs = None
        else:
            train_val_dirs = None

        # 调用dataloader方法加载数据
        self.dataloader(val_split=self.val_split, short=self.short, seed=self.seed, train_val_dirs=train_val_dirs)

        # TODO: 将num_training_steps硬编码为轨迹数量而不是图像数量
        self.num_training_steps = self.train_trajlength.shape[0]  # 训练步骤数
        self.num_val_steps = self.val_trajlength.shape[0]  # 验证步骤数
        self.lr_warmup_iters = self.lr_warmup_epochs * self.num_training_steps  # 学习率预热迭代次数

        ##################################
        ## 定义网络和优化器 ##
        ##################################

        self.mylogger('[SETUP] 建立模型和优化器。')
        # 根据模型类型初始化不同的模型
        if self.model_type == 'LSTMNet':
            self.model = model_library.LSTMNet().to(self.device).float()
        elif self.model_type == 'ConvNet':
            self.model = model_library.ConvNet().to(self.device).float()
        elif self.model_type == 'ViT':
            self.model = model_library.ViT().to(self.device).float()
        elif self.model_type == 'ViTLSTM':
            self.model = model_library.LSTMNetVIT().to(self.device).float()
        elif self.model_type == 'UNet':
            self.model = model_library.UNetConvLSTMNet().to(self.device).float()
        else:
            self.mylogger(f'[SETUP] 无效的模型类型 {self.model_type}。退出。')
            exit()

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.num_eps_trained = 0  # 已训练的轮数
        if self.load_checkpoint:
            self.load_from_checkpoint(self.checkpoint_path)  # 从检查点加载模型

        self.total_its = self.num_eps_trained * self.num_training_steps  # 总迭代次数

    def mylogger(self, msg):
        """
        自定义日志记录函数，将消息打印到控制台并写入日志文件。
        
        参数:
            msg (str): 要记录的消息。
        """
        print(msg)
        self.logfile.write(msg + '\n')

    def load_from_checkpoint(self, checkpoint_path):
        """
        从检查点加载模型参数和已训练的轮数。
        
        参数:
            checkpoint_path (str): 检查点文件路径。
        """
        try:
            # 尝试从检查点路径中解析已训练的轮数
            self.num_eps_trained = int(checkpoint_path[-10:-4])
        except:
            self.num_eps_trained = 0
            self.mylogger(f'[SETUP] 无法从检查点路径解析已训练的轮数 {checkpoint_path}，使用0')
        self.mylogger(f'[SETUP] 从 {checkpoint_path} 加载检查点，已训练 {self.num_eps_trained} 轮')
        # 加载模型参数
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def dataloader(self, val_split, short=0, seed=None, train_val_dirs=None):
        """
        加载和预处理数据，包括训练集和验证集的划分。
        
        参数:
            val_split (float): 验证集比例。
            short (int): 数据集子集大小，如果大于0，则只加载前short个文件夹。
            seed (int): 随机种子。
            train_val_dirs (tuple): 训练和验证文件夹列表。
        """
        self.mylogger(f'[DATALOADER] 从 {self.dataset_dir} 加载数据')
        # 调用dataloader函数加载数据
        train_data, val_data, is_png, (self.train_dirs, self.val_dirs) = dataloader(
            opj(self.basedir, self.dataset_dir),
            val_split=val_split,
            short=short,
            seed=seed,
            train_val_dirs=train_val_dirs
        )
        # 解包训练集和验证集数据
        self.train_meta, self.train_ims, self.train_trajlength, self.train_desvel, self.train_currquat, self.train_currctbr = train_data
        self.val_meta, self.val_ims, self.val_trajlength, self.val_desvel, self.val_currquat, self.val_currctbr = val_data
        self.mylogger(f'[DATALOADER] 数据加载完成 | 训练图像形状 {self.train_ims.shape}, 验证图像形状 {self.val_ims.shape}')

        # 预加载数据到指定设备
        self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr = preload(
            (self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr),
            self.device
        )
        self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr = preload(
            (self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr),
            self.device
        )
        self.mylogger(f'[DATALOADER] 预加载到设备 {self.device} 完成')

        # 确保图像已归一化到[0.0, 1.0]范围
        assert self.train_ims.max() <= 1.0 and self.train_ims.min() >= 0.0, '图像未归一化（值超出 [0.0, 1.0]）'
        # 确保图像有足够的亮度（最大值大于0.5）
        assert self.train_ims.max() > 0.50, "图像未正确归一化（所有值均低于0.10，可能是因为未对'旧'数据集进行归一化）"

        # 提取速度指令，根据is_png确定列范围
        self.train_velcmd = self.train_meta[:, range(13, 16) if is_png else range(12, 15)]
        self.val_velcmd = self.val_meta[:, range(13, 16) if is_png else range(12, 15)]

        # 保存训练和验证文件夹名称到工作空间，以便后续使用
        np.save(opj(self.workspace, 'train_val_dirs.npy'), np.array((self.train_dirs, self.val_dirs), dtype=object))

    def lr_scheduler(self, it):
        """
        学习率调度函数，根据迭代次数调整学习率。
        
        参数:
            it (int): 当前迭代次数。
        
        返回:
            float: 调整后的学习率。
        """
        if it < self.lr_warmup_iters:
            # 线性预热学习率
            lr = (0.9 * self.lr) / self.lr_warmup_iters * it + 0.1 * self.lr
        else:
            if self.lr_decay:
                # 学习率指数衰减
                lr = self.lr * (0.1 ** ((it - self.lr_warmup_iters) / (self.N_eps * self.num_training_steps)))
            else:
                lr = self.lr
        return lr

    def save_model(self, ep):
        """
        保存当前模型参数到工作空间。
        
        参数:
            ep (int): 当前轮数。
        """
        self.mylogger(f'[SAVE] 在轮数 {ep} 保存模型')
        path = self.workspace
        torch.save(self.model.state_dict(), opj(path, f'model_{str(ep).zfill(6)}.pth'))
        self.mylogger(f'[SAVE] 模型已保存到 {path}')

    def weighted_mse_loss(self, input, target, weight):
        """
        加权均方误差损失函数。
        
        参数:
            input (torch.Tensor): 模型预测输出。
            target (torch.Tensor): 目标值。
            weight (torch.Tensor): 权重。
        
        返回:
            torch.Tensor: 计算得到的损失值。
        """
        return torch.mean(weight * (input - target) ** 2)

    def train(self):
        """
        主训练循环，执行模型的训练过程。
        """
        self.mylogger(f'[TRAIN] 开始训练，共 {self.N_eps} 轮')
        train_start = time.time()

        # 计算训练集中每条轨迹的起始索引
        self.train_traj_starts = np.cumsum(self.train_trajlength) - self.train_trajlength
        train_traj_lengths = self.train_trajlength

        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):
            # 按照指定频率保存模型
            if ep % self.save_model_freq == 0 and ep - self.num_eps_trained > 0:
                self.save_model(ep)

            # 按照指定频率进行验证
            if ep % self.val_freq == 0:
                self.validation(ep)

            ep_loss = 0  # 记录本轮损失
            gradnorm = 0  # 记录梯度范数

            # 随机打乱训练数据的轨迹顺序
            shuffled_traj_indices = np.random.permutation(len(self.train_traj_starts))
            train_traj_starts = self.train_traj_starts[shuffled_traj_indices]
            train_traj_lengths = self.train_trajlength[shuffled_traj_indices]

            ### 训练循环 ###
            self.model.train()  # 设置模型为训练模式
            for it in range(self.num_training_steps):
                self.optimizer.zero_grad()  # 清空梯度

                # 获取当前迭代的输入数据
                traj_input = self.train_ims[train_traj_starts[it] + 1 : train_traj_starts[it] + train_traj_lengths[it], :, :].unsqueeze(1)  # 添加维度以匹配模型输入
                desvel = self.train_desvel[train_traj_starts[it] + 1 : train_traj_starts[it] + train_traj_lengths[it]].view(-1, 1)  # 期望速度，调整形状
                currquat = self.train_currquat[train_traj_starts[it] + 1 : train_traj_starts[it] + train_traj_lengths[it]]  # 当前四元数

                # 模型前向传播，得到预测输出
                pred, _ = self.model([traj_input, desvel, currquat])  # 模型可能返回多个值，这里只取第一个输出

                # 获取当前的速度指令
                cmd = self.train_velcmd[train_traj_starts[it] + 1 : train_traj_starts[it] + train_traj_lengths[it], :]
                # 将速度指令标准化
                cmd_norm = cmd / desvel  # 按每个desvel元素归一化每行

                # 计算均方误差损失
                loss = F.mse_loss(cmd_norm, pred)
                ep_loss += loss  # 累加损失

                # 反向传播计算梯度
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                gradnorm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=torch.inf)
                # 优化器更新参数
                self.optimizer.step()
                # 调整学习率
                new_lr = self.lr_scheduler(self.total_its - self.num_eps_trained * self.num_training_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

                self.total_its += 1  # 增加总迭代次数

            # 计算本轮平均损失和梯度范数
            ep_loss /= self.num_training_steps
            gradnorm /= self.num_training_steps

            # 记录日志
            self.mylogger(f'[TRAIN] 完成轮数 {ep + 1}/{self.num_eps_trained + self.N_eps}, 轮损失 = {ep_loss:.6f}, 用时 = {time.time() - train_start:.2f}s, 每轮用时 = {(time.time() - train_start)/(ep + 1 - self.num_eps_trained):.2f}s')

            # 将损失和梯度范数记录到TensorBoard
            self.writer.add_scalar('train/loss', ep_loss, ep)
            self.writer.add_scalar('train/gradnorm', gradnorm, ep)
            self.writer.add_scalar('train/lr', new_lr, self.total_its)
            self.writer.flush()

        # 训练完成后记录总用时并保存模型
        self.mylogger(f'[TRAIN] 训练完成，总用时 = {time.time() - train_start:.2f}s')
        self.save_model(ep)

    def validation(self, ep):
        """
        验证函数，在验证集上评估模型性能。
        
        参数:
            ep (int): 当前轮数。
        """
        self.mylogger(f'[VAL] 开始验证，验证集大小 {self.val_ims.shape[0]} 张图像')

        val_start = time.time()
        it = 1

        with torch.no_grad():  # 禁用梯度计算，提高效率
            ep_loss = 0  # 记录验证损失

            # 计算验证集中每条轨迹的起始索引
            val_traj_starts = np.cumsum(self.val_trajlength) - self.val_trajlength
            val_traj_starts = np.hstack((val_traj_starts, -1))  # -1作为结束标志

            ### 验证循环 ###
            self.model.eval()  # 设置模型为评估模式

            for it in range(self.num_val_steps):
                # 获取当前验证迭代的输入数据
                traj_input = self.val_ims[val_traj_starts[it] + 1 : val_traj_starts[it] + self.val_trajlength[it], :, :].unsqueeze(1)
                desvel = self.val_desvel[val_traj_starts[it] + 1 : val_traj_starts[it] + self.val_trajlength[it]].view(-1, 1)
                currquat = self.val_currquat[val_traj_starts[it] + 1 : val_traj_starts[it] + self.val_trajlength[it]]
                
                # 模型前向传播，得到预测输出
                pred, _ = self.model([traj_input, desvel, currquat])

                # 获取当前的速度指令并标准化
                cmd = self.val_velcmd[val_traj_starts[it] + 1 : val_traj_starts[it] + self.val_trajlength[it], :]
                cmd_norm = cmd / desvel  # 按每个desvel元素归一化每行

                # 计算均方误差损失
                loss = F.mse_loss(cmd_norm, pred)
                ep_loss += loss  # 累加损失

            # 计算平均验证损失
            ep_loss /= (it + 1)

            # 记录日志
            self.mylogger(f'[VAL] 完成验证，验证损失 = {ep_loss:.6f}, 用时 = {time.time() - val_start:.2f} s')
            # 将验证损失记录到TensorBoard
            self.writer.add_scalar('val/loss', ep_loss, ep)

def argparsing():
    """
    解析命令行参数和配置文件。
    
    返回:
        argparse.Namespace: 解析后的参数对象。
    """
    import configargparse
    parser = configargparse.ArgumentParser()

    # 通用参数
    parser.add_argument('--config', is_config_file=True, help='配置文件相对路径')
    parser.add_argument('--basedir', type=str, default=f'/home/{uname}/agile_ws/src/agile_flight', help='仓库路径')
    parser.add_argument('--logdir', type=str, default='learner/logs', help='相对日志目录路径')
    parser.add_argument('--datadir', type=str, default=f'/home/{uname}/agile_ws/src/agile_flight', help='相对数据集目录路径')
    
    # 实验级别和学习器参数
    parser.add_argument('--ws_suffix', type=str, default='', help='工作空间名称的后缀（如果有）')
    parser.add_argument('--model_type', type=str, default='LSTMNet', help='与lstmArch.py中的模型名称匹配的字符串')
    parser.add_argument('--dataset', type=str, default='5-2', help='数据集名称')
    parser.add_argument('--short', type=int, default=0, help='如果非零，加载多少个轨迹文件夹')
    parser.add_argument('--val_split', type=float, default=0.2, help='用于验证的训练集比例')
    parser.add_argument('--seed', type=int, default=None, help='用于python random, numpy和torch的随机种子 -- 警告，可能未完全实现')
    parser.add_argument('--device', type=str, default='cuda', help='通用CUDA设备；具体GPU应在CUDA_VISIBLE_DEVICES中指定')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='是否从模型检查点加载')
    parser.add_argument('--checkpoint_path', type=str, default=f'/home/{uname}/agile_ws/src/agile_flight/learner/logs/d05_10_t03_13/model_000499.pth', help='模型检查点的绝对路径')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--N_eps', type=int, default=100, help='训练的轮数')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='学习率预热的轮数')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='是否使用学习率衰减，硬编码为训练结束时指数衰减到0.01 * lr')
    parser.add_argument('--save_model_freq', type=int, default=25, help='保存模型检查点的频率（以轮为单位）')
    parser.add_argument('--val_freq', type=int, default=10, help='在验证集上评估的频率（以轮为单位）')

    args = parser.parse_args()
    print(f'[CONFIGARGPARSE] 从配置文件 {args.config} 解析参数')

    return args

if __name__ == '__main__':
    # 设置默认的张量类型为CUDA Float Tensor
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # 解析命令行参数
    args = argparsing()
    print(args)

    # 初始化训练器并开始训练
    learner = TRAINER(args)
    learner.train()
