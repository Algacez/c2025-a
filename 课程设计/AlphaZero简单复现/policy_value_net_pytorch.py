# -*- coding: utf-8 -*-
"""
PyTorch策略价值网络实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def set_learning_rate(optimizer, lr):
    """设置学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """策略-价值网络模块"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # 公共层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 动作策略层 (Action Policy Layers)
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # 状态价值层 (State Value Layers)
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 公共层前向传播
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 策略头
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # 价值头
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """策略价值网络封装类"""
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # L2 正则化系数
        
        # 自动选择设备
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            try:
                # 确保加载模型到正确的设备
                net_params = torch.load(model_file, map_location=self.device)
                self.policy_value_net.load_state_dict(net_params)
            except Exception as e:
                print("模型加载失败，将使用随机初始化权重。错误信息:", e)

    def policy_value(self, state_batch):
        """
        输入: 一批状态
        输出: 一批动作概率和状态价值
        """
        self.policy_value_net.eval() # 切换到评估模式
        with torch.no_grad():
            state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.cpu().numpy())
            return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """
        输入: 棋盘对象
        输出: (action, probability) 元组列表，以及当前局面的分数
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        
        self.policy_value_net.eval()
        with torch.no_grad():
            current_state_tensor = torch.from_numpy(current_state).float().to(self.device)
            log_act_probs, value = self.policy_value_net(current_state_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            value = value.item()
            
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行一步训练"""
        self.policy_value_net.train() # 切换到训练模式
        
        # 数据转 Tensor 并移至 GPU
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch)).to(self.device)

        # 参数梯度清零
        self.optimizer.zero_grad()
        # 设置学习率
        set_learning_rate(self.optimizer, lr)

        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        # 定义损失函数: Value损失(均方误差) + Policy损失(交叉熵)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # 反向传播和优化
        loss.backward()
        self.optimizer.step()
        # 计算策略熵，仅用于监控
        with torch.no_grad():
            entropy = -torch.mean(
                    torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                    )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """保存模型参数到文件"""
        net_params = self.get_policy_param() 
        torch.save(net_params, model_file)