# -*- coding: utf-8 -*-
"""
训练流程实现
"""

from __future__ import print_function
import random
import numpy as np
import os
import pickle
import logging
import sys
from datetime import datetime
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet # 使用 PyTorch 版本
import torch

class TrainPipeline():
    def __init__(self, init_model=None, checkpoint_file=None, resume_training=False, log_file=None):
        # 棋盘和游戏参数
        self.board_width = 9
        self.board_height = 9
        self.n_in_row = 5  # 5子连珠
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # 训练参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 根据 KL 散度自适应调整学习率
        self.temp = 1.0  # 温度参数
        self.n_playout = 400  # 每次落子的模拟次数
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 2048  # 训练的 mini-batch 大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # 每次更新的训练步数
        self.kl_targ = 0.02
        self.check_freq = 50  # 每多少次对局检查一次性能
        self.game_batch_num = 1000000 # 总训练对局数
        self.best_win_ratio = 0.0
        # 纯 MCTS 的模拟次数，作为评估基准
        self.pure_mcts_playout_num = 1000

        self.model_dir = './models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 训练状态变量
        self.current_batch = 0
        self.training_history = {
            'losses': [],
            'entropies': [],
            'win_ratios': [],
            'kl_divergences': [],
            'lr_multipliers': [],
            'episode_lengths': []
        }

        # 设置日志
        self.setup_logging(log_file)

        # 检查 GPU 可用性
        use_gpu = torch.cuda.is_available()
        self.logger.info(f"训练使用设备: {'GPU' if use_gpu else 'CPU'}")

        # 尝试从checkpoint恢复训练状态
        if resume_training and checkpoint_file and os.path.exists(checkpoint_file):
            self.load_checkpoint(checkpoint_file)
            self.logger.info(f"从断点恢复训练: 批次 {self.current_batch}/{self.game_batch_num}")
        elif resume_training and os.path.exists('training_checkpoint.pkl'):
            self.load_checkpoint('training_checkpoint.pkl')
            self.logger.info(f"从默认断点恢复训练: 批次 {self.current_batch}/{self.game_batch_num}")

        if init_model:
            # 从现有模型开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model,
                                                   use_gpu=use_gpu)
        else:
            # 从头开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   use_gpu=use_gpu)
        
        # 这里的 MCTSPlayer 使用了 policy_value_net，也就是 AlphaZero 的核心
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        通过旋转和翻转扩充数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集自我对弈数据用于训练"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 数据增强
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

            # 使用日志记录训练信息
            self.log_training_info(self.current_batch, self.episode_len)

    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            
            # 计算 KL 散度
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # 如果 KL 散度发散严重则提前停止
                break
                
        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        
        # 使用日志系统记录详细的训练信息
        self.log_policy_update(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new)

        # 记录KL散度到训练历史
        self.training_history['kl_divergences'].append(kl)

        return loss, entropy

    def save_checkpoint(self, checkpoint_file='training_checkpoint.pkl'):
        """保存训练状态到checkpoint文件"""
        checkpoint_data = {
            'current_batch': self.current_batch,
            'best_win_ratio': self.best_win_ratio,
            'lr_multiplier': self.lr_multiplier,
            'pure_mcts_playout_num': self.pure_mcts_playout_num,
            'data_buffer': list(self.data_buffer),  # deque转换为list
            'training_history': self.training_history,
            'model_params': self.policy_value_net.get_policy_param(),
            'optimizer_state': self.policy_value_net.optimizer.state_dict(),
            'episode_len': getattr(self, 'episode_len', 0)
        }

        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            self.log_checkpoint_save(self.current_batch)
        except Exception as e:
            self.logger.error(f"保存checkpoint失败: {e}")

    def load_checkpoint(self, checkpoint_file='training_checkpoint.pkl'):
        """从checkpoint文件恢复训练状态"""
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # 恢复训练状态
            self.current_batch = checkpoint_data['current_batch']
            self.best_win_ratio = checkpoint_data['best_win_ratio']
            self.lr_multiplier = checkpoint_data['lr_multiplier']
            self.pure_mcts_playout_num = checkpoint_data['pure_mcts_playout_num']
            self.data_buffer = deque(checkpoint_data['data_buffer'], maxlen=self.buffer_size)
            self.training_history = checkpoint_data['training_history']
            self.episode_len = checkpoint_data.get('episode_len', 0)

            # 恢复模型参数和优化器状态
            if 'model_params' in checkpoint_data:
                self.policy_value_net.policy_value_net.load_state_dict(checkpoint_data['model_params'])
            if 'optimizer_state' in checkpoint_data and hasattr(self.policy_value_net, 'optimizer'):
                self.policy_value_net.optimizer.load_state_dict(checkpoint_data['optimizer_state'])

            self.logger.info(f"成功加载断点: 批次 {self.current_batch}")
            return True

        except Exception as e:
            self.logger.error(f"加载checkpoint失败: {e}")
            self.logger.info("将从头开始训练")
            return False

    def setup_logging(self, log_file=None):
        """设置日志系统"""
        # 如果没有指定日志文件，使用默认文件名
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"training_log_{timestamp}.log"

        self.log_file = log_file

        # 创建日志记录器
        self.logger = logging.getLogger('AlphaZeroTraining')
        self.logger.setLevel(logging.INFO)

        # 清除之前的处理器
        self.logger.handlers.clear()

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("="*60)
        self.logger.info("AlphaZero 五子棋训练开始 (9x9)")
        self.logger.info("="*60)
        self.logger.info(f"日志文件: {log_file}")

    def log_training_info(self, batch_idx, episode_len, loss=None, entropy=None, win_ratio=None):
        """记录训练信息"""
        info_msg = f"批次 {batch_idx+1}: 对局长度={episode_len}"

        if loss is not None and entropy is not None:
            info_msg += f", 损失={loss:.4f}, 熵={entropy:.4f}"

        if win_ratio is not None:
            info_msg += f", 胜率={win_ratio:.2f}"

        self.logger.info(info_msg)

    def log_policy_update(self, kl, lr_multiplier, loss, entropy, explained_var_old, explained_var_new):
        """记录策略更新详细信息"""
        self.logger.info(
            f"策略更新 - KL散度: {kl:.5f}, "
            f"学习率调整: {lr_multiplier:.3f}, "
            f"损失: {loss:.4f}, "
            f"熵: {entropy:.4f}, "
            f"解释方差(旧): {explained_var_old:.3f}, "
            f"解释方差(新): {explained_var_new:.3f}"
        )

    def log_evaluation_result(self, win_cnt, win_ratio, pure_mcts_playout_num):
        """记录评估结果"""
        self.logger.info(
            f"模型评估 - MCTS模拟次数: {pure_mcts_playout_num}, "
            f"胜利: {win_cnt[1]}, 失败: {win_cnt[2]}, 平局: {win_cnt[-1]}, "
            f"胜率: {win_ratio:.2f}"
        )

    def log_checkpoint_save(self, batch_idx):
        """记录断点保存"""
        self.logger.info(f"训练断点已保存: 批次 {batch_idx+1}")

    def log_training_complete(self):
        """记录训练完成"""
        self.logger.info("="*60)
        self.logger.info("五子棋训练完成 (10x10)!")
        self.logger.info(f"总训练批次: {self.current_batch}")
        self.logger.info(f"最佳胜率: {self.best_win_ratio:.2f}")
        self.logger.info("="*60)

    def policy_evaluate(self, n_games=10):
        """
        通过与纯 MCTS 对战来评估训练好的策略
        注意：这仅用于监控训练进度
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            # 切换先后手，i%2==0时AlphaZero先手(0)，==1时纯MCTS先手
            # start_play 返回的 winner 是 1 或 2 (或 -1)
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        
        # 计算胜率：胜利 + 0.5 * 平局
        # 在 start_play 中，start_player 0 对应玩家1，1 对应玩家2
        # 我们要计算的是 current_mcts_player 的胜率
        # player1 (index 1) 始终是 start_play 的第一个参数 (current_mcts_player)
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games

        # 使用日志系统记录评估结果
        self.log_evaluation_result(win_cnt, win_ratio, self.pure_mcts_playout_num)

        return win_ratio

    def run(self):
        """运行训练流程"""
        try:
            # 如果从断点恢复，跳过已经训练过的批次
            start_batch = self.current_batch

            for i in range(start_batch, self.game_batch_num):
                self.current_batch = i
                self.collect_selfplay_data(self.play_batch_size)

                # 记录对局长度
                self.training_history['episode_lengths'].append(self.episode_len)

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    # 记录训练指标
                    self.training_history['losses'].append(loss)
                    self.training_history['entropies'].append(entropy)
                    self.training_history['lr_multipliers'].append(self.lr_multiplier)

                # 检查当前模型性能，并保存模型参数
                if (i+1) % self.check_freq == 0:
                    self.logger.info(f"当前自我对弈批次: {i+1}")
                    win_ratio = self.policy_evaluate()
                    self.training_history['win_ratios'].append(win_ratio)

                    self.policy_value_net.save_model('./current_policy.model')
                    
                    model_file_name = f'policy_batch_{i+1}.model'
                    save_path = os.path.join(self.model_dir, model_file_name)
                    self.policy_value_net.save_model(save_path)
                    self.logger.info(f"备份模型已保存至: {save_path}")

                    if win_ratio > self.best_win_ratio:
                        self.logger.info("发现新最佳策略!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # 更新最佳策略
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

                    # 保存训练断点
                    self.save_checkpoint()

                # 定期保存断点（根据设置的频率）
                if (i+1) % self.checkpoint_freq == 0:
                    self.save_checkpoint()

            # 训练完成日志
            self.log_training_complete()

        except KeyboardInterrupt:
            self.logger.info('\n训练中断')
            self.logger.info('保存训练断点...')
            self.save_checkpoint()
            self.logger.info('训练断点已保存，下次可以使用 --resume 参数继续训练')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AlphaZero Gomoku 训练程序')
    parser.add_argument('--init_model', type=str, default=None,
                        help='初始化模型文件路径 (默认: None)')
    parser.add_argument('--resume', action='store_true',
                        help='从断点继续训练')
    parser.add_argument('--checkpoint', type=str, default='training_checkpoint.pkl',
                        help='checkpoint文件路径 (默认: training_checkpoint.pkl)')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='checkpoint保存频率 (默认: 10个批次)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='训练日志文件路径 (默认: 自动生成带时间戳的日志文件)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志文件保存目录 (默认: logs)')

    args = parser.parse_args()

    # 创建日志目录
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # 如果指定了日志目录但没有指定日志文件，在指定目录中生成日志文件
    if args.log_file is None and args.log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join(args.log_dir, f"training_log_{timestamp}.log")

    print("="*60)
    print("AlphaZero 五子棋训练程序 (10x10)")
    print("="*60)

    if args.resume:
        print(f"从断点恢复训练模式: {args.checkpoint}")
    else:
        print("从头开始训练模式")

    if args.init_model:
        print(f"使用预训练模型初始化: {args.init_model}")

    if args.log_file:
        print(f"日志文件: {args.log_file}")

    print("="*60)

    training_pipeline = TrainPipeline(
        init_model=args.init_model,
        checkpoint_file=args.checkpoint,
        resume_training=args.resume,
        log_file=args.log_file
    )

    # 设置checkpoint保存频率
    if hasattr(training_pipeline, 'checkpoint_freq'):
        training_pipeline.checkpoint_freq = args.checkpoint_freq
    else:
        training_pipeline.checkpoint_freq = 10

    try:
        training_pipeline.run()
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("保存训练断点...")
        training_pipeline.save_checkpoint(args.checkpoint)
        print("训练断点已保存，下次可以使用 --resume 参数继续训练")
        raise