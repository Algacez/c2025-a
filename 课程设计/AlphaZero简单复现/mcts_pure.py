# -*- coding: utf-8 -*-
"""
纯 MCTS 实现
用于在训练过程中作为基准对手评估模型强度
"""

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """随机漫步策略：在可用位置中随机选择一步"""
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """
    纯 MCTS 的策略值函数
    返回均匀概率和 0 分数
    """
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """MCTS 树节点"""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 动作 -> TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """扩展子节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """选择最大价值的子节点"""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """更新节点值"""
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归更新祖先节点"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算节点价值 (UCB)"""
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """蒙特卡洛树搜索主类"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """执行一次模拟"""
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # 贪婪选择
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # 检查是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        
        # 纯 MCTS 的核心：随机模拟直到游戏结束 (Rollout)
        leaf_value = self._evaluate_rollout(state)
        # 更新节点
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """使用随机策略模拟直到游戏结束"""
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                if winner == -1:  # 平局
                    return 0
                else:
                    return 1 if winner == player else -1
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("警告: 模拟达到上限次数")
            return 0

    def get_move(self, state):
        """执行所有模拟并返回访问次数最多的移动"""
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """更新树根"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """纯 MCTS 玩家类"""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("警告: 棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)