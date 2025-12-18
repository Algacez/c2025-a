# -*- coding: utf-8 -*-
"""
AlphaGo Zero 的 MCTS
使用策略价值网络来指导树搜索并评估叶子节点
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
    MCTS树中的节点。
    每个节点记录其自身的价值 Q，先验概率 P，以及经过访问次数调整后的先验分数 u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从动作到TreeNode的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """
        通过创建新子节点来扩展树。
        action_priors: 由策略函数生成的 (动作, 先验概率) 元组列表。
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        在子节点中选择能产生最大 (Q + u(P)) 值的动作。
        返回: (action, next_node) 元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        根据叶子节点的评估值更新节点值。
        leaf_value: 从当前玩家视角看子树的评估值。
        """
        # 增加访问次数
        self._n_visits += 1
        # 更新 Q 值，即所有访问值的移动平均
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        类似于 update()，但递归地应用于所有祖先节点。
        """
        # 如果不是根节点，先更新父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        计算并返回此节点的值。
        它是叶子节点评估值 Q 和根据访问次数调整后的先验 u 的组合。
        c_puct: 控制价值 Q 和先验概率 P 相对影响的参数。
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """检查是否为叶子节点（即该节点下没有扩展出子节点）。"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """蒙特卡洛树搜索的实现"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: 一个函数，接收棋盘状态，输出 (动作, 概率) 列表和
            当前玩家视角的[-1, 1]之间的分数。
        c_puct: 控制探索收敛速度的参数。值越高越依赖先验概率。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        运行一次从根到叶子的模拟，获取叶子节点的价值并反向传播回父节点。
        State 会被就地修改，所以必须提供一个副本。
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # 贪婪地选择下一步
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 使用网络评估叶子节点，输出 (action, probability) 元组 p 和 分数 v
        action_probs, leaf_value = self._policy(state)
        # 检查游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 对于结束状态，返回“真实”的叶子节点值
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 更新本次遍历路径上节点的值和访问计数
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        顺序运行所有模拟并返回可用动作及其对应的概率。
        state: 当前游戏状态
        temp: 温度参数，(0, 1]，控制探索程度
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 根据根节点的访问次数计算动作概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """
        在树中前进一步，保留已知的子树信息。
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """基于 MCTS 的 AI 玩家"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # 按照 AlphaGo Zero 论文中 MCTS 返回的 pi 向量
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 增加 Dirichlet 噪声用于探索 (自我对弈训练需要)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新根节点并复用搜索树
                self.mcts.update_with_move(move)
            else:
                # 默认 temp=1e-3 时，几乎等同于选择概率最大的动作
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("警告: 棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)