"""
粒子群优化算法

目标函数只有一个
"""

import numpy as np
from compare.equation import objection_1


class PSOEvolutor:
    """
    粒子群优化算法，根据一个目标函数优化预测结果
    """

    def __init__(self, S):
        self.S = S

    def evolve(self, X, max_steps=10):
        """
        开始进化

        :param X:初始粒子位置
        :return 进化后的粒子位置
        """
        max_steps = max_steps  # 最大迭代次数

        dim = (len(X), 2)  # 粒子的维度
        init_weight = 0.6  # 初始惯性权重与当前惯性权重
        end_weight = 0.1  # 结束惯性权重
        c1 = c2 = 2  # 个体学习因子、社会学习因子

        # 评估每个粒子并得到全局最优
        pBest = X  # 存放每个粒子的历史最优位置，默认初始位置为最优位置
        gBest = self.get_gBest(pBest)  # 获取这一届总体最优位置

        # 随机初始化每一个粒子的速度
        v = np.random.rand(dim[0], dim[1])

        iter = 0
        for step in range(max_steps):
            iter += 1

            # 计算本次迭代惯性因子
            cur_weight = init_weight - (iter - 1) * (init_weight - end_weight) / (max_steps - 1)

            # 生成两个随机数，分别代表飞向当前粒子历史最佳位置、全局历史最佳位置的程度
            r1 = np.random.rand(dim[0], dim[1])
            r2 = np.random.rand(dim[0], dim[1])

            # 更新每个粒子的速度和位置
            v = np.array(v)
            X = np.array(X)
            pBest = np.array(pBest)
            gBest = np.array(gBest)
            v = cur_weight * v + c1 * r1 * (pBest - X) + c2 * r2 * (gBest - X)
            X = X + v  # 粒子跑到新位置

            # 超出范围的粒子位置要进行限制
            # 预测概率不可能大于1，也不可能小于0
            X[X > 1] = 1
            X[X < 0] = 0

            # 新位置不一定是好位置，还得和之前的个体粒子最优位置进行比较，比之前好才能更新
            pBest = self.get_pBest(pBest, X)
            gBest = self.get_gBest(pBest)

        return X

    def get_gBest(self, X):
        """
        从历届最优个体位置中选择一个最好的

        :return:
        """
        # 首先计算每个粒子对应的函数值
        values1 = [objection_1(self.S, xi) for xi in X]

        return max(values1)

    def get_pBest(self, old_pos, new_pos):
        """
        比较新旧位置，返回最好位置

        :param old_pos: 所有粒子历史最优位置
        :param new_pos: 新位置
        :return: 所有粒子的新的历史最优位置
        """
        pBest = []
        for i in range(len(old_pos)):
            # 比较两个位置谁好
            optiaml = self.get_optimal(old_pos[i], new_pos[i])
            pBest.append(optiaml)

        return pBest

    def get_optimal(self, x1, x2):
        """
        比较两个位置谁好

        :param x1:某个粒子1
        :param x2:某个粒子2
        :return: 位置好的粒子
        """
        x1_fun_value = [objection_1(self.S, x1)]
        x2_fun_value = [objection_1(self.S, x2)]

        if x1_fun_value[0] <= x2_fun_value[0]:
            return x1
        else:
            return x2

