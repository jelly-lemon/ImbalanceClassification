import numpy as np
from equation import objection_1, objection_2

class PSOEvolutor:
    """
    粒子群优化算法，根据两个目标函数优化预测结果

    """

    def __init__(self, S, R, Q):
        """
        初始化粒子群优化器

        :param S:样本相似度矩阵
        :param R:样本聚类结果矩阵
        :param Q:各个分类器对聚类质心的分类结果
        """
        self.S = S
        self.R = R
        self.Q = Q

    def evolve(self, X, max_steps):
        """
        开始进化

        :param X:初始粒子位置
        :return 进化后的粒子位置
        """
        max_steps = max_steps  # 最大迭代次数

        dim = (len(X[0]), 2)  # 粒子的维度
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
            print("\r进化进度 %d/%d" % (iter, max_steps), end="")

            # 计算本次迭代惯性因子
            cur_weight = init_weight - (iter - 1) * (init_weight - end_weight)  / (max_steps-1)

            # 生成两个随机数，分别代表飞向当前粒子历史最佳位置、全局历史最佳位置的程度
            r1 = np.random.rand(dim[0], dim[1])
            r2 = np.random.rand(dim[0], dim[1])

            # 更新每个粒子的速度和位置
            v = np.array(v)
            X = np.array(X)
            pBest = np.array(pBest)
            gBest = np.array(gBest)
            v = cur_weight * v + c1 * r1 * (pBest - X) + c2 * r2 * (gBest - X)
            X = X + v   # 粒子跑到新位置

            # 超出范围的粒子位置要进行限制
            # 预测概率不可能大于1，也不可能小于0
            X[X > 1] = 1
            X[X < 0] = 0

            # 新位置不一定是好位置，还得和之前的个体粒子最优位置进行比较，比之前好才能更新
            pBest = self.get_pBest(pBest, X)
            gBest = self.get_gBest(pBest)

        print("")
        return X

    def get_gBest(self, X):
        """
        从历届最优个体位置中选择一个最好的

        :return:
        """
        # 首先计算每个粒子对应的函数值
        values1 = [objection_1(self.S, xi) for xi in X]
        values2 = [objection_2(len(self.Q[0]), self.R, X[i], self.Q[i]) for i in range(len(X))]

        # 找帕累托最优前沿
        # 找到不受支配的点，所有不受支配的点，就是帕累托最优前沿
        # 不受支配的点：找不到两个函数值都比这个点小的点，那这个点就是不受支配的点
        best_front = []
        for i in range(len(values1)):
            is_found = False
            for j in range(len(values1)):
                if i == j:
                    continue
                else:
                    if values1[j] < values1[i] and values2[j] < values2[i]:
                        # 找到支配 i 的点
                        is_found = True
                        break
            if is_found is False:
                best_front.append(i)

        # 计算最优前沿中哪个点函数值之和最小
        i_min = 0
        sum = values1[best_front[i_min]] + values2[best_front[i_min]]
        for i in range(1, len(best_front)):
            if values1[i] + values2[i] < sum:
                i_min = i
                sum = values1[i] + values2[i]

        return X[best_front[i_min]]

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
            optiaml = self.get_optimal(old_pos[i], new_pos[i], self.Q[i])
            pBest.append(optiaml)

        return pBest

    def get_optimal(self, x1, x2, q):
        """
        比较两个位置谁好

        :param x1:某个粒子1
        :param x2:某个粒子2
        :return:位置好的粒子
        """
        x1_fun_value = [objection_1(self.S, x1), objection_2(len(self.Q[0]), self.R, x1, q)]
        x2_fun_value = [objection_1(self.S, x2), objection_2(len(self.Q[0]), self.R, x2, q)]

        # 如果 x1 支配 x2，则返回 x1
        if x1_fun_value[0] < x2_fun_value [0] and x1_fun_value[1] < x2_fun_value[1]:
            return x1

        # 如果 x2 支配 x1，则返回 x2
        if x1_fun_value[0] > x2_fun_value[0] and x1_fun_value[1] > x2_fun_value[1]:
            return x2

        # 如果两者非支配关系，选一个总和最小的返回
        if x1_fun_value[0] + x1_fun_value[1] < x2_fun_value[0] + x2_fun_value[1]:
            return x1
        else:
            return x2