import numpy as np
from equation import objection_1, objection_2

class PSOEvolutor:
    """
    粒子群优化算法

    """

    def __init__(self, S, R, Q):
        self.S = S
        self.R = R
        self.Q = Q

    def evolve(self, X, max_steps):
        """

        :param X:初始粒子位置
        """
        max_steps = max_steps  # 最大迭代次数

        dim = (len(X[0]), 2)  # 粒子的维度
        init_weight = 0.6  # 初始惯性权重与当前惯性权重
        end_weight = 0.1  # 结束惯性权重
        c1 = c2 = 2  # 个体学习因子、社会学习因子

        # 评估每个粒子并得到全局最优
        pBest = X  # 存放每个粒子的历史最优位置，默认初始位置为最优位置
        gBest = self.get_best(pBest)  # 获取这一届总体最优位置

        # 随机初始化每一个粒子的速度
        v = np.random.rand(dim[0], dim[1])

        iter = 0
        for step in range(max_steps):
            iter += 1
            print("\r%d/%d" % (iter, max_steps), end="")

            # 计算本次迭代惯性因子
            cur_weight = (init_weight - end_weight) * (max_steps - iter) / max_steps + end_weight

            # 生成两个随机数，分别代表飞向当前粒子历史最佳位置、全局历史最佳位置的程度
            r1 = np.random.rand(dim[0], dim[1])
            r2 = np.random.rand(dim[0], dim[1])

            # 更新每个粒子的速度和位置
            v = np.array(v)
            X = np.array(X)
            pBest = np.array(pBest)
            gBest = np.array(gBest)
            v = cur_weight * v + c1 * r1 * (pBest - X) + c2 * r2 * (gBest - X)

            X = X + v

            # 超出范围的粒子位置要进行限制
            X[X > 1] = 1
            X[X < 0] = 0

            # 新位置不一定是好位置，还得和之前的个体粒子最优位置进行比较，比之前好才能更新
            pBest = self.compare(pBest, X)
            gBest = self.get_best(pBest)
        print("")
        return X

    def get_best(self, X):
        """
        从历届最优个体位置中选择一个最好的

        :return:
        """
        # 首先计算每个粒子对应的函数值
        values1 = [objection_1(self.S, xi) for xi in X]
        values2 = [objection_2(len(self.Q[0]), self.R, X[i], self.Q[i]) for i in range(len(X))]

        S = [[] for i in range(0, len(values1))]
        front = [[]]
        n = [0 for i in range(0, len(values1))]
        rank = [0 for i in range(0, len(values1))]

        for p in range(0, len(values1)):
            S[p] = []
            n[p] = 0
            for q in range(0, len(values1)):
                if (values1[p] > values1[q] and values2[p] > values2[q]) or (
                        values1[p] >= values1[q] and values2[p] > values2[q]) or (
                        values1[p] > values1[q] and values2[p] >= values2[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                        values1[q] >= values1[p] and values2[q] > values2[p]) or (
                        values1[q] > values1[p] and values2[q] >= values2[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while (front[i] != []):
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            front.append(Q)

        del front[len(front) - 1]


        return X[front[0][0]]

    def compare(self, old_pos, new_pos):
        """
        比较新旧位置，返回最好位置

        :param old_pos: 粒子历史最优位置
        :param new_pos: 新位置
        :return: 新的历史最优位置
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
        x2_fun_value = [objection_1(self.S, x2), objection_2(len(self.Q[0]), self.R, x1, q)]

        # 判定x1是不是最优解
        for i in range(len(x1_fun_value)):
            if x1_fun_value[i] < x2_fun_value[i]:
                continue
            else:
                break
        if i == len(x1_fun_value):
            return x1

        # 判定 x2 是不是最优解
        for i in range(len(x2_fun_value)):
            if x2_fun_value[i] < x1_fun_value[i]:
                continue
            else:
                break
        if i == len(x2_fun_value):
            return x2

        # 如果两者不相上下，该如何取舍呢？
        if x1_fun_value[0] < x2_fun_value[0]:
            return x1
        else:
            return x2












