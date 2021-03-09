"""
多目标粒子群优化算法

关键点：
选择 pBest(个体历史最优)：新旧位置各好一半时，随机选择新旧作为 pBest
选择 gBest(全局历史最优)：非支配排序 -> 拥挤度计算，选择优先级最高并且拥挤读最小的位置，再进行新旧比较。

"""
import numpy as np


class mopso:
    """
    多目标粒子群优化算法
    """

    def __init__(self, R, S, U, Q):
        self.R = R
        self.S = S
        self.U = U
        self.Q = Q

    def fast_non_dominated_sort(self, P):
        """
        快速非支配排序

        :param P: 种群
        :return: 第一个非支配层个体集合
        """
        S = {}  # 支配个体集合。例如，p 支配的个体集合 S[p]
        n = {}  # 支配个体数量。例如，支配 p 的个体数量 n[p]
        rank = {}   # 个体所属层级。例如, p 的所属层级 rank[p]
        F = {}  # 某一层级个体集合。例如，第一非支配层 F[1]

        # 计算得到第一非支配层个体集合
        for p in P:
            S[p] = []   # p 支配的个体集合
            n[p] = 0    # 支配 p 的个体数量
            for q in P:
                if (self.compare(p, q) > 0):
                    S[p].append(q)
                elif (self.compare(p, q) < 0):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 1
                F[1].append(p)

        # 计算剩余个体的所属非支配层
        # i = 1
        # while len(F[i]) != 0:
        #     Q = []
        #     for p in F[i]:
        #         for q in S[p]:
        #             n[q] -= 1
        #             if n[q] == 0:
        #                 rank[q] = i + 1
        #                 Q.append(q)
        #     i += 1
        #     F[i] = Q

        return F[1]

    def compare(self, p, q):
        """
        比较个体 p 和 q 谁支配谁
        :param p:
        :param q:
        :return:
        """
        p_func_value = (self.func_1(p), self.func_2(p))
        q_func_value = (self.func_1(p), self.func_2(p))

        # p 支配 q
        if p_func_value[0] < q_func_value[0] and p_func_value[1] < q_func_value[1]:
            return 1

        # q 支配 p
        if p_func_value[0] > q_func_value[0] and p_func_value[1] > q_func_value[1]:
            return -1

        # p、q 势均力敌
        return 0

    def get_best(self, P):
        """
        获取种群中最好的个体

        :param P: 种群
        :return: 最好的个体
        """
        # 获取最优前沿，也就是第一非支配层个体集合
        best_front = self.fast_non_dominated_sort(P)

        # 获取最优前沿里面的最优个体
        best_p = self.get_front_best(best_front)

        return best_p.copy()

    def func_1(self, S, U):
        """
        目标函数 1，值越小越好

        思想：两个样本越相似，那么同属于一个类别的概率也越大
        我们认为，所有样本的 “样本相似度 * 样本类别差距” 加起来，
        值越小，预测结果越准确。

        :param S:样本相似度矩阵
        :param U:分类器预测结果概率矩阵
        :return:目标函数值
        """

        n_sample = len(U)
        sum = 0
        for i in range(n_sample):
            for j in range(n_sample):
                sum += S[i][j] * np.linalg.norm(U[i] - U[j])
        return sum

    def func_2(self, n_cluster_center, R, U, Q):
        """
        目标函数 2，值越小越好

        :param n_cluster_center:聚类质心数量
        :param R:所有样本聚类结果矩阵
        :param U:分类器预测结果概率矩阵
        :param Q:分类器对聚类质心的分类结果
        :return:
        """
        n_sample = len(U)
        sum = 0
        for i in range(n_sample):
            for j in range(n_cluster_center):
                sum += R[i][j] * np.linalg.norm(U[i] - Q[j])
        return sum

    def sort(self, individual_set, func_value):
        """
        对某一种群集合按目标函数 1 的值进行排序

        :param individual_set: 种群
        :param func_value:
        :return:
        """
        # 冒泡升序
        n = len(func_value)
        for i in range(n-1):
            for j in range(n-i):
                if func_value[j][0] > func_value[j+1][0]:
                    t = func_value[j]
                    func_value[j] = func_value[j+1]
                    func_value[j+1] = t

                    t = individual_set[j]
                    individual_set[j] = individual_set[j+1]
                    individual_set[j+1] = t

        return individual_set, func_value

    def get_front_best(self, individual_set):
        """
        获取某一非支配层中最好的个体

        :param individual_set: 某一非支配层个体集合
        :return: 最好的个体
        """
        # 计算每个个体的目标函数 1 和 目标函数 2 的值
        func_value = []
        for p in individual_set:
            func_value.append((self.func_1(p), self.func_2(p)))

        # 按目标函数 1 的值升序排序
        individual_set, func_value = self.sort(individual_set, func_value)

        # 计算每个个体的拥挤度
        crowding_dis = [-1 for i in range(len(func_value))] # -1 代表拥挤度无穷大
        for i, value in enumerate(func_value):
            if i == 0 or i == len(func_value) - 1:
                continue
            else:
                crowding_dis[i] = ((value[i+1][0]-value[i-1][1])+(value[i+1][1]-value[i-1][1]))*2

        # 找到拥挤度最小值下标
        min_pos = 0
        for i in range(len(crowding_dis)):
            if crowding_dis[i] < crowding_dis[min_pos]:
                min_pos = i

        return individual_set[min_pos]

    def evolute(self, pso, max_steps=10):
        """
        粒子群优化算法

        :param pso: 粒子群的初始位置
        :return: 进化后的粒子位置
        """
        max_steps = max_steps  # 最大迭代次数

        particle_dim = pso[0].shape  # 粒子的维度
        init_weight = 0.6  # 初始惯性权重与当前惯性权重
        end_weight = 0.1  # 结束惯性权重
        c1 = c2 = 2  # 个体学习因子、社会学习因子

        # 评估每个粒子并得到全局最优
        pBest = pso.copy()  # 存放每个粒子的历史最优位置，默认初始位置为最优位置
        gBest = self.get_best(pBest)  # 获取这一届总体最优位置

        # 随机初始化每一个粒子的速度
        v = np.random.rand(particle_dim[0], particle_dim[1])

        # 迭代计算
        iter = 0
        for step in range(max_steps):
            iter += 1

            # 计算本次迭代惯性因子
            cur_weight = init_weight - (iter - 1) * (init_weight - end_weight) / (max_steps - 1)

            # 生成两个随机数，分别代表飞向当前粒子历史最佳位置、全局历史最佳位置的程度
            r1 = np.random.rand(particle_dim[0], particle_dim[1])
            r2 = np.random.rand(particle_dim[0], particle_dim[1])

            # 更新每个粒子的速度和位置
            v = np.array(v)
            pso = np.array(pso)
            pBest = np.array(pBest)
            gBest = np.array(gBest)
            v = cur_weight * v + c1 * r1 * (pBest - pso) + c2 * r2 * (gBest - pso)
            pso = pso + v  # 粒子跑到新位置

            # 超出范围的粒子位置要进行限制
            # 预测概率不可能大于1，也不可能小于0
            pso[pso > 1] = 1
            pso[pso < 0] = 0

            # 新位置不一定是好位置，还得和之前的个体粒子最优位置进行比较，比之前好才能更新
            for i in range(len(pBest)):
                pBest[i] = self.get_best([pBest[i], pso[i]])
            gBest = self.get_best(pBest)

        return pso



