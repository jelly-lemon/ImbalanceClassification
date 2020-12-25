# 描述：NSAG-II 遗传算法
# 作者：Jelly Lemon

import math
import random
import matplotlib.pyplot as plt
import numpy as np

from equation import objection_1, objection_2



class NSGAIIEvolutor():
    """
    遗传算法 NSGA-II 的实现，用来进化预测结果

    """

    def __init__(self, max_gen, solution, S, R, all_q):
        """
        初始化 NSGA-II 进化算法

        :param max_gen:迭代次数
        :param solution:初始种群
        :param S:相似度矩阵
        :param R:聚类结果矩阵
        :param all_q:所有基分类器对聚类质心的分类结果
        """
        self.max_gen = max_gen
        self.pop_size = len(solution)
        self.solution = solution
        self.gen_no = 0  # 当前迭代次数
        self.R = R
        self.S = S
        self.all_q = all_q

    def sort_by_values(self, list1, values):
        """

        :param list1:
        :param values:
        :return:
        """
        sorted_list = []
        while (len(sorted_list) != len(list1)):
            if self.index_of(min(values), values) in list1:
                sorted_list.append(self.index_of(min(values), values))
            values[self.index_of(min(values), values)] = math.inf
        return sorted_list

    def fast_non_dominated_sort(self, values1, values2):
        """
        快速非支配排序

        :param values1:目标函数1的值
        :param values2:目标函数2的值
        :return:
        """
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
        return front

    def crowding_distance(self, values1, values2, front):
        """
        计算拥挤距离

        :param values1: 目标函数1值集合
        :param values2: 目标函数2值集合
        :param front:
        :return:
        """
        # 拥挤距离初始化为 0
        distance = [0 for i in range(0, len(front))]

        sorted1 = self.sort_by_values(front, values1[:])
        sorted2 = self.sort_by_values(front, values2[:])

        # 边界两个元素的拥挤距离为无穷大
        distance[0] = float('inf')
        distance[len(front) - 1] = float('inf')

        for k in range(1, len(front) - 1):
            # 分母可能出现0的情况
            distance[k] = distance[k] + np.abs(values1[sorted1[k + 1]] - values1[sorted1[k - 1]])

        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + np.abs(values2[sorted1[k + 1]] - values2[sorted1[k - 1]])

        return distance

    def crossover(self, father, mother):
        """
        随机交换变异

        :param father:个体a
        :param mother:个体b
        :return:
        """
        child = father.copy()
        crossover_rate = 0.8
        if np.random.rand() < crossover_rate:
            cross_points = np.random.randint(low=0, high=len(child))  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]

        return child

    def mutation(self, child):
        """
        变异

        :param child:
        :return:
        """
        mutation_rate = 0.03
        if np.random.rand() < mutation_rate:
            mutate_point = np.random.randint(0, len(child))  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point][1], child[mutate_point][0] = child[mutate_point][0], child[mutate_point][1]

        return child


    def index_of(self, a, list):
        """

        :param a:
        :param list:
        :return:
        """
        for i in range(0, len(list)):
            if list[i] == a:
                return i
        return -1

    def evolute(self):
        """
        进化

        :return:返回和初始种群大小一样的进化后的种群
        """
        while (self.gen_no < self.max_gen):
            print("\rgen_no %d/%d" % (self.gen_no, self.max_gen), end="")

            # 交叉变异，产生后代，得到新的种群
            while (len(self.solution) != 2 * self.pop_size):
                # 交叉变异
                a1 = b1 = 0
                while a1 == b1:
                    a1 = random.randint(0, self.pop_size - 1)
                    b1 = random.randint(0, self.pop_size - 1)
                new_one = self.crossover(self.solution[a1], self.solution[b1])
                new_one = self.mutation(new_one)
                self.solution.append(new_one)
                self.all_q.append(self.all_q[a1].copy())

            # 计算合并后的种群目标函数值
            function1_values2 = [objection_1(self.S, self.solution[i]) for i in range(0, 2 * self.pop_size)]
            function2_values2 = [objection_2(len(self.all_q[0]), self.R, self.solution[i], self.all_q[i]) for i in
                                 range(0, 2 * self.pop_size)]

            # 非支配性排序
            non_dominated_sorted_solution2 = self.fast_non_dominated_sort(function1_values2[:], function2_values2[:])

            # 计算拥挤度
            crowding_distance_values2 = []
            for i in range(0, len(non_dominated_sorted_solution2)):
                crowding_distance_values2.append(
                    self.crowding_distance(function1_values2[:], function2_values2[:],
                                           non_dominated_sorted_solution2[i][:]))

            # 适者生存，得到新的种群，只保留 pop_size 个个体
            new_solution = []   # 里面只存编号
            for i in range(0, len(non_dominated_sorted_solution2)):
                non_dominated_sorted_solution2_1 = [
                    self.index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                    range(0, len(non_dominated_sorted_solution2[i]))]
                front22 = self.sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
                front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                         range(0, len(non_dominated_sorted_solution2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if (len(new_solution) == self.pop_size):
                        break
                if (len(new_solution) == self.pop_size):
                    break

            # 新种群的数据
            self.solution = [self.solution[i] for i in new_solution]
            self.all_q = [self.all_q[i] for i in new_solution]

            self.gen_no = self.gen_no + 1

        print("")
        return self.solution


def show_objection(function1_values, function2_values, title):
    """
    画个图看一下

    :param function1_values:目标函数1的值
    :param function2_values:目标函数2的值
    :param title:图片标题
    """
    function1 = [i for i in function1_values]
    function2 = [j for j in function2_values]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.title(title)
    plt.scatter(function1, function2)
    plt.show()
