
import math
import random
import matplotlib.pyplot as plt
from equation import objection_1, objection_2



class NSGAII():
    """
    遗传算法 NSGA-II 的实现

    """

    def __init__(self, max_gen, pop_size, solution, n_cluster, R, S, Q):
        """

        :param max_gen:迭代次数
        :param pop_size:种群大小
        :param solution:初始种群
        :param R: 聚类结果矩阵
        :param S:相似度矩阵
        :param Q:聚类质心矩阵
        """
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.solution = solution
        self.n_cluster = n_cluster
        self.gen_no = 0  # 当前迭代
        self.all_R = R
        self.all_S = S
        self.all_Q = Q

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

        :param values1:
        :param values2:
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

        :param values1:
        :param values2:
        :param front:
        :return:
        """
        distance = [0 for i in range(0, len(front))]
        sorted1 = self.sort_by_values(front, values1[:])
        sorted2 = self.sort_by_values(front, values2[:])
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (
                    max(values1) - min(values1))
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                    max(values2) - min(values2))
        return distance

    def crossover(self, a, b):
        """
        随机交换变异

        :param a:个体a
        :param b:个体b
        :param ua: 个体a所有元素对应的预测概率
        :param ub:
        :return:
        """
        a = a.copy()

        n = len(a)
        start = n
        end = 0
        while start >= end or end - start < 0.2 * n:
            start = random.randint(0, n)
            end = random.randint(0, n)

        for i in range(start, end):
            a[i] = b[i]

        return a

    # Function to find index of list
    def index_of(self, a, list):
        for i in range(0, len(list)):
            if list[i] == a:
                return i
        return -1

    def evolute(self):
        """
        进化
        :return:
        """
        while (self.gen_no < self.max_gen):
            print("gen_no %d" % self.gen_no)
            # 交叉变异，产生后代，得到新的种群
            while (len(self.solution) != 2 * self.pop_size):
                # 交叉变异
                a1 = b1 = 0
                while a1 == b1:
                    a1 = random.randint(0, self.pop_size - 1)
                    b1 = random.randint(0, self.pop_size - 1)
                new_one = self.crossover(self.solution[a1], self.solution[b1])
                self.solution.append(new_one)
                self.all_S.append(self.all_S[a1].copy())
                self.all_R.append(self.all_R[a1].copy())
                self.all_Q.append(self.all_Q[a1].copy())

            # 计算合并后的种群目标函数值
            function1_values2 = [objection_1(self.all_S[i], self.solution[i]) for i in range(0, 2 * self.pop_size)]
            function2_values2 = [objection_2(self.n_cluster, self.all_R[i], self.solution[i], self.all_Q[i]) for i in
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
            new_solution = []
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

            self.solution = [self.solution[i] for i in new_solution]
            self.all_S = [self.all_S[i] for i in new_solution]
            self.all_R = [self.all_R[i] for i in new_solution]
            self.all_Q = [self.all_Q[i] for i in new_solution]

            self.gen_no = self.gen_no + 1

        return self.solution


def show_objection(function1_values, function2_values, title):
    function1 = [i for i in function1_values]
    function2 = [j for j in function2_values]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    # plt.xlim((30,100))
    # plt.ylim((0,5))
    plt.title(title)
    plt.scatter(function1, function2)
    plt.show()
