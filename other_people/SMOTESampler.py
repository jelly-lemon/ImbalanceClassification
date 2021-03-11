import random

import numpy as np

from data import read_data


class SmoteSampler:

    def __init__(self, k):
        """

        :param k: 邻居数
        """
        self.k = k

    def resample(self, x, N):
        """
        smote 上采样

        :param x: 原始样本集
        :param N: 上采样的个数
        :return: 采样后的总集合
        """
        if self.k > len(x):
            self.k = len(x)

        # 计算每个样本的之间的欧式距离
        dis = self.get_distance(x)

        T = []

        while len(T) < N:
            # 遍历每一个样本
            for i in range(len(x)):
                # 得到样本 i 的 k 个邻居
                k_neighbor = self.get_neighbor(dis[i])
                choose_neighbor = random.choice(k_neighbor)
                x_new = x[i] + random.random() * (x[choose_neighbor] - x[i])
                T.append(x_new)
                if len(T) == N:
                    break
            x = np.concatenate((x, T), axis=0)

        return x

    def get_distance(self, x):
        """
        获取集合中各个点之间的欧式距离

        :param x: 集合
        :return: 欧式距离，矩阵
        """

        dis = np.zeros((len(x), len(x)))
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                dis[i][j] = np.sqrt(np.sum(np.square(xi - xj)))

        return dis

    def get_neighbor(self, d):
        """
        获取最近的 k 个邻居编号

        :param d: 所有邻居的距离
        :return: k 个最近邻居的编号，从近到远
        """
        t = sorted(enumerate(d), key=lambda x: x[1])
        t = [x[0] for x in t]

        return t[:self.k]

if __name__ == '__main__':
    x, y = read_data.get_data([1], -1, "banana.dat", show_info=True)

    x_neg = x[y == 0]
    print("采样前：%d" % len(x_neg))
    x_neg = SmoteSampler(5).resample(x_neg, 100)
    print("采样后：%d" % len(x_neg))