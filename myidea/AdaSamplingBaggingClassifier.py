# 描述：自适应采样率集成分类器
import random

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class AdaSamplingBaggingClassifier:
    def __init__(self, n_estimator):
        """

        :param n_estimator:基分类器的数量
        """
        self.n_estimator = n_estimator
        self.classifiers = []
        for i in range(self.n_estimator):
            self.classifiers.append(KNeighborsClassifier())

    def fit(self, x, y, sampling="under", show_info=False):
        """
        训练集成器

        :param x:样本
        :param y:标签
        """
        # 下采样
        if sampling == "under":
            IR = len(y[y == 1]) / len(y[y == 0])
            sampling_interval = 1 / (IR * np.log2(IR))  # 采样间隔
            balance_rate = 1 / IR   # 平衡采样率
            sampling_rate = balance_rate - sampling_interval / 2
            if show_info:
                print("平衡采样率 %.4f 采样间隔 %.4f" % (balance_rate, sampling_interval))
            for i in range(self.n_estimator):
                sampling_rate += i * (sampling_interval / (self.n_estimator - 1))

                # # 基于密度采样
                # x_train, y_train = DBUSampler(sampling_rate=sampling_rate).fit_resample(x, y)

                # # 随机下采样
                while True:
                    data = list(zip(x, y))
                    data = random.sample(data, int(sampling_rate * len(x)))
                    x_train, y_train = zip(*data)
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    if len(y_train[y_train == 1]) == len(y_train):
                        continue
                    else:
                        break

                self.classifiers[i].fit(x_train, y_train)
        else:
            # 上采样
            balance_rate = len(y[y == 1]) / len(y[y == 0]) - 1
            for i in range(self.n_estimator):
                sampling_rate = balance_rate / self.n_estimator

    def predict_proba(self, x):
        """
        对给定数据 x 进行预测

        :param x:输入样本
        :return:预测概率；列表，[[0.1 0.9], [0.8. 0.1], ...]
        """
        all_y_prob = []
        for i, clf in enumerate(self.classifiers):
            # 基分类器预测
            y_prob = clf.predict_proba(x)
            all_y_prob.append(y_prob)
            # print("i=%d" % i)

        # 所有基分类器预测结果求平均
        y_proba = np.mean(np.array(all_y_prob), axis=0)

        return y_proba

    def predict_proba_2(self, x):
        all_y_prob = []
        for i, clf in enumerate(self.classifiers):
            # 基分类器预测
            y_prob = clf.predict_proba(x)
            all_y_prob.append(y_prob)
            # print("i=%d" % i)

        return all_y_prob