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

    def fit(self, x, y):
        """
        训练集成器

        :param x:样本
        :param y:标签
        """
        IR = len(y[y == 1]) / len(y[y == 0])
        sampling_interval = 1 / (IR * np.log2(IR))
        balance_rate = 1 / IR
        balance_point = 5
        start_rate = balance_rate - sampling_interval / balance_point
        print("平衡采样率点位：1/%d" % balance_point)
        print("平衡采样率 %.4f 采样间隔 %.4f" % (balance_rate, sampling_interval))
        for i in range(self.n_estimator):
            sampling_rate = start_rate + i * (sampling_interval / (self.n_estimator - 1))
            print("下采样率 %.4f 采样个数 %d" % (sampling_rate, int(sampling_rate * len(x))))

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

        y_proba = np.mean(np.array(all_y_prob), axis=0)

        return y_proba
