# 描述：混合采样集成器的实现类

import random

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from other.read_data import shuffle_data


class HSBaggingClassifier():
    """
    Hybrid Sampling Bagging Classifier
    混合采样集成分类器. 该分类器自带各种上下采样器.

    """

    def __init__(self, n_random_sampler=5, n_down_sampler=1, n_up_sampler=1, n_no_sampler=10, sampling_rate_start=0.1, sampling_rate_end=1.0):
        """
        初始化集成器

        采样率是指从多数类样本中采取的比例.
        随机采样器就是根据采样率随机采取.
        下采样器使用的 RandomUnderSampler.
        上采样器使用的 SMOTE.
        不使用采样率表示使用原始数据训练分类器.

        :param n_random_sampler: 随机采样器的个数
        :param n_down_sampler: 下采样器的个数
        :param n_up_sampler: 上采样器的个数
        :param n_no_sampler: 不使用采样器的个数
        :param sampling_rate_start:开始采样率
        :param sampling_rate_end:结束采样率
        """
        self.n_random_sampler = n_random_sampler
        self.n_down_sampler = n_down_sampler
        self.n_up_sampler = n_up_sampler
        self.n_no_sampler = n_no_sampler
        self.sampling_rate_start = sampling_rate_start
        self.sampling_rate_end = sampling_rate_end

        # 采样器的总个数
        number_sampler = n_random_sampler + n_down_sampler + n_up_sampler + n_no_sampler

        # 创建基分类器
        self.classifier_ensemble = []
        for i in range(int(number_sampler)):
            self.classifier_ensemble.append(KNeighborsClassifier())  # 使用 KNN 作为基分类器

        # 保存每个分类器的预测结果
        self.all_U = []

    def fit(self, x, y):
        """
        训练集成器

        :param x:样本
        :param y:标签
        """
        # 记录评估结果
        val_history = {}
        val_history["val_acc"] = []
        val_history["val_precision"] = []
        val_history["val_recall"] = []
        val_history["val_f1"] = []
        val_history["val_auc"] = []
        val_history["val_gmean"] = []

        # 训练每一个基分类器
        start = 0
        end = self.n_random_sampler
        for i in range(start, end):
            # 随机抽样
            x_train, y_train = shuffle_data(x, y)  # 打乱数据集
            sampling_rate = self.sampling_rate_start + ((self.sampling_rate_end - self.sampling_rate_start) / (self.n_random_sampler - 1)) * i
            print("采样率：%.2f" % sampling_rate)
            data = list(zip(x_train, y_train))
            data = random.sample(data, int(sampling_rate * len(x_train)))
            x_train, y_train = zip(*data)

            # 训练对应的分类器
            self.classifier_ensemble[i].fit(x_train, y_train)

        start = end
        end = self.n_random_sampler + self.n_down_sampler
        for i in range(start, end):
            # 下采样
            x_train, y_train = shuffle_data(x, y)  # 打乱数据集
            x_train, y_train = RandomUnderSampler().fit_resample(x_train, y_train)  # 抽样

            self.classifier_ensemble[i].fit(x_train, y_train)

        start = end
        end = self.n_random_sampler + self.n_down_sampler + self.n_up_sampler
        for i in range(start, end):
            # 上采样
            x_train, y_train = shuffle_data(x, y)  # 打乱数据集
            x_train, y_train = SMOTE().fit_resample(x_train, y_train)

            self.classifier_ensemble[i].fit(x_train, y_train)

        start = end
        end = self.n_random_sampler + self.n_down_sampler + self.n_up_sampler + self.n_no_sampler
        for i in range(start, end):
            # 不采样
            x_train, y_train = shuffle_data(x, y)  # 打乱数据集
            self.classifier_ensemble[i].fit(x_train, y_train)


    def predict_proba(self, x):
        """
        对给定数据 x 进行预测

        :param x:输入样本
        :return:预测概率；列表，[[0.1 0.9], [0.8. 0.1], ...]
        """
        for i, clf in enumerate(self.classifier_ensemble):
            # 基分类器预测
            U = clf.predict_proba(x)
            self.all_U.append(U)

        y_proba = np.mean(self.all_U, axis=0)

        return y_proba
