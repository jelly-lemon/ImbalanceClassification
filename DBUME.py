# 描述：基于密度的下采样多目标优化集成器
# 作者：Jelly Lemon
import random

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

from equation import get_R_matrix, get_S_matrix
from read_data import shuffle_data


class DBUME():
    """
    Density-based undersampling multi-objective optimization ensemble
    基于密度的下采样多目标优化集成器，但是这个类没有集成下采样器

    """

    def __init__(self, random_sampler, down_sampler, up_sampler, no_sampler, start, end):
        """
        初始化集成器

        :param n_classifier:基分类器数量
        """
        self.random_sampler = random_sampler
        self.down_sampler = down_sampler
        self.up_sampler = up_sampler
        self.no_sampler = no_sampler
        self.start = start
        self.end = end
        number_clf = random_sampler + down_sampler + up_sampler + no_sampler

        # 基分类器
        self.classifier_ensemble = []
        for i in range(number_clf):
            self.classifier_ensemble.append(DecisionTreeClassifier())  # 使用 KNN 作为基分类器

        # 保存预测、聚类结果
        self.all_U = []  # 预测结果概率 [0.5,0.6],...,[0.7,0.1]]
        self.S = []  # 样本相似度
        self.R = []  # 聚类结果，one-hot矩阵
        self.all_q = []

    def fit(self, x, y):
        """
        训练集成器

        :param x:样本
        :param y:标签
        """
        # 训练每一个基分类器
        for i in range(0, self.random_sampler):
            x_train, y_train = shuffle_data(x, y)  # 打乱数据集
            x_train, y_train = RandomUnderSampler().fit_resample(x_train, y_train)  # 抽样

            # cur_weight = (init_weight - end_weight) * (max_steps - iter) / max_steps + end_weight
            sampling_rate = self.start + ((self.end - self.start) / (self.random_sampler - 1)) * i
            print("采样率：%.2f" % sampling_rate)

            data = list(zip(x_train, y_train))
            data = random.sample(data, int(sampling_rate * len(x_train)))
            x_train, y_train = zip(*data)

            self.classifier_ensemble[i].fit(x_train, y_train)

        for i in range(self.random_sampler, self.random_sampler + self.down_sampler):
            x_train, y_train = shuffle_data(x, y)  # 打乱数据集
            x_train, y_train = RandomUnderSampler().fit_resample(x_train, y_train)  # 抽样
            self.classifier_ensemble[i].fit(x_train, y_train)

        for i in range(self.random_sampler + self.down_sampler,
                       self.random_sampler + self.down_sampler + self.up_sampler):
            x_train, y_train = shuffle_data(x, y)  # 打乱数据集
            x_train, y_train = SMOTE().fit_resample(x_train, y_train)
            self.classifier_ensemble[i].fit(x_train, y_train)

        for i in range(self.random_sampler + self.down_sampler + self.up_sampler,
                       self.random_sampler + self.down_sampler + self.up_sampler + self.no_sampler):
            x_train, y_train = shuffle_data(x, y)  # 打乱数据集
            self.classifier_ensemble[i].fit(x_train, y_train)

    def predict_proba(self, x, y_val):
        """
        对给定数据 x 进行预测

        :param x:输入样本
        :param y: 我用来验证的
        :param use_genetic:
        :return:预测概率；列表，[[0.1 0.9], [0.8. 0.1], ...]
        """
        # 计算样本相似度
        self.S = get_S_matrix(x)  # 样本相似度

        # 对样本进行聚类
        kmeans_cluster = KMeans(n_clusters=30)
        kmeans_cluster.fit(x)
        cluster_res = kmeans_cluster.predict(x)  # 聚类结果
        n_cluster_center = len(kmeans_cluster.cluster_centers_)
        self.R = get_R_matrix(cluster_res, n_cluster_center)  # 聚类结果矩阵
        c = kmeans_cluster.cluster_centers_  # 获取聚类质心

        # 计算相似度矩阵，聚类结果，聚类质心，基分类器预测结果
        for i, clf in enumerate(self.classifier_ensemble):
            # 基分类器预测
            U = clf.predict_proba(x)    # 对样本进行分类
            q = clf.predict_proba(c)    # 对聚类质心进行分类

            # 保存结果
            self.all_U.append(U)
            self.all_q.append(q)

        # 进化----------------------------------------------
        # print("进化之前的预测结果表现：")
        # y_proba = np.mean(self.all_U, axis=0)
        # y_pred = np.argmax(y_proba, axis=1)
        # val_acc = metrics.accuracy_score(y_val, y_pred)
        # val_precision = metrics.precision_score(y_val, y_pred)
        # val_recall = metrics.recall_score(y_val, y_pred)
        # val_f1 = metrics.f1_score(y_val, y_pred)
        # auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
        # val_gmean = gmean(y_val, y_pred)
        #
        # print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f gmean:%.2f" %
        #       (val_acc, val_precision, val_recall, val_f1, auc_value, val_gmean))

        # 遗传算法进行优化
        # print("开始进化...")
        # 遗传算法
        # self.all_U = NSGAII(100, solution=self.all_U, R=self.R, S=self.S, all_q=self.all_q).evolute()   # NSGA-II


        # 粒子群
        # self.all_U = PSO(self.S, self.R, self.all_q).evolve(self.all_U, 10)
        # 进化---------------------------------------------------------

        # 优化的结果求平均
        y_prob = np.mean(self.all_U, axis=0)

        return y_prob


