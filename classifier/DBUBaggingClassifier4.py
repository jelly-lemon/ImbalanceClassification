import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from evolutor.PSOEvolutor2 import PSOEvolutor2
from other.metrics import gmean
from sampler.DBUSampler import DBUSampler


class DBUBaggingClassifier4:
    """
    DBUBaggingClassifier + 粒子群优化每个分类器的权重
    """

    def __init__(self, n_estimator):
        """

        :param n_estimator:基分类器的数量
        """
        self.n_estimator = n_estimator
        self.classifiers = []
        for i in range(self.n_estimator):
            self.classifiers.append(KNeighborsClassifier())


        # 保存预测、聚类结果
        self.all_U = []  # 预测结果概率 [0.5,0.6],...,[0.7,0.1]]


    def fit(self, x, y):
        """
        训练集成器

        :param x:样本
        :param y:标签
        """
        for i in range(self.n_estimator):
            sampling_rate = 0.1 + i*(0.6-0.1)/(self.n_estimator-1)
            x_train, y_train = DBUSampler(sampling_rate=sampling_rate).fit_resample(x, y)
            # x_train, y_train =  RandomUnderSampler().fit_resample(x, y)
            self.classifiers[i].fit(x_train, y_train)

        # 基分类器预测
        all_U = []
        for clf in self.classifiers:
            U = clf.predict_proba(x)  # 对样本进行分类
            # 保存结果
            all_U.append(U)

        # 先看看进化之前的效果
        y_val = y
        y_proba = np.mean(np.array(all_U), axis=0)
        y_pred = np.argmax(y_proba, axis=1)
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
        val_gmean = gmean(y_val, y_pred)

        print("---------训练集上的的表现---------")
        print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f val_gmean:%.2f" %
              (val_acc, val_precision, val_recall, val_f1, auc_value, val_gmean))




        self.weight = PSOEvolutor2(all_U, y_val).evolve(max_steps=200)

        # 进化后
        all_y_prob = all_U
        y_proba = None
        for i, w in enumerate(self.weight):
            if y_proba is None:
                y_proba = all_y_prob[i] * w
            else:
                y_proba += all_y_prob[i] * w

        y_pred = np.argmax(y_proba, axis=1)
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
        val_gmean = gmean(y_val, y_pred)

        print("---------进化后的的表现---------")
        print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f val_gmean:%.2f" %
              (val_acc, val_precision, val_recall, val_f1, auc_value, val_gmean))

    def predict_proba(self, x, y_val):
        """
        对给定数据 x 进行预测

        :param x:输入样本
        :return:预测概率；列表，[[0.1 0.9], [0.8. 0.1], ...]
        """
        # 基分类器预测
        for clf in self.classifiers:
            U = clf.predict_proba(x)  # 对样本进行分类
            # 保存结果
            self.all_U.append(U)

        # y_proba = np.mean(np.array(self.all_U), axis=0)

        # # 先看看进化之前的效果
        # y_proba = np.mean(np.array(self.all_U), axis=0)
        # y_pred = np.argmax(y_proba, axis=1)
        # val_acc = metrics.accuracy_score(y_val, y_pred)
        # val_precision = metrics.precision_score(y_val, y_pred)
        # val_recall = metrics.recall_score(y_val, y_pred)
        # val_f1 = metrics.f1_score(y_val, y_pred)
        # auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
        # val_gmean = gmean(y_val, y_pred)
        # print("---------进化前的表现---------")
        # print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f val_gmean:%.2f" %
        #       (val_acc, val_precision, val_recall, val_f1, auc_value, val_gmean))
        #
        # # 优化
        # all_y_prob = self.all_U
        # self.weight = PSOEvolutor2(all_y_prob, y_val).evolve(max_steps=500)
        #

        # 根据权重求预测结果
        all_y_prob = self.all_U
        y_proba = None
        for i, w in enumerate(self.weight):
            if y_proba is None:
                y_proba = all_y_prob[i] * w
            else:
                y_proba += all_y_prob[i] * w


        return y_proba

