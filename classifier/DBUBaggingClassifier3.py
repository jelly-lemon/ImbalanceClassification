import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from compare.equation import get_R_matrix, get_S_matrix
from compare import PSOEvolutor
from compare.mymetrics import gmean
from myidea.DBUSampler import DBUSampler


class DBUBaggingClassifier3:
    """
    DBUBaggingClassifier + 粒子群优化预测结果
    设定采样间隔范围
    """

    def __init__(self, n_estimator, sampling_interval=0.2):
        """

        :param n_estimator:基分类器的数量
        """
        print("sampling_interval %.2f" % sampling_interval)
        self.n_estimator = n_estimator
        self.sampling_interval = sampling_interval
        self.classifiers = []
        for i in range(self.n_estimator):
            self.classifiers.append(KNeighborsClassifier())

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
        for i in range(self.n_estimator):
            y = np.array(y)
            balance_rate = len(y[y == 0]) / len(y[y == 1])

            interval = self.sampling_interval

            start_rate = balance_rate - interval
            if start_rate < 0.1:
                start_rate = 0.1
            end_rate = balance_rate + interval
            # 采样率太高，采样时间很长
            if end_rate > 0.7:
                end_rate = 0.7

            # 采样率
            sampling_rate = start_rate + i * (end_rate - start_rate) / (self.n_estimator - 1)
            print("%.2f" % sampling_rate)

            # 基于密度采样
            x_train, y_train = DBUSampler(sampling_rate=sampling_rate).fit_resample(x, y)

            # 随机采样
            # data = list(zip(x, y))
            # data = random.sample(data, int(sampling_rate * len(x)))
            # x_train, y_train = zip(*data)
            # x_train, y_train = np.array(x_train), np.array(y_train)

            # 根据采样结果训练模型
            self.classifiers[i].fit(x_train, y_train)

    def predict_proba(self, x, y_val=None):
        """
        对给定数据 x 进行预测

        :param x:输入样本
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

        # 基分类器预测
        for clf in self.classifiers:
            U = clf.predict_proba(x)  # 对样本进行分类
            q = clf.predict_proba(c)  # 对聚类质心进行分类
            # 保存结果
            self.all_U.append(U)
            self.all_q.append(q)

        # 先看看进化之前的效果
        if y_val is not None:
            y_proba = np.mean(np.array(self.all_U), axis=0)
            y_pred = np.argmax(y_proba, axis=1)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            val_precision = metrics.precision_score(y_val, y_pred)
            val_recall = metrics.recall_score(y_val, y_pred)
            val_f1 = metrics.f1_score(y_val, y_pred)
            auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
            val_gmean = gmean(y_val, y_pred)
            print("进化前：")
            print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f val_gmean:%.2f" %
                  (val_acc, val_precision, val_recall, val_f1, auc_value, val_gmean))

            # 优化
            self.all_U = PSOEvolutor(self.S, self.R, self.all_q).evolve(self.all_U, 100)

        # 求平均
        y_proba = np.mean(np.array(self.all_U), axis=0)

        return y_proba
