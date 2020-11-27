# 描述：基于密度的下采样多目标优化集成器
# 作者：Jelly Lemon

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from equation import get_R_matrix, get_S_matrix
from read_data import yeast1
from NSGAII import NSGAII


class DBUME():
    """
    Density-based undersampling multi-objective optimization ensemble
    基于密度的下采样多目标优化集成器

    """

    def __init__(self, n_classifier, n_cluster):
        """
        初始化集成器

        :param n_classifier:分类器数量
        :param n_cluster:聚类器数量
        """
        self.n_classifier = n_classifier
        self.n_cluster = n_cluster

        # 基分类器
        self.classifier_ensemble = []
        for i in range(self.n_classifier):
            self.classifier_ensemble.append(KNeighborsClassifier())  # 使用 KNN 作为基分类器

        # 保存预测、聚类结果
        self.all_U = []  # 预测结果概率 [0.5,0.6],...,[0.7,0.1]]
        self.all_S = []  # 样本相似度
        self.all_R = []  # 聚类结果，one-hot矩阵
        self.all_Q = []  # 聚类质心

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
        val_history["auc_value"] = []

        # 训练每一个基分类器
        for i, clf in enumerate(self.classifier_ensemble):
            # 欠采样，训练
            x_train, y_train = RandomUnderSampler().fit_resample(x, y)  # 随机抽样
            clf.fit(x_train, y_train)

    def predict_proba(self, x):
        """
        对给定数据 x 进行预测

        :param x:输入样本
        :return:预测概率；列表，[[0.1 0.9], [0.8. 0.1], ...]
        """
        # 单独预测
        for i, clf in enumerate(self.classifier_ensemble):
            U = y_prob = clf.predict_proba(x)

            # k-means 进行聚类
            kmeans_cluster = KMeans(n_clusters=self.n_cluster)
            kmeans_cluster.fit(y_prob)
            cluster_res = kmeans_cluster.predict(y_prob)
            R = get_R_matrix(cluster_res)  # 聚类结果矩阵
            Q = kmeans_cluster.cluster_centers_  # 获取聚类质心
            S = get_S_matrix(y_prob)  # 样本相似度

            # 保存结果
            self.all_U.append(U)
            self.all_R.append(R)
            self.all_Q.append(Q)
            self.all_S.append(S)

        # 遗传算法进行优化
        all_y_prob = self.evolute(self.all_U)

        # 优化的结果求平均
        y_prob = np.mean(all_y_prob, axis=0)

        return y_prob

    def evolute(self, all_y_prob):
        """
        进化预测结果

        :param all_y_prob:所有基分类器的预测结果
        :return:进化后的预测结果；返回数据和输入数据长得一样
        """
        evolutor = NSGAII(10, pop_size=self.n_classifier, solution=all_y_prob, n_cluster=self.n_cluster,
                          R=self.all_R, S=self.all_S, Q=self.all_Q)
        new_y_prob = evolutor.evolute()

        return new_y_prob


if __name__ == '__main__':
    # 原始数据
    x, y = yeast1()
    x = np.array(x)
    y = np.array(y)
    print("总数据 pos:%d neg:%d" % (len(y[y == 1]), len(y[y == 0])))

    # 记录评估结果
    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []

    # k折交叉
    kf = KFold(n_splits=10, shuffle=True)
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        # 划分数据
        cur_k += 1
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]
        print("k = %d" % cur_k)

        # 构建模型，训练
        clf = DBUME(10, 6)
        clf.fit(x_train, y_train)

        # 测试
        y_proba = clf.predict_proba(x_val)
        y_pred = np.argmax(y_proba, axis=1)

        # 评估测试集
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_proba[:, 1])
        auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
        print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f" %
              (val_acc, val_precision, val_recall, val_f1, auc_value))

        # 保存在列表中
        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))
