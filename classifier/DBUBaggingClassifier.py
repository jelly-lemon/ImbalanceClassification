import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from myidea.DBUSampler import DBUSampler


class DBUBaggingClassifier:
    """
    Density-based Under-Sampling Bagging Classifier
    基于密度的下采样集成分类器，自带下采样器.
    """

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
        for i in range(self.n_estimator):
            sampling_rate = 0.1 + i*(0.6-0.1)/(self.n_estimator-1)
            x_train, y_train = DBUSampler(sampling_rate=sampling_rate).fit_resample(x, y)
            self.classifiers[i].fit(x_train, y_train)


    def predict_proba(self, x):
        """
        对给定数据 x 进行预测

        :param x:输入样本
        :return:预测概率；列表，[[0.1 0.9], [0.8. 0.1], ...]
        """
        all_y_prob = []
        for clf in self.classifiers:
            # 基分类器预测
            y_prob = clf.predict_proba(x)
            all_y_prob.append(y_prob)

        y_proba = np.mean(np.array(all_y_prob), axis=0)

        return y_proba

