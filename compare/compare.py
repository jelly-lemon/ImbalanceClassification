# 描述：仅仅用来运行常见算法

import random

from classifier.AdaC2Classifier import AdaC2Classifier
from classifier.AdaSamplingBaggingClassifier import AdaSamplingBaggingClassifier
from classifier.HSBaggingClassifier import HSBaggingClassifier
from other.metrics import gmean
from data.read_data import get_data, upsampling
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sampler.DBUSampler import DBUSampler


def get_balance(x, y):
    """
    抽取多数类样本中的部分，使其数量和少数类相同

    :param x:样本
    :param y:标签
    :return:抽样后的(x,y)
    """
    # 先转为数组
    x, y = np.array(x), np.array(y)

    # 少数类和多数类分家
    x_majority = x[y == 1]
    x_minority = x[y == 0]

    # 随机从多数类样本中抽取少数类样本个数
    x_majority = random.sample(list(x_majority), len(x_minority))

    # 再组合成新的样本集
    x_majority, x_minority = list(x_majority), list(x_minority)
    x = x_majority + x_minority
    y = [1 for i in range(len(x_majority))] + [0 for i in range(len(x_minority))]
    x, y = np.array(x), np.array(y)

    return x, y


def kFoldTest(x, y, sampler, classifier, k=10):
    """
    k折交叉验证

    :param x:样本
    :param y:标签
    :param sampler:采样器
    :param classifier:分类器
    :param k:交叉验证折数
    """
    print("-"*60)
    print("%s-%s" % (sampler, classifier))

    # 记录评估结果
    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []
    val_history["val_gmean"] = []

    # k折交叉
    kf = KFold(n_splits=k, shuffle=True)  # 混洗数据
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        # 划分数据
        cur_k += 1  # 当前第几折次交叉验证
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]
        print("k = %d" % cur_k)
        print("训练 正样本：%d 负样本：%d" % (len(y_train[y_train == 1]), len(y_train[y_train == 0])))
        IR = len(y_train[y_train == 1]) / len(y_train[y_train == 0])
        print("IR = %.2f" % IR)

        # 采样器
        if sampler == "DBU":
            x_train, y_train = DBUSampler(sampling_rate=0.1).fit_resample(x_train, y_train)  # 抽样
        elif sampler == "RUS":
            x_train, y_train = RandomUnderSampler(replacement=False).fit_resample(x_train, y_train)  # 抽样
        elif sampler == "SMOTE":
            x_train, y_train = SMOTE(k_neighbors=2).fit_resample(x_train, y_train)

        # 分类器
        if classifier == "KNN":
            clf = KNeighborsClassifier()
        elif classifier == "DT":
            clf = DecisionTreeClassifier()
        elif classifier == "RandomForestClassifier":
            clf = RandomForestClassifier(n_estimators=15)
        elif classifier == "BaggingClassifier":
            clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=15, bootstrap=True)
        elif classifier == "AdaBoostClassifier":
            clf = AdaBoostClassifier()
        elif classifier == "EasyEnsembleClassifier":
            clf = EasyEnsembleClassifier(base_estimator=KNeighborsClassifier(), n_estimators=15)
        elif classifier == "BalancedBaggingClassifier":
            clf = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=15)
        elif classifier == "AdaSamplingBaggingClassifier":
            clf = AdaSamplingBaggingClassifier(15)


        # 训练
        clf.fit(x_train, y_train)

        # 测试
        print("测试 正样本：%d 负样本：%d" % (len(y_val[y_val == 1]), len(y_val[y_val == 0])))
        y_proba = clf.predict_proba(x_val)
        y_pred = np.argmax(y_proba, axis=1)

        # 评估测试集
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
        val_gmean = gmean(y_val, y_pred)
        # auc_value = 0
        # val_gmean = 0

        # 存储评估结果
        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)
        val_history["val_gmean"].append(val_gmean)

        # 打印输出每折的评估情况
        print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f val_gmean:%.2f" %
              (val_acc, val_precision, val_recall, val_f1, auc_value, val_gmean))

    # 统计，求平均值和标准差
    # print("%s-%s" % (sampler, classifier))
    # for k in val_history.keys():
    #     # print("%.4f" % (np.mean(val_history[k])))
    #     print("%.4f ±%.4f" % (np.mean(val_history[k]), np.std(val_history[k])))
    # print("-" * 60)

    return val_history




if __name__ == '__main__':
    # 获取原始数据
    x, y = get_data([7], [1,4,5,8],  "1到5/yeast.dat")

    x, y = upsampling(x, y, 90)


    # 多次交叉验证
    history = None
    for i in range(10):
        val_history = kFoldTest(x, y, "NO", "AdaSamplingBaggingClassifier", k=2)
        if history is None:
            history = val_history
        else:
            for k in val_history.keys():
                history[k] += val_history[k]
    val_history = history
    print("-" * 60)
    for k in val_history.keys():
        print("%.4f ±%.4f" % (np.mean(val_history[k]), np.std(val_history[k])))
    for k in val_history.keys():
        print(k)
        print(val_history[k])
    print("-" * 60)

