# Description: RUS 进行欠采样，使数据平衡；用平衡后的数据训练分类器 DT
# Author: Lin Meng

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from read_data import yeast1

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

        # 构建模型，训练
        x_train, y_train = RandomUnderSampler().fit_resample(x_train, y_train)  # 随机抽样
        clf = DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        # 测试
        y_pred = clf.predict(x_val)
        y_proba = clf.predict_proba(x_val)

        # 评估测试集
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_proba[:, 1])
        auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])

        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))






