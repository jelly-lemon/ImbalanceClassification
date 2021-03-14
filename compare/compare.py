"""
在数据集上运行不平衡分类算法
"""

import random

import mymetrics
from myidea.AdaSamplingBaggingClassifier import AdaSamplingBaggingClassifier
from data import read_data
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from other_people.DBUSampler import DBUSampler



def get_balance(x, y):
    """
    随机抽取多数类样本中的部分，使其数量和少数类相同

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



def kFoldTest(x, y, sampler, classifier, k=10, show_info=False):
    """
    k折交叉验证(该函数会打乱数据顺序，无需手动再打乱)

    :param x:样本
    :param y:标签
    :param sampler:采样器
    :param classifier:分类器
    :param k:交叉验证折数
    """
    if show_info:
        print("-" * 60)
        print("%s-%s" % (sampler, classifier))

    # 记录评估结果
    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []
    val_history["val_gmean"] = []
    val_history["bAcc"] = []

    # k折交叉
    kf = KFold(n_splits=k, shuffle=True)  # 混洗数据
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        # 划分数据
        cur_k += 1  # 当前第几折次交叉验证
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]
        if sampler != "" and show_info:
            print("采样前：%d/%d=%.2f" % (len(y_train[y_train == 1]), len(y_train[y_train == 0]),
                                      (len(y_train[y_train == 1]) / len(y_train[y_train == 0]))))

        # 采样器
        if sampler in ("DBU", "DUS"):
            x_train, y_train = DBUSampler(show_info=True).fit_resample(x_train, y_train)  # 抽样
        elif sampler == "RUS":
            # replacement=False 表示不放回抽样
            x_train, y_train = RandomUnderSampler(replacement=False).fit_resample(x_train, y_train)  # 抽样
        elif sampler == "SMOTE":
            x_train, y_train = SMOTE().fit_resample(x_train, y_train)
        if sampler != "" and show_info:
            print("采样后：%d/%d=%.2f" % (len(y_train[y_train == 1]), len(y_train[y_train == 0]),
                                      (len(y_train[y_train == 1]) / len(y_train[y_train == 0]))))
        if show_info:
            print("[k = %d]" % cur_k)
            print("训练 正样本：%d 负样本：%d IR = %.2f" % (len(y_train[y_train == 1]), len(y_train[y_train == 0]),
                                                  len(y_train[y_train == 1]) / len(y_train[y_train == 0])))

        # 分类器
        if classifier.lower() == "knn":
            clf = KNeighborsClassifier()
        elif classifier.lower() == "dt":
            clf = DecisionTreeClassifier()
        elif classifier.lower() == "svc":
            # probability=True 表示可以计算得到概率
            clf = SVC(probability=True)
        elif classifier in ("RandomForestClassifier", "RFC", "RandomForest"):
            clf = RandomForestClassifier(n_estimators=3)
        elif classifier == "BaggingClassifier":
            clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), bootstrap=True)
        elif classifier in ("AdaBoostClassifier", "AdaBoost"):
            clf = AdaBoostClassifier(n_estimators=3)
        elif classifier in ("EasyEnsembleClassifier", "EasyEnsemble"):
            clf = EasyEnsembleClassifier(n_estimators=3)
        elif classifier in ("BalancedBaggingClassifier", "BalancedBagging"):
            clf = BalancedBaggingClassifier(n_estimators=5)
        elif classifier == "AdaSamplingBaggingClassifier":
            clf = AdaSamplingBaggingClassifier(15)

        # 训练
        clf.fit(x_train, y_train)

        # 测试
        if show_info:
            print("测试 正样本：%d 负样本：%d IR = %.2f" % (
                len(y_val[y_val == 1]), len(y_val[y_val == 0]), len(y_val[y_val == 1]) / len(y_val[y_val == 0])))
        y_proba = clf.predict_proba(x_val)
        y_pred = np.argmax(y_proba, axis=1)

        # 评估测试集
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
        val_gmean = mymetrics.gmean(y_val, y_pred)
        val_bAcc = metrics.balanced_accuracy_score(y_val, y_pred)

        # 存储评估结果
        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)
        val_history["val_gmean"].append(val_gmean)
        val_history['bAcc'].append(val_bAcc)


        # 打印输出每折的评估情况
        if show_info:
            print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f val_gmean:%.2f" %
                  (val_acc, val_precision, val_recall, val_f1, auc_value, val_gmean))

    # 统计，求平均值和标准差
    header = ""
    value = ""
    for k in val_history.keys():
        header += "%-20s" % k
        value += "%-20s" % ("%.4f ±%.4f" % (np.mean(val_history[k]), np.std(val_history[k])))
    if show_info:
        print("%s-%s 平均数据" % (sampler, classifier))
        print(header)
        print(value)

    # 打印出关键输出，方便复制到 Markdown
    if sampler != "":
        model_name = "%s-%s" % (sampler, classifier)
    else:
        model_name = classifier

    all_data = "|%-20s" % model_name
    key_data = "|%-20s" % model_name
    for k in val_history.keys():
        t = "|%-20s" % ("%.4f ±%.4f" % (np.mean(val_history[k]), np.std(val_history[k])))
        all_data += t
        if k in ("val_f1", "auc_value", "val_gmean", "bAcc"):
            key_data += t

    if show_info:
        print(all_data)
        print(key_data)
        print("-" * 60)

    return all_data, key_data


def one_step():
    """
    一步到位运行所有对比方法
    """
    x, y = read_data.get_data([6], -1, "ecoli.dat", show_info=True)

    k = 5   # 交叉验证次数
    # 期望每折交叉验证样本数量 >= 100
    # while len(y) / k < 100:
    #     x, y = read_data.upsampling_copy(x, y, 1)
    #     print("复制一份后：%d/%d" % (len(y[y == 1]), len(y[y == 0])))

    print("|%-20s|%-20s|%-20s|%-20s|%-20s" % ("", "f1score", "auc", "gmean", "bACC"))
    print("|%-20s|%-20s|%-20s|%-20s|%-20s" % ("----", "----", "----", "----", "----"))

    method = ("KNN", "DT", "RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging")
    sampling = ("RUS", "SMOTE")
    for m in method:
        if m in ("KNN", "DT"):
            for s in sampling:
                result = kFoldTest(x.copy(), y.copy(), sampler=s, classifier=m, k=k)
                print(result[1])
        else:
            result = kFoldTest(x.copy(), y.copy(), sampler="", classifier=m, k=k)
            print(result[1])



if __name__ == '__main__':
    one_step()

    # # 获取原始数据
    # x, y = read_data.get_data(neg_no=[1], pos_no=-1,  file_name="banana.dat")
    #
    # # k 折交叉验证
    # result = kFoldTest(x, y, sampler="", other_people="BalancedBagging", k=5)
    # print(result)
