"""
在数据集上运行不平衡分类算法
"""

import random

from classifier.AdaSamplingBaggingClassifier import AdaSamplingBaggingClassifier
from metrics import gmean
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
from myidea.DBUSampler import DBUSampler



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


def my_print(s):
    global kFold_show_info
    if kFold_show_info:
        print(s)


def kFoldTest(x, y, sampler, classifier, k=10):
    """
    k折交叉验证(该函数会打乱数据顺序)

    :param x:样本
    :param y:标签
    :param sampler:采样器
    :param classifier:分类器
    :param k:交叉验证折数
    """

    my_print("-"*60)
    my_print("%s-%s" % (sampler, classifier))

    # 记录评估结果
    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []
    val_history["val_gmean"] = []

    # 采样器
    if sampler == "DBU":
        x, y = DBUSampler(sampling_rate=0.1).fit_resample(x, y)  # 抽样
    elif sampler == "RUS":
        # replacement=False 表示不放回抽样
        x, y = RandomUnderSampler(replacement=False).fit_resample(x, y)  # 抽样
    elif sampler == "SMOTE":
        x, y = SMOTE().fit_resample(x, y)
    if sampler != "":
        my_print("平衡后：%d/%d=%.2f" % (len(y[y == 1]), len(y[y == 0]), (len(y[y == 1]) / len(y[y == 0]))))

    # k折交叉
    kf = KFold(n_splits=k, shuffle=True)  # 混洗数据
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        # 划分数据
        cur_k += 1  # 当前第几折次交叉验证
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]
        my_print("[k = %d]" % cur_k)
        my_print("训练 正样本：%d 负样本：%d IR = %.2f" % (len(y_train[y_train == 1]), len(y_train[y_train == 0]), len(y_train[y_train == 1]) / len(y_train[y_train == 0])))

        # 分类器
        if classifier.lower() == "knn":
            clf = KNeighborsClassifier()
        elif classifier.lower() == "dt":
            clf = DecisionTreeClassifier()
        elif classifier.lower() == "svc":
            # probability=True 表示可以计算得到概率
            clf = SVC(probability=True)
        elif classifier == "RandomForestClassifier" or "RFC" or "RandomForest":
            clf = RandomForestClassifier()
        elif classifier == "BaggingClassifier":
            clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), bootstrap=True)
        elif classifier == "AdaBoostClassifier" or "AdaBoost":
            clf = AdaBoostClassifier()
        elif classifier == "EasyEnsembleClassifier" or "EasyEnsemble":
            clf = EasyEnsembleClassifier()
        elif classifier == "BalancedBaggingClassifier" or "BalancedBagging":
            clf = BalancedBaggingClassifier()
        elif classifier == "AdaSamplingBaggingClassifier":
            clf = AdaSamplingBaggingClassifier(15)

        # 训练
        clf.fit(x_train, y_train)

        # 测试
        my_print("测试 正样本：%d 负样本：%d IR = %.2f" % (len(y_val[y_val == 1]), len(y_val[y_val == 0]), len(y_val[y_val == 1]) / len(y_val[y_val == 0])))
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
        my_print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f val_gmean:%.2f" %
              (val_acc, val_precision, val_recall, val_f1, auc_value, val_gmean))

    # 统计，求平均值和标准差
    my_print("%s-%s 平均数据" % (sampler, classifier))
    header = ""
    value = ""
    for k in val_history.keys():
        header += "%-20s" % k
        value += "%-20s" % ("%.4f ±%.4f" % (np.mean(val_history[k]), np.std(val_history[k])))
    my_print(header)
    my_print(value)

    # 打印出关键输出，方便复制到 Markdown
    if sampler != "":
        model_name = "%s-%s" % (sampler, classifier)
    else:
        model_name = classifier
    f1score = "%.4f ±%.4f" % (np.mean(val_history["val_f1"]), np.std(val_history["val_f1"]))
    auc = "%.4f ±%.4f" % (np.mean(val_history["auc_value"]), np.std(val_history["auc_value"]))
    gmean_value = "%.4f ±%.4f" % (np.mean(val_history["val_gmean"]), np.std(val_history["val_gmean"]))
    copy_data = "|%-20s|%-20s|%-20s|%-20s|" % (model_name, f1score, auc, gmean_value)
    my_print("[copy]")
    my_print(copy_data)
    my_print("-" * 60)

    return copy_data

def one_step():
    global kFold_show_info
    kFold_show_info = False

    x, y = read_data.get_data([3], -1, "page-blocks.dat", show_info=True)

    k = 5
    while len(y)/k < 100:
        x, y = read_data.upsampling_copy(x, y, 1)
        print("复制一份后：%d/%d" % (len(y[y == 1]), len(y[y == 0])))

    print("|%-20s|%-20s|%-20s|%-20s|" % ("", "f1score", "auc", "gmean"))
    print("|%-20s|%-20s|%-20s|%-20s|" % ("----", "----", "----", "----"))
    method = ("KNN", "DT", "SVC", "RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging")
    for key in method:
        result = kFoldTest(x.copy(), y.copy(), sampler="", classifier=key, k=k)
        print(result)




kFold_show_info = False # k折交叉验证是否打印中间信息
if __name__ == '__main__':
    one_step()

    # # 获取原始数据
    # x, y = read_data.get_data(neg_no=[1], pos_no=-1,  file_name="banana.dat")
    #
    # # k 折交叉验证
    # result = kFoldTest(x, y, sampler="", classifier="BalancedBagging", k=5)
    # print(result)

