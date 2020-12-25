from classifier.DBUBaggingClassifier3 import DBUBaggingClassifier3
from other.metrics import gmean
from other.read_data import get_data
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from multiprocessing import Process, Manager


def kFoldTest(procnum, x, y, k, n_estimator, sampling_interval, return_dict):
    """
    k折交叉验证

    :param x:样本
    :param y:标签
    :param sampler:采样器
    :param classifier:分类器
    :param k:交叉验证折数
    """
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
        print("")
        print("-"*50)
        print("k = %d" % cur_k)
        print("训练 正样本：%d 负样本：%d" % (len(y_train[y_train == 1]), len(y_train[y_train == 0])))

        # 使其平衡
        # x_val, y_val = get_balance(x_val, y_val)

        # 分类器
        clf = DBUBaggingClassifier3(n_estimator, sampling_interval)


        # 训练
        clf.fit(x_train, y_train)

        # 测试
        print("测试 正样本：%d 负样本：%d" % (len(y_val[y_val == 1]), len(y_val[y_val == 0])))
        y_proba = clf.predict_proba(x_val, y_val=y_val)
        y_pred = np.argmax(y_proba, axis=1)

        # 评估测试集
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
        val_gmean = gmean(y_val, y_pred)


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
    for k in val_history.keys():
        # print("%.4f" % (np.mean(val_history[k])))
        print("%.4f ±%.4f" % (np.mean(val_history[k]), np.std(val_history[k])))

    return_dict[procnum] = val_history['auc_value']




if __name__ == '__main__':
    # 获取原始数据
    x, y = get_data([0], -1,  "/大于2小于5/vehicle.dat")




    # 一次性完成
    # sampling_rate = [0.1, 0.2, 0.3, 0.4]
    # return_dict = [Manager().dict() for i in range(len(sampling_rate))]
    # all_process = []
    # for i, rate in enumerate(sampling_rate):
    #     for j in range(10):
    #         p = Process(target=kFoldTest, args=(j, x, y, 2, 15, rate, return_dict[i]))
    #         all_process.append(p)
    #         p.start()
    # # 等待所有进程完成
    # for p in all_process:
    #     p.join()
    # # 最后的结果是多个进程返回值的集合
    # for i, rate in enumerate(sampling_rate):
    #     print("sampling_rate:%.2f" % sampling_rate[i])
    #     a = return_dict[i].values()
    #     a = np.array(a).reshape((len(a) * len(a[0]),))
    #     print("[", end="")
    #     for auc in a:
    #         print("%.2f," % auc, end="")
    #     print("]")



    # 多进程并发
    return_dict = Manager().dict()    # 用于保存各个进程计算结果
    all_process = []
    for i in range(8):
        p = Process(target=kFoldTest, args=(i, x, y, 3, 15, 0.2, return_dict))
        all_process.append(p)
        p.start()
    # 等待所有进程完成-
    for p in all_process:
        p.join()
    a = return_dict.values()
    a = np.array(a).reshape((len(a) * len(a[0]),))
    print("[", end="")
    for auc in a:
        print("%f," % auc, end="")
    print("]")






