import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from compare import mymetrics, equation
from data import read_data
from experiment.is_PSO_useful.PSOEvolutor import PSOEvolutor


def save_metric(history: dict, y_val, y_pred, y_proba):
    if len(history.keys()) == 0:
        history["val_acc"] = []
        history["val_precision"] = []
        history["val_recall"] = []
        history["val_f1"] = []
        history["auc_value"] = []
        history["val_gmean"] = []

    val_acc = metrics.accuracy_score(y_val, y_pred)
    val_precision = metrics.precision_score(y_val, y_pred)
    val_recall = metrics.recall_score(y_val, y_pred)
    val_f1 = metrics.f1_score(y_val, y_pred)
    auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
    val_gmean = mymetrics.gmean(y_val, y_pred)

    # 存储结果
    history["val_acc"].append(val_acc)
    history["val_precision"].append(val_precision)
    history["val_recall"].append(val_recall)
    history["val_f1"].append(val_f1)
    history["auc_value"].append(auc_value)
    history["val_gmean"].append(val_gmean)


def show_last_data(history):
    print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f val_gmean:%.2f" %
          (history["val_acc"][-1], history["val_precision"][-1],
           history["val_recall"][-1], history["val_f1"][-1], history["auc_value"][-1],
           history["val_gmean"][-1]))


def show_mean_data(history):
    s = ""
    for k in history.keys():
        s += "|%-20s" % ("%.4f ±%.4f" % (np.mean(history[k]), np.std(history[k])))
    print(s)


def kFoldEvolution(x, y):
    # 记录评估结果
    val_history = {}
    evo_history = {}

    kf = KFold(n_splits=5, shuffle=True)  # 混洗数据
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        # 划分数据
        cur_k += 1  # 当前第几折次交叉验证
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        # 分类器
        clf = KNeighborsClassifier()

        # 训练
        clf.fit(x_train, y_train)

        # 测试
        y_proba = clf.predict_proba(x_val)
        y_pred = np.argmax(y_proba, axis=1)

        # 进化前的表现
        save_metric(val_history, y_val, y_pred, y_proba)
        print("进化前：")
        show_last_data(val_history)

        # 进化
        s = equation.get_S_matrix(x_val)
        y_proba = PSOEvolutor(s).evolve(y_proba, max_steps=200)
        y_pred = np.argmax(y_proba, axis=1)

        # 进化后的表现
        save_metric(evo_history, y_val, y_pred, y_proba)
        print("进化后：")
        show_last_data(evo_history)

    # 统计，求平均值和标准差
    print("进化前平均：")
    show_mean_data(val_history)
    print("进化后平均：")
    show_mean_data(evo_history)


if __name__ == '__main__':
    x, y = read_data.get_data([0, 6], -1, "yeast.dat", show_info=True)

    kFoldEvolution(x, y)
