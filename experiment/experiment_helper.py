from sklearn.model_selection import KFold

from compare import mymetrics

from sklearn import metrics
import numpy as np



def save_metric(history: dict, y_val, y_pred, y_proba):
    """
    保存评估指标到字典中

    :param history: 字典对象
    :param y_val: 真实标签
    :param y_pred: 预测标签
    :param y_proba: 预测概率，如(0.2, 0.8)
    """
    # 创建列表
    if len(history.keys()) == 0:
        # history["acc"] = []
        # history["precision"] = []
        # history["recall"] = []
        history["f1score"] = []
        history["auc"] = []
        history["gmean"] = []
        history["bACC"] = []

    # 计算评估指标
    # val_acc = metrics.accuracy_score(y_val, y_pred)
    # val_precision = metrics.precision_score(y_val, y_pred)
    # val_recall = metrics.recall_score(y_val, y_pred)
    val_f1 = metrics.f1_score(y_val, y_pred)
    auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])
    val_gmean = mymetrics.gmean(y_val, y_pred)
    val_bAcc = metrics.balanced_accuracy_score(y_val, y_pred)

    # 存储结果
    # history["acc"].append(val_acc)
    # history["precision"].append(val_precision)
    # history["recall"].append(val_recall)
    history["f1score"].append(val_f1)
    history["auc"].append(auc_value)
    history["gmean"].append(val_gmean)
    history['bACC'].append(val_bAcc)



def show_last_data(history, blank_col=False):
    """
    打印输出最后一条记录
    """
    if blank_col:
        header = "|%-20s" % ""
        split_line = "|%-20s" % "---"
        value = "|%-20s" % ""
    else:
        header = ""
        split_line = ""
        value = ""
    for key in history.keys():
        header += "|%-20s" % key
        split_line += "|%-20s" % "---"
        value += "|%-20.4f" % history[key][-1]
    print(header)
    print(split_line)
    print(value)

def show_mean_data(history, blank_col=False):
    """
    打印评估指标平均值
    """
    if blank_col:
        header = "|%-20s" % ""
        split_line = "|%-20s" % "---"
        value = "|%-20s" % ""
    else:
        header = ""
        split_line = ""
        value = ""
    for key in history.keys():
        header += "|%-20s" % key
        split_line += "|%-20s" % "---"
        value += "|%-20s" % ("%.4f ±%.4f" % (np.mean(history[key]), np.std(history[key])))
    print(header)
    print(split_line)
    print(value)

