# 描述：一些评估指标

def gmean(y_true, y_pred):
    """
    g-mean 评估指标

    :param y_true:真正的标签
    :param y_pred:预测的标签
    :return:
    """
    tp_fn = len(y_true[y_true == 1])
    tn_fp = len(y_true) - tp_fn
    tp_fp = len(y_pred[y_pred == 1])

    # 计算 tp 和 tn
    t = y_true*y_pred
    tp = len(t[t == 1])
    tn = len(y_true) - (tp_fn + tp_fp - tp)

    # TPR = TP/(TP+FN) = Recall
    tpr = tp / tp_fn

    # TNR = TN/(TN+FP)
    if tn_fp == 0:
        tnr = 0
    else:
        tnr = tn / tn_fp
    g_mean = (tpr * tnr) ** 0.5

    return g_mean
