import numpy as np


def gmean(y_true, y_pred):
    tp_fn = len(y_true[y_true == 1])
    tn_fp = len(y_true) - tp_fn
    tp_fp = len(y_pred[y_pred == 1])

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
