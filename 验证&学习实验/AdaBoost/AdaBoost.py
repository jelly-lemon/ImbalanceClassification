import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from data.read_data import get_data


def my_adaboost_clf(Y_train, X_train, Y_test, X_test, M=20, weak_clf=DecisionTreeClassifier(max_depth=1)):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        weak_clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)

        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        print("weak_clf_%02d train acc: %.4f"
              % (i + 1, 1 - sum(miss) / n_train))

        # Error
        err_m = np.dot(w, miss)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        miss2 = [x if x == 1 else -1 for x in miss]  # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)

        # Add to prediction
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    pred_train = (pred_train > 0) * 1
    pred_test = (pred_test > 0) * 1

    print("My AdaBoost clf train accuracy: %.4f" % (sum(pred_train == Y_train) / n_train))
    print("My AdaBoost clf test accuracy: %.4f" % (sum(pred_test == Y_test) / n_test))

if __name__ == '__main__':
    x, y = get_data([1], -1, "/可调节/movement_libras.dat")

    # k折交叉
    kf = KFold(n_splits=2, shuffle=True)  # 混洗数据
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        # 划分数据
        cur_k += 1  # 当前第几折次交叉验证
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        print("k = %d" % cur_k)
        print("训练 正样本：%d 负样本：%d" % (len(y_train[y_train == 1]), len(y_train[y_train == 0])))

        my_adaboost_clf(y_train, x_train, y_val, x_val)