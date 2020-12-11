import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from read_data import get_data, shuffle_data


class AdaC2Classifier:
    def __init__(self, T, C):
        self.T = T  # 总迭代次数
        self.C = C
        self.all_h = [] # 所有基分类器
        self.all_alpha = [-1 for i in range(T)]  # 所有 α

    def fit(self, X, Y):
        C = self.C

        M = len(Y)
        # 初始化样本权重
        D = [1/M for j in range(M)]


        # 开始迭代
        for t in range(self.T):

            # 训练基分类器
            clf = DecisionTreeClassifier(class_weight=C)
            clf.fit(X, Y, sample_weight=D)
            y_pred = clf.predict(X)
            self.all_h.append(clf)

            # 计算阿尔法
            alpha = self.get_alpha(D, C, Y, y_pred)
            self.all_alpha[t] = alpha

            # 计算归一化因子
            Z = self.get_normalization_factor(D, C, alpha, Y, y_pred)

            # 更新权重
            D = self.get_new_weights(D, C, alpha, Y, y_pred, Z)   # 更新权重


    def get_new_weights(self, D, C, alpha, Y, y_pred, Z):
        M = len(Y)

        # 新的权重
        new_D = [-1 for i in range(M)]
        for i in range(M):
            if Y[i] == 0:
                y = -1
            else:
                y = 1
            if y_pred[i] == 0:
                h = -1
            else:
                h = 1

            new_D[i] = C[Y[i]] * D[i] * np.exp(-alpha*y*h) / Z

        return new_D


    def predict(self, X):
        all_y_pred = []
        for i in range(self.T):
            y_pred = self.all_h[i].predict(X)
            all_y_pred.append(y_pred)

        Y = []
        for i, x in enumerate(X):
            sum = 0
            for j in range(self.T):
                if all_y_pred[j][i] == 0:
                    h = -1
                else:
                    h = 1

                sum += self.all_alpha[j] * h

            Y.append(np.sign(sum))
        Y = np.array(Y, dtype=np.int8)
        Y[Y == -1] = 0

        return Y



    def get_alpha(self, D, C, Y, y_pred):
        # 样本数量
        M = len(Y)

        # 先求分子分母
        sum_up = 0  # 分子
        sum_down = 0  # 分母
        for i in range(M):
            if Y[i] == y_pred[i]:
                sum_up += C[y[i]] * D[i]
            else:
                sum_down += C[y[i]] * D[i]

        # 如果分母为0
        if sum_down == 0:
            sum_down = 0.1

        alpha = np.log(sum_up/sum_down)/2

        return alpha





    def get_normalization_factor(self, D, C, alpha, Y, y_pred):
        # 样本数量
        M = len(Y)

        Z = 0
        for i in range(M):
            if Y[i] == 0:
                y = -1
            else:
                y = 1
            if y_pred[i] == 0:
                h = -1
            else:
                h = 1

            Z += C[Y[i]] * D[i] * np.exp(-alpha*y*h)


        return Z




if __name__ == '__main__':
    x, y = get_data([1], -1, "yeast.dat")

    x, y = shuffle_data(x, y)
    x = np.array(x)
    y = np.array(y, dtype=np.int8)
    # y[y == 0] = -1

    # 代价项
    C = {1: 0.1, 0: 1}

    x_train = x[:-100]
    y_train = y[:-100]

    x_val = x[-100:]
    y_val = y[-100:]

    clf = AdaC2Classifier(15, C)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    print("预测结果：")
    print(y_pred)

    print("原本标签：")
    print(y_val)

    val_acc = metrics.accuracy_score(y_val, y_pred)
    val_precision = metrics.precision_score(y_val, y_pred)
    val_recall = metrics.recall_score(y_val, y_pred)
    val_f1 = metrics.f1_score(y_val, y_pred)

    print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f" %
          (val_acc, val_precision, val_recall, val_f1))