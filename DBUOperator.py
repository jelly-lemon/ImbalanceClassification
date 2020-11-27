# 描述：基于密度的下采样器
# 作者：Jelly Lemon

import random
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from read_data import yeast1


class DBUOperator:
    """
    Density-Based Undersampling Operator 基于密度的下采样器

    默认正样本为1（多数类），负样本为0（少数类）
    经过下采样后，正样本数量==负样本数数量

    Examples
    --------
    >>> dub = DBUOperator()
    >>> new_x, new_y = dub.fit(x, y)
    """

    def __init__(self, K_Neighbor=5, H_Neighbor=5):
        """
        初始化采样器

        :param K_Neighbor: 正样本中邻居参考数
        :param H_Neighbor: 负样本中邻居参考数
        """
        self.K_Neighbor = K_Neighbor
        self.H_Neighbor = H_Neighbor

    def fit(self, x, y):
        """
        下采样数据集，返回的格式和输入相同，只是量变少了

        :return: 采样后的数据集
        """
        # 样本分类
        self.T_p = [list(x[i]) for i in range(len(y)) if y[i] == 1]  # 正样本集
        self.T_n = [list(x[i]) for i in range(len(y)) if y[i] == 0]  # 负样本集
        self.R = len(self.T_p) / len(self.T_n)  # 样本比例（多比少，大于1）
        self.P = len(self.T_p)  # 正样本数量
        self.N = len(self.T_n)  # 负样本数量

        # 初始化相关变量
        self.delt_star = None  # 累计密度因子
        self.all_Ri = [[] for i in range(self.P)]  # 采样间隔范围
        self.all_delt_p_i = [-1] * self.P  # 样本xi在正样本集中的密度
        self.all_delt_n_i = [-1] * self.P  # 样本xi在负样本集中的密度
        self.all_Dpi = [[] for i in range(self.P)]  # 样本xi在正样本集中k邻居距离之和
        self.all_Dni = [[] for i in range(self.P)]  # 样本xi在负样本集中h邻居距离之和
        self.all_delt_i = [-1] * self.P  # 样本xi的采样边界
        self.dis_p = [[-1] * self.P] * self.P  # 正样本集合中各点之间的距离
        self.dis_n = [[-1] * self.N] * self.P  # 正样本xi到负样本集中各点之间的距离

        # 求正样本集中各点之间的距离
        print("计算正样本集中各点之间的距离...")
        for i, p1 in enumerate(self.T_p):
            for j, p2 in enumerate(self.T_p):
                d = np.sqrt(np.sum(np.square(np.array(p1) - np.array(p2))))
                self.dis_p[i][j] = d

        # 正样本xi到负样本集中各点之间的距离
        print("计算正样本xi到负样本集中各点之间的距离...")
        for i, p1 in enumerate(self.T_p):
            for j, p2 in enumerate(self.T_n):
                d = np.sqrt(np.sum(np.square(np.array(p1) - np.array(p2))))
                self.dis_n[i][j] = d

        # 计算 Ri
        print("计算Ri...")
        for i in range(1, self.P + 1):
            self.Ri(i)  # i表示样本编号，从1开始

        # 开始采样
        print("开始采样...")
        count = 0  # 当前采样个数
        T_p_new = []  # 存放采样样本的集合
        while count < self.N:
            # 随机生成 [0,1] 的一个数
            r = random.uniform(0, 1)
            for i in range(1, self.P + 1):  # j表示样本编号，从1开始
                start, end = self.Ri(i)
                if start < r <= end:
                    if self.T_p[i - 1] not in T_p_new:
                        count += 1
                        # print("加入一个样本（编号%d），还差%d个" % (i, self.N - count))
                        T_p_new.append(self.T_p[i - 1])
                        if count == self.N:
                            break

        T_new = T_p_new + self.T_n
        y_p = np.ones((len(T_p_new),), dtype=np.uint8)
        y_n = np.zeros((len(self.T_n),), dtype=np.uint8)
        y = np.concatenate((y_p, y_n))

        return T_new, y

    def D_p_i(self, i):
        """
        在正样本集中，样本i到k个邻居的距离

        例如：Dpi = [di1,di2,...,dik] = [1,1.5,...,0.8]
        :param i: 正样本中的样本i，从1开始表示
        :return: 样本i到k个邻居的距离，是一个列表
        """
        # 如果之前没有计算过
        if len(self.all_Dpi[i - 1]) == 0:
            # 求出样本i到k个邻居的距离
            dis = self.dis_p[i - 1][:]  # 切片复制一份
            dis.sort()  # 然后升序排序
            self.all_Dpi[i - 1] = dis[: self.K_Neighbor]  # 取前k个

        return self.all_Dpi[i - 1]

    def D_n_i(self, i):
        """
        正样本集中的样本i到负样本集h个邻居的距离

        :param i: 正样本中的某个数据下标，从1开始
        :return:样本i到负样本集h个邻居的距离
        """
        if len(self.all_Dni[i - 1]) == 0:
            dis = self.dis_n[i - 1][:]  # 切片复制一份，然后升序排序
            dis.sort()
            self.all_Dni[i - 1] = dis[:self.H_Neighbor]

        return self.all_Dni[i - 1]

    def Ri(self, i):
        """
        interval range 采样间隔

        例如：(0.1, 0.8]
        :param i: 样本编号，从1开始
        :return:采样间隔，列表
        """
        if len(self.all_Ri[i - 1]) == 0:
            # 把所有的 Ri 都计算出来
            for j in range(1, self.P + 1):
                start = self.delt_i(j - 1) / self.get_delt_star()
                end = self.delt_i(j) / self.get_delt_star()
                self.all_Ri[j - 1] = (start, end)

        return self.all_Ri[i - 1]

    def delt_p_i(self, i):
        """
        正样本集中的xi在正样本集的密度

        :param i: 来自正样本集合的样本xi
        :return:xi在正样本集的密度
        """
        if self.all_delt_p_i[i - 1] == -1:
            res = 1 / np.mean(self.D_p_i(i))
            self.all_delt_p_i[i - 1] = res

        return self.all_delt_p_i[i - 1]

    def delt_n_i(self, i):
        """
        正样本集中的xi在负样本集的密度

        :param i: 来自正样本集合的样本xi
        :return:xi在负样本集的密度
        """
        if self.all_delt_n_i[i - 1] == -1:
            res = 1 / np.mean(self.D_n_i(i))
            self.all_delt_n_i[i - 1] = res

        return self.all_delt_n_i[i - 1]

    def delt_i(self, i):
        """
        采样边界 interval boundary

        :param i:来自正样本集合的样本xi
        :return:采样边界; 一个数值
        """
        if i == 0:
            return 0

        if self.all_delt_i[i - 1] == -1:
            res = self.delt_i(i - 1) + self.delt_p_i(i) + self.delt_n_i(i)
            self.all_delt_i[i - 1] = res

        return self.all_delt_i[i - 1]

    def get_delt_star(self):
        """
        归一化密度因子

        :return:归一化密度因子; 一个数值
        """
        if self.delt_star is not None:
            return self.delt_star
        else:
            res = 0
            for i in range(1, len(self.T_p) + 1):
                res += self.delt_p_i(i)
            for i in range(1, len(self.T_n) + 1):
                res += self.delt_n_i(i)
            self.delt_star = res

            return res


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
        print("k = %d" % cur_k)

        # 构建模型，训练
        x_train, y_train = DBUOperator().fit(x_train, y_train)  # 随机抽样
        clf = KNeighborsClassifier()
        print("开始训练模型...")
        clf.fit(x_train, y_train)

        # 测试
        print("开始评估模型...")
        y_proba = clf.predict_proba(x_val)
        y_pred = np.argmax(y_proba, axis=1)

        # 评估测试集
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_proba[:, 1])

        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)

        print("val_acc:%.2f val_precision:%.2f val_recall:%.2f val_f1:%.2f auc_value:%.2f" %
              (val_acc, val_precision, val_recall, val_f1, auc_value))

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))
