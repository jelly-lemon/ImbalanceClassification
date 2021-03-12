"""
多目标粒子群优化算法

关键点：
选择 pBest(个体历史最优)：新旧位置各好一半时，随机选择新旧作为 pBest
选择 gBest(全局历史最优)：非支配排序 -> 拥挤度计算，选择优先级最高并且拥挤读最小的位置，再进行新旧比较。

"""
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from data import read_data
from experiment import experiment_helper
from myidea.HybridBaggingClassifier import hybridBaggingClassifier
import warnings


# 忽略找不到那么多聚类质心的警告
warnings.filterwarnings('ignore')


class mopso():
    """
    多目标粒子群优化算法
    """

    def __init__(self, x):
        # 样本相似度矩阵
        self.S = self.get_S_matrix(x)

    def clustering(self, u):
        """
        聚类

        :param u:
        :return:
        """
        kmeans_cluster = KMeans()
        cluster_res = kmeans_cluster.fit_predict(u)  # 获得聚类结果
        n_cluster_center = len(kmeans_cluster.cluster_centers_)  # 取得簇数量
        r = self.get_r_matrix(cluster_res, n_cluster_center)  # 聚类结果矩阵
        q = kmeans_cluster.cluster_centers_[:n_cluster_center]  # 获取聚类质心

        return r, q

    def gaussian(self, a, b, sigma):
        """
        高斯函数
        """
        t = np.exp(-np.linalg.norm(a - b) / (2 * sigma ** 2))
        return t

    def sim(self, a, b):
        """
        计算相似度

        :param a: 数据 a
        :param b: 数据 b
        :return: 相似度
        """
        return self.gaussian(a, b, 1)

    def get_S_matrix(self, x):
        """
        计算相似度矩阵(计算样本之间的相似度或预测结果之间的相似度)

        例如：
        样本的预测结果（属于类别0的概率，属于类别1的概率）：
        [[0.8 0.2]
         [0.4 0.6]
         [0.  1. ]]

        得到相似度矩阵 3x3：
        [[1.         0.32259073 0.10406478]
         [0.32259073 1.         0.32259073]
         [0.10406478 0.32259073 1.]]

        :param x: 预测结果概率
        :return: 相似度矩阵
        """
        mat = np.zeros((len(x), len(x)))
        for i, i_value in enumerate(x):
            for j, j_value in enumerate(x):
                if i > j:
                    mat[j][i] = mat[i][j] = self.sim(i_value, j_value)
                elif i == j:
                    mat[i][j] = 1
                else:
                    break

        return mat

    def get_r_matrix(self, cluster_result, n_cluster_center):
        """
        将获得的聚类结果转 one-hot 矩阵

        例如：
        聚类结果 [1 2 0 0 1 2 0]
        转成 one-hot 矩阵
        [[0 1 0]
         [0 0 1]
         [1 0 0]
         ...
         [1 0 0]]

        :param cluster_result: 聚类结果
        :return:
        """

        mat = np.zeros((len(cluster_result), n_cluster_center), np.uint8)
        for index, num in enumerate(cluster_result):
            mat[index][num] = 1

        return mat

    def fast_non_dominated_sort(self, pso, func_value):
        """
        快速非支配排序

        :param pso: 种群
        :return: 第一个非支配层个体集合
        """
        F1_func_value = []
        n = np.zeros((len(pso),), dtype=np.uint8)  # 支配个体数量。例如，支配 p 的个体数量 n[p]
        F1 = []  # 某一层级个体集合。例如，第一非支配层 F[1]

        # 计算得到第一非支配层个体集合
        for i, p in enumerate(pso):
            n[i] = 0  # 支配 p 的个体数量
            for j, q in enumerate(pso):
                if self.compare(func_value[i], func_value[j]) < 0:
                    n[i] += 1
            if n[i] == 0:
                F1.append(p)
                F1_func_value.append(func_value[i])

        return F1, F1_func_value

    def compare(self, func_value_1, func_value_2):
        # p 支配 q
        if func_value_1[0] < func_value_2[0] and func_value_1[1] < func_value_2[1]:
            return 1

        # q 支配 p
        if func_value_1[0] > func_value_2[0] and func_value_1[1] > func_value_2[1]:
            return -1

        # p、q 势均力敌
        return 0

    def get_gBest(self, pso, func_value):
        # 计算每个粒子的函数值

        F1, F1_func_value = self.fast_non_dominated_sort(pso, func_value)

        p = self.get_front_best(F1, F1_func_value)

        return p

    def func_1(self, u):
        """
        目标函数 1，值越小越好

        思想：两个样本越相似，那么同属于一个类别的概率也越大
        我们认为，所有样本的 “两个样本相似度 * 两个样本预测类别差距” 加起来，
        值越小，预测结果越准确。

        :param u: 一个个体，也就是一个分类器的预测概率矩阵
        """

        n_sample = len(u)
        sum = 0
        for i in range(n_sample):
            for j in range(n_sample):
                if i >= j:
                    sum += self.S[i][j] * np.linalg.norm(u[i] - u[j])
                else:
                    break
        return sum

    def func_2(self, u):
        """
        目标函数 2，值越小越好

        :param r:所有样本聚类结果 one-hot 矩阵
        :param u:分类器预测概率矩阵
        :param q:分类器对聚类质心的分类概率
        :return:
        """
        r, q = self.clustering(u)

        n_sample = len(u)
        n_cluster_center = len(r[0])
        sum = 0
        for i in range(n_sample):
            for j in range(n_cluster_center):
                sum += r[i][j] * np.linalg.norm(u[i] - q[j])

        return sum

    def sort(self, pso, func_value):
        """
        对某一种群集合按目标函数 1 的值进行排序

        :param pso: 种群
        :param func_value:
        :return:
        """
        # 冒泡升序
        n = len(func_value)
        for i in range(n - 1):
            for j in range(n - i - 1):
                if func_value[j][0] > func_value[j + 1][0]:
                    t = func_value[j]
                    func_value[j] = func_value[j + 1]
                    func_value[j + 1] = t

                    t = pso[j]
                    pso[j] = pso[j + 1]
                    pso[j + 1] = t

        return pso, func_value

    def get_front_best(self, pso, func_value):
        """
        获取某一非支配层中最好的个体

        :param pso: 某一非支配层个体集合
        :return: 最好的个体
        """
        # 按目标函数 1 的值升序排序
        pso, func_value = self.sort(pso, func_value)

        # 计算每个个体的拥挤度
        crowding_dis = [-1 for i in range(len(func_value))]  # -1 代表拥挤度无穷大
        for i in range(len(func_value)):
            if i == 0 or i == len(func_value) - 1:
                continue
            else:
                crowding_dis[i] = ((func_value[i + 1][0] - func_value[i - 1][1]) + (
                            func_value[i + 1][1] - func_value[i - 1][1])) * 2

        # 找到拥挤度最小值下标
        min_pos = 0
        for i in range(len(crowding_dis)):
            if crowding_dis[i] < crowding_dis[min_pos]:
                min_pos = i

        return pso[min_pos]

    def evolute(self, pso, max_steps=10, show_info=False):
        """
        粒子群优化算法

        :param pso: 粒子群的初始位置
        :return: 进化后的粒子位置
        """
        max_steps = max_steps  # 最大迭代次数
        particle_dim = pso[0].shape  # 粒子的维度
        init_weight = 0.6  # 初始惯性权重与当前惯性权重
        end_weight = 0.1  # 结束惯性权重
        c1 = 2  # 个体学习因子
        c2 = 2  # 社会学习因子

        # 评估每个粒子并得到全局最优
        pBest = pso.copy()  # 存放每个粒子的历史最优位置，默认初始位置为最优位置
        pBest_func_value = [(self.func_1(u), self.func_2(u)) for u in pBest]  # 计算每个粒子的目标函数值
        gBest = self.get_gBest(pBest, pBest_func_value)  # 获取这一届总体最优位置

        # 随机初始化每一个粒子的速度
        v = np.random.rand(particle_dim[0], particle_dim[1])

        # 迭代计算
        if show_info:
            print("开始进化")
        for step in range(max_steps):
            if show_info:
                if step == max_steps - 1:
                    print("\r%d/%d" % (step + 1, max_steps))
                else:
                    print("\r%d/%d" % (step + 1, max_steps), end="")

            # 计算本次迭代惯性因子
            cur_weight = init_weight - step * (init_weight - end_weight) / (max_steps - 1)

            # 生成两个随机数，分别代表飞向当前粒子历史最佳位置、全局历史最佳位置的程度
            r1 = np.random.rand(particle_dim[0], particle_dim[1])
            r2 = np.random.rand(particle_dim[0], particle_dim[1])

            # 更新每个粒子的速度和位置
            v = np.array(v)
            pso = np.array(pso)
            pBest = np.array(pBest)
            gBest = np.array(gBest)
            v = cur_weight * v + c1 * r1 * (pBest - pso) + c2 * r2 * (gBest - pso)
            pso = pso + v  # 粒子跑到新位置

            # 超出范围的粒子位置要进行限制
            # 预测概率不可能大于1，也不可能小于0
            pso[pso > 1] = 1
            pso[pso < 0] = 0

            # 新位置不一定是好位置，还得和之前的个体粒子最优位置进行比较，比之前好才能更新
            pso_func_value = [(self.func_1(u), self.func_2(u)) for u in pso]
            for i in range(len(pBest)):
                if self.compare(pBest_func_value[i], pso_func_value[i]) < 0:
                    pBest[i] = pso[i].copy

            # 获取全局最优位置
            gBest_func_value = (self.func_1(gBest), self.func_2(gBest))
            for i in range(len(pBest)):
                if self.compare(pBest_func_value[i], gBest_func_value) > 0:
                    gBest = pBest[i].copy()
                    gBest_func_value = pBest_func_value[i]

        # 对所有粒子求平均，这就是进化后的预测结果
        y_prob = np.mean(pso, axis=0)
        # y_prob = np.mean(pBest, axis=0)

        return y_prob

def kFoldEvolution(x, y, evolution=False):
    # 记录评估结果
    val_history = {}  # 进化前的预测结果
    evo_history = {}  # 进化后的预测结果
    mean_history = {}

    k = 5
    kf = KFold(n_splits=k, shuffle=True)  # 混洗数据
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        # 划分数据
        cur_k += 1  # 当前第几折次交叉验证
        print("%d/%d 交叉验证" % (cur_k, k))
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        # 分类器
        # clf = KNeighborsClassifier()
        #clf = AdaSamplingBaggingClassifier(3)
        clf = hybridBaggingClassifier(20, 5)

        # 训练
        #clf.fit(x_train, y_train, sampling="under", show_info=True)
        clf.fit(x_train, y_train)

        # 测试
        all_y_proba = clf.predict_proba_2(x_val)
        y_proba = np.mean(all_y_proba, axis=0)
        y_pred = np.argmax(y_proba, axis=1)

        if evolution:
            # 进化前的表现
            experiment_helper.save_metric(val_history, y_val, y_pred, y_proba)
            print("进化前：")
            experiment_helper.show_last_data(val_history)

            # 进化
            y_proba_evo = mopso(x_val).evolute(all_y_proba, max_steps=5, show_info=True)
            y_pred_evo = np.argmax(y_proba_evo, axis=1)

            # 进化后的表现
            experiment_helper.save_metric(evo_history, y_val, y_pred_evo, y_proba_evo)
            print("进化后：")
            experiment_helper.show_last_data(evo_history)
            print("-" * 60)

            all_y_proba = [y_proba, y_proba_evo]
            y_proba_mean = np.mean(all_y_proba, axis=0)
            y_pred_mean = np.argmax(y_proba_mean, axis=1)
            experiment_helper.save_metric(mean_history, y_val, y_pred_mean, y_proba_mean)
            print("结合后：")
            experiment_helper.show_last_data(mean_history)
            print("-" * 60)

        else:
            experiment_helper.save_metric(val_history, y_val, y_pred, y_proba)
            experiment_helper.show_last_data(val_history)

    if evolution:
        # 统计，求平均值和标准差
        print("进化前平均：")
        experiment_helper.show_mean_data(val_history)
        print("进化后平均：")
        experiment_helper.show_mean_data(evo_history)
        print("结合后平均：")
        experiment_helper.show_mean_data(mean_history)
    else:
        experiment_helper.show_mean_data(val_history)


if __name__ == '__main__':
    x, y = read_data.get_data([1], -1, "yeast.dat", show_info=True)

    kFoldEvolution(x, y, evolution=True)