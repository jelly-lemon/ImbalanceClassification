# 描述：基于密度的下采样器
# 作者：Jelly Lemon

import random
import numpy as np


class DBUSampler:
    """
    Density-Based Under-Sampling Operator 基于密度的下采样器

    默认正样本为1（多数类），负样本为0（少数类）

    Examples
    --------
    >> dub = DBUOperator()
    >> new_x, new_y = dub.fit(x, y)
    """

    def __init__(self, K_Neighbor=5, H_Neighbor=5, sampling_rate=None):
        """
        初始化采样器

        :param K_Neighbor: 正样本中邻居参考数
        :param H_Neighbor: 负样本中邻居参考数
        """
        self.K_Neighbor = K_Neighbor
        self.H_Neighbor = H_Neighbor

        self.sampling_rate = sampling_rate

        # 初始化相关变量
        self.delt_star = None  # 累计密度因子
        self.all_Ri = None  # 采样间隔范围
        self.all_delt_p_i = None  # 样本xi在正样本集中的密度
        self.all_delt_n_i = None  # 样本xi在负样本集中的密度
        self.all_Dpi = None  # 样本xi在正样本集中k邻居距离之和
        self.all_Dni = None  # 样本xi在负样本集中h邻居距离之和
        self.all_delt_i = None  # 样本xi的采样边界
        self.dis_p = None  # 正样本集合中各点之间的距离
        self.dis_n = None  # 正样本xi到负样本集中各点之间的距离

    def get_dis_p(self, dis_p, T_p):
        """
        计算正样本中各点之间的距离

        :param dis_p: 保存距离的变量
        :param T_p: 正样本集
        :return:
        """
        # dis_p 是一个对称方阵
        # 所以算上三角就好了，下三角直接复制
        # i,j 表示样本编号，从1开始
        for i in range(1, len(T_p) + 1):
            for j in range(i, len(T_p) + 1):
                # 上三角
                dis_p[i - 1][j - 1] = np.sqrt(np.sum(np.square(np.array(T_p[i - 1]) - np.array(T_p[j - 1]))))
                # 下三角
                dis_p[j - 1][i - 1] = dis_p[i - 1][j - 1]

        return dis_p

    def get_dis_n(self, dis_n, T_p, T_n):
        """
        计算正样本中到负样本中各点的距离

        :param dis_n: 保存距离的变量
        :param T_p:正样本
        :param T_n:负样本
        :return:
        """
        for i in range(1, len(T_p) + 1):
            for j in range(1, len(T_n) + 1):
                dis_n[i - 1][j - 1] = np.sqrt(np.sum(np.square(np.array(T_p[i - 1]) - np.array(T_n[j - 1]))))

        return dis_n

    def fit_resample(self, x, y):
        """
        下采样数据集，返回的格式和输入相同，只是量变少了

        :return: 采样后的数据集
        """
        # 统计样本信息
        self.N = len(y)  # 样本个数
        self.T_p = [list(x[i]) for i in range(len(y)) if y[i] == 1]  # 正样本集
        self.T_n = [list(x[i]) for i in range(len(y)) if y[i] == 0]  # 负样本集
        self.N_pos = len(self.T_p)  # 正样本数量
        self.N_neg = len(self.T_n)  # 负样本数量



        # 初始化相关变量
        self.delt_star = None  # 累计密度因子
        self.all_Ri = [[] for i in range(self.N_pos)]  # 采样间隔范围
        self.all_delt_p_i = [-1 for i in range(self.N_pos)]  # 样本xi在正样本集中的密度
        self.all_delt_n_i = [-1 for i in range(self.N_pos)]  # 样本xi在负样本集中的密度
        self.all_Dpi = [[] for i in range(self.N_pos)]  # 样本xi在正样本集中k邻居距离之和
        self.all_Dni = [[] for i in range(self.N_pos)]  # 样本xi在负样本集中h邻居距离之和
        self.all_delt_i = [-1 for i in range(self.N_pos)]  # 样本xi的采样边界
        self.dis_p = [[-1 for i in range(self.N_pos)] for i in range(self.N_pos)]  # 正样本集合中各点之间的距离
        self.dis_n = [[-1 for i in range(self.N_neg)] for j in range(self.N_pos)]  # 正样本xi到负样本集中各点之间的距离

        # 求正样本集中各点之间的距离
        # print("计算正样本集中各点之间的距离...", end="")
        self.dis_p = self.get_dis_p(self.dis_p, self.T_p)

        # 正样本xi到负样本集中各点之间的距离
        # print("\r计算正样本xi到负样本集中各点之间的距离...", end="")
        self.dis_n = self.get_dis_n(self.dis_n, self.T_p, self.T_n)

        # 计算 Ri
        # print("\r计算Ri...", end="")
        for i in range(1, self.N_pos + 1):
            self.get_R(i)  # i表示样本编号，从1开始

        if self.sampling_rate is None:
            # 根据不平衡比设定采样率
            if self.N_pos / self.N_neg <= 5:
                self.sampling_rate = 0.3
            else:
                self.sampling_rate = 0.1

        # 开始采样
        print("\r开始采样（采样率%.2f）..." % self.sampling_rate)
        count = 0  # 当前采样个数
        T_p_new = []  # 存放采样样本的集合
        while count < int(self.N_pos * self.sampling_rate): # 下采样数量=正样本数量*采样率
            # 随机生成 [0,1] 的一个数
            r = random.uniform(0, 1)
            # print("r=%f" % r)
            for i in range(1, self.N_pos + 1):  # j表示样本编号，从1开始
                start, end = self.get_R(i)
                if start < r <= end:
                    if self.T_p[i - 1] not in T_p_new:
                        count += 1
                        T_p_new.append(self.T_p[i - 1])

                        # 第1种break方式
                        break

                        # 第2种break方式
                        # if count == self.N:
                        #     break

        T_new = T_p_new + self.T_n
        y_p = np.ones((len(T_p_new),), dtype=np.uint8)
        y_n = np.zeros((len(self.T_n),), dtype=np.uint8)
        y = np.concatenate((y_p, y_n))

        return T_new, y

    def get_Dp(self, i):
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

    def get_Dn(self, i):
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

    def get_R(self, i):
        """
        计算采样采样间隔 interval range

        例如：(0.1, 0.8]

        :param i: 样本编号，从1开始
        :return:采样间隔，列表
        """
        if len(self.all_Ri[i - 1]) == 0:
            # 把所有的 Ri 都计算出来
            for j in range(1, self.N_pos + 1):
                start = self.get_delt(j - 1) / self.get_delt_star()
                end = self.get_delt(j) / self.get_delt_star()
                self.all_Ri[j - 1] = (start, end)

        return self.all_Ri[i - 1]

    def get_delt_p(self, i):
        """
        正样本集中的xi在正样本集的密度

        :param i: 来自正样本集合的样本xi
        :return:xi在正样本集的密度
        """
        if self.all_delt_p_i[i - 1] == -1:
            if np.mean(self.get_Dp(i)) == 0:
                res = 1
            else:
                res = 1 / np.mean(self.get_Dp(i))
            self.all_delt_p_i[i - 1] = res

        return self.all_delt_p_i[i - 1]

    def get_delt_n(self, i):
        """
        正样本集中的xi在负样本集的密度

        :param i: 来自正样本集合的样本xi
        :return:xi在负样本集的密度
        """
        if self.all_delt_n_i[i - 1] == -1:
            res = 1 / np.mean(self.get_Dn(i))
            self.all_delt_n_i[i - 1] = res

        return self.all_delt_n_i[i - 1]

    def get_delt(self, i):
        """
        采样边界 interval boundary

        :param i:来自正样本集合的样本xi
        :return:采样边界; 一个数值
        """
        if i == 0:
            return 0

        if self.all_delt_i[i - 1] == -1:
            res = self.get_delt(i - 1) + self.get_delt_p(i) + self.get_delt_n(i)
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
                res += self.get_delt_p(i)
            for i in range(1, len(self.T_n) + 1):
                res += self.get_delt_n(i)
            self.delt_star = res

            return res
