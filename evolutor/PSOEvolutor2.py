import numpy as np
from sklearn import metrics

class PSOEvolutor2:
    """
    粒子群优化算法，优化每个分类器的权重

    """
    def __init__(self, all_y_prob, y_val):
        self.all_y_prob = all_y_prob
        self.y_val = y_val

    def evolve(self, max_steps):
        """

        :param X:初始粒子位置
        """
        max_steps = max_steps  # 最大迭代次数

        X = []
        # X.append(np.ones(len(self.all_y_prob)))
        for i in range(len(self.all_y_prob)):
            # 循环创建每一个粒子
            length = len(self.all_y_prob)
            X.append( np.random.rand(length))


        init_weight = 0.6  # 初始惯性权重与当前惯性权重
        end_weight = 0.1  # 结束惯性权重
        c1 = c2 = 2  # 个体学习因子、社会学习因子

        # 评估每个粒子并得到全局最优
        pBest = X  # 存放每个粒子的历史最优位置，默认初始位置为最优位置
        gBest = self.get_gBest(pBest)  # 获取这一届总体最优位置

        # 随机初始化每一个粒子的速度
        v = np.random.rand(len(X), len(X[0]))

        iter = 0
        for step in range(max_steps):
            iter += 1
            print("\r%d/%d" % (iter, max_steps), end="")

            # 计算本次迭代惯性因子
            cur_weight = init_weight - (init_weight - end_weight)  / (max_steps-1)




            # 更新每个粒子的速度和位置
            v = np.array(v)
            X = np.array(X)
            pBest = np.array(pBest)
            gBest = np.array(gBest)


            for i, x in enumerate(X):
                # 生成两个随机数，分别代表飞向当前粒子历史最佳位置、全局历史最佳位置的程度
                r1 = np.random.rand(1)
                r2 = np.random.rand(1)
                v[i] = cur_weight * v[i] + c1 * r1 * (pBest[i] - X[i]) + c2 * r2 * (gBest - X[i])

            X = X + v

            # 超出范围的粒子位置要进行限制
            X = np.array(X)
            X[X > 1] = 1
            X[X < -1] = -1

            # 新位置不一定是好位置，还得和之前的个体粒子最优位置进行比较，比之前好才能更新
            pBest = self.get_pBest(pBest, X)
            gBest = self.get_gBest(pBest)
        print("")

        return gBest

    def get_gBest(self, X):
        """
        从历届最优个体位置中选择一个最好的

        :return:
        """
        i_best = 0
        for i in range(1, len(X)):
            if self.get_optimal(X[i_best], X[i]) == 1:
                i_best = i


        return X[i_best]

    def get_pBest(self, old_pos, new_pos):
        """
        比较新旧位置，返回最好位置

        :param old_pos: 所有粒子历史最优位置
        :param new_pos: 新位置
        :return: 所有粒子的新的历史最优位置
        """
        pBest = []
        for i in range(len(old_pos)):
            # 比较两个位置谁好
            if self.get_optimal(old_pos[i], new_pos[i]) == 0:
                pBest.append(old_pos[i])
            else:
                pBest.append(new_pos[i])

        return pBest

    def get_optimal(self, weight1, weight2):
        """
        比较两个位置谁好

        :param weight1:某个粒子1
        :param weight2:某个粒子2
        :return:位置好的粒子
        """
        all_y_prob = np.array(self.all_y_prob)

        # 先计算两个权重对应的概率
        prob_1 = None
        for i, y_prob in enumerate(all_y_prob):
            y_prob = np.array(y_prob)
            if prob_1 is None:
                prob_1 = weight1[i] * y_prob
            else:
                prob_1 += weight1[i] * y_prob

        prob_2 = None
        for i, y_prob in enumerate(all_y_prob):
            y_prob = np.array(y_prob)
            if prob_2 is None:
                prob_2 = weight2[i] * y_prob
            else:
                prob_2 += weight2[i] * y_prob

        # 计算两个 AUC
        auc_1 = metrics.roc_auc_score(self.y_val, prob_1[:, 1])
        auc_2 = metrics.roc_auc_score(self.y_val, prob_2[:, 1])

        # 第一个权重大，就返回0，否则返回1
        if auc_1 > auc_2:
            return 0
        else:
            return 1















