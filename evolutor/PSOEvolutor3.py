import numpy as np
from sklearn import metrics

from classifier.HSBaggingClassifier import HSBaggingClassifier


class PSOEvolutor3:
    """
    粒子群优化算法，优化基分类器的数量

    粒子就是一个元组，元组里面的数字代表每个基分类器的数量
    根据这些元组创建一个分类器，使用训练数据训练
    比较不同训练器得到的AUC
    然后返回最优的一个粒子

    """
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def evolve(self, max_steps):
        max_steps = max_steps  # 最大迭代次数


        # 生成10个粒子
        X = []
        for i in range(10):
            X.append(np.random.randint(0, 21, size=4))

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

        all_auc = []
        for x in X:
            clf = HSBaggingClassifier(x[0], x[1], x[2], x[3])
            clf.fit(self.x_train, self.y_train)
            y_prob = clf.predict_proba(self.x_val)
            auc_value = metrics.roc_auc_score(self.y_val, y_prob[:, 1])
            all_auc.append(auc_value)

        # 最大值下标
        i_best = np.argmax(all_auc)
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

    def get_optimal(self, x1, x2):
        """
        比较两个位置谁好

        :param x1:某个粒子1
        :param x2:某个粒子2
        :return:位置好的粒子
        """
        clf_1 = HSBaggingClassifier(x1[0], x1[1], x1[2], x1[3])
        clf_1.fit(self.x_train, self.y_train)
        prob_1 = clf_1.predict_proba(self.x_val)

        clf_2 = HSBaggingClassifier(x2[0], x2[1], x2[2], x2[3])
        clf_2.fit(self.x_train, self.y_train)
        prob_2 = clf_2.predict_proba(self.x_val)

        # 计算两个 AUC
        auc_1 = metrics.roc_auc_score(self.y_val, prob_1[:, 1])
        auc_2 = metrics.roc_auc_score(self.y_val, prob_2[:, 1])

        # 第一个权重大，就返回0，否则返回1
        if auc_1 > auc_2:
            return 0
        else:
            return 1















