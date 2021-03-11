import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from data import read_data
from experiment import experiment_helper
from myidea.HybridBaggingClassifier import hybridBaggingClassifier
import warnings


# 忽略找不到那么多聚类质心的警告
warnings.filterwarnings('ignore')

def kFoldEvolution(x, y, evolution=False):
    # 记录评估结果
    val_history = {}  # 进化前的预测结果

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
        clf = hybridBaggingClassifier(5, 5)

        # 训练
        #clf.fit(x_train, y_train, sampling="under", show_info=True)
        clf.fit(x_train, y_train)

        # 测试
        all_y_proba = clf.predict_proba_2(x_val)
        y_proba = np.mean(all_y_proba, axis=0)
        y_pred = np.argmax(y_proba, axis=1)


        experiment_helper.save_metric(val_history, y_val, y_pred, y_proba)
        experiment_helper.show_last_data(val_history)


    experiment_helper.show_mean_data(val_history)


if __name__ == '__main__':
    x, y = read_data.get_data([0, 6], -1, "yeast.dat", show_info=True)

    kFoldEvolution(x, y)