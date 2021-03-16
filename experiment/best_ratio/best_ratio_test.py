import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from data import read_data
from experiment import experiment_helper
from experiment.PSO_two_object.MOPSO import mopso
from myidea.HybridBaggingClassifier import hybridBaggingClassifier
import warnings


# 忽略找不到那么多聚类质心的警告
warnings.filterwarnings('ignore')


def kFoldEvolution(x, y, evolution=False, n_under=5, n_up=10):
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
        clf = hybridBaggingClassifier(n_under, n_up)

        # 训练
        #clf.fit(x_train, y_train, sampling="under", show_info=True)
        clf.fit(x_train, y_train, show_info=False)

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
            y_proba_evo = mopso(x_val).evolute(all_y_proba, max_steps=5, show_info=False)
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

        break

    if evolution:
        # 统计，求平均值和标准差
        print("进化前平均：")
        experiment_helper.show_mean_data(val_history)
        print("进化后平均：")
        experiment_helper.show_mean_data(evo_history)
        print("结合后平均：")
        experiment_helper.show_mean_data(mean_history)
        auc_record["%d:%d" % (n_under, n_up)] = get_auc(mean_history)
        print(auc_record)
    else:
        experiment_helper.show_mean_data(val_history)

def get_auc(history):
        return np.mean(history["auc"])




auc_record = {}
if __name__ == '__main__':
    N_under = (5, 10, 15, 20)
    N_up = (5, 10, 15, 20)
    for i in N_under:
        for j in N_up:
            x, y = read_data.get_data([6], -1, "yeast.dat", show_info=False)

            # 期望每折交叉验证样本数量 >= 100
            # for i in range(1):
            #     x, y = read_data.upsampling_copy(x, y, 1)
            #     print("复制一份后：%d/%d" % (len(y[y == 1]), len(y[y == 0])))

            kFoldEvolution(x, y, evolution=True, n_under=i, n_up=j)

