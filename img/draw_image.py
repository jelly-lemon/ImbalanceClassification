import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_box(x, y, title):
    """
    箱线图

    :param x:
    :param y:
    :param title:
    :return:
    """
    plt.boxplot(x, labels=y, vert=False)  # 水平显示箱线图
    plt.title(title, fontdict={'weight': 'normal', 'size': 20})
    plt.xlabel("AUC", fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel("Bags", fontdict={'weight': 'normal', 'size': 16})
    plt.show()  # 显示该图


def draw_hot(data, title="", save_name=""):
    """
    热力图

    :param data:
    :param title:
    :param save_name:
    :return:
    """
    f, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_title(title, fontsize=16)

    ticklabel = [5, 10, 15, 20]
    # annot 是否在每个小框中写上具体的数值
    # 返回值是 matplotlib Axes
    # YlGnBu == 黄绿蓝
    # Greens == 绿色
    ax1 = sns.heatmap(data, annot=True, fmt=".4f", annot_kws={'size': 16, 'weight': 'normal'},
                      linewidths=0.05,
                      cmap='YlGnBu',
                      ax=ax1)
    ax1.set_xlabel("$N_{under}$", fontsize=16)
    ax1.set_ylabel("$N_{up}$", fontsize=16, rotation=0)
    ax1.set_xticklabels(ticklabel, fontsize=16)
    ticklabel.reverse()
    ax1.set_yticklabels(ticklabel, fontsize=16, rotation=0)  # 加上 rotation=0， heatmap 会旋转 90

    if save_name != "":
        # savefig 必须在 show 之前，因为 show 会默认打开一个新的画板，导致 savefig 为空白
        plt.savefig(save_name + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(save_name + ".eps", dpi=300, bbox_inches='tight')

    plt.show()


def draw_barh(title="", save_name="", x_label=""):
    """
    水平条状图

    # AUC 进化前后对比
    # data[0] 是进化前，data[1] 是进化后
    data = [[0.7890, 0.8351, 0.7266, 0.8323, 0.9019, 0.8082, 0.8249, 0.6738, 0.8296, 0.9327,
             0.9909, 0.7289, 0.8938, 0.9772, 0.9994, 0.9204, 0.9732, 0.9903, 0.7007, 0.8475,
             0.9589, 0.7007, 0.9884, 0.9075, 0.4934, 0.7994, 0.9441, 0.9733],
            [0.8027, 0.8474, 0.7530, 0.8568, 0.9090, 0.8065, 0.8423, 0.7078, 0.8321, 0.9392,
             0.9966, 0.7350, 0.9133, 0.9780, 0.9997, 0.9253, 0.9734, 0.9969, 0.7057, 0.8755,
             0.9605, 0.7038, 0.9894, 0.9119, 0.5079, 0.8012, 0.9472, 0.9840]]

    # 数据集标签
    y_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]

    """
    title = "Comparison of AUC before and after evolution"
    save_name = "AUC_Comparison_evolution"
    x_label = "AUC"

    bar_height = 3  # 条状图高度
    unit_inner_space = 0.5  # 单元内间距
    unit_outer_space = 4  # 单元间间距

    # data[0] 是进化前，data[1] 是进化后
    data = [[0.7890, 0.8351, 0.7266, 0.8323, 0.9019, 0.8082, 0.8249, 0.6738, 0.8296, 0.9327,
             0.9909, 0.7289, 0.8938, 0.9772, 0.9994, 0.9204, 0.9732, 0.9903, 0.7007, 0.8475,
             0.9589, 0.7007, 0.9884, 0.9075, 0.4934, 0.7994, 0.9441, 0.9733],
            [0.8027, 0.8474, 0.7530, 0.8568, 0.9090, 0.8065, 0.8423, 0.7078, 0.8321, 0.9392,
             0.9966, 0.7350, 0.9133, 0.9780, 0.9997, 0.9253, 0.9734, 0.9969, 0.7057, 0.8755,
             0.9605, 0.7038, 0.9894, 0.9119, 0.5079, 0.8012, 0.9472, 0.9840]]

    # 数据集标签
    y_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    # y 轴上标签位置
    unit_width = bar_height * len(data) + unit_inner_space * (len(data) - 1)
    total_height = unit_width * len(data[0]) + unit_outer_space * (len(data[0]) - 1)
    y_label_ticks = []
    label_tick = total_height - unit_width / 2
    y_label_ticks.append(label_tick)
    for i in range(len(data[0]) - 1):
        label_tick = label_tick - unit_width - unit_outer_space
        y_label_ticks.append(label_tick)
    plt.xlim([0.45, 1.2])
    plt.xlabel(x_label)
    plt.yticks(ticks=y_label_ticks, labels=y_labels)

    # 开始画每组数据
    draw_point = []  # 每组数据中每条画点
    point = total_height - bar_height / 2
    draw_point.append(point)
    for i in range(len(data[0]) - 1):
        point = point - unit_width - unit_outer_space
        draw_point.append(point)
    draw_point = np.array(draw_point)

    all_hbar = []
    for i, width in enumerate(data):
        hbar = plt.barh(y=draw_point, width=width, height=bar_height)
        draw_point = draw_point - bar_height - unit_inner_space
        all_hbar.append(hbar)

    bar_legend = ["Before optimization", "After optimization"]
    # upper right
    plt.legend(all_hbar, bar_legend, loc="best")

    plt.title(title)

    if save_name != "":
        # savefig 必须在 show 之前，因为 show 会默认打开一个新的画板，导致 savefig 为空白
        plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
        plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')
        plt.savefig("./svg_img/" + save_name + ".svg", dpi=300, bbox_inches='tight')

    plt.show()


def draw_bar(x_tick_labels, y_data, y_ticks=None, save_name=None, title=None, y_label=None, bar_legend=None,
             x_label_rotation=90):
    """
    条形图
    """
    bar_width = 3  # 条状图高度
    unit_inner_space = 0.5  # 单元内间距
    unit_outer_space = 4  # 单元间间距

    # y 轴上标签位置
    unit_width = bar_width * len(y_data) + unit_inner_space * (len(y_data) - 1)
    total_width = unit_width * len(y_data[0]) + unit_outer_space * (len(y_data[0]) - 1)
    x_label_ticks = []
    label_tick = unit_width / 2
    x_label_ticks.append(label_tick)
    for i in range(len(y_data[0]) - 1):
        label_tick += unit_outer_space + unit_width
        x_label_ticks.append(label_tick)
    # plt.xlim([0.45, 1.2])
    if y_label is not None:
        plt.ylabel(y_label)
    plt.xticks(ticks=x_label_ticks, labels=x_tick_labels, rotation=x_label_rotation, fontsize=8)
    if y_ticks is not None:
        plt.yticks(y_ticks)

    # 开始画每组数据
    draw_point = []  # 每组数据中每条画点
    point = bar_width / 2
    draw_point.append(point)
    for i in range(len(y_data[0]) - 1):
        point += unit_outer_space + unit_width
        draw_point.append(point)
    draw_point = np.array(draw_point)

    all_bar = []
    for i, height in enumerate(y_data):
        bar = plt.bar(x=draw_point, width=bar_width, height=height)
        draw_point += unit_inner_space + bar_width
        all_bar.append(bar)

    if bar_legend is not None:
        plt.legend(all_bar, bar_legend, loc="best")  # 还有 upper right 等

    if title is not None:
        plt.title(title)

    if save_name is not None:
        # savefig 必须在 show 之前，因为 show 会默认打开一个新的画板，导致 savefig 为空白
        plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
        plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')
        plt.savefig("./svg_img/" + save_name + ".svg", dpi=300, bbox_inches='tight')

    plt.show()


def draw_line_chart(y_data, line_labels, x_tick_labels, title=None, y_label=None, bar_legend=None, save_name=None):
    markers = ["^", "P", "x", "v", "p", "o", "s", "d", "D"]
    x_ticks = [i for i in range(len(x_tick_labels))]
    for i, data in enumerate(y_data):
        # 第一个参数是横坐标，第二个参数纵坐标，第三个参数
        plt.plot(x_ticks, data, marker=markers[i], label=line_labels[i])
        plt.xticks(x_ticks, labels=x_tick_labels, rotation=90)
    plt.legend(loc="best")

    if title is not None:
        plt.title(title)
    if y_label is not None:
        plt.ylabel(y_label)
    if save_name is not None:
        # savefig 必须在 show 之前，因为 show 会默认打开一个新的画板，导致 savefig 为空白
        plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
        plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')
        plt.savefig("./svg_img/" + save_name + ".svg", dpi=300, bbox_inches='tight')

    plt.show()


def fig_1():
    """
    各数据集正负样本数量对比条状图
    """
    title = "Number of positive and negative samples in each dataset"
    y_label = "Number of samples"
    bar_legend = ["Number of positive samples", "Number of negative samples"]
    save_name = "Number_of_sample"

    # x 轴标签
    x_tick_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                     "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                     "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                     "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                     "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    # data[0]正样本数量，data[1]负样本数量
    y_data = [
        [153, 144, 102, 1055, 259, 85, 1205, 243, 457, 284, 185, 262, 301, 791, 900, 307, 5028, 197, 576, 429, 316, 663,
         2259, 1433, 1546, 638, 1449, 5385],
        [90, 70, 49, 429, 77, 21, 279, 54, 90, 52, 30, 35, 35, 90, 90, 29, 444, 17, 49, 30, 20, 30, 90, 51, 53, 18, 35,
         87]]
    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name)


def fig_2():
    title = "Number of attributes in each dataset"
    y_label = "Number of attributes"
    bar_legend = None
    save_name = "Number_of_attributes"

    # x 轴标签
    x_tick_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                     "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                     "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                     "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                     "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    # data[0]每个类属性数量
    y_data = [[10, 9, 5, 8, 7, 7, 8, 13, 8, 7, 5, 13, 7, 10, 13, 7, 10, 9, 4, 8, 7, 8, 16, 8, 11, 2, 10, 5]]

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name)


def fig_3():
    title = "Times of optimal F1-Score"
    y_label = "times"
    bar_legend = None
    save_name = "Times_of_optimal_F1-Score_Single"

    # x 轴标签
    x_tick_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[0, 2, 0, 3, 23]]
    y_ticks = np.arange(0, 25, 2)

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)


def fig_4():
    title = "Times of optimal F1-Score"
    y_label = "times"
    save_name = "Times_of_optimal_F1-Score_Ensemble"

    # x 轴标签
    x_tick_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[8, 6, 0, 1, 13]]
    y_ticks = np.arange(0, 20, 2)

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=title, y_label=y_label,
             save_name=save_name, x_label_rotation=0)


def fig_5():
    title = "Times of optimal AUC"
    y_label = "times"
    bar_legend = None
    save_name = "Times_of_optimal_AUC_Single"

    # x 轴标签
    x_tick_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[2, 5, 1, 2, 19]]
    y_ticks = np.arange(0, 25, 2)

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)


def fig_6():
    title = "Times of optimal AUC"
    y_label = "times"
    bar_legend = None
    save_name = "Times_of_optimal_AUC_Ensemble"

    # x 轴标签
    x_tick_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[1, 1, 1, 5, 20]]
    y_ticks = np.arange(0, 25, 2)

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)


def fig_7():
    title = "Performance on each dataset of single classifier and HABC "
    y_label = "F1-Score"
    save_name = "Performance_F1-Score_Single"
    y_data = [
        [0.623, 0.732, 0.554, 0.755, 0.899, 0.854, 0.833, 0.545, 0.831, 0.879, 0.975, 0.695, 0.892, 0.957, 0.936, 0.940,
         0.970, 0.792, 0.673, 0.769, 0.939, 0.700, 0.900, 0.888, 0.678, 0.772, 0.935, 0.975],
        [0.658, 0.814, 0.616, 0.771, 0.912, 0.830, 0.848, 0.589, 0.850, 0.911, 0.980, 0.783, 0.930, 0.970, 0.996, 0.966,
         0.981, 0.981, 0.835, 0.871, 0.968, 0.862, 0.991, 0.947, 0.847, 0.885, 0.960, 0.994],
        [0.717, 0.821, 0.751, 0.729, 0.910, 0.778, 0.802, 0.650, 0.804, 0.889, 0.954, 0.762, 0.848, 0.963, 0.972, 0.857,
         0.966, 0.951, 0.680, 0.779, 0.926, 0.719, 0.962, 0.879, 0.758, 0.807, 0.886, 0.976],
        [0.728, 0.843, 0.802, 0.789, 0.895, 0.861, 0.870, 0.719, 0.888, 0.942, 0.970, 0.797, 0.922, 0.977, 0.992, 0.962,
         0.987, 1.000, 0.800, 0.924, 0.976, 0.934, 0.992, 0.962, 0.942, 0.959, 0.980, 0.996],
        [0.788, 0.869, 0.830, 0.841, 0.908, 0.888, 0.899, 0.786, 0.918, 0.961, 0.979, 0.873, 0.939, 0.982, 0.998, 0.979,
         0.993, 0.971, 0.879, 0.948, 0.977, 0.931, 0.993, 0.972, 0.950, 0.969, 0.979, 0.998]]
    line_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    x_tick_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                     "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                     "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                     "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                     "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=title, y_label=y_label, save_name=save_name)


def fig_8():
    title = "Performance on each dataset of ensemble classifier and HABC "
    y_label = "F1-Score"
    save_name = "Performance_F1-Score_Ensemble"
    y_data = [
        [0.730, 0.871, 0.754, 0.847, 0.911, 0.915, 0.900, 0.763, 0.916, 0.951, 0.980, 0.872, 0.933, 0.974, 0.995, 0.964,
         0.988, 0.998, 0.836, 0.939, 0.976, 0.930, 0.983, 0.979, 0.979, 0.972, 0.986, 0.997],
        [0.748, 0.823, 0.796, 0.844, 0.903, 0.901, 0.910, 0.798, 0.925, 0.955, 0.980, 0.837, 0.949, 0.966, 0.981, 0.972,
         0.980, 0.967, 0.859, 0.945, 0.961, 0.937, 0.980, 0.972, 0.983, 0.972, 0.981, 0.997],
        [0.676, 0.819, 0.682, 0.779, 0.877, 0.750, 0.858, 0.645, 0.852, 0.939, 0.974, 0.845, 0.889, 0.963, 0.971, 0.904,
         0.974, 0.951, 0.545, 0.826, 0.971, 0.705, 0.961, 0.889, 0.785, 0.834, 0.918, 0.989],
        [0.728, 0.867, 0.754, 0.788, 0.925, 0.854, 0.874, 0.704, 0.866, 0.937, 0.980, 0.830, 0.918, 0.975, 0.982, 0.942,
         0.984, 0.951, 0.808, 0.861, 0.972, 0.838, 0.974, 0.916, 0.869, 0.868, 0.937, 0.988],
        [0.788, 0.869, 0.830, 0.841, 0.908, 0.888, 0.899, 0.786, 0.918, 0.961, 0.979, 0.873, 0.939, 0.982, 0.998, 0.979,
         0.993, 0.971, 0.879, 0.948, 0.977, 0.931, 0.993, 0.972, 0.950, 0.969, 0.979, 0.998]]
    line_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    x_tick_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                     "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                     "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                     "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                     "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=title, y_label=y_label, save_name=save_name)

def fig_9():
    """AUC"""
    title = "Performance on each dataset of single classifier and HABC "
    y_label = "AUC"
    save_name = "Performance_AUC_Single"
    y_data = [[0.593, 0.841, 0.526, 0.754, 0.910, 0.844, 0.842, 0.466, 0.840, 0.897, 0.987, 0.644, 0.889, 0.963, 0.975, 0.896, 0.972, 0.896, 0.649, 0.733, 0.920, 0.615, 0.974, 0.891, 0.516, 0.536, 0.917, 0.959],
              [0.653, 0.860, 0.618, 0.751, 0.926, 0.789, 0.832, 0.362, 0.807, 0.902, 0.969, 0.630, 0.863, 0.947, 1.000, 0.887, 0.965, 1.000, 0.573, 0.699, 0.962, 0.652, 0.996, 0.885, 0.491, 0.684, 0.897, 0.969],
              [0.659, 0.773, 0.731, 0.643, 0.840, 0.766, 0.711, 0.461, 0.701, 0.834, 0.915, 0.616, 0.753, 0.927, 0.962, 0.797, 0.952, 0.954, 0.492, 0.679, 0.932, 0.693, 0.944, 0.835, 0.553, 0.719, 0.817, 0.969],
              [0.639, 0.805, 0.704, 0.658, 0.793, 0.667, 0.706, 0.488, 0.718, 0.876, 0.909, 0.577, 0.705, 0.918, 0.976, 0.811, 0.944, 1.000, 0.453, 0.591, 0.966, 0.548, 0.908, 0.709, 0.533, 0.652, 0.767, 0.928],
              [0.803, 0.842, 0.760, 0.857, 0.906, 0.828, 0.851, 0.684, 0.831, 0.934, 0.996, 0.738, 0.959, 0.980, 0.999, 0.927, 0.996, 0.995, 0.704, 0.874, 0.961, 0.715, 0.991, 0.923, 0.507, 0.816, 0.942, 0.990]]
    line_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    x_tick_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                     "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                     "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                     "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                     "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=title, y_label=y_label,
                    save_name=save_name)

def fig_10():
    title = "Performance on each dataset of ensemble classifier and HABC "
    y_label = "AUC"
    save_name = "Performance_AUC_Ensemble"
    y_data = [[0.661, 0.850, 0.641, 0.801, 0.895, 0.757, 0.768, 0.509, 0.778, 0.917, 0.951, 0.652, 0.874, 0.965, 0.982, 0.834, 0.954, 0.983, 0.474, 0.649, 0.999, 0.497, 0.964, 0.723, 0.575, 0.747, 0.751, 0.968],
              [0.616, 0.837, 0.656, 0.792, 0.893, 0.743, 0.799, 0.537, 0.818, 0.906, 0.985, 0.677, 0.902, 0.930, 0.958, 0.906, 0.976, 0.899, 0.290, 0.782, 0.947, 0.667, 0.936, 0.903, 0.669, 0.793, 0.877, 0.991],
              [0.693, 0.848, 0.669, 0.784, 0.874, 0.797, 0.832, 0.503, 0.826, 0.916, 0.992, 0.697, 0.869, 0.980, 0.996, 0.920, 0.984, 0.990, 0.374, 0.842, 0.985, 0.693, 0.983, 0.876, 0.713, 0.804, 0.917, 0.985],
              [0.680, 0.889, 0.725, 0.771, 0.906, 0.788, 0.813, 0.501, 0.817, 0.917, 0.995, 0.756, 0.919, 0.982, 0.981, 0.928, 0.987, 0.990, 0.601, 0.753, 0.956, 0.643, 0.984, 0.874, 0.672, 0.753, 0.903, 0.985],
              [0.803, 0.842, 0.760, 0.857, 0.906, 0.828, 0.851, 0.684, 0.831, 0.934, 0.996, 0.738, 0.959, 0.980, 0.999, 0.927, 0.996, 0.995, 0.704, 0.874, 0.961, 0.715, 0.991, 0.923, 0.507, 0.816, 0.942, 0.990]]

    line_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    x_tick_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                     "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                     "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                     "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                     "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=title, y_label=y_label,
                    save_name=save_name)

def fig_11():
    title = "Average F1-score on each data set"
    y_label = "F1-score"
    bar_legend = None
    save_name = "Average_F1-score_Single"

    # x 轴标签
    x_tick_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[0.817, 0.876, 0.839, 0.907, 0.928]]
    # y_ticks = np.arange(0, 25, 2)
    y_ticks = None

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)

def fig_12():
    title = "Average F1-score on each data set"
    y_label = "F1-score"
    bar_legend = None
    save_name = "Average_F1-score_Ensemble"

    # x 轴标签
    x_tick_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[0.922, 0.922, 0.849, 0.888, 0.928]]
    y_ticks = None

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)

def fig_13():
    title = "Average AUC on each data set"
    y_label = "AUC"
    bar_legend = None
    save_name = "Average_AUC_Single"

    # x 轴标签
    x_tick_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[0.801, 0.806, 0.772, 0.748, 0.868]]
    # y_ticks = np.arange(0, 25, 2)
    y_ticks = None

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)

def fig_14():
    title = "Average AUC on each data set"
    y_label = "AUC"
    bar_legend = None
    save_name = "Average_AUC_Ensemble"

    # x 轴标签
    x_tick_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[0.790, 0.810, 0.833, 0.838, 0.868]]
    y_ticks = None

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)

def fig_15():
    total_title = "The relationship between the number of iterations and AUC"
    sub_title = ["yeast-0-6 IR=4", "ecoli-7 IR=5", "glass-2 IR=12", "yeast-4 IR=28"]
    x_label = "number of iterations"
    x_ticks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    y_label = "AUC"
    save_name = "iterations_and_AUC"

    y_data = [[]]


if __name__ == '__main__':
    fig_14()
