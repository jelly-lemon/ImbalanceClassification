import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np


def draw_box(x, y, title):
    """
    箱线图
    """
    plt.boxplot(x, labels=y, vert=False)  # 水平显示箱线图
    plt.title(title, fontdict={'weight': 'normal', 'size': 20})
    plt.xlabel("AUC", fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel("Bags", fontdict={'weight': 'normal', 'size': 16})
    plt.show()  # 显示该图


def draw_hot(data, title="", save_name=""):
    """
    热力图
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


def draw_barh(y_tick_labels, x_data, x_ticks=None, save_name=None, title=None, x_label=None, bar_legend=None,
              y_label_rotation=0):
    bar_height = 5  # 条状图高度
    unit_inner_space = 4  # 单元内间距
    unit_outer_space = 15  # 单元间间距

    # 设置图片比例
    plt.figure(figsize=(9, 20))

    # y 轴上标签位置
    unit_width = bar_height * len(x_data) + unit_inner_space * (len(x_data) - 1)
    total_width = unit_width * len(x_data[0]) + unit_outer_space * (len(x_data[0]) - 1)
    y_label_ticks = []
    label_tick = unit_width / 2
    y_label_ticks.append(label_tick)
    for i in range(len(x_data[0]) - 1):
        label_tick += unit_outer_space + unit_width
        y_label_ticks.append(label_tick)
    if x_label is not None:
        plt.xlabel(x_label)
    plt.yticks(ticks=y_label_ticks, labels=y_tick_labels, rotation=y_label_rotation)
    if x_ticks is not None:
        plt.xticks(x_ticks)

    # 开始画每组数据
    draw_point = []  # 每组数据中每条画点
    point = bar_height / 2
    draw_point.append(point)
    for i in range(len(x_data[0]) - 1):
        point += unit_outer_space + unit_width
        draw_point.append(point)
    draw_point = np.array(draw_point)

    all_bar = []
    for i, width in enumerate(x_data):
        bar = plt.barh(y=draw_point, width=width, height=bar_height)
        draw_point += unit_inner_space + bar_height
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

    plt.grid(axis='y', linestyle='--') # 设置网格线

    # 开始画每组数据
    draw_point = []  # 每组数据中每条画点
    point = bar_width / 2
    draw_point.append(point)
    for i in range(len(y_data[0]) - 1):
        point += unit_outer_space + unit_width
        draw_point.append(point)
    draw_point = np.array(draw_point)

    my_color = ['lightgrey', 'darkgrey', 'grey', 'dimgrey', 'black']
    # my_color = ['lightgrey', 'darkgrey', 'grey', 'dimgrey', 'black']

    all_bar = []
    for i, height in enumerate(y_data):
        bar = plt.bar(x=draw_point, width=bar_width, height=height,
                      color=my_color, edgecolor='black') # 绘制条状图
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
    """
    折线图
    """
    markers = ["^", "P", "x", "v", "p", "o", "s", "d", "D"]
    x_ticks = [i for i in range(len(x_tick_labels))]
    for i, data in enumerate(y_data):
        # 第一个参数是横坐标，第二个参数纵坐标，第三个参数表示该数据的标签，legend 用得着
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


def draw_some_line_chart(x_ticks, y_data, n_row, n_col, title, sub_title, x_label, y_label, save_name=None):
    """
    画多个对比折现图
    """
    figure, axes = plt.subplots(n_row, n_col, figsize=(16, 6), constrained_layout=True)
    for i in range(n_row):
        for j in range(n_col):
            axes[i][j].plot(x_ticks, y_data[n_col * i + j])
            axes[i][j].set_title(sub_title[n_col * i + j])
            axes[i][j].set_xlabel(x_label)
            axes[i][j].set_ylabel(y_label)
            axes[i][j].set_xticks(x_ticks)
            # axes[i][j].set_ylim([0.9, 1])
            axes[i][j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))

    plt.suptitle(title, fontsize=16)

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
    """
    每个数据集的属性数量
    """
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
    """
    单分类器最优 F1-Score 出现次数
    """
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
    """
    集成学习最优 F1-Score 出现次数
    """
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
    """
    单分类器最优 AUC 出现次数
    """
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
    """
    集成学习最优 AUC 出现次数
    """
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
    """
    单分类器在每个数据集上的 F1-Score
    """
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
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=title, y_label=y_label,
                    save_name=save_name)


def fig_8():
    """
    单分类器在每个数据集上的 F1-Score
    """
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
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=title, y_label=y_label,
                    save_name=save_name)


def fig_9():
    """
    单分类器在每个数据集上的 AUC
    """
    title = "Performance on each dataset of single classifier and HABC "
    y_label = "AUC"
    save_name = "Performance_AUC_Single"
    y_data = [
        [0.593, 0.841, 0.526, 0.754, 0.910, 0.844, 0.842, 0.466, 0.840, 0.897, 0.987, 0.644, 0.889, 0.963, 0.975, 0.896,
         0.972, 0.896, 0.649, 0.733, 0.920, 0.615, 0.974, 0.891, 0.516, 0.536, 0.917, 0.959],
        [0.653, 0.860, 0.618, 0.751, 0.926, 0.789, 0.832, 0.362, 0.807, 0.902, 0.969, 0.630, 0.863, 0.947, 1.000, 0.887,
         0.965, 1.000, 0.573, 0.699, 0.962, 0.652, 0.996, 0.885, 0.491, 0.684, 0.897, 0.969],
        [0.659, 0.773, 0.731, 0.643, 0.840, 0.766, 0.711, 0.461, 0.701, 0.834, 0.915, 0.616, 0.753, 0.927, 0.962, 0.797,
         0.952, 0.954, 0.492, 0.679, 0.932, 0.693, 0.944, 0.835, 0.553, 0.719, 0.817, 0.969],
        [0.639, 0.805, 0.704, 0.658, 0.793, 0.667, 0.706, 0.488, 0.718, 0.876, 0.909, 0.577, 0.705, 0.918, 0.976, 0.811,
         0.944, 1.000, 0.453, 0.591, 0.966, 0.548, 0.908, 0.709, 0.533, 0.652, 0.767, 0.928],
        [0.803, 0.842, 0.760, 0.857, 0.906, 0.828, 0.851, 0.684, 0.831, 0.934, 0.996, 0.738, 0.959, 0.980, 0.999, 0.927,
         0.996, 0.995, 0.704, 0.874, 0.961, 0.715, 0.991, 0.923, 0.507, 0.816, 0.942, 0.990]]
    line_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    x_tick_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                     "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                     "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                     "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                     "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=title, y_label=y_label,
                    save_name=save_name)


def fig_10():
    """
    集成学习在每个数据上的 AUC
    """
    title = "Performance on each dataset of ensemble classifier and HABC "
    y_label = "AUC"
    save_name = "Performance_AUC_Ensemble"
    y_data = [
        [0.661, 0.850, 0.641, 0.801, 0.895, 0.757, 0.768, 0.509, 0.778, 0.917, 0.951, 0.652, 0.874, 0.965, 0.982, 0.834,
         0.954, 0.983, 0.474, 0.649, 0.999, 0.497, 0.964, 0.723, 0.575, 0.747, 0.751, 0.968],
        [0.616, 0.837, 0.656, 0.792, 0.893, 0.743, 0.799, 0.537, 0.818, 0.906, 0.985, 0.677, 0.902, 0.930, 0.958, 0.906,
         0.976, 0.899, 0.290, 0.782, 0.947, 0.667, 0.936, 0.903, 0.669, 0.793, 0.877, 0.991],
        [0.693, 0.848, 0.669, 0.784, 0.874, 0.797, 0.832, 0.503, 0.826, 0.916, 0.992, 0.697, 0.869, 0.980, 0.996, 0.920,
         0.984, 0.990, 0.374, 0.842, 0.985, 0.693, 0.983, 0.876, 0.713, 0.804, 0.917, 0.985],
        [0.680, 0.889, 0.725, 0.771, 0.906, 0.788, 0.813, 0.501, 0.817, 0.917, 0.995, 0.756, 0.919, 0.982, 0.981, 0.928,
         0.987, 0.990, 0.601, 0.753, 0.956, 0.643, 0.984, 0.874, 0.672, 0.753, 0.903, 0.985],
        [0.803, 0.842, 0.760, 0.857, 0.906, 0.828, 0.851, 0.684, 0.831, 0.934, 0.996, 0.738, 0.959, 0.980, 0.999, 0.927,
         0.996, 0.995, 0.704, 0.874, 0.961, 0.715, 0.991, 0.923, 0.507, 0.816, 0.942, 0.990]]

    line_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    x_tick_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                     "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                     "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                     "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                     "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=title, y_label=y_label,
                    save_name=save_name)


def fig_11():
    """
    单分类器平均 F1-Score
    """
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
    """
    集成学习平均 F1-Score
    """
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
    """
    单分类器平均 AUC
    :return:
    """
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
    """
    集成学习平均 AUC
    """
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
    """
    迭代次数与 F1-score
    """
    total_title = "The relationship between the number of iterations and F1-Score"
    sub_title = ["yeast-0-6 IR=4", "ecoli-7 IR=5", "ecoli-4 IR=9", "glass-2 IR=12", "ecoli-5 IR=16", "yeast-4 IR=28"]
    x_label = "number of iterations"
    x_ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    y_label = "F1-Score"
    save_name = "iterations_and_F1-Score"

    y_data = [[0.8698, 0.8787, 0.8800, 0.8805, 0.8776, 0.8861, 0.8842, 0.8950, 0.8908, 0.8937, 0.8953, 0.8999, 0.8947,
               0.8981, 0.8928, 0.8994, 0.8958, 0.8984, 0.8947, 0.8994, 0.8961],
              [0.9262, 0.9491, 0.9381, 0.9155, 0.9462, 0.9462, 0.9569, 0.9581, 0.9635, 0.9539, 0.9569, 0.9550, 0.9581,
               0.9519, 0.9581, 0.9550, 0.9562, 0.9555, 0.9550, 0.9555, 0.9585],
              [0.9138, 0.9138, 0.9231, 0.9238, 0.9212, 0.9238, 0.9322, 0.9231, 0.9222, 0.9222, 0.9238, 0.9231, 0.9212,
               0.9238, 0.9238, 0.9231, 0.9238, 0.9200, 0.9244, 0.9231, 0.9322],
              [0.9739, 0.9731, 0.9731, 0.9787, 0.9751, 0.9851, 0.9867, 0.9887, 0.9851, 0.9851, 0.9887, 0.9887, 0.9887,
               0.9867, 0.9820, 0.9867, 0.9867, 0.9851, 0.9811, 0.9800, 0.9850],
              [0.9677, 0.9669, 0.9627, 0.9754, 0.9769, 0.9712, 0.9783, 0.9769, 0.9796, 0.9783, 0.9712, 0.9769, 0.9796,
               0.9754, 0.9769, 0.9754, 0.9769, 0.9712, 0.9727, 0.9740, 0.9783],
              [0.9643, 0.9771, 0.9754, 0.9754, 0.9719, 0.9735, 0.9789, 0.9701, 0.9735, 0.9718, 0.9718, 0.9718, 0.9736,
               0.9754, 0.9719, 0.9754, 0.9807, 0.9842, 0.9789, 0.9735, 0.9754]]
    draw_some_line_chart(x_ticks, y_data, 2, 3, title=total_title, sub_title=sub_title, x_label=x_label,
                         y_label=y_label, save_name=save_name)


def fig_16():
    """
    迭代次数与 AUC
    """
    total_title = "The relationship between the number of iterations and AUC"
    sub_title = ["yeast-0-6 IR=4", "ecoli-7 IR=5", "ecoli-4 IR=9", "glass-2 IR=12", "ecoli-5 IR=16", "yeast-4 IR=28"]
    x_label = "number of iterations"
    x_ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    y_label = "AUC"
    save_name = "iterations_and_AUC"
    y_data = [[0.8249, 0.8017, 0.8274, 0.8297, 0.8344, 0.8462, 0.8446, 0.8434, 0.8443, 0.8423, 0.8427, 0.8451, 0.8448,
               0.8455, 0.8477, 0.8447, 0.8411, 0.8417, 0.8410, 0.8458, 0.8427],
              [0.9327, 0.9229, 0.9276, 0.9321, 0.9351, 0.9341, 0.9301, 0.9323, 0.9301, 0.9306, 0.9376, 0.9355, 0.9346,
               0.9331, 0.9339, 0.9377, 0.9360, 0.9369, 0.9398, 0.9383, 0.9332],
              [0.8938, 0.8952, 0.9167, 0.9167, 0.9241, 0.9232, 0.9262, 0.9255, 0.9194, 0.9255, 0.9298, 0.9267, 0.9232,
               0.9209, 0.9228, 0.9235, 0.9201, 0.9235, 0.9247, 0.9282, 0.9259],
              [0.9903, 0.9913, 0.9956, 0.9933, 0.9938, 0.9936, 0.9933, 0.9928, 0.9992, 0.9976, 0.9949, 0.9987, 0.9972,
               0.9962, 0.9969, 0.9926, 0.9982, 0.9996, 0.9990, 0.9949, 0.9954],
              [0.9589, 0.9614, 0.9632, 0.9677, 0.9729, 0.9639, 0.9674, 0.9624, 0.9643, 0.9680, 0.9613, 0.9609, 0.9698,
               0.9651, 0.9659, 0.9639, 0.9676, 0.9646, 0.9687, 0.9636, 0.9631],
              [0.9075, 0.9059, 0.9045, 0.9080, 0.9157, 0.9146, 0.9178, 0.9209, 0.9204, 0.9186, 0.9206, 0.9232, 0.9171,
               0.9179, 0.9129, 0.9168, 0.9111, 0.9146, 0.9180, 0.9180, 0.9163]]

    draw_some_line_chart(x_ticks, y_data, 2, 3, title=total_title, sub_title=sub_title, x_label=x_label,
                         y_label=y_label, save_name=save_name)


def fig_17():
    """
    进化前后 F1-Score 对比
    """
    title = "Comparison of F1-Score before and after evolution"
    save_name = "F1-Score_Comparison_evolution"
    x_label = "F1-Score"

    # data[0] 是进化前，data[1] 是进化后
    data = [[0.7842, 0.8612, 0.8081, 0.8362, 0.8920, 0.8824, 0.8698, 0.7713, 0.9004,
             0.9262, 0.9663, 0.8682, 0.9138, 0.9670, 0.9972, 0.9455, 0.9746, 0.9739,
             0.8536, 0.9330, 0.9677, 0.9122, 0.9801, 0.9643, 0.9199, 0.9624, 0.9878,
             0.9868],
            [0.7975, 0.8672, 0.8114, 0.8424, 0.9023, 0.8824, 0.8746, 0.7955, 0.9087,
             0.9305, 0.9790, 0.8784, 0.9221, 0.9713, 0.9978, 0.9655, 0.9845, 0.9831,
             0.8695, 0.9458, 0.9756, 0.9353, 0.9899, 0.9723, 0.9593, 0.9683, 0.9895,
             0.9940]]

    # 数据集标签
    y_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appen-1",
                "yeast-06", "cleve-1", "yeast-0", "ecoli-7", "newth-0", "cleve-2",
                "ecoli-4", "page-1234", "vowel-0", "ecoli-2356", "page-123", "glass-2",
                "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1458", "letter-1",
                "yeast-4", "wine-4", "wine-8vs.6", "yeast-6", "page-3"]

    bar_legend = ["Before optimization", "After optimization"]
    draw_barh(y_tick_labels=y_labels, x_data=data, save_name=save_name, title=title, x_label=x_label,
              bar_legend=bar_legend)


def fig_18():
    """
    进化前后 AUC 对比
    """
    title = "Comparison of AUC before and after evolution"
    save_name = "AUC_Comparison_evolution"
    x_label = "AUC"

    # data[0] 是进化前，data[1] 是进化后
    data = [[0.7890, 0.8351, 0.7266, 0.8323, 0.9019, 0.8082, 0.8249, 0.6738, 0.8296, 0.9327,
             0.9909, 0.7289, 0.8938, 0.9772, 0.9994, 0.9204, 0.9732, 0.9903, 0.7007, 0.8475,
             0.9589, 0.7007, 0.9884, 0.9075, 0.4934, 0.7994, 0.9441, 0.9733],
            [0.8027, 0.8474, 0.7530, 0.8568, 0.9090, 0.8065, 0.8423, 0.7078, 0.8321, 0.9392,
             0.9966, 0.7350, 0.9133, 0.9780, 0.9997, 0.9253, 0.9734, 0.9969, 0.7057, 0.8755,
             0.9605, 0.7038, 0.9894, 0.9119, 0.5079, 0.8012, 0.9472, 0.9840]]

    # 数据集标签
    y_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appen-1",
                "yeast-06", "cleve-1", "yeast-0", "ecoli-7", "newth-0", "cleve-2",
                "ecoli-4", "page-1234", "vowel-0", "ecoli-2356", "page-123", "glass-2",
                "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1458", "letter-1",
                "yeast-4", "wine-4", "wine-8vs.6", "yeast-6", "page-3"]

    bar_legend = ["Before optimization", "After optimization"]
    draw_barh(y_tick_labels=y_labels, x_data=data, save_name=save_name, title=title, x_label=x_label,
              bar_legend=bar_legend)


if __name__ == '__main__':
    fig_6()
