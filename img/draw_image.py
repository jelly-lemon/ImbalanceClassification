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


def draw_bar(x_tick_labels, y_data, save_name=None, title=None, y_label=None, bar_legend=None, x_label_rotation=90):
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
    if y_label != None:
        plt.ylabel(y_label)
    plt.xticks(ticks=x_label_ticks, labels=x_tick_labels, rotation=x_label_rotation, fontsize=8)

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

    if bar_legend != None:
        plt.legend(all_bar, bar_legend, loc="best")  # 还有 upper right 等

    if title != None:
        plt.title(title)

    if save_name != None:
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
    y_data = [[10,9,5,8,7,7,8,13,8,7,5,13,7,10,13,7,10,9,4,8,7,8,16,8,11,2,10,5]]

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
    y_data = [[0,2,0,3,23]]

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)

def fig_4():
    title = "Times of optimal F1-Score"
    y_label = "times"
    bar_legend = None
    save_name = "Times_of_optimal_F1-Score_Ensemble"

    # x 轴标签
    x_tick_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[8,6,0,1,13]]

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, title=title, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0)


if __name__ == '__main__':
    fig_4()
