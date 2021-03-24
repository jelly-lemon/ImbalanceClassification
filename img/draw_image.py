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


def draw_barh(title="", save_name=""):
    """
    水平条状图
    """
    title = "Comparison of AUC before and after evolution"
    save_name = "AUC_Comparison_evolution"
    x_label = "AUC"

    bar_height = 3  # 条状图高度
    unit_inner_space = 0.5  # 单元内间距
    unit_outer_space = unit_inner_space * 8  # 单元间间距

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


def draw_bar(save_name="AUC_compare"):
    """
    条形图

    :param save_name: 保存文件名
    :return:
    """
    bar_width = 0.3
    # 数组表示有多少分类器
    # 子数组表示一个分类器在所有数据集的分类结果
    data = [[0.7890, 0.8351, 0.7266, 0.8323, 0.9019, 0.8182, 0.8249, 0.6738, 0.8296, 0.9427,
             0.9989, 0.7289, 0.8938, 0.9792, 0.9994, 0.9204, 0.9732, 0.9903, 0.7007, 0.8475,
             0.9589, 0.7007, 0.9884, 0.9075, 0.4934, 0.7994, 0.9441, 0.9733],
            [0.8027, 0.8474, 0.7530, 0.8568, 0.9090, 0.8065, 0.8423, 0.7078, 0.8321, 0.9392,
             0.9966, 0.7350, 0.9133, 0.9780, 0.9997, 0.9253, 0.9734, 0.9969, 0.7057, 0.8755,
             0.9605, 0.7038, 0.9894, 0.9119, 0.5079, 0.8012, 0.9472, 0.9840]]
    x_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    x_ticks = np.array([i * 1.5 for i in range(1, len(x_labels) + 1)], dtype=np.float)

    # ticks 表明在哪个位置，labels 表明显示什么
    plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=6, rotation=90)
    plt.ylabel("AUC", fontsize=16)

    plt.ylim([0.45, 1.05])

    space = 0.2  # 组内间距
    data_width = bar_width * len(data) + space * (len(data) - 1)  # 一组对比条状图所占宽度

    start = x_ticks - data_width / 2 + bar_width / 2  # 第一个条状图绘制位置
    all_b = []
    color = ["white", "grey"]
    hatch = ["", ""]
    for i, d in enumerate(data):
        # edgecolor="black", linestyle=':'
        b = plt.bar(x=start, height=d, width=bar_width, edgecolor="black", color=color[i], hatch=hatch[i])
        start += bar_width + space
        all_b.append(b)

    bar_legend = ["Before optimization", "After optimization"]
    # upper right
    plt.legend(all_b, bar_legend, loc="best", fontsize=6)

    if save_name != "":
        # savefig 必须在 show 之前，因为 show 会默认打开一个新的画板，导致 savefig 为空白
        plt.savefig(save_name + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(save_name + ".eps", dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # data = [[0.9289, 0.9351, 0.8829, 0.9296],
    #         [0.9483, 0.9169, 0.9138, 0.9287],
    #         [0.9323, 0.9077, 0.9353, 0.9216],
    #         [0.8625, 0.9170, 0.9163, 0.9184]]
    # draw_hot(data, title="yeast-6 1449/35=41.40", save_name="yeast_6")
    draw_barh()
    # draw_bar()
