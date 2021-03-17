import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_box(x, y, title):
    plt.boxplot(x, labels=y, vert=False)  # 水平显示箱线图
    plt.title(title, fontdict={'weight': 'normal', 'size': 20})
    plt.xlabel("AUC", fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel("Bags", fontdict={'weight': 'normal', 'size': 16})
    plt.show()  # 显示该图


def draw_hot(data, title="", save_name=""):
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


def draw_barh():
    bar_width = 0.15
    # 数组表示有多少分类器
    # 子数组表示一个分类器在所有数据集的分类结果
    data = [[0.7842, 0.8612, 0.8081, 0.8362, 0.8920, 0.8824, 0.8898, 0.7713, 0.9004,
             0.9362, 0.9863, 0.8682, 0.9138, 0.9770, 0.9972, 0.9655, 0.9846, 0.9739,
             0.8536, 0.9330, 0.9677, 0.9122, 0.9801, 0.9643, 0.9199, 0.9624, 0.9878,
             0.9868],
            [0.7975, 0.8672, 0.8114, 0.8424, 0.9023, 0.8824, 0.8746, 0.7955, 0.9087,
             0.9305, 0.9790, 0.8784, 0.9221, 0.9713, 0.9978, 0.9655, 0.9845, 0.9831,
             0.8695, 0.9458, 0.9756, 0.9353, 0.9899, 0.9723, 0.9593, 0.9683, 0.9895,
             0.9940]]
    x_labels = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appendicitis-1",
                "yeast-0-6", "cleveland-1", "yeast-0", "ecoli-7", "newthyroid-0", "cleveland-2",
                "ecoli-4", "page-blo-1-2-3-4", "vowel-0", "ecoli-2-3-5-6", "page-blo-1-2-3", "glass-2",
                "balance-1", "yeast-7vs.1", "ecoli-5", "yeast-7vs.1-4-5-8", "letter-img-1",
                "yeast-4", "wine-red-4", "wine-red-8vs.6", "yeast-6", "page-blo-3"]
    x_ticks = np.array([i / 2 for i in range(1, len(x_labels) + 1)], dtype=np.float)

    # ticks 表明在哪个位置，labels 表明显示什么
    # plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=10, rotation=90)
    # plt.ylabel("F1-score", fontsize=16)

    plt.xlabel("F1-score", fontsize=16)
    plt.yticks(ticks=x_ticks, labels=x_labels, fontsize=10)

    space = 0.05  # 间距
    data_width = bar_width * len(data) + space * (len(data) - 1)  # 一组对比条状图所占宽度

    start = x_ticks - data_width / 2 + bar_width / 2  # 第一个条状图绘制位置
    all_b = []
    for i, d in enumerate(data):
        # b = plt.bar(x=start, height=d, width=bar_width)
        b = plt.barh(y=start, width=d, height=bar_width)
        start += bar_width + space
        all_b.append(b)

    bar_legend = ["Before optimization", "After optimization"]
    # upper right
    plt.legend(all_b, bar_legend, loc="best")

    plt.show()


def draw_bar(save_name="AUC_compare"):
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

    draw_bar()
