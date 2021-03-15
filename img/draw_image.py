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
        plt.savefig(save_name, dpi=300, bbox_inches='tight')

    plt.show()


def draw_bar():
    legend = ("kNN", "DT", "Our")
    bar_width = 0.09
    # 数组表示有多少分类器
    # 子数组表示一个分类器在所有数据集的分类结果
    data = [[0.79, 0.83, 0.81, 0.84], [0.91, 0.89, 0.93, 0.90], [0.56, 0.60, 0.63, 0.59]]
    x_ticks = np.array([i for i in range(1, len(data[0]) + 1)], dtype=np.float)
    # ticks 表明在哪个位置，labels 表明显示什么
    plt.xticks(ticks=x_ticks, labels=x_ticks)
    space = 0.01
    data_width = bar_width * len(data) + space * (len(data) - 1)

    start = x_ticks - data_width / 2 + bar_width / 2
    all_b = []
    for i, d in enumerate(data):
        b = plt.bar(x=start, height=d, width=bar_width)
        start += bar_width + space
        all_b.append(b)

    # upper right
    plt.legend(all_b, legend, loc="best")

    plt.show()


if __name__ == '__main__':
    yeast_0_6 = [[0.9015, 0.8729, 0.9055, 0.8981],
                 [0.9066, 0.8911, 0.9027, 0.8908],
                 [0.8932, 0.8917, 0.8952, 0.8848],
                 [0.8716, 0.8875, 0.8875, 0.8900]]

    draw_hot(yeast_0_6, title="ecoli-1 IR=3.36", save_name="ecoli_1")

