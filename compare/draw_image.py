import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.faker import Faker
import random
import numpy as np
import seaborn as sns

def draw_box(x, y, title):
    plt.boxplot(x, labels=y, vert=False)  # 水平显示箱线图
    plt.title(title, fontdict={'weight': 'normal', 'size': 20})
    plt.xlabel("AUC", fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel("Bags", fontdict={'weight': 'normal', 'size': 16})
    plt.show()  # 显示该图

def draw_hot(data, title="yeast-0-6 IR=2.46"):
    f, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_title(title)


    ticklabel = [5,10,15,20]
    # annot 是否在每个小框中写上具体的数值
    # 返回值是 matplotlib Axes
    ax1 = sns.heatmap(data, annot=True,
                linewidths=0.05,
                cmap='YlGnBu',
                ax=ax1)
    ax1.set_xlabel("N1")
    ax1.set_ylabel("N2", rotation=0)
    ax1.set_xticklabels(ticklabel)
    ax1.set_yticklabels(ticklabel, rotation=0)  # 加上 rotation=0， heatmap 会旋转 90

    plt.show()



if __name__ == '__main__':
    data = [[0.7617, 0.7635, 0.7606, 0.7586],
            [0.7777, 0.7730, 0.7668, 0.7501],
            [0.7711, 0.7695, 0.7460, 0.7668],
            [0.7686, 0.7630, 0.7631, 0.7601]]


    draw_hot(data)