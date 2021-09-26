import matplotlib.pyplot as plt
# plt.style.use('ggplot') # 图像风格
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np

data_name = ["bands-0", "glass-0", "tae-0", "yeast-1", "ecoli-1", "appen-1",
             "yeast-06", "cleve-1", "yeast-0", "ecoli-7", "newth-0", "cleve-2",
             "ecoli-4", "page-1234", "vowel-0", "ecoli-2356", "page-123", "glass-2",
             "balance-1", "yst-7vs.1", "ecoli-5", "yst-7v1458", "letter-1",
             "yeast-4", "wine-4", "wine-8vs.6", "yeast-6", "page-3"]

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 24,
         }

font1_bold = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 24,
              }

large_width_fig_size = (24, 9)
middle_width_fig_size = (21, 9)
small_width_fig_size = (16, 9)  # 4:3


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
        plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
        plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')

    plt.show()


def draw_barh(y_tick_labels, x_data, x_ticks=None, save_name=None, title=None, x_label=None, bar_legend=None,
              y_label_rotation=0):
    """
    横状条形图

    :param y_tick_labels:
    :param x_data:
    :param x_ticks:
    :param save_name:
    :param title:
    :param x_label:
    :param bar_legend:
    :param y_label_rotation:
    :return:
    """
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
             x_label_rotation=90, figsize=(16, 9)):
    """
    条形图
    """
    # 设置条状图属性
    bar_width = 10  # 条状图高度
    unit_inner_space = 3  # 单元内间距
    unit_outer_space = 6  # 单元间间距

    # 设置图片比例
    plt.figure(figsize=figsize)

    # y 轴上标签位置
    unit_width = bar_width * len(y_data) + unit_inner_space * (len(y_data) - 1)
    total_width = unit_width * len(y_data[0]) + unit_outer_space * (len(y_data[0]) - 1)
    x_label_ticks = []
    label_tick = unit_width / 2
    x_label_ticks.append(label_tick)
    for i in range(len(y_data[0]) - 1):
        label_tick += unit_outer_space + unit_width
        x_label_ticks.append(label_tick)

    # y 轴刻度范围
    # plt.xlim([0.45, 1.2])

    # y 轴标签
    if y_label is not None:
        plt.ylabel(y_label, fontdict=font1)

    # y 轴刻度
    plt.yticks(fontproperties=font1)
    if y_ticks is not None:
        plt.yticks(y_ticks)

    # x 轴刻度
    plt.xticks(ticks=x_label_ticks, labels=x_tick_labels, rotation=x_label_rotation, fontproperties=font1)

    # 设置背景网格线
    plt.grid(axis='y', linestyle='--')

    # 计算横坐标起始绘制位置
    draw_point = []  # 每组数据中每条画点
    point = bar_width / 2
    draw_point.append(point)
    for i in range(len(y_data[0]) - 1):
        point += unit_outer_space + unit_width
        draw_point.append(point)
    draw_point = np.array(draw_point)

    # 每组数据的颜色
    # my_color = ['lightgrey', 'darkgrey', 'grey', 'dimgrey', 'black']
    # my_color = ['lightgrey', 'dimgrey']
    # my_color = ['white', 'darkgrey']
    my_color = ['white']

    # 填充物
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = (' ', '//')

    # 绘制条状图
    all_bar = []
    for i, height in enumerate(y_data):
        # 需要颜色就添加 color=my_color 参数
        # 填充物  hatch=patterns，每组数据一种填充类型
        if len(y_data) > 1:
            bar = plt.bar(x=draw_point, width=bar_width, height=height,
                          hatch=patterns[i],
                          color=my_color,
                          edgecolor='black', zorder=9)
        elif len(y_data) == 1:
            # 为了绘制每一个“条”都有填充物，这里用一个 for 循环
            for j, h in enumerate(height):
                bar = plt.bar(x=draw_point[j], width=bar_width, height=height[j],
                              hatch=patterns[j % 2],
                              color=my_color,
                              edgecolor='black', zorder=9)  # 绘制条状图，zorder 越大，表示绘制顺序越靠后，就会覆盖之前内容，用于覆盖虚线

        # 绘制注释文字
        for xy in zip(draw_point, height):
            plt.annotate(text=xy[1], xy=xy, xytext=(-20, 6), textcoords='offset points', fontproperties=font1)

        draw_point += unit_inner_space + bar_width  # 下次绘制起始位置
        all_bar.append(bar)

    #
    # 图例
    #
    if bar_legend is not None:
        plt.legend(all_bar, bar_legend, loc="best", prop=font1)  # 还有 upper right 等

    #
    # 标题
    #
    if title is not None:
        plt.title(title)

    #
    # 保存文件
    #
    plt.tight_layout()  # 自动调整布局
    if save_name is not None:
        # savefig 必须在 show 之前，因为 show 会默认打开一个新的画板，导致 savefig 为空白
        plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
        plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')
        plt.savefig("./svg_img/" + save_name + ".svg", dpi=300, bbox_inches='tight')

    #
    # 立即显示图片
    #
    plt.show()


def draw_line_chart(y_data, line_labels, x_tick_labels, title=None, y_label=None, bar_legend=None, save_name=None,
                    figsize=large_width_fig_size):
    """
    折线图
    """
    # 设置图片比例
    plt.figure(figsize=figsize)

    # 网格线
    plt.grid(axis='y', linestyle='--')

    # 记号
    markers = ["^", "P", "x", "v", "p", "o", "s", "d", "D"]

    # 折线风格
    line_style = ['--', '-', '--', '-']
    # line_style = ['-', '--', '-.', ':', '-', '--']

    # 颜色
    line_color = ['red', 'green', 'black', 'orange']
    # line_color = ['red', 'tomato', 'black', 'orange', 'green']

    # 绘制折线图
    x_ticks = [i for i in range(len(x_tick_labels))]
    for i, data in enumerate(y_data):
        # 第一个参数是横坐标，第二个参数纵坐标，第三个参数表示该数据的标签，legend 用得着
        plt.plot(x_ticks, data, marker=markers[i], label=line_labels[i], linewidth=3,
                 markersize=16, linestyle=line_style[i], color=line_color[i])

    # 横坐标
    plt.xticks(x_ticks, labels=x_tick_labels, rotation=30, fontproperties=font1)

    # 图例
    plt.legend(loc="best", prop=font1)

    # 标题
    if title is not None:
        plt.title(title)

    # 纵坐标
    plt.yticks(fontproperties=font1)
    if y_label is not None:
        plt.ylabel(y_label, fontdict=font1)

    # 保存图片
    if save_name is not None:
        save_img(plt.gcf(), save_name)

    plt.show()


def draw_some_line_chart(x_ticks, y_data, n_row, n_col, title, sub_title, x_label, y_label, save_name=None,
                         figsize=large_width_fig_size):
    """
    画多个对比折现图
    """
    figure, axes = plt.subplots(n_row, n_col, figsize=(21, 9), constrained_layout=True)

    """
    for i in range(n_row):
        for j in range(n_col):
            axes[i][j].plot(x_ticks, y_data[n_col * i + j])

            axes[i][j].set_title(sub_title[n_col * i + j], fontdict=font1)
            # axes[i][j].set_xlabel(x_label)
            # axes[i][j].set_ylabel(y_label)
            axes[i][j].set_xticks(x_ticks)
            # axes[i][j].set_ylim([0.9, 1])
            axes[i][j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
    """

    for j in range(n_col):
        axes[0][j].plot(x_ticks, y_data[j][0])
        axes[0][j].set_title(sub_title[j], fontdict=font1)
        axes[0][j].set_xticks(x_ticks)
        axes[0][j].set_ylim([0, 20])
        axes[0][j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%2.2f%%'))
        #axes[0][j].set_xlabel("Base classifier number")
        axes[0][j].set_ylabel("under-sampling rate")

        axes[1][j].plot(x_ticks, y_data[j][1])
        axes[1][j].set_xticks(x_ticks)
        axes[1][j].set_ylim([0, 105])
        axes[1][j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%2.2f%%'))
        axes[1][j].set_xlabel("Base classifier number", fontdict=font1)
        axes[1][j].set_ylabel("over-sampling rate")

    # axes[i][j].set_xlabel(x_label)
    # axes[i][j].set_ylabel(y_label)
    # axes[i][j].set_ylim([0.9, 1])

    # 主标题
    plt.suptitle(title, fontsize=24)

    if save_name is not None:
        save_img(plt.gcf(), save_name)

    plt.show()


def save_img(fig, save_name):
    """
    保存图片
    """
    # 【注意】savefig 必须在 show 之前，因为 show 会默认打开一个新的画板，导致 savefig 为空白
    fig.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
    fig.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight', format='eps', transparent=True)


def fig_1():
    """
    各数据集正负样本数量对比条状图
    """
    title = "Number of positive and negative samples in each dataset"
    y_label = "number of samples"
    bar_legend = ["number of positive samples", "number of negative samples"]
    save_name = "Number_of_sample"

    # x 轴标签
    x_tick_labels = data_name

    # data[0]正样本数量，data[1]负样本数量
    y_data = [
        [153, 144, 102, 1055, 259, 85, 1205, 243, 457, 284, 185, 262, 301, 791, 900, 307, 5028, 197, 576, 429, 316, 663,
         2259, 1433, 1546, 638, 1449, 5385],
        [90, 70, 49, 429, 77, 21, 279, 54, 90, 52, 30, 35, 35, 90, 90, 29, 444, 17, 49, 30, 20, 30, 90, 51, 53, 18, 35,
         87]]

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=30, figsize=large_width_fig_size)


def fig_2():
    """
    每个数据集的属性数量
    """
    title = "Number of attributes in each dataset"
    y_label = "Number of attributes"
    bar_legend = None
    save_name = "Number_of_attributes"

    # x 轴标签
    x_tick_labels = data_name
    # data[0]每个类属性数量
    y_data = [[10, 9, 5, 8, 7, 7, 8, 13, 8, 7, 5, 13, 7, 10, 13, 7, 10, 9, 4, 8, 7, 8, 16, 8, 11, 2, 10, 5]]

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=30, figsize=large_width_fig_size)


def fig_3():
    """
    单分类器最优 F1-Score 出现次数
    """
    title = "Times of optimal F1-Score"
    y_label = "times of optimal F1-Score"
    bar_legend = None
    save_name = "Times_of_optimal_F1-Score_Single"

    # x 轴标签
    x_tick_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    # data[0]每个方法最优次数
    y_data = [[0, 2, 0, 3, 23]]
    y_ticks = np.arange(0, 25, 2)

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0, figsize=small_width_fig_size)


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

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=None, y_label=y_label,
             save_name=save_name, x_label_rotation=0, figsize=small_width_fig_size)


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
    y_data = [[2, 5, 1, 1, 20]]
    y_ticks = np.arange(0, 25, 2)

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0, figsize=small_width_fig_size)


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

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0, figsize=small_width_fig_size)


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

    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=data_name, title=None, y_label=y_label,
                    save_name=save_name, figsize=large_width_fig_size)


def fig_8():
    """
    集成学习在每个数据集上的 F1-Score
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
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=data_name, title=None, y_label=y_label,
                    save_name=save_name, figsize=large_width_fig_size)


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
         0.996, 0.995, 0.704, 0.874, 0.961, 0.715, 0.991, 0.923, 0.697, 0.816, 0.942, 0.990]]
    line_labels = ["RUS-KNN", "SMOTE-KNN", "RUS-DT", "SMOTE-DT", "HABC"]
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=data_name, title=None, y_label=y_label,
                    save_name=save_name, figsize=large_width_fig_size)


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
         0.996, 0.995, 0.704, 0.874, 0.961, 0.715, 0.991, 0.923, 0.697, 0.816, 0.942, 0.990]]

    line_labels = ["RandomForest", "AdaBoost", "EasyEnsemble", "BalancedBagging", "HABC"]
    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=data_name, title=None, y_label=y_label,
                    save_name=save_name, figsize=large_width_fig_size)


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

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0, figsize=small_width_fig_size)


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

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0, figsize=small_width_fig_size)


def fig_13():
    """
    单分类器平均 AUC
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

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0, figsize=small_width_fig_size)


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

    draw_bar(x_tick_labels=x_tick_labels, y_data=y_data, y_ticks=y_ticks, title=None, y_label=y_label,
             bar_legend=bar_legend, save_name=save_name, x_label_rotation=0, figsize=small_width_fig_size)


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
    draw_some_line_chart(x_ticks, y_data, 2, 3, title=None, sub_title=sub_title, x_label=x_label,
                         y_label=y_label, save_name=save_name, figsize=large_width_fig_size)


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

    draw_some_line_chart(x_ticks, y_data, 2, 3, title=None, sub_title=sub_title, x_label=x_label,
                         y_label=y_label, save_name=save_name, figsize=large_width_fig_size)


def fig_17():
    """
    进化前后 F1-Score 对比
    """
    title = "Comparison of F1-Score before and after evolution"
    save_name = "F1-Score_Comparison_evolution"
    y_label = "F1-Score"

    # data[0] 是进化前，data[1] 是进化后
    y_data = [[0.7842, 0.8612, 0.8081, 0.8362, 0.8920, 0.8824, 0.8698, 0.7713, 0.9004,
               0.9262, 0.9663, 0.8682, 0.9138, 0.9670, 0.9972, 0.9455, 0.9746, 0.9739,
               0.8536, 0.9330, 0.9677, 0.9122, 0.9801, 0.9643, 0.9199, 0.9624, 0.9878,
               0.9868],
              [0.7975, 0.8672, 0.8114, 0.8424, 0.9023, 0.8824, 0.8746, 0.7955, 0.9087,
               0.9305, 0.9790, 0.8784, 0.9221, 0.9713, 0.9978, 0.9655, 0.9845, 0.9831,
               0.8695, 0.9458, 0.9756, 0.9353, 0.9899, 0.9723, 0.9593, 0.9683, 0.9895,
               0.9940]]

    line_labels = ["before", "after optimized"]
    x_tick_labels = data_name

    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=None, y_label=y_label,
                    save_name=save_name, figsize=large_width_fig_size)


def fig_18():
    """
    进化前后 AUC 对比
    """
    title = "Comparison of AUC before and after evolution"
    save_name = "AUC_Comparison_evolution"
    y_label = "AUC"

    # data[0] 是进化前，data[1] 是进化后
    y_data = [[0.7890, 0.8351, 0.7266, 0.8323, 0.9019, 0.8082, 0.8249, 0.6738, 0.8296, 0.9327,
               0.9909, 0.7289, 0.8938, 0.9772, 0.9994, 0.9204, 0.9732, 0.9903, 0.7007, 0.8475,
               0.9589, 0.7007, 0.9884, 0.9075, 0.4934, 0.7994, 0.9441, 0.9733],
              [0.8027, 0.8474, 0.7530, 0.8568, 0.9090, 0.8065, 0.8423, 0.7078, 0.8321, 0.9392,
               0.9966, 0.7350, 0.9133, 0.9780, 0.9997, 0.9253, 0.9734, 0.9969, 0.7057, 0.8755,
               0.9605, 0.7038, 0.9894, 0.9119, 0.5079, 0.8012, 0.9472, 0.9840]]

    # 数据集标签
    line_labels = ["before", "after optimized"]
    x_tick_labels = data_name

    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=None, y_label=y_label,
                    save_name=save_name)


def fig_19():
    """
    8 个热力图
    """
    data1 = [[0.9015, 0.8729, 0.9055, 0.8981],
             [0.9066, 0.8911, 0.9027, 0.8908],
             [0.8932, 0.8917, 0.8952, 0.8848],
             [0.8716, 0.8875, 0.8875, 0.8900]]
    draw_hot(data1, "ecoli-1 IR=3", "hot1_ecoli-1")

    data2 = [[0.8545, 0.8422, 0.8442, 0.8476],
             [0.8599, 0.8520, 0.8514, 0.8533],
             [0.8526, 0.8563, 0.8548, 0.8582],
             [0.8402, 0.8443, 0.8443, 0.8515]]
    draw_hot(data2, "yeast-06 IR=4", "hot2_yeast-06")

    data3 = [[0.7295, 0.7086, 0.7003, 0.6940],
             [0.7333, 0.7225, 0.7041, 0.7246],
             [0.7233, 0.6839, 0.6927, 0.6859],
             [0.7148, 0.7078, 0.7022, 0.7250]]
    draw_hot(data3, "cleve-2 IR=7", "hot3_cleve-2")

    data4 = [[0.8334, 0.8179, 0.8191, 0.7728],
             [0.8002, 0.8130, 0.7626, 0.7546],
             [0.7790, 0.7775, 0.7605, 0.7610],
             [0.7528, 0.7605, 0.7629, 0.7593]]
    draw_hot(data4, "balance-1 IR=12", "hot4_balance-1")

    data5 = [[0.8712, 0.8601, 0.8348, 0.8110],
             [0.8758, 0.8705, 0.7149, 0.8323],
             [0.7265, 0.7613, 0.7390, 0.7292],
             [0.6741, 0.6860, 0.7494, 0.6789]]
    draw_hot(data5, "yeast-7vs.1 IR=14", "hot5_yeast-7vs.1")

    data6 = [[0.9568, 0.9696, 0.9582, 0.9503],
             [0.9648, 0.9546, 0.9529, 0.9586],
             [0.9570, 0.9396, 0.9538, 0.9521],
             [0.9384, 0.9348, 0.9487, 0.9257]]
    draw_hot(data6, "ecoli-5 IR=16", "hot6_ecoli-5")

    data7 = [[0.9278, 0.9250, 0.8980, 0.9144],
             [0.9266, 0.9294, 0.9128, 0.8808],
             [0.9180, 0.8715, 0.8901, 0.9005],
             [0.9052, 0.9222, 0.9171, 0.9230]]
    draw_hot(data7, "yeast-4 IR=28", "hot7_yeast-4")

    data8 = [[0.9289, 0.9351, 0.8829, 0.9296],
             [0.9483, 0.9169, 0.9138, 0.9287],
             [0.9323, 0.9077, 0.9353, 0.9216],
             [0.8625, 0.9170, 0.9163, 0.9184]]
    draw_hot(data8, "yeast-6 IR=41", "hot8_yeast-6")


def fig_20():
    """
    进化前后对比图，F1-Score，AUC
    """
    title = "Comparison of before and after evolution"
    save_name = "Comparison_before_after_evolution"
    y_label = ""

    # data[0] 是进化前，data[1] 是进化后
    F1_y_data = [[0.7842, 0.8612, 0.8081, 0.8362, 0.8920, 0.8824, 0.8698, 0.7713, 0.9004,
                  0.9262, 0.9663, 0.8682, 0.9138, 0.9670, 0.9972, 0.9455, 0.9746, 0.9739,
                  0.8536, 0.9330, 0.9677, 0.9122, 0.9801, 0.9643, 0.9199, 0.9624, 0.9878,
                  0.9868],
                 [0.7975, 0.8672, 0.8114, 0.8424, 0.9023, 0.8824, 0.8746, 0.7955, 0.9087,
                  0.9305, 0.9790, 0.8784, 0.9221, 0.9713, 0.9978, 0.9655, 0.9845, 0.9831,
                  0.8695, 0.9458, 0.9756, 0.9353, 0.9899, 0.9723, 0.9593, 0.9683, 0.9895,
                  0.9940]]
    AUC_y_data = [[0.7890, 0.8351, 0.7266, 0.8323, 0.9019, 0.8082, 0.8249, 0.6738, 0.8296, 0.9327,
                   0.9909, 0.7289, 0.8938, 0.9772, 0.9994, 0.9204, 0.9732, 0.9903, 0.7007, 0.8475,
                   0.9589, 0.7007, 0.9884, 0.9075, 0.4934, 0.7994, 0.9441, 0.9733],
                  [0.8027, 0.8474, 0.7530, 0.8568, 0.9090, 0.8065, 0.8423, 0.7078, 0.8321, 0.9392,
                   0.9966, 0.7350, 0.9133, 0.9780, 0.9997, 0.9253, 0.9734, 0.9969, 0.7057, 0.8755,
                   0.9605, 0.7038, 0.9894, 0.9119, 0.5079, 0.8012, 0.9472, 0.9840]]
    y_data = F1_y_data + AUC_y_data

    # 数据集标签
    line_labels = ["F1 before", "F1 after evolution", "AUC before", "AUC after evolution"]
    x_tick_labels = data_name

    draw_line_chart(y_data, line_labels=line_labels, x_tick_labels=x_tick_labels, title=None, y_label=y_label,
                    save_name=save_name)


def fig_21():
    """
    自适应采样率折线图
    """
    x_ticks = [i for i in range(1, 16)]
    y_data = [
        [[9.999816265871788, 9.999632531743574, 9.99926506348715, 9.998530126974298, 9.997060253948595,
          9.99412050789719, 9.988241015794376, 9.976482031588752, 9.952964063177504, 9.905928126355008,
          9.811856252710012, 9.623712505420023, 9.247425010840047, 8.494850021680096, 6.989700043360189],
         [10.005919781496788, 10.011839562993574, 10.02367912598715, 10.047358251974298, 10.094716503948595,
          10.18943300789719, 10.378866015794376, 10.757732031588752, 11.515464063177504, 13.030928126355008,
          16.06185625271001, 22.123712505420023, 34.24742501084005, 58.49485002168009, 106.98970004336019]],

        [[3.9999474271779696, 3.999894854355939, 3.9997897087118783, 3.9995794174237562, 3.9991588348475133,
          3.9983176696950258, 3.996635339390052, 3.993270678780103, 3.9865413575602067, 3.973082715120413,
          3.9461654302408258, 3.892330860481652, 3.784661720963303, 3.569323441926607, 3.138646883853214],
         [4.00605094280297, 4.012101885605939, 4.024203771211878, 4.048407542423757, 4.096815084847513,
          4.193630169695026, 4.3872603393900516, 4.774520678780103, 5.549041357560207, 7.0980827151204124,
          10.196165430240825, 16.392330860481653, 28.7846617209633, 53.569323441926606, 103.13864688385321]],

        [[1.6666494452100018, 1.6666322237533369, 1.6665977808400072, 1.6665288950133472, 1.6663911233600281,
          1.6661155800533893, 1.665564493440112, 1.6644623202135576, 1.6622579737604484, 1.6578492808542302,
          1.6490318950417935, 1.6313971234169204, 1.5961275801671744, 1.5255884936676822, 1.3845103206686975],
         [1.6727529608350018, 1.6788392550033369, 1.6910118433400072, 1.7153570200133472, 1.7640473733600281,
          1.8614280800533893, 2.056189493440112, 2.4457123202135578, 3.224757973760448, 4.78284928085423,
          7.899031895041794, 14.131397123416923, 26.596127580167177, 51.525588493667684, 101.38451032066871]]
    ]
    title = "Adaptive sampling rate corresponding to different IR"
    sub_title = ["IR=10", "IR=25", "IR=60"]
    x_label = "Base classifier number"
    y_label = "sampling rate"
    save_name = "sampling_rate_different_IR"
    draw_some_line_chart(x_ticks, y_data, 2, 3, title, sub_title, x_label, y_label,
                         save_name)


if __name__ == '__main__':
    fig_19()
