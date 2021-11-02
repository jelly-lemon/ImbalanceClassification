from matplotlib import pyplot as plt
import seaborn

data = [[[0.9015, 0.8729, 0.9055, 0.8981],
         [0.9066, 0.8911, 0.9027, 0.8908],
         [0.8932, 0.8917, 0.8952, 0.8848],
         [0.8716, 0.8875, 0.8875, 0.8900]],
        [[0.8545, 0.8422, 0.8442, 0.8476],
         [0.8599, 0.8520, 0.8514, 0.8533],
         [0.8526, 0.8563, 0.8548, 0.8582],
         [0.8402, 0.8443, 0.8443, 0.8515]],

        [[0.7295, 0.7086, 0.7003, 0.6940],
         [0.7333, 0.7225, 0.7041, 0.7246],
         [0.7233, 0.6839, 0.6927, 0.6859],
         [0.7148, 0.7078, 0.7022, 0.7250]],

        [[0.8334, 0.8179, 0.8191, 0.7728],
         [0.8002, 0.8130, 0.7626, 0.7546],
         [0.7790, 0.7775, 0.7605, 0.7610],
         [0.7528, 0.7605, 0.7629, 0.7593]],

        [[0.8712, 0.8601, 0.8348, 0.8110],
         [0.8758, 0.8705, 0.7149, 0.8323],
         [0.7265, 0.7613, 0.7390, 0.7292],
         [0.6741, 0.6860, 0.7494, 0.6789]],

        [[0.9568, 0.9696, 0.9582, 0.9503],
         [0.9648, 0.9546, 0.9529, 0.9586],
         [0.9570, 0.9396, 0.9538, 0.9521],
         [0.9384, 0.9348, 0.9487, 0.9257]],

        [[0.9278, 0.9250, 0.8980, 0.9144],
         [0.9266, 0.9294, 0.9128, 0.8808],
         [0.9180, 0.8715, 0.8901, 0.9005],
         [0.9052, 0.9222, 0.9171, 0.9230]],

        [[0.9289, 0.9351, 0.8829, 0.9296],
         [0.9483, 0.9169, 0.9138, 0.9287],
         [0.9323, 0.9077, 0.9353, 0.9216],
         [0.8625, 0.9170, 0.9163, 0.9184]]]

title = ["ecoli-1 IR=3", "yeast-06 IR=4", "cleve-2 IR=7", "balance-1 IR=12", "yeast-7vs1 IR=14", "ecoli-5 IR=16", "yeast-4 IR=28",
         "yeast-6 IR=41"]

sub_title_font = {'family': 'Times New Roman',
                  'weight': 'bold',
                  'size': 24,
                  }

nrows = 2
ncols = 4
fig, axis = plt.subplots(nrows, ncols, figsize=(32, 14))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
for i in range(nrows):
    for j in range(ncols):
        axis[i][j].set_title(title[i*ncols+j], fontdict=sub_title_font)

        ticklabel = [5, 10, 15, 20]
        # annot 是否在每个小框中写上具体的数值
        # 返回值是 matplotlib Axes
        # YlGnBu == 黄绿蓝
        # Greens == 绿色
        axis[i][j] = seaborn.heatmap(data[i*ncols+j], annot=True, fmt=".4f", annot_kws={'size': 16, 'weight': 'normal'},
                          linewidths=0.05,
                          cmap='YlGnBu',
                          ax=axis[i][j])
        axis[i][j].set_xlabel("$N_{under}$", fontsize=24)
        axis[i][j].set_ylabel("$N_{over}$", fontsize=24, rotation=0)
        axis[i][j].set_xticklabels(ticklabel, fontsize=24)
        ticklabel.reverse()
        axis[i][j].set_yticklabels(ticklabel, fontsize=24, rotation=0)  # 加上 rotation=0， heatmap 会旋转 90

save_name = "heat_map"
# savefig 必须在 show 之前，因为 show 会默认打开一个新的画板，导致 savefig 为空白
plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')

plt.show()
