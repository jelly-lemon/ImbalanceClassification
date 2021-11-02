"""
自适应采样率折现图
"""

from math import log

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

title_font = {'family': 'Times New Roman',
           'weight': 'normal',  # normal/bold
           'size': 24,
              }

ticks_font = {'family': 'Times New Roman',
           'weight': 'normal',  # normal/bold
           'size': 24,
              }


x_IR = [i for i in range(2, 21)]

# 欠采样
all_delta = [1/(IR*log(IR, 2)) for IR in x_IR]
y_r_balance = [1/IR for IR in x_IR]
y_r_min = np.array(y_r_balance) - np.array(all_delta)/2
y_r_max = np.array(y_r_balance) + np.array(all_delta)/2

fig, ax = plt.subplots(1, 2, figsize=(21, 9))

ax[0].plot(x_IR, y_r_balance, label="balanced under-sampling rate")
ax[0].plot(x_IR, y_r_min,  linestyle='--', label="min under-sampling rate")
ax[0].plot(x_IR, y_r_max,  linestyle='--', label="max under-sampling rate")
ax[0].set_xlabel("IR", fontdict=title_font)
ax[0].set_ylabel("under-sampling rate", fontdict=title_font)
ax[0].legend(prop=title_font)
ax[0].set_xticks(x_IR)
ax[0].set_xticklabels(labels=x_IR, fontname="Times New Roman", fontsize=24)
for size in ax[0].get_yticklabels():
    size.set_fontname('Times New Roman')
    size.set_fontsize('24')

# 过采样率
x_IR = [i for i in range(2, 21)]
all_delta = [1/log(IR, 2) for IR in x_IR]
y_r_balance = np.array(x_IR) - 1
y_r_min = np.array(y_r_balance) - np.array(all_delta)/2
y_r_max = np.array(y_r_balance) + np.array(all_delta)/2

ax[1].plot(x_IR, y_r_balance, label="balanced over-sampling rate")
ax[1].plot(x_IR, y_r_min,  linestyle='--', label="min over-sampling rate")
ax[1].plot(x_IR, y_r_max,  linestyle='--', label="max over-sampling rate")
ax[1].set_xlabel("IR", fontdict=title_font)
ax[1].set_ylabel("over-sampling rate", fontdict=title_font)
ax[1].legend(prop=title_font)
ax[1].set_xticks(x_IR)
ax[1].set_xticklabels(labels=x_IR, fontname="Times New Roman", fontsize=24)
for size in ax[1].get_yticklabels():
    size.set_fontname('Times New Roman')
    size.set_fontsize('24')

save_name = "adaptive_sampling_rate"
plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')

fig.show()

