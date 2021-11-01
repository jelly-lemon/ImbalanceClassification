from math import log

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# delta
# x_IR = [i for i in range(2, 61)]
# all_delta = [1/(IR*log(IR, 2)) for IR in x_IR]


my_font = {'family': 'Times New Roman',
           'weight': 'normal',
           'size': 16,
           }


x_IR = [i for i in range(2, 41)]

# 欠采样
all_delta = [1/(IR*log(IR, 2)) for IR in x_IR]
y_r_balance = [1/IR for IR in x_IR]
y_r_min = np.array(y_r_balance) - np.array(all_delta)/2
y_r_max = np.array(y_r_balance) + np.array(all_delta)/2

fig, ax = plt.subplots(1, 2, figsize=(21, 9))

ax[0].plot(x_IR, y_r_balance, label="balanced under-sampling rate")
ax[0].plot(x_IR, y_r_min,  linestyle='--', label="min under-sampling rate")
ax[0].plot(x_IR, y_r_max,  linestyle='--', label="max under-sampling rate")
ax[0].set_xlabel("IR", fontdict=my_font)
ax[0].set_ylabel("under-sampling rate", fontdict=my_font)
ax[0].legend(prop=my_font)

# 过采样率
x_IR = [i for i in range(2, 21)]
all_delta = [1/log(IR, 2) for IR in x_IR]
y_r_balance = np.array(x_IR) - 1
y_r_min = np.array(y_r_balance) - np.array(all_delta)/2
y_r_max = np.array(y_r_balance) + np.array(all_delta)/2

ax[1].plot(x_IR, y_r_balance, label="balanced over-sampling rate")
ax[1].plot(x_IR, y_r_min,  linestyle='--', label="min over-sampling rate")
ax[1].plot(x_IR, y_r_max,  linestyle='--', label="max under-sampling rate")
ax[1].set_xlabel("IR", fontdict=my_font)
ax[1].set_ylabel("over-sampling rate", fontdict=my_font)
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax[1].legend(prop=my_font)


save_name = "adaptive_sampling_rate"
plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')

fig.show()

