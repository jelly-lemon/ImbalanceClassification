import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import math

# import warnings
# # 解决(MatplotlibDeprecationWarning: Using default event loop until function specific to this GUI is implemented warnings.warn(str, mplDeprecation)
# warnings.filterwarnings("ignore",".*GUI is implemented.*")


fig = plt.figure()
# 连续性的画图
ax = fig.add_subplot(1, 1, 1)
# 设置图像显示的时候XY轴比例
ax.axis("equal")
# 开启一个画图的窗口
plt.ion()
print('开始仿真')
try:
    for i in range(20):
        x = np.random.randn()
        y = np.random.randn()
        if x < 0.5:
            ax.scatter(x, y, c='r', marker='v')  # 散点图
        else:
            ax.scatter(x, y, c='b', marker='.')  # 散点图
        plt.axvline(0.5)  # 画出竖线 (y=0.5)
        plt.axvline(-0.5)  # 画出竖线 (y=-.5)
        # 停顿时间
        plt.pause(0.1)
except Exception as err:
    print(err)
plt.show()
# 防止运行结束时闪退
# plt.pause(0)