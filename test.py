from imblearn.over_sampling import SMOTE

from data import read_data

x, y = read_data.get_data([1], -1, "banana.dat", show_info=True)

x_neg = x[y == 0]
print("采样前：%d" % len(x_neg))
SMOTE()
print("采样后：%d" % len(x_neg))