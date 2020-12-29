# 描述：读取指定文件里面的数据

import random
import numpy as np


def shuffle_data(x, y):
    """
    先将样本和标签组合，然后打乱

    不会改变样本对应的标签，只是打乱样本与样本之间的顺序，对应标签会跟着一起打乱
    :param x:样本
    :param y:标签
    :return:打乱后的数据
    """
    data = list(zip(x, y))
    random.shuffle(data)
    x, y = zip(*data)

    return x, y


def get_data(neg_no, pos_no, file_name, shuffle=False):
    """
    按指定标签读取文件里的数据，正类为1，负类为0

    :param neg_no: 负标签下标
    :param pos_no: 正标签下标
    :param file_name:文件名
    :param shuffle: 是否随机打乱
    :return:打包好的数据

    Notes
    -----
    返回的数据类似 MNIST 数据格式。x 中既包含了正样本，也包含了负样本。y 就是对应的标签.

    Examples
    --------
    > x, y = get_data([1], [-1],  "yeast.dat")
    表示读取 yeast.dat 文件，其中标签数组中下标为1的标签作为负样本标签，
    -1表示剩余全部标签作为正样本标签.
    yeast.dat 文件里标签有这些：[MIT, NUC, CYT, ME1, ME2, ME3, EXC, VAC, POX, ERL]
    具体有哪些标签，需要自己打开数据文件查看，文件里面的 @attribute Class 属性就明确地
    写出了有哪些标签.

    > x, y = get_data([1], [0],  "yeast.dat")
    表示读取 yeast.dat 文件，NUC 标签样本作为负样本，MIT 标签样本作为正样本.
    """
    # 数据根目录
    root_dir = "/0-0-pycharm/ImbalanceClassification/data"

    # 将负样本标签下标转 List (针对只给了一个数字的情况)
    if type(neg_no) is not list:
        neg_no = [neg_no]

    # -1 表示取除了负标签以外的全部标签作为正标签
    if pos_no == -1:
        pass
    elif type(pos_no) is not list:
        pos_no = [pos_no]

    # 读取数据
    file_dir = root_dir + "/" + file_name  # 文件路径
    x_pos = []
    x_neg = []
    neg_label = []
    pos_label = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            # @ 开头的行都是一些描述行
            if line[0] == "@":
                if line.find("Class") != -1 and line.find("{") != -1:
                    s = line[line.find("{") + 1:-2]
                    s = s.replace(" ", "")
                    all_label = s.split(",")

                    if pos_no == -1:
                        pos_no = []
                        for i in range(len(all_label)):
                            if i not in neg_no:
                                pos_no.append(i)

                    for i in range(len(all_label)):
                        if i in neg_no:
                            neg_label.append(all_label[i])
                        if i in pos_no:
                            pos_label.append(all_label[i])
                pass
            else:
                # 按逗号分割，去掉末尾的标签（标签占一个单词），提取数据
                label = line.split(",")[-1].replace("\n", "").replace(" ", "")
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                # 如果该行包含负标签
                if label in neg_label:
                    x_neg.append(t)  # 添加到负样本列表中
                elif label in pos_label:
                    x_pos.append(t)
            line = file.readline()

    IR = len(x_pos) / len(x_neg)  # 计算不平衡率：正（多）/负（少）
    e = len(x_pos) + len(x_neg)  # 计算总样本数

    # 当数据太多了，只用其中一部分
    x_neg_expected = 90
    if len(x_neg) > x_neg_expected:
        x_neg = random.sample(x_neg, x_neg_expected)
    if int(len(x_neg) * IR) < len(x_pos):
        x_pos = random.sample(x_pos, int(len(x_neg) * IR))

    # 合并数据
    x = np.array(x_pos + x_neg)  # 合并正负样本
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)  # 生成对应样本的二分类标签
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    # 打印输出样本详细信息
    print("-"*60)
    print("数据集简报")
    print("%s label=%d m=%d IR=%.2f pos=%d neg=%d e=%d" % (
    file_name, len(all_label), len(x[0]), IR, len(x_pos), len(x_neg), e))
    print("neg_no", neg_no)
    print("pos_no", pos_no)

    # 打印样本简报
    dataset_name = file_name.split(".")[0]
    for k in neg_no:
        dataset_name += "-" + str(k)
    if len(pos_no) != len(all_label) - len(neg_no):
        dataset_name += " vs. "
        for i, k in enumerate(pos_no):
            if i == 0:
                dataset_name += str(k)
            else:
                dataset_name += "-" + str(k)
    print("数据集\t类别数量\t属性数量\t不平衡比")
    print("%s\t%d\t%d\t%d/%d=%.2f" % (dataset_name, len(pos_label)+len(neg_label), len(x[0]), len(x_pos), len(x_neg), IR))
    print("-" * 60)

    # 是否随机打乱
    if shuffle:
        x, y = shuffle_data(x, y)

    # 转为数组
    x = np.array(x)
    y = np.array(y)

    return x, y


if __name__ == '__main__':
    # 读取数据
    get_data([8], [6,7], "25到30/winequality-red.dat")
