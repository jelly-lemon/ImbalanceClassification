"""
读取指定文件里面的数据
"""
import os
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


def upsampling_copy(x: np.array, y: np.array, times):
    """
    所有数据拷贝几份，增加数据量

    :param x:
    :param y:
    :param times: 复制份数
    :return:
    """
    x_t = x.copy()
    y_t = y.copy()
    for i in range(times):
        x_t = np.concatenate((x_t, x.copy()))
        y_t = np.concatenate((y_t, y.copy()))

    return x_t, y_t


def upsampling_random(x, y, target_number=90):
    """
    过采样，随机选中某些样本进行复制，然后添加到原数据集中

    扩充过后，多数类/少数类比例不变

    :param x:原数据集
    :param y:原标签
    :return:扩充后的数据集
    """

    # x, y = list(x), list(y)
    x_neg = [x[i] for i in range(len(y)) if y[i] == 0]
    x_pos = [x[i] for i in range(len(y)) if y[i] == 1]

    if len(x_neg) >= target_number:
        return np.array(x), np.array(y)

    # 计算不平衡率
    IR = len(x_pos) / len(x_neg)

    # 为了保持不平衡率，正样本需要复制扩充多少
    y1_add_number = int(IR * target_number - len(x_pos))
    y0_need_number = target_number - len(x_neg)  # 需要复制多少个

    # 不放回随机抽样
    for i in range(y1_add_number):
        t = random.sample(x_pos, 1).copy()
        x_pos.append(t[0])

    for i in range(y0_need_number):
        t = random.sample(x_neg, 1).copy()
        x_neg.append(t[0])

    # 合并
    x = np.concatenate((np.array(x_pos), np.array(x_neg)))
    y = [1 for i in range(len(x_pos))] + [0 for i in range(len(x_neg))]

    print("复制扩充后：%d/%d=%.2f" % (len(x_pos), len(x_neg), len(x_pos) / len(x_neg)))

    return np.array(x), np.array(y)


def list_dir(dir):
    dir_name = []
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isdir(path):
            dir_name.append(path)

    return dir_name


def get_file_path(file_name):
    """
    在本程序目录下找到指定文件，返回其路径

    :param file_name:文件名
    :return:该文件的路径
    """
    file_path = None
    if "./raw_data" in list_dir("./"):
        file_path = os.path.join("./raw_data", file_name)

    if "./data" in list_dir("./"):
        file_path = os.path.join("./data/raw_data", file_name)

    if "../data" in list_dir("../"):
        file_path = os.path.join("../data/raw_data", file_name)

    if "../../data" in list_dir("../../"):
        file_path = os.path.join("../../data/raw_data", file_name)

    if file_path is not None:
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(file_path, "不存在")


def get_data(neg_no, pos_no, file_name, shuffle=False, show_info=False, need_copy=False):
    """
    按指定标签读取文件里的数据，正类为1，负类为0

    :param neg_no: 负标签下标
    :param pos_no: 正标签下标(-1表示除了指定的正类标签外，其余都为负类)
    :param file_name: 文件名
    :param shuffle: 是否随机打乱
    :return: 打包好的数据(array 类型)

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
    # 将负样本标签下标转 List (针对只给了一个数字的情况)
    if type(neg_no) is not list:
        neg_no = [neg_no]

    # -1 表示取除了负标签以外的全部标签作为正标签
    if pos_no == -1:
        pass
    elif type(pos_no) is not list:
        pos_no = [pos_no]

    # 获取文件真实路径
    file_path = get_file_path(file_name)

    x_pos = []
    x_neg = []
    neg_label = []
    pos_label = []
    with open(file_path) as file:
        line = file.readline()
        while line:
            # @ 开头的行都是一些描述行
            if line[0] == "@":
                if (line.find("Class") != -1 or line.find("Type") != -1) and line.find("{") != -1:
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
                # 如果标签末尾含有一个 "."，把它给删掉
                if label[-1] == '.':
                    label = label.replace(".", "")
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                # 如果该行包含负标签
                if label in neg_label:
                    x_neg.append(t)  # 添加到负样本列表中
                elif label in pos_label:
                    x_pos.append(t)
            line = file.readline()

    # 统计样本信息
    if len(x_neg) != 0:
        IR = len(x_pos) / len(x_neg)  # 计算不平衡率：正（多）/负（少）
    else:
        print("不存在负样本")
        IR = len(x_pos)
    e = len(x_pos) + len(x_neg)  # 计算总样本数

    #
    # 当数据太多了，只用其中一部分
    #
    # x_neg_expected = 90 # 期望负样本数量只有90个
    # if len(x_neg) > x_neg_expected:
    #     # 如果负样本数量超过了90个，就随机采样为90个
    #     x_neg = random.sample(x_neg, x_neg_expected)
    # # 保持不平衡比不变，正样本也进行随机采样
    # if int(len(x_neg) * IR) < len(x_pos):
    #     x_pos = random.sample(x_pos, int(len(x_neg) * IR))

    # 合并数据
    x = np.array(x_pos + x_neg)  # 合并正负样本
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)  # 生成对应样本的二分类标签
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    # 打印输出样本详细信息
    if show_info:
        print("-" * 60)
        print("%s label=%d m=%d IR=%.2f pos=%d neg=%d e=%d" % (
            file_name, len(all_label), len(x[0]), IR, len(x_pos), len(x_neg), e))
        print("neg_no", neg_no)
        print("pos_no", pos_no)

        # 打印样本简报
        print("")
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
        print("%-20s%-20s%-20s%-20s" % ("name", "class", "attribute", "imbalance"))
        print("%-20s%-20d%-20d%-20s" % (dataset_name, len(pos_label) + len(neg_label), len(x[0]),
                                        "%d/%d=%.2f" % (len(x_pos), len(x_neg), IR)))
        print("-" * 60)

    # 是否随机打乱
    if shuffle:
        x, y = shuffle_data(x, y)

    # 转为数组
    x = np.array(x)
    y = np.array(y)

    # 如果数据集较小，是否复制扩充
    if need_copy:
        while len(y[y == 0]) / 5 < 50:
            x, y = upsampling_copy(x, y, 1)
            if show_info:
                print("1次复制扩充")

    return x, y


if __name__ == '__main__':
    #---------------------------------------------
    # 读取数据
    x, y = get_data([1], -1, "yeast.dat", show_info=True)

    print(len(y[y == 0]))
    print(len(y[y == 1]))
