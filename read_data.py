import random
import numpy as np


root_dir = "./data"

def shuffle_data(x, y):
    """
    打乱数据

    :param x:
    :param y:
    :return:
    """
    data = list(zip(x, y))
    random.shuffle(data)
    x, y = zip(*data)

    return x, y

def ecoli0_1_4_7_vs_2_3_5_6():
    label = ("cp", "im", "imS", "imL", "imU", "om", "omL", "pp")
    pos_label = (label[0], label[1], label[4], label[7])
    neg_label = (label[2], label[3], label[5], label[6])
    file_dir = root_dir + "/ecoli.dat"
    x_pos = []
    x_neg = []

    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                tag = line.split(",")[-1]
                tag = tag.replace(" ", "")
                tag = tag.replace("\n", "")

                if tag in pos_label:
                    x_pos.append(t)
                elif tag in neg_label:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def ecoli3():
    label = ("cp", "im", "imS", "imL", "imU", "om", "omL", "pp")
    pos_label = label[3]
    file_dir = root_dir + "/ecoli.dat"
    x_pos = []
    x_neg = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def ecoli1():
    label = ("cp", "im", "imS", "imL", "imU", "om", "omL", "pp")
    pos_label = label[1]
    file_dir = root_dir + "/ecoli.dat"
    x_pos = []
    x_neg = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def ecoli2():
    label = ("cp", "im", "imS", "imL", "imU", "om", "omL", "pp")
    pos_label = label[2]
    file_dir = root_dir + "/ecoli.dat"
    x_pos = []
    x_neg = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def vehicle2():
    label = ("van", "saab", "bus", "opel")
    pos_label = label[2]
    file_dir = root_dir + "/vehicle.dat"
    x_pos = []
    x_neg = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def vehicle1():
    label = ("van", "saab", "bus", "opel")
    pos_label = label[1]
    file_dir = root_dir + "/vehicle.dat"
    x_pos = []
    x_neg = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def yeast1():
    """
    yeast 数据中，"NUC"作为负样本，其余全做正样本，已打乱

    :return: 整理好的数据，类似 MNIST
    """
    label = ("MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL")
    neg_label = label[1]                # 负样本标签
    file_dir = root_dir + "/yeast.dat"  # 文件路径
    x_pos = []
    x_neg = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            # @ 开头的行都是一些描述行
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签（标签占一个单词），提取数据
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                # 如果该行包含负标签
                if line.find(neg_label) != -1:
                    # 添加到负样本列表中
                    x_neg.append(t)
                else:
                    x_pos.append(t)
            line = file.readline()
    # 计算不平衡率：正（多）/负（少）
    IR = len(x_pos) / len(x_neg)
    # 计算总样本数
    e = len(x_pos) + len(x_neg)
    # 合并正负样本
    x = np.array(x_pos + x_neg)
    # 生成对应样本的二分类标签
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    x, y = shuffle_data(x, y)

    return x, y


def yeast_7_vs_1():
    label = ("MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL")
    pos_label = label[7]
    neg_label = label[1]
    file_dir = root_dir + "/yeast.dat"
    x_pos = []
    x_neg = []

    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                elif line.find(neg_label) != -1:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def yeast6():
    label = ("MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL")
    pos_label = label[6]
    file_dir = root_dir + "/yeast.dat"
    x_pos = []
    x_neg = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def yeast4():
    label = ("MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL")
    pos_label = label[4]
    file_dir = root_dir + "/yeast.dat"
    x_pos = []
    x_neg = []
    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def yeast_7_vs_1_4_5_8():
    label = ("MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL")
    pos_label = label[7]
    neg_label = (label[1], label[4], label[5], label[8])
    file_dir = root_dir + "/yeast.dat"
    x_pos = []
    x_neg = []

    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 按逗号分割，去掉末尾的标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                if line.find(pos_label) != -1:
                    x_pos.append(t)
                # 因为最有有个 \n，所以是 -4
                elif line[-4:-1] in neg_label:
                    x_neg.append(t)
            line = file.readline()
    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y

def thoraric_surgery():
    file_dir = root_dir + "/thoraric-surgery.arff"

    x_pos = []
    x_neg = []

    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 提取标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                label = line.split(",")[-1]
                label = label.replace(" ", "")
                label = label.replace("\n", "")

                if label == "positive":
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()

    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)



    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("IR=%.2f e=%d" % (IR, e))

    return x, y

def fertility():
    file_dir = root_dir + "/fertility_Diagnosis.txt"

    x_pos = []
    x_neg = []

    with open(file_dir) as file:
        line = file.readline()
        while line:

            # 提取标签
            t = line.split(",")[:-1]
            t = [float(x) for x in t]
            t = np.array(t)

            label = line.split(",")[-1]
            label = label.replace(" ", "")
            label = label.replace("\n", "")

            if label == "N":
                x_pos.append(t)
            else:
                x_neg.append(t)

            line = file.readline()

    # IR = len(x_neg) / len(x_pos)
    IR = len(x_pos) / len(x_neg)

    e = len(x_pos) + len(x_neg)



    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("IR=%.2f e=%d" % (IR, e))

    return x, y

def haberman():
    file_dir = root_dir + "/haberman.dat"

    x_pos = []
    x_neg = []

    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 提取标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                label = line.split(",")[-1]
                label = label.replace(" ", "")
                label = label.replace("\n", "")

                if label == "positive":
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()

    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y


def page_blocks0():
    # TODO 这个还没改
    file_dir = root_dir + "/page-blocks.dat"

    x_pos = []
    x_neg = []

    with open(file_dir) as file:
        line = file.readline()
        while line:
            if line[0] == "@":
                pass
            else:
                # 提取标签
                t = line.split(",")[:-1]
                t = [float(x) for x in t]
                t = np.array(t)

                label = line.split(",")[-1]
                label = label.replace(" ", "")
                label = label.replace("\n", "")

                if label == "0":
                    x_pos.append(t)
                else:
                    x_neg.append(t)
            line = file.readline()

    IR = len(x_neg) / len(x_pos)
    e = len(x_pos) + len(x_neg)
    print("IR=%.2f e=%d" % (IR, e))
    x = np.array(x_pos + x_neg)
    y_pos = np.ones((len(x_pos),), dtype=np.uint8)
    y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
    y = np.concatenate((y_pos, y_neg))

    print("x.shape", x.shape)
    print(x)
    print("y.shape", y.shape)
    print(y)

    return x, y

if __name__ == '__main__':
    x, y = yeast1()
