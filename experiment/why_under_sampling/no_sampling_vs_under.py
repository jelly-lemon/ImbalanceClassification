"""
当数据集不平衡时，探究不采样和欠采样对分类器的影响。

【实验设计】
数据集：yeast-0-6 1205/279=4.32
分类器：KNN、DT、SVC
欠采样方法：RUS (随机欠采样)
实验方法：5 折交叉验证

【注意】
直接降原始数据进行 5 折分开。
每一折数据又分为训练集和测试集。本实验所说的欠采样，是针对每一折里面训练集进行操作，不会对测试集操作。



"""
from compare import compare
from data import read_data

def one_step():
    """
    一步到位运行所有对比方法

    """
    x, y = read_data.get_data([2], -1, "zoo.dat", show_info=True)

    k = 5
    while len(y[y == 0]) / k < 50:
        x, y = read_data.upsampling_copy(x, y, 1)
        print("复制一份后：%d/%d" % (len(y[y == 1]), len(y[y == 0])))

    print("|%-20s|%-20s|%-20s|%-20s|%-20s|%-20s|%-20s" % ("", "val_acc", "val_precision", "val_recall", "val_f1", "auc_value", "val_gmean"))
    print("|%-20s|%-20s|%-20s|%-20s|%-20s|%-20s|%-20s" % ("----", "----", "----", "----", "----", "----", "----"))
    classify_method = ("KNN", "DT", "SVC")
    sampling_method = ("",)

    for classifier in classify_method:
        for sampler in sampling_method:
            result = compare.kFoldTest(x.copy(), y.copy(), sampler=sampler, classifier=classifier, k=k)
            print(result[0])


if __name__ == '__main__':
    one_step()
