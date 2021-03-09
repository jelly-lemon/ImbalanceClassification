"""
目标函数
"""

import numpy as np


def objection_1(S, U):
    """
    目标函数 1

    思想：两个样本越相似，那么同属于一个类别的概率也越大
    我们认为，所有样本的 “样本相似度 * 样本类别差距” 加起来，
    值越小，预测结果越准确。

    :param S:样本相似度矩阵
    :param U:某个基分类器预测结果概率矩阵
    :return:目标函数值
    """
    n_sample = len(U)
    sum = 0
    for i in range(n_sample):
        for j in range(n_sample):
            sum += S[i][j] * np.linalg.norm(U[i] - U[j])
    return sum


def objection_2(n_cluster_center, R, U, Q):
    """
    目标函数 2

    :param n_cluster_center:聚类质心数量
    :param R:所有样本聚类结果矩阵
    :param U:某个分类器预测结果概率矩阵
    :param Q:某个分类器对聚类质心的分类结果
    :return:
    """
    n_sample = len(U)
    sum = 0
    for i in range(n_sample):
        for j in range(n_cluster_center):
            sum += R[i][j] * np.linalg.norm(U[i] - Q[j])
    return sum


def gaussian(x, y, sigma):
    """
    高斯函数
    """
    t = np.exp(-np.linalg.norm(x - y) / (2 * sigma ** 2))
    return t


def sim(i, j):
    """
    计算相似度

    :param i:样本 i
    :param j:样本 j
    :return: 相似度
    """
    return gaussian(i, j, 1)


def get_S_matrix(x):
    """
    计算相似度矩阵

    例如：
    样本的预测结果（属于类别0的概率，属于类别1的概率）：
    [[0.8 0.2]
     [0.4 0.6]
     [0.  1. ]]

    得到相似度矩阵 3x3：
    [[1.         0.32259073 0.10406478]
     [0.32259073 1.         0.32259073]
     [0.10406478 0.32259073 1.]]

    :param x:预测结果概率
    :return: 相似度矩阵
    """
    mat = np.zeros((len(x), len(x)))
    for i, i_value in enumerate(x):
        for j, j_value in enumerate(x):
            mat[i][j] = sim(i_value, j_value)
    return mat


def get_R_matrix(result, n_cluster_center):
    """
    将获得的聚类结果转 one-hot 矩阵

    例如：
    聚类结果 [1 2 0 0 1 2 0]
    转成 one-hot 矩阵
    [[0 1 0]
     [0 0 1]
     [1 0 0]
     ...
     [1 0 0]]

    :param result: 聚类结果
    :return:
    """

    mat = np.zeros((len(result), n_cluster_center), np.uint8)
    for index, num in enumerate(result):
        mat[index][num] = 1

    return mat
