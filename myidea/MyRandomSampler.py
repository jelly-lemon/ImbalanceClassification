import random

import numpy as np


class myRandomSampler():

    def random_sample(self, x, y, sampling_rate):
        x_pos = x[y == 1]
        x_neg = x[y == 0]

        # 采样率大于1就说明是上采样
        if sampling_rate > 1:
            x_neg = self.resample(x_neg, sampling_rate)
        else:
            x_pos = self.resample(x_pos, sampling_rate)

        x = np.concatenate((x_pos, x_neg), axis=0)
        y_pos = np.ones((len(x_pos),), dtype=np.uint8)
        y_neg = np.zeros((len(x_neg),), dtype=np.uint8)
        y = np.concatenate((y_pos, y_neg), axis=0)
        data = list(zip(x, y))
        random.shuffle(data)
        x, y = zip(*data)
        x = np.array(x)
        y = np.array(y)

        return x, y

    def up_sampling(self, x, y, sampling_rate):
        return self.random_sample(x, y, sampling_rate)

    def under_sampling(self, x, y, sampling_rate):
        return self.random_sample(x, y, sampling_rate)


    def resample(self, data, sampling_rate, replacement=True):
        """
        随机采样

        :param data: 数据
        :param sampling_rate: 采样率
        :param replacement: 是否放回
        :return:
        """
        if replacement:
            data = random.choices(data, k=int(sampling_rate * len(data)))
            return data





