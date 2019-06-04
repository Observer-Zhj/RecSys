# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2019/3/25 19:08
# @Project  : recsys
# @File     : datasets.py
# @Software : PyCharm

import numpy as np


class DataSet:
    def __init__(self, trainx, trainy, seed=None):
        self.train_data = (trainx, trainy)
        self.is_shuffling = 0
        self.end = 0
        self.l = len(trainx)
        self.seed = seed

    def next_batch(self, batch):
        """
        打乱数据并生成一个迭代器。

        数据打乱的规则见初始化函数的参数介绍。
        如果是第一次抽样，则会先打乱数据，然后从中抽取一个`batch`的数据。后面每次调用`next_batch`方法都会在打乱后的数据中
        取出一个`batch`大小的数据。在一轮数据被取完前，取出的数据不会重复。如果取到最后剩余数据小于`batch`，则全部取出，再
        一次抽样的时候所有数据重新打乱。

        Args:
            batch：int, 每次抽样取出的数据样本个数。

        Returns:
            抽样后的数据。
        """
        if self.is_shuffling == 0:
            np.random.seed(self.seed)
            self.randomSeries = np.random.permutation(self.l)
            self.is_shuffling = 1

        end = min(self.end + batch, self.l)
        r = list(range(self.end, end))
        r = self.randomSeries[r]
        batch_trainx = self.train_data[0][r]
        batch_trainy = self.train_data[1][r]
        self.end += batch
        if self.end >= self.l:
            self.end = 0
            self.is_shuffling = 0

        return batch_trainx, batch_trainy