# -*- coding: utf-8 -*-
# @Author  : ZhengHj
# @Time    : 2019/6/10 13:46
# @Project : recsys
# @File    : mydeepfm.py
# @IDE     : PyCharm


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from fm.datasets import DataSet, DataParser
from fm.log import set_logger
from sklearn.model_selection import train_test_split
import random
import pickle
import gc

"""
deepfm简单实现，
只处理了离散单值特征，连续特征额外添加一个权重矩阵即可，多值特征因为每一行长度不同，需要补0至长度一致。
avazu数据集未做特征工程，仅做程序测试。
"""


class MyDeepFM:
    """
        DeepFM with tensorflow
        :param feature_nums: int, discrete feature size
        :param K: int, factor dimension, default 8
        :param deep_units: tuple, deep layer shape
        :param max_iter: int, maximum iterations, equivalent to the epochs, default 30
        :param eta: float, learning rate, default 0.0001
        :param batch: int, minibatch size, default 256
        :param decay: float, learning rate decay rate, default 0.99
        :param optimizer: str, optimizer, dufault "Adam"
        :param log_name: str, log name, default "deepfm"
        """
    def __init__(self, feature_nums, K=8, deep_units=(32, 32), max_iter=30,
                 eta=0.0001, batch=256, decay=0.99, optimizer="Adam", log_name="deepfm"):
        self.feature_nums = feature_nums
        self.K = K
        self.deep_units = deep_units
        self.max_iter = max_iter
        self.eta = eta
        self.batch = batch
        self.decay = decay
        self.optimizer = optimizer
        tf.reset_default_graph()
        self.g = tf.get_default_graph()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(log_device_placement=True))
        self.logger = set_logger(name=log_name)
        self.logger.info("arguments: {}".format({"feature_nums": feature_nums, "max_iter": max_iter, "eta": eta,
                                                 "batch": batch, "decay": decay, "k": K, "optimizer": optimizer}))
        self.weights = {}

    def inference(self, X, reuse=False):
        with self.g.as_default():
            with tf.variable_scope("deepfm", reuse=reuse):
                # init weights
                weights_first_order = tf.get_variable("weights_first_order", shape=[self.feature_nums, 1],
                                                      dtype=tf.float32,
                                                      initializer=tf.truncated_normal_initializer(stddev=0.1))  # F*1
                weights_embedding = tf.get_variable("weights_embedding", shape=[self.feature_nums, self.K],
                                                    dtype=tf.float32,
                                                    initializer=tf.truncated_normal_initializer(stddev=0.1))  # F*K
                weights_output = tf.get_variable("weights_output", shape=[X.shape[1]+self.K+self.deep_units[-1], 1],
                                                 dtype=tf.float32,
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1))  # (f+K+u)*1
                bias = tf.get_variable("bias", shape=1, dtype=tf.float32,
                                       initializer=tf.initializers.zeros())
                self.weights = {"weights_first_order": weights_first_order,
                                "weights_embedding": weights_embedding,
                                "weights_output": weights_output,
                                "bias": bias}
                # fm first order
                fm_first_order = tf.nn.embedding_lookup(weights_first_order, X)  # None*f*1
                fm_first_order = tf.reshape(fm_first_order, (-1, X.shape[1]))  # None*f
                # embedding
                embedding = tf.nn.embedding_lookup(weights_embedding, X)  # None*f*K
                # fm second_order
                fm_second_order = tf.reduce_sum(embedding, axis=1) ** 2 - \
                                  tf.reduce_sum(embedding ** 2, axis=1)  # None*K
                # deep
                deep = tf.reshape(embedding, (-1, X.shape[1] * self.K))
                for i, units in enumerate(self.deep_units):
                    deep = tf.layers.dense(deep, units, activation=tf.nn.relu, name="deep_layer{}".format(i+1))
                y_ = tf.concat([fm_first_order, fm_second_order, deep], axis=1)
                y_ = tf.matmul(y_, weights_output) + bias
                y_logit = tf.sigmoid(y_)
                return tf.squeeze(y_), tf.squeeze(y_logit)

    def build(self, X, y, reuse=False):
        with self.g.as_default():
            y_, y_logit = self.inference(X, reuse)
            loss = tf.losses.sigmoid_cross_entropy(y, y_)
            global_step = tf.Variable(0, trainable=False)
            l_r = tf.train.exponential_decay(self.eta, global_step, 100, self.decay)
            train_op = tf.contrib.layers.optimize_loss(loss, global_step, l_r, self.optimizer)
            return y_logit, loss, train_op

    def fit(self, X, y, vali=None):
        """
        mini batch optimize
        :param X: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                  indi_j is the feature index of feature field j of sample i in the training set
        :param y: array-like, shape [n_samples]
        :param vali: tuple, Validation set, [vali_X, vali_y]
        """
        with self.g.as_default():
            self.X = tf.placeholder(tf.int32, [None, X.shape[1]], name="input_X")
            self.y = tf.placeholder(tf.float32, [None], name="output_y")
            y_logit, loss, train_op = self.build(self.X, self.y)

            self.sess.run(tf.global_variables_initializer())
            ds = DataSet(X, y)

            for it in range(self.max_iter):
                for _ in range(len(X) // self.batch):
                    batch_X, batch_y = ds.next_batch(self.batch)
                    _, train_loss = self.sess.run([train_op, loss], feed_dict={self.X: batch_X, self.y: batch_y})
                train_len = X.shape[0]
                bl = 10000
                train_losses = []
                for tl in range(train_len // bl):
                    _, train_loss = self.sess.run([y_logit, loss],
                                                  feed_dict={self.X: X[tl*bl:(tl+1)*bl],
                                                             self.y: y[tl*bl:(tl+1)*bl]})
                    train_losses.append(train_loss)
                train_losses = np.mean(train_losses)
                if vali:
                    test_len = vali[0].shape[0]
                    test_losses = []
                    for tl in range(test_len // bl):
                        _, test_loss = self.sess.run([y_logit, loss],
                                                     feed_dict={self.X: vali[0][tl*bl:(tl+1)*bl],
                                                                self.y: vali[1][tl*bl:(tl+1)*bl]})
                        test_losses.append(test_loss)
                    test_losses = np.mean(test_losses)
                    self.logger.info("epoch {} train loss: {} vail loss: {}".format(
                        it, train_losses, test_losses))
                else:
                    self.logger.info("epoch {} train loss: {}".format(it, train_losses))

    def predict(self, X):
        """
        predict
        :param X: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                  indi_j is the feature index of feature field j of sample i in the training set
        :return array-like, shape [n_samples]
        """
        with self.g.as_default():
            _, y_logit = self.inference(self.X, True)
            y_ = self.sess.run(y_logit, feed_dict={self.X: X})
            return y_


if __name__ == '__main__':
    # train = pd.read_csv("../data/avazu/train.csv")
    # label = np.array(train.click)
    # ignore_features = ["id", "click", "device_id", "device_ip"]
    # col = list(train.columns)
    # col.remove("id")
    # col.remove("click")
    # col.remove("device_id")
    # col.remove("device_ip")
    # for i in col:
    #     print("{} unique nums : {}".format(i, len(train[i].unique())))
    # dp = DataParser(train.drop(ignore_features, axis=1))
    # data = np.array(dp.parse())
    # sparse_train_data = (dp.feat_dim, data, label)
    # with open("../data/avazu/sparse_train_data", "wb") as f:
    #     pickle.dump(sparse_train_data, f)
    with open("../data/avazu/sparse_train_data", "rb") as f:
        feature_nums, data, label = pickle.load(f)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)
    del data, label
    gc.collect()
    df_model = MyDeepFM(feature_nums, K=8)
    df_model.fit(X_train, y_train, (X_test, y_test))