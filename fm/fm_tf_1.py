# -*- coding: utf-8 -*-
# @Author  : ZhengHj
# @Time    : 2019/5/11 9:15
# @Project : recsys
# @File    : facmac_tf.py
# @IDE     : PyCharm

import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import os
from fm.log import set_logger
from fm.datasets import DataSet, DataParser
from scipy.sparse import hstack


class FM:
    """
    Factorization Machine with tensorflow
    :param max_iter: maximum iterations, equivalent to the epochs, default 3000
    :param eta: learning rate, default 0.0001
    :param batch: minibatch size, default 10000
    :param decay: learning rate decay rate, default 0.99
    :param k: factor dimension, default 30
    :param alpha: coefficient of L2 regularization, default 30
    :param optimizer: optimizer, dufault "SGD"
    """
    def __init__(self, feature_nums, max_iter=100, eta=0.0001, batch=10000, decay=0.99, k=30, alpha=0.01, optimizer="SGD", log_name="fm_tf"):
        self.feature_nums = feature_nums
        self.max_iter = max_iter
        self.eta = eta
        self.batch = batch
        self.decay = decay
        self.k = k
        self.alpha = alpha
        self.optimizer = optimizer
        self.w0 = None
        self.w = None
        self.V = None
        tf.reset_default_graph()
        self.g = tf.get_default_graph()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0，1'
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(log_device_placement=True))
        self.logger = set_logger(name=log_name)
        self.logger.info("arguments: {}".format({"max_iter": max_iter, "eta": eta, "batch": batch,
                                                 "decay": decay, "k": k, "alpha": alpha, "optimizer": optimizer}))

    def inference(self, X, reuse=False):
        """ Feedforward process """
        with self.g.as_default():
            with tf.variable_scope("fm", reuse=reuse):
                self.b0 = tf.get_variable("b", 1, initializer=tf.initializers.zeros())
                self.w = tf.get_variable("w", [self.feature_nums, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.V = tf.get_variable("v", [self.feature_nums, self.k], initializer=tf.truncated_normal_initializer(stddev=0.1))
                # def inner(elems):
                #     xw = tf.reduce_sum(tf.nn.embedding_lookup(self.w, elems))
                #     xv = tf.nn.embedding_lookup(self.V, elems)
                #     return self.b0 + xw + 0.5 * tf.reduce_sum(
                #         tf.reduce_sum(xv, axis=0) ** 2 - tf.reduce_sum(xv ** 2, axis=0))
                # y_ = tf.map_fn(inner, X, dtype=tf.float32)
                xw = tf.nn.embedding_lookup(self.w, X)
                xw = tf.reduce_sum(xw, axis=1)
                xw = tf.reshape(xw, (-1, ))
                xv = tf.nn.embedding_lookup(self.V, X)
                y_ = self.b0 + xw + 0.5 * tf.reduce_sum(tf.reduce_sum(xv, axis=1)**2 - tf.reduce_sum(xv**2, axis=1), axis=1)
                unique_X = tf.unique(tf.reshape(X, (-1, ))).y
                rl = tf.contrib.layers.l2_regularizer(self.alpha)(tf.nn.embedding_lookup(self.w, unique_X)) + \
                     tf.contrib.layers.l2_regularizer(self.alpha)(tf.nn.embedding_lookup(self.V, unique_X))
                return y_, rl

    def build(self, X, y, reuse=False):
        """ Calculate the loss, generate the optimizer """
        with self.g.as_default():
            y_, rl = self.inference(X, reuse)
            loss = tf.losses.mean_squared_error(y, y_)
            loss += rl
            global_step = tf.Variable(0, trainable=False)
            l_r = tf.train.exponential_decay(self.eta, global_step, 100, self.decay)
            train_op = tf.contrib.layers.optimize_loss(loss, global_step, l_r, self.optimizer)
            return y_, loss, train_op

    def fit(self, X, y, vali=None):
        """
        mini batch optimize
        :param X: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                  indi_j is the feature index of feature field j of sample i in the training set
        :param y: array-like, shape [n_samples]
        :param vali: tuple, Validation set, [vali_X, vali_y]
        """
        with self.g.as_default():
            self.X = tf.placeholder(tf.int32, [None, None], name="input_X")
            self.y = tf.placeholder(tf.float32, [None], name="output_y")
            y_, loss, train_op = self.build(self.X, self.y)

            self.sess.run(tf.global_variables_initializer())

            ds = DataSet(X, y)
            for it in range(self.max_iter):
                for _ in range(len(X) // self.batch):
                    batch_X, batch_y = ds.next_batch(self.batch)
                    _, train_loss = self.sess.run([train_op, loss], feed_dict={self.X: batch_X, self.y: batch_y})
                train_pre, train_losses = self.sess.run([y_, loss], feed_dict={self.X: X, self.y: y})
                train_rmse = np.sqrt(np.mean((np.array(train_pre) - y) ** 2))
                if vali:
                    test_pre, test_losses = self.sess.run([y_, loss], feed_dict={self.X: vali[0], self.y: vali[1]})
                    test_rmse = np.sqrt(np.mean((np.array(test_pre) - vali[1]) ** 2))
                    self.logger.info("epoch {} train loss: {} train rmse: {} vail loss: {} vail rmse: {}".
                                     format(it, train_losses, train_rmse, test_losses, test_rmse))
                else:
                    self.logger.info("epoch {} train loss: {} train rmse: {}".format(it, train_losses, train_rmse))

    def transform(self, X):
        return self._transform(X)

    def _transform(self, X):
        with self.g.as_default():
            output, _ = self.inference(self.X, True)
            y_ = self.sess.run(output, feed_dict={self.X: X})
            return y_


def pro_gender(x):
    if x == "M":
        return 0
    if x == "F":
        return 1


def pro_age(x):
    ages = [1, 18, 25, 35, 45, 50, 56]
    return ages.index(x)


def split_data(x, rate=0.2, seed=None):
    random.seed(seed)
    gb = x.movie_id.groupby(x.user_id)
    train_idx = []
    test_idx = []
    for uid, iids in gb:
        idx = set(iids.index)
        test = set(random.sample(idx, int(len(idx)*rate)))
        train = idx - test
        test_idx += list(test)
        train_idx += list(train)
    return train_idx, test_idx


if __name__ == '__main__':
    ohe = OneHotEncoder()
    mlb = MultiLabelBinarizer(sparse_output=True)
    # load data
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table("../data/ml-1m/ratings.dat", header=None, sep="::", names=rnames, engine="python")
    mnames = ["movie_id", "title", "genres"]
    movies = pd.read_table("../data/ml-1m/movies.dat", header=None, sep="::", names=mnames, engine="python")
    unames = ["user_id", "gender", "age", "occupation", "zipcode"]
    users = pd.read_table("../data/ml-1m/users.dat", header=None, sep="::", names=unames, engine="python")

    mdata = pd.merge(ratings, movies, on="movie_id")
    mdata = pd.merge(mdata, users, on="user_id")
    mdata["time"] = pd.to_datetime(mdata.timestamp, unit="s")
    mdata["year"] = mdata.time.dt.year
    mdata["month"] = mdata.time.dt.month
    mdata["weekday"] = mdata.time.dt.weekday
    mdata["hour"] = mdata.time.dt.hour

    # preprocessing features
    mdata.gender = mdata.gender.map(pro_gender)
    mdata.age = mdata.age.map(pro_age)
    mdata.genres = mdata.genres.map(lambda x: x.split("|"))

    ignore_cols = ["user_id", "movie_id", "rating", "timestamp", "title", "zipcode", "time"]
    dp = DataParser(mdata.drop(ignore_cols, axis=1), ["genres"])
    data = dp.parse()
    maxlen = max(map(len, data))
    data = [x + [-1]*(maxlen-len(x)) for x in data]
    data = np.array(data)
    # Each user randomly selects 20% of the records as the validation set
    train_idx, test_idx = split_data(mdata, rate=0.2)

    y = np.array(ratings.rating)

    fm_model = FM(feature_nums=dp.feat_dim, max_iter=100, batch=10000, optimizer="Adam", log_name="fm_tf_1")
    # trainX = data[train_idx].tolist()
    # testX = data[test_idx].tolist()
    fm_model.fit(data[train_idx], y[train_idx], (data[test_idx], y[test_idx]))

    pre = fm_model.transform(data[test_idx])
    rmse = np.sqrt(np.mean((np.array(pre) - y[test_idx])**2))
    fm_model.logger.info("after {} epochs, final rmse: {}".format(fm_model.max_iter, rmse))