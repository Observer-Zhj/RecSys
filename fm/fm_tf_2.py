# -*- coding: utf-8 -*-
# @Author  : ZhengHj
# @Time    : 2020/5/20 23:15
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
from fm.datasets import DataSet, DataParser, to_sparse
from scipy.sparse import hstack


class FM:
    """
    Factorization Machine with tensorflow
    :param feature_nums: int, discrete feature size
    :param max_iter: int, maximum iterations, equivalent to the epochs, default 30
    :param eta: float, learning rate, default 0.0001
    :param batch: int, minibatch size, default 256
    :param k: factor dimension, default 30
    :param alpha: coefficient of L2 regularization, default 0.01
    :param optimizer: optimizer, dufault "Adam"
    :param log_name: str, log name, default "fm_tf_1"
    """
    def __init__(self, feature_nums,
                 max_iter=30, eta=0.0001,
                 batch=256, k=30,
                 alpha=0.01, optimizer="Adam",
                 log_name="fm_tf_1"):
        self.feature_nums = feature_nums
        self.max_iter = max_iter
        self.eta = eta
        self.batch = batch
        self.k = k
        self.alpha = alpha
        self.optimizer = optimizer
        self.w0 = None
        self.w = None
        self.V = None
        tf.reset_default_graph()
        self.g = tf.get_default_graph()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(log_device_placement=True))
        self.logger = set_logger(name=log_name)
        self.logger.info("arguments: {}".format({"max_iter": max_iter, "eta": eta, "batch": batch,
                                                 "k": k, "alpha": alpha, "optimizer": optimizer}))

    def inference(self, indices, values, dense_shape, reuse=False):
        """ Feedforward process """
        with self.g.as_default():
            with tf.variable_scope("fm", reuse=reuse):
                X = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
                self.b0 = tf.get_variable("b", 1, initializer=tf.initializers.zeros())
                self.w = tf.get_variable("w", [self.feature_nums, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.V = tf.get_variable("v", [self.feature_nums, self.k], initializer=tf.truncated_normal_initializer(stddev=0.1))
                xw = tf.nn.embedding_lookup_sparse(self.w, X, sp_weights=None, combiner='sum')
                xw = tf.reshape(xw, (-1, ))
                xv_1 = tf.nn.embedding_lookup_sparse(self.V, X, sp_weights=None, combiner='sum') ** 2
                xv_2 = tf.nn.embedding_lookup_sparse(self.V ** 2, X, sp_weights=None, combiner='sum')
                y_ = self.b0 + xw + 0.5 * tf.reduce_sum(xv_1 - xv_2, axis=1)
                unique_X = tf.unique(values).y
                rl = tf.contrib.layers.l2_regularizer(self.alpha)(tf.nn.embedding_lookup(self.w, unique_X)) \
                     + tf.contrib.layers.l2_regularizer(self.alpha)(tf.nn.embedding_lookup(self.V, unique_X))
                return y_, rl

    def build(self, indices, values, dense_shape, y, reuse=False):
        """ Calculate the loss, generate the optimizer """
        with self.g.as_default():
            y_, rl = self.inference(indices, values, dense_shape, reuse)
            loss = tf.losses.mean_squared_error(y, y_)
            loss += rl
            global_step = tf.Variable(0, trainable=False)
            train_op = tf.contrib.layers.optimize_loss(loss, global_step, self.eta, self.optimizer)
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
            self.indices = tf.placeholder(tf.int64, [None, 2], name="indices")
            self.values = tf.placeholder(tf.int32, [None], name="value")
            self.dense_shape = tf.placeholder(tf.int64, [2], name="dense_shape")
            self.y = tf.placeholder(tf.float32, [None], name="output_y")
            y_, loss, train_op = self.build(self.indices, self.values, self.dense_shape, self.y)

            self.sess.run(tf.global_variables_initializer())

            ds = DataSet(X, y)
            total_indices, total_values = to_sparse(X)
            for it in range(self.max_iter):
                for _ in range(len(X) // self.batch):
                    batch_X, batch_y = ds.next_batch(self.batch)
                    indices, values = to_sparse(batch_X)
                    _, train_loss = self.sess.run([train_op, loss], feed_dict={
                        self.indices: indices,
                        self.values: values,
                        self.dense_shape: [len(batch_X), self.feature_nums],
                        self.y: batch_y})
                train_pre, train_losses = self.sess.run([y_, loss], feed_dict={
                    self.indices: total_indices,
                    self.values: total_values,
                    self.dense_shape: [len(X), self.feature_nums],
                    self.y: y})
                train_rmse = np.sqrt(np.mean((np.array(train_pre) - y) ** 2))
                if vali:
                    indices, values = to_sparse(vali[0])
                    vail_pre, test_losses = self.sess.run([y_, loss], feed_dict={
                        self.indices: indices,
                        self.values: values,
                        self.dense_shape: [len(vali[0]), self.feature_nums],
                        self.y: vali[1]})
                    vail_rmse = np.sqrt(np.mean((np.array(vail_pre) - vali[1]) ** 2))
                    self.logger.info("epoch {} train loss: {} train rmse: {} vail loss: {} vail rmse: {}".
                                     format(it, train_losses, train_rmse, test_losses, vail_rmse))
                else:
                    self.logger.info("epoch {} train loss: {} train rmse: {}".format(it, train_losses, train_rmse))

    def predict(self, X):
        """
        predict
        :param X: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                  indi_j is the feature index of feature field j of sample i in the training set
        :return array-like, shape [n_samples]
        """
        return self._predict(X)

    def _predict(self, X):
        indices, values = to_sparse(X)
        with self.g.as_default():
            output, _ = self.inference(self.indices, self.values, self.dense_shape, True)
            y_ = self.sess.run(output, feed_dict={
                self.indices: indices,
                self.values: values,
                self.dense_shape: [len(X), self.feature_nums]
            })
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
    data = np.array(data)
    # Each user randomly selects 20% of the records as the validation set
    train_idx, test_idx = split_data(mdata, rate=0.2)

    y = np.array(ratings.rating)

    fm_model = FM(feature_nums=dp.feat_dim, max_iter=30, batch=512, optimizer="Adam", log_name="fm_tf_1")

    fm_model.fit(data[train_idx], y[train_idx], (data[test_idx], y[test_idx]))

    pre = fm_model.predict(data[test_idx])
    rmse = np.sqrt(np.mean((np.array(pre) - y[test_idx])**2))
    fm_model.logger.info("after {} epochs, final rmse: {}".format(fm_model.max_iter, rmse))
