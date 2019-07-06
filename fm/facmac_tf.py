# -*- coding: utf-8 -*-
# @Author  : ZhengHj
# @Time    : 2019/3/25 19:08
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
from fm.datasets import DataSet


class FM:
    """
    Factorization Machine with tensorflow
    :param max_iter: int, maximum iterations, equivalent to the epochs, default 30
    :param eta: float, learning rate, default 0.0001
    :param batch: int, minibatch size, default 256
    :param k: factor dimension, default 30
    :param alpha: coefficient of L2 regularization, default 0.01
    :param optimizer: optimizer, dufault "Adam"
    :param log_name: str, log name, default "fm_tf"
    """
    def __init__(self, max_iter=0,
                 eta=0.0001, batch=256,
                 k=30, alpha=0.01,
                 optimizer="Adam",
                 log_name="fm_tf"):
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

    def reference(self, X, reuse=False):
        """ Feedforward process """
        with self.g.as_default():
            with tf.variable_scope("fm", reuse=reuse):
                self.b0 = tf.get_variable("b", 1, initializer=tf.initializers.zeros())
                self.w = tf.get_variable("w", [X.shape[1], 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.V = tf.get_variable("v", [X.shape[1], self.k], initializer=tf.truncated_normal_initializer(stddev=0.1))
                xw = tf.matmul(X, self.w)
                xw = tf.reshape(xw, (-1,))
                y_ = self.b0 + xw + 0.5 * tf.reduce_sum(tf.matmul(X, self.V)**2 - tf.matmul(X**2, self.V**2), axis=1)
                return y_

    def build(self, X, y, reuse=False):
        """ Calculate the loss, generate the optimizer """
        with self.g.as_default():
            y_ = self.reference(X, reuse)
            loss = tf.losses.mean_squared_error(y, y_)
            loss += tf.contrib.layers.l2_regularizer(self.alpha)(self.w)
            loss += tf.contrib.layers.l2_regularizer(self.alpha)(self.V)
            global_step = tf.Variable(0, trainable=False)
            train_op = tf.contrib.layers.optimize_loss(loss, global_step, self.eta, self.optimizer)
            return y_, loss, train_op

    def fit(self, X, y, vali=None):
        """
        mini batch optimize
        :param X: array-like, shape [n_samples, n_feature]
        :param y: array-like, shape [n_samples]
        :param vali: tuple, Validation set, [vali_X, vali_y]
        """
        with self.g.as_default():
            self.X = tf.placeholder(tf.float32, [None, X.shape[1]], name="input_X")
            self.y = tf.placeholder(tf.float32, [None], name="output_y")
            y_, loss, train_op = self.build(self.X, self.y)

            self.sess.run(tf.global_variables_initializer())

            ds = DataSet(X, y)
            for it in range(self.max_iter):
                for _ in range(X.shape[0] // self.batch):
                    batch_X, batch_y = ds.next_batch(self.batch)
                    _, train_loss = self.sess.run([train_op, loss], feed_dict={self.X: batch_X, self.y: batch_y})
                train_pre, train_losses = self.sess.run([y_, loss], feed_dict={self.X: X, self.y: y})
                train_rmse = np.sqrt(np.mean((np.array(train_pre) - y) ** 2))
                if vali:
                    vali_pre, vali_losses = self.sess.run([y_, loss], feed_dict={self.X: vali[0], self.y: vali[1]})
                    vali_rmse = np.sqrt(np.mean((np.array(vali_pre) - vali[1]) ** 2))
                    self.logger.info("epoch {} train loss: {} train rmse: {} vail loss: {} vail rmse: {}".
                                     format(it, train_losses, train_rmse, vali_losses, vali_rmse))
                else:
                    self.logger.info("epoch {} train loss: {} train rmse: {}".format(it, train_losses, train_rmse))

    def predict(self, X):
        return self._predict(X)

    def _predict(self, X):
        with self.g.as_default():
            output = self.reference(self.X, True)
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
    ohe = OneHotEncoder(sparse=False)
    mlb = MultiLabelBinarizer()
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
    gender = mdata.gender.map(pro_gender)
    age = mdata.age.map(pro_age)
    genres = mdata.genres.map(lambda x: x.split("|"))
    # feature one hot encoding
    gender = ohe.fit_transform(np.array(gender)[:, np.newaxis])  # 2
    age = ohe.fit_transform(np.array(age)[:, np.newaxis])  # 7
    occupation = ohe.fit_transform(np.array(mdata.occupation)[:, np.newaxis])  # 21
    year = ohe.fit_transform(np.array(mdata.year)[:, np.newaxis])
    month = ohe.fit_transform(np.array(mdata.month)[:, np.newaxis])
    weekday = ohe.fit_transform(np.array(mdata.weekday)[:, np.newaxis])
    hour = ohe.fit_transform(np.array(mdata.hour)[:, np.newaxis])

    # feature genres multi labels encoding
    genres = mlb.fit_transform(np.array(genres.values))  # 18

    # concat features, ignore ID feature
    data = np.concatenate([gender, age, occupation, genres, year, month, weekday, hour], axis=1)  #48
    # Each user randomly selects 20% of the records as the validation set
    train_idx, test_idx = split_data(mdata, rate=0.2)

    y = np.array(ratings.rating)

    fm_model = FM(max_iter=30, batch=512, optimizer="Adam", log_name="fm_tf")
    fm_model.fit(data[train_idx], y[train_idx], (data[test_idx], y[test_idx]))

    pre = fm_model.predict(data[test_idx])
    rmse = np.sqrt(np.mean((np.array(pre) - y[test_idx])**2))
    fm_model.logger.info("after {} epochs, final rmse: {}".format(fm_model.max_iter, rmse))