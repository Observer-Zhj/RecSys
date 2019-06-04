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
    :param max_iter: maximum iterations, default 3000
    :param eta: learning rate, default 0.0001
    :param batch: minibatch size, default 10000
    :param decay: learning rate decay rate, default 0.99
    :param k: factor dimension, default 30
    :param alpha: coefficient of L2 regularization, default 30
    :param optimizer: optimizer, dufault "SGD"
    """
    def __init__(self, max_iter=100, eta=0.0001, batch=10000, decay=0.99, k=30, alpha=0.001, optimizer="SGD"):
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
        self.logger = set_logger(name="fm_tf")
        self.logger.info("arguments: {}".format({"max_iter": max_iter, "eta": eta, "batch": batch,
                                                 "decay": decay, "k": k, "alpha": alpha, "optimizer": optimizer}))

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
            l_r = tf.train.exponential_decay(self.eta, global_step, 100, self.decay)
            train_op = tf.contrib.layers.optimize_loss(loss, global_step, l_r, self.optimizer)
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
                train_losses = self.sess.run(loss, feed_dict={self.X: X, self.y: y})
                if vali:
                    test_losses = self.sess.run(loss, feed_dict={self.X: vali[0], self.y: vali[1]})
                    self.logger.info("epoch {} train rmse: {} test rmse: {}".format(it, train_losses, test_losses))
                else:
                    self.logger.info("epoch {} train: {}".format(it, train_losses))

    def transform(self, X):
        return self._transform(X)

    def _transform(self, X):
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


def split_data(x, rate=0.2):
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


    # preprocessing features
    gender = mdata.gender.map(pro_gender)
    age = mdata.age.map(pro_age)
    genres = mdata.genres.map(lambda x: x.split("|"))
    # feature one hot encoding
    gender = ohe.fit_transform(np.array(gender.values).reshape((-1, 1)))  # 2
    age = ohe.fit_transform(np.array(age.values).reshape((-1, 1)))  # 7
    occupation = ohe.fit_transform(np.array(mdata.occupation.values).reshape((-1, 1)))  # 21

    # feature genres multi labels encoding
    genres = mlb.fit_transform(np.array(genres.values))  # 18

    # concat features
    data = np.concatenate([gender, age, occupation, genres], axis=1)  #48
    # Each user randomly selects 20% of the records as the validation set
    train_idx, test_idx = split_data(mdata, rate=0.2)

    y = np.array(ratings.rating)

    fm_model = FM(max_iter=100, optimizer="Adam")
    fm_model.fit(data[train_idx], y[train_idx], (data[test_idx], y[test_idx]))

    pre = fm_model.transform(data[test_idx])
    rmse = np.mean((np.array(pre) - y[test_idx])**2)
    fm_model.logger.info("after {} epochs, final rmse: {}".format(fm_model.max_iter, rmse))