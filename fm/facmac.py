# -*- coding: utf-8 -*-
# @Author  : ZhengHj
# @Time    : 2019/3/25 17:08
# @Project : recsys
# @File    : facmac.py
# @IDE     : PyCharm

import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from fm.log import set_logger


class FM:
    """
    Factorization Machine
    :param max_iter: maximum iterations, equivalent to the epochs
    :param eta: learning rate
    :param decay: learning rate decay rate
    :param k: factor dimension
    :param alpha: coefficient of L2 regularization
    :param seed: Seed for `RandomState`
    """
    def __init__(self, max_iter=30, eta=0.0001, decay=0.999, k=30, alpha=0.001, seed=None):

        self.max_iter = max_iter
        self.eta = eta
        self.decay = decay
        self.k = k
        self.alpha = alpha
        self.seed = seed
        self.w0 = None
        self.w = None
        self.V = None
        self.logger = set_logger()
        self.logger.info("arguments: {}".format({"max_iter": max_iter, "eta": eta, "decay": decay,
                                                 "k": k, "alpha": alpha}))

    def fit(self, X, y, vali=None):
        """
        SGD, each input sample updates the gradient
        delta_w = delta * X[i].T - self.alpha * self.w
        DV = X[i].T * X[i] * V - np.multiply(np.multiply(X[i], X[i]).T, V)
        delta_V = delta * DV - alpha * self.V
        :param X: array-like, shape [n_samples, n_feature]
        :param y: array-like, shape [n_samples]
        :param vali: tuple, Validation set, [vali_X, vali_y]
        """
        np.random.seed(self.seed)
        X = np.mat(X)
        y = np.mat(np.reshape(y, (-1, 1)))
        if vali:
            vali = (np.mat(vali[0]), np.mat(np.reshape(vali[1], (-1, 1))))
        self.w0 = 0.0
        self.w = np.mat(np.random.normal(0, 0.1, [X.shape[1], 1]))
        self.V = np.mat(np.random.normal(0, 0.1, [X.shape[1], self.k]))
        eta = self.eta
        for it in range(self.max_iter):
            for i in range(X.shape[0]):
                delta = y[i] - self.transform(X[i])
                delta = delta[0, 0]
                self.w0 += eta * delta
                self.w += eta * (delta * X[i].T - self.alpha * self.w)

                DV = X[i].T * X[i] * self.V - np.multiply(np.multiply(X[i], X[i]).T, self.V)
                self.V += eta * (delta * DV - self.alpha * self.V)
                # for f in range(self.k):
                #     delta_vf = np.multiply(X[i] * self.V[:, f], X[i].T) - np.multiply(np.multiply(X[i], X[i]).T, self.V[:, f])
                #     self.V[:, f] += eta * (delta * delta_vf - self.alpha * self.V[:, f])
            eta = eta * self.decay
            loss = y - self.transform(X)
            loss = np.multiply(loss, loss)
            rmse = np.sqrt(np.mean(loss))
            if vali:
                vali_loss = vali[1] - self.transform(vali[0])
                vali_loss = np.multiply(vali_loss, vali_loss)
                vali_rmse = np.sqrt(np.mean(vali_loss))
                self.logger.info("epoch {} train rmse: {} test rmse: {}".format(it, rmse, vali_rmse))
            else:
                self.logger.info("epoch {} train: {}".format(it, rmse))

    def transform(self, X):
        XV = X * self.V
        X2V2 = np.multiply(X, X) * np.multiply(self.V, self.V)
        return X * self.w + 0.5 * np.sum(np.multiply(XV, XV) - X2V2, axis=1)


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

    fm_model = FM()
    fm_model.fit(data[train_idx], y[train_idx], (data[test_idx], y[test_idx]))

    pre = fm_model.transform(data[test_idx])
    rmse = np.mean((np.squeeze(np.array(pre)) - y[test_idx])**2)
    fm_model.logger.info("after {} epochs, final rmse: {}".format(fm_model.max_iter, rmse))