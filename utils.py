#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# etips
#
# Copyright (c) Siemens AG, 2020
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import random
import gzip
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf


class InvokeTimes(object):
    """a helper class to keep the times of invocation of a funciton
    """
    def __init__(self, init=0):
        self.number = init

    def add_one(self):
        self.number += 1


def fix_random_seed(n=0):
    """fix the random seed to facilitate reproducibility

    Args:
        n (int): the random seed

    Returns:
        None. The randomness of the program is away.

    """
    random.seed(n)
    np.random.seed(n)
    tf.random.set_random_seed(n)


def convert_to_onehot(target, n_classes=None):
    """convert categorical integers into onehot array

    Args:
        target(np.ndarray): containing categorical integers, like 1,...9
        n_classes(int): contains the information of possible classes

    Returns:
        onehot_target(np.ndarray): onehot encoded array, shape of `(taget.size, #categories)`

    """
    if n_classes is None:
        onehot_target = np.zeros((target.size, int(target.max())+1))
    else:
        onehot_target = np.zeros((target.size, n_classes))

    onehot_target[np.arange(target.size), target.astype(int)] = 1

    return onehot_target


def load_counting_data(fp=Path('./data/'), fn='Dataset_10k.pickle'):
    """load and preprocess the dataset

    Args:
        fp(pathlib.PosixPath): Path of the dataset
        fn(str): name of the dataset

    Returns:
        x_data, y_target(np.ndarray): shape of (#samples, #timesteps, #features): (10,000, ?, 784)

    """
    with gzip.open(fp / fn, 'rb') as f:
        data = pickle.load(f)
    tensor, target, sequences = data

    y_target = convert_to_onehot(target=target)

    x_data = tensor[:, :, 1:]
    x_data /= 255

    return x_data, y_target


def load_mnist_data():
    """load and preprocess the mnist dataset

    Returns:
        x_data, y_target(np.ndarray): shape of (#samples, #timesteps, #features): (70,000, 28, 28)

    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_data = np.concatenate([x_train, x_test], axis=0)
    x_data = x_data.astype('float') / 255

    y_target = np.concatenate([y_train, y_test], axis=0)
    y_target = convert_to_onehot(y_target)

    return x_data, y_target


def load_bandit_data(fp=Path('./data/'), fn='Bandit_1.pickle'):
    """"

    Returns:
        x(np.ndarray): context like sequence of images, shape would be (#samples, #timesteps, #features)
        y(np.ndarray): ground-truth label for the context x, one-hot encoded, (#samples, #classes)
        a(np.ndarray): action taken according to some policy, one-hot encoded, (#samples, #classes)
        scores(np.ndarray): generalized propensity score for the observed action, (#samples, )
        delta(np.ndarray): loss of the correspnding action, 0 is no loss (correct) while 1 is high loss (wrong action),
                           (#samples, )

    """
    with gzip.open(fp / fn, 'rb') as f:
        data = pickle.load(f)
    x, y, actions, scores, delta = data

    a = convert_to_onehot(actions)

    return x, y, a, scores, delta


def load_ebandit_data(fp=Path('./data/'), fn='eBandit_1.pickle'):
    """"

    Returns:
        x(np.ndarray): context like sequence of images, shape would be (#samples, #timesteps, #features)
        y(np.ndarray): ground-truth label for the context x, one-hot encoded, (#samples, #classes)
        a(np.ndarray): action taken according to some policy, one-hot encoded, (#samples, #classes)
        scores(np.ndarray): generalized propensity score for the observed action, (#samples, )
        delta(np.ndarray): loss of the correspnding action, 0 is no loss (correct) while 1 is high loss (wrong action),
                           (#samples, )
        scores_hat(np.ndarray): estimated propensity score for the observed action, (#samples, )
        baseline(np.ndarray): estimated baseline for the observed context, (#samples, )

    """
    with gzip.open(fp / fn, 'rb') as f:
        data = pickle.load(f)
    x, y, actions, delta, scores_hat = data

    a = convert_to_onehot(actions)

    return x, y, a, delta, scores_hat


if __name__ == '__main__':
    # x, y = load_counting_data(fp=Path('./data/'), fn='Dataset_10k.pickle')
    # x, y = load_mnist_data()
    # x, y, a, s, d = load_bandit_data(fp=Path('./data/'), fn='Bandit_1.pickle')
    x, y, a, d, s_hat = load_ebandit_data(fp=Path('../Dataset/'),
                                          fn='eBandit_1.pickle')
