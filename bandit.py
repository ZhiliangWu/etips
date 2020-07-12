#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# etips
#
# Copyright (c) Siemens AG, 2020
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import gzip
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from lstm import build_lstm_classifier, build_callbacks
from utils import fix_random_seed, load_counting_data, load_mnist_data


class LoggingPolicy(object):
    def __init__(self, model_path=None, x_train=None, y_train=None, x_test=None, y_test=None, rate=0.05):
        """logging policy generation based on `rate` of the complete dataset

        Args:
            model_path(Nonetype, str): place to load the trained model (if yes, then the followings are irrelevant)
            x_train(np.ndarray): complete training data of x
            y_train(np.ndarray): complete training data of y
            x_test(np.ndarray): complete test data of x
            y_test(np.ndarray): complete test data of y
            rate(float): rate used to train the logging policy
        """

        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None
            n_samples = np.int((x_train.shape[0] + x_test.shape[0]) * rate)
            print(f'{n_samples} samples')
            self.x_train = x_train[:n_samples, :, :]
            self.y_train = y_train[:n_samples, :]
            self.x_test = x_test
            self.y_test = y_test

    def train_the_policy(self, **kwargs):
        """train the logging policy based on one part of the dataset

        Args:
            **kwargs(dict): hyperparamters for specifying the LSTM

        Returns:
            val_acc(float): validation accuracy after training

        """
        cbs = build_callbacks(monitor='val_loss', save=False)
        self.model = build_lstm_classifier(timesteps=self.x_train.shape[1], feature_size=self.x_train.shape[2],
                                           output_shape=self.y_train.shape[1], **kwargs)
        history = self.model.fit(x=self.x_train, y=self.y_train, epochs=100, verbose=0, validation_split=0.1,
                                 callbacks=cbs)

        return history

    def compute_performance(self):
        """evaluate the performance of the trained model with the test data

        Returns:
            test_acc(float): accuracy of the logging policy evaluated on the test data

        Notes:
            The returned value is only for inspection purposes
        """
        _, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        return test_acc


def generate_bandit_dataset(fp, model=None, source_number=1):
    """generate bandit dataset with a trained model

    Args:
        fp(pathlib.PosixPath): path to read/save the dataset
        model(tf.keras.Model): model of the logging policy, loading a model in `.h5` file is preferred
        source_number(int): 1 for counting mnist, 2 for sequential mnist

    Returns:
        The bandit dataset is saved in the given path

    """
    if source_number == 1:
        x, y = load_counting_data(fp)
    elif source_number == 2:
        x, y = load_mnist_data()
    else:
        raise ValueError('Source data is not found')

    predictions = model.predict(x)

    actions = np.argmax(predictions, axis=1)
    scores = predictions[np.arange(predictions.shape[0]), actions]

    # if the chosen action is the same as the ground-truth label, the loss is 0;
    # otherwise, the loss is 1, the goal is to minimize the loss
    delta = (actions != np.argmax(y, axis=1)).astype('int')
    print(f'Accuracy of observed actions is {1 - delta.sum() / delta.size}')

    data_list = [x, y, actions, scores, delta]

    with gzip.open(fp / f"Bandit_{source_number}.pickle", "wb") as f:
        pickle.dump(data_list, f, protocol=-1)


def generate_estimated_bandit_dataset(fp, ps=None, source_number=1):
    """generate estimated bandit dataset with propensity score model

    Args:
        fp(pathlib.PosixPath): file path to read/save the dataset
        ps(tf.keras.Model): propensity score estimation model
        source_number(int): 1 for zeros counting MNIST, 2 for row-by-row MNIST

    Returns:
        The ebandit dataset is saved in the given path

    """

    with gzip.open(fp / f'Bandit_{source_number}.pickle', 'rb') as f:
        data = pickle.load(f)
    x, y, actions, scores, delta = data

    # as a sanity check
    _, ps_acc = ps.evaluate(x, y)
    print(f'ps acc on the whole dataset: {ps_acc}')

    predictions_hat = ps.predict(x)
    scores_hat = predictions_hat[np.arange(actions.size), actions]

    data_list = [x, y, actions, scores_hat, delta]

    with gzip.open(fp / f"eBandit_{source_number}.pickle", "wb") as f:
        pickle.dump(data_list, f, protocol=-1)


if __name__ == '__main__':
    fix_random_seed(0)
    dataset_fp = Path('./test/')
    # generate logging policy by running tuning.py

    # lp = LoggingPolicy(model_path='./models/####.h5')
    # generate_bandit_dataset(fp=dataset_fp, model=lp.model, source_number=1)

    p_model = tf.keras.models.load_model('./models/####.h5')
    generate_estimated_bandit_dataset(fp=dataset_fp, ps=p_model, source_number=1)
