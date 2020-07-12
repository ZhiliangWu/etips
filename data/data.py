#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# etips
#
# Copyright (c) Siemens AG, 2020
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import pickle
import gzip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataSet(object):
    def __init__(self, fpath, n_samples=10000, n_zeros=2):
        """a class to generate sequential images for classification tasks.

        Example::

            ds = Dataset(fpath='./', n_samples=10000, n_zeros=3)
            ds.visualize_length_distribution()
            ds.generate_sequence(binary=False)
            ds.visualize_one_sample(sample_index=2)

        Args:
            fpath(str): path of the mnist files.
            n_samples(int): number of samples of the generated dataset.
            n_zeros(int): maximum number of zeros in each sequence.
        """
        fname1 = 'mnist_train.zip'
        fname2 = 'mnist_test.zip'
        train_data = pd.read_csv(fpath + fname1)
        test_data = pd.read_csv(fpath + fname2)
        data = np.concatenate([train_data.values, test_data.values])

        np.random.shuffle(data)
        self.zero_features = data[data[:, 0] == 0]
        self.nonzero_features = data[data[:, 0] != 0]

        # tensor and target is the final outcome
        # note, the first dimension is the digit,
        # it is kept here for the ease of visualization
        # it is removed when loaded to training in utils.load_counting_data
        self.tensor = np.zeros((n_samples, 32, 785))
        self.target = np.zeros(n_samples)

        self.zero_lengths = np.random.choice(range(n_zeros + 1), size=n_samples, replace=True).astype(int)

        self.sequence_lengths = np.random.normal(20, 5, n_samples).round().clip(3, 32).astype(int)

    def visualize_length_distribution(self):
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        _ = ax[0].hist(self.zero_lengths)
        ax[0].set_xlabel('zeros length distribution')
        ax[0].set_ylabel('Count')

        _ = ax[1].hist(self.sequence_lengths)
        ax[1].set_xlabel('sequence length distribution')
        ax[1].set_ylabel('Count')
        plt.show()

    def generate_sequences(self, binary=False):
        """generate sequences

        Args:
            binary(Boolen): whether the output label is binary or not.
            By default the task is defined as a multiclass (3) classification problem.

        Returns:
            None.
            Generated dataset is saved in the instance attributes `.tensor` and `.target`.

        Note:
            `.tensor` is of the shape `(samples, timestep, features)`

        """
        for i, (zlen, seq) in enumerate(zip(self.zero_lengths, self.sequence_lengths)):
            zero_ids = np.random.choice(self.zero_features.shape[0], size=zlen, replace=False)
            nonzero_ids = np.random.choice(self.nonzero_features.shape[0], size=seq-zlen, replace=False)
            zeros = self.zero_features[zero_ids, :]
            nonzeros = self.nonzero_features[nonzero_ids, :]

            sequence = np.concatenate([zeros, nonzeros], axis=0)
            np.random.shuffle(sequence)

            self.tensor[i, -sequence.shape[0]:, -sequence.shape[1]:] = sequence

            # here could be modified if output is binary
            if binary:
                self.target[i] = int(bool(zlen))
            else:
                self.target[i] = zlen

    def visualize_one_sample(self, sample_index=1):
        """

        Args:
            sample_index(int): index of the sample to be visualized.

        Returns:
            None.
            THe visualized sequence is plotted by the matplotlib and its label is print to the std output.

        """
        n_sequence = self.sequence_lengths[sample_index]
        fig, ax = plt.subplots(1, n_sequence)
        fig.set_size_inches(n_sequence * 3, 4)
        sequence_sample = self.tensor[sample_index, -n_sequence:, :]
        print(sequence_sample[:, 1:].shape)
        for i in range(n_sequence):
            ax[i].imshow(sequence_sample[i, 1:].reshape(28, 28), cmap='gray')
            ax[i].set_axis_off()
            ax[i].set_title(int(sequence_sample[i, 0]))
        plt.show()
        print(self.target[sample_index])

    def save_dataset(self, fpath='../Dataset/', info='first'):
        """save the dataset in the given place

        Args:
            fpath(str): place to save the generated dataset
            info(str): additional information to name the file

        Returns:
            None. Dataset is saved.

        Note:
            To use the dataset, `pickle` has to be used asgain and the output is of the form of a list

        """
        data_list = [self.tensor, self.target, self.sequence_lengths]

        with gzip.open(fpath+f"Dataset_{info}.pickle", "wb") as f:
            pickle.dump(data_list, f, protocol=-1)


if __name__ == '__main__':
    fp = './MNIST/'
    np.random.seed(0)
    ds = DataSet(fpath=fp, n_samples=10000, n_zeros=2)
    ds.visualize_length_distribution()
    ds.generate_sequences(binary=False)
    for i in range(10):
        ds.visualize_one_sample(sample_index=i)
    # ds.save_dataset(fpath='./test/', info='10k')
