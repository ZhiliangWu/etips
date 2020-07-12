#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# etips
#
# Copyright (c) Siemens AG, 2020
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from custom import IpsLossLayer


def plot_the_model(model, fp=Path('./models'), fn='model.png', shape=False):
    """plot the model in the given position

    Args:
        model (tf.keras.models.Model): instance of the model
        fp (pathlib.PosixPath): the file path to save the plot
        fn (str): the file name of the plot
        shape (bool): whether to display shape information or not

    Returns:
        None. The plot of the model is saved to the given position

    """
    fp.mkdir(parents=True, exist_ok=True)
    plot_model(model, to_file=str(fp / fn), show_shapes=shape)


def build_lstm_classifier(timesteps=32, feature_size=784, output_shape=3,
                          repr_size=64, activation='tanh',inp_drop=0.0,
                          re_drop=0.0, l2_coef=1e-3, lr=3e-4):

    seq_inputs = layers.Input(shape=(timesteps, feature_size), name='Sequential_Input')
    x = layers.Masking(mask_value=0, name='Masking')(seq_inputs)
    x = layers.LSTM(repr_size, activation=activation, use_bias=True, dropout=inp_drop, recurrent_dropout=re_drop,
                    return_sequences=False, name='Sequential_Representation')(x)
    class_pred = layers.Dense(output_shape, activation='softmax', use_bias=True, kernel_regularizer=l2(l2_coef),
                              name='Class_Prediction')(x)

    m = Model(inputs=[seq_inputs], outputs=class_pred)
    m.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    print('model is built and compiled')

    return m


def build_direct_method_classifier(timesteps=32, feature_size=784, repr_size=64,
                                   action_size=3, act_repr_size=4,
                                   activation='tanh', inp_drop=0.0,
                                   re_drop=0.0, l2_coef=1e-3, lr=3e-4):

    seq_inputs = layers.Input(shape=(timesteps, feature_size), name='Sequential_Input')
    x = layers.Masking(mask_value=0, name='Masking')(seq_inputs)
    x = layers.LSTM(repr_size, activation=activation, use_bias=True, dropout=inp_drop, recurrent_dropout=re_drop,
                    return_sequences=False, name='Sequential_Representation')(x)

    action_input = layers.Input(shape=(action_size,), name='Action_Input')
    y = layers.Dense(act_repr_size, activation=activation, use_bias=False, kernel_regularizer=l2(l2_coef),
                     name='Action_Representation')(action_input)

    z = layers.Concatenate(axis=-1, name='Total_Representation')([x, y])
    class_pred = layers.Dense(1, activation='sigmoid', use_bias=True, kernel_regularizer=l2(l2_coef),
                              name='Binary_Classification')(z)

    m = Model(inputs=[seq_inputs, action_input], outputs=class_pred)
    m.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

    print('model is built and compiled')

    return m


def build_bandit_lstm_classifier(timesteps=32, feature_size=784, output_shape=3, repr_size=64, activation='tanh',
                                 inp_drop=0.0, re_drop=0.0, l2_coef=1e-3, lr=3e-4, translation=0.0):
    seq_inputs = layers.Input(shape=(timesteps, feature_size), name='Sequential_Input')
    x = layers.Masking(mask_value=0, name='Masking')(seq_inputs)
    x = layers.LSTM(repr_size, activation=activation, use_bias=True, dropout=inp_drop, recurrent_dropout=re_drop,
                    return_sequences=False, name='Sequential_Representation')(x)
    class_pred = layers.Dense(output_shape, activation='softmax', use_bias=True, kernel_regularizer=l2(l2_coef),
                              name='Class_Prediction')(x)
    action = layers.Input(shape=(output_shape,), name='Action_Input', dtype=tf.int32)
    propen = layers.Input(shape=(), name='Propensity_Input', dtype=tf.float32)
    delta = layers.Input(shape=(), name='Delta_Input', dtype=tf.float32)
    ips_loss = IpsLossLayer(translation=translation,
                            name='ipsloss')([class_pred, action, propen, delta])
    m = Model(inputs=[seq_inputs, action, propen, delta], outputs=ips_loss, name='training')
    m.add_loss(ips_loss)
    m.compile(optimizer=Adam(lr=lr))
    test_m = Model(inputs=m.get_layer('Sequential_Input').input, outputs=m.get_layer('Class_Prediction').output,
                   name='testing')
    test_m.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    print('model is built and compiled')

    return m, test_m


def build_callbacks(fp='./Exps/', monitor='val_loss', save=True):
    """build a list of useful callbacks

    Args:
        fp (str): place to save the information generated by Callbacks, e.g. './logs/'
        monitor (str): value to be monitored with earlystopper
        save(bool): save the result or not

    Returns:
        list_cbs (list): a list of tf.keras.Callbacks

    """

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                     patience=5,
                                                     restore_best_weights=True)

    if save:
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(fp, 'training.log'))

        list_cbs = [
                    early_stopper,
                    csv_logger
                    ]
    else:
        list_cbs = [early_stopper,
                    ]

    return list_cbs


if __name__ == '__main__':
    # tf.enable_eager_execution()
    m1, m2 = build_bandit_lstm_classifier()
    plot_the_model(m1, fn='training_with_propensity.png', shape=False)
    plot_the_model(m2, fn='testing_with_propensity.png', shape=False)
