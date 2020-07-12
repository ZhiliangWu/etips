#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# etips
#
# Copyright (c) Siemens AG, 2020
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT


import numpy as np
import tensorflow as tf


class IpsLossLayer(tf.keras.layers.Layer):
    def __init__(self, translation=0, **kwargs):
        """customized lambda-translated loss layer

        Args:
            translation(float): lambda value to translate the delta
        """
        super(IpsLossLayer, self).__init__(**kwargs)
        self.translation = translation

    def call(self, inputs, **kwargs):
        """compute the (lambda-translated) IPS risk with bandit feedback

        Args:
            inputs(list): all computation related tf.Tensors
                prediction(tf.Tensor): predicted probability distribution of different actions,
                                       shape is of (#samples, #classes)
                action(tf.Tensor): observed action/treatment, used to retrieve a specific entry
                                   in the prediction distribution
                propensity(tf.Tensor): (estimated) propensity value for the observed action
                delta(tf.Tensor): loss for the observed action, either 0 or 1
            **kwargs(dict): not used, only to fulfill the parent's function signiture

        Returns:
            ips_loss(tf.Tensor): scalar value, the results of the loss computation

        """
        prediction, action, propensity, delta = inputs
        # onehot_action = tf.one_hot(action, depth=self.depth)
        prob_action = tf.boolean_mask(prediction, action, name='probability_of_chosen_action')
        importance_sampling_ratio = tf.divide(prob_action, propensity, name='importance_sampling_ratio')
        lambda_delta = tf.subtract(delta, self.translation, name='lambda_translated_delta')
        risks = tf.multiply(importance_sampling_ratio, lambda_delta, name='risks')
        ips_loss = tf.reduce_mean(risks)

        return ips_loss


if __name__ == '__main__':
    tf.enable_eager_execution()

    p = tf.convert_to_tensor(np.random.rand(4, 3), dtype=tf.float32)
    print(p)
    a = tf.convert_to_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=tf.int32)
    print(a)
    pro = tf.convert_to_tensor(np.random.rand(4), dtype=tf.float32)
    print(pro)
    delt = tf.convert_to_tensor([0, 1, 1, 0], dtype=tf.float32)
    layer = IpsLossLayer(translation=0)
    layer_output = layer([p, a, pro, delt])
    # print(layer_output)
    print(layer_output.numpy())
