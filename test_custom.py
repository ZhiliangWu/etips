#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# etips
#
# Copyright (c) Siemens AG, 2020
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

# python -m pytest [TEST_SCRIPT]

import numpy as np
import pytest
import tensorflow as tf
from custom import IpsLossLayer

tf.enable_eager_execution()


def _compute_ips_output(p, a, propen, delta):
    p = tf.convert_to_tensor(p, dtype=tf.float32)
    a = tf.convert_to_tensor(a, dtype=tf.int32)
    propen = tf.convert_to_tensor(propen, dtype=tf.float32)
    delta = tf.convert_to_tensor(delta, dtype=tf.float32)
    losslayer = IpsLossLayer(translation=0)
    layer_output = losslayer([p, a, propen, delta])

    return layer_output


class TestIpsLoss(object):
    def test_shape(self):
        p = np.random.rand(4, 3)
        a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        propen = np.random.rand(4)
        delta = [0, 1, 1, 0]
        ipsloss = _compute_ips_output(p, a, propen, delta)
        assert np.isscalar(ipsloss.numpy())

    def test_value(self):
        p = np.random.rand(4, 3)
        a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        propen = np.random.rand(4)
        delta = [0, 1, 1, 0]
        # computation with numpy
        # pro_a = p[np.arange(p.shape[0]), a]
        pro_a = p[a.nonzero()]
        imp_ratio = np.divide(pro_a, propen)
        risks = np.multiply(delta, imp_ratio)
        loss = np.mean(risks)

        ipsloss = _compute_ips_output(p, a, propen, delta).numpy()

        assert np.allclose(loss, ipsloss, rtol=1e-3)




