#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# etips
#
# Copyright (c) Siemens AG, 2020
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from pathlib import Path
from joblib import dump

import numpy as np
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier

from utils import fix_random_seed, load_counting_data, load_mnist_data

if __name__ == '__main__':
    fix_random_seed(0)

    data_fp = Path('../data/')
    exp_name = 'RD1'  # or RD2
    cv_index = 0    # 0-4
    exp_fp = Path(f'./Exps/{exp_name}/CV{cv_index}/')
    exp_fp.mkdir(parents=True, exist_ok=True)

    x, y = load_counting_data(fp=data_fp, fn='Dataset_10k.pickle')
    # x, y = load_mnist_data()
    y = np.argmax(y, axis=1)

    test_size = int(0.1 * x.shape[0])
    x_tr, x_te = x[test_size:, :, :], x[:test_size, :, :]
    y_tr, y_te = y[test_size:], y[:test_size]

    print(f'shape of x_tr, x_te: {x_tr.shape}, {x_te.shape}')
    print(f'shape of y_tr, y_te: {y_tr.shape}, {y_te.shape}')

    kf = KFold(n_splits=5, shuffle=False, random_state=None)

    tr_idx, val_idx = list(kf.split(x_tr))[cv_index]

    data_list = [x_tr[tr_idx, :, :], y_tr[tr_idx],
                 x_tr[val_idx, :, :], y_tr[val_idx],
                 x_te, y_te]

    rc = DummyClassifier(strategy='uniform')
    rc.fit(data_list[0], data_list[1])
    acc = rc.score(data_list[-2], data_list[-1])
    print(acc)
    dump(rc, exp_fp / 'model_trial_random.joblib')
    print('The model is saved.')
