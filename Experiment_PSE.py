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
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from hyperopt import fmin, hp, tpe, Trials
from hyperopt.pyll import scope

from tuning import hyperopt_score_estimation
from utils import fix_random_seed, InvokeTimes, load_bandit_data


if __name__ == '__main__':
    fix_random_seed(0)

    data_fp = Path('./data/')
    exp_name = 'PS1'  #propensity score estimation model
    cv_index = 0    # 0-4

    x, y, a, _, _ = load_bandit_data(fp=data_fp, fn='Bandit_1.pickle')

    test_size = int(0.1 * x.shape[0])
    x_tr, x_te = x[test_size:, :, :], x[:test_size, :, :]
    y_tr, y_te = y[test_size:, :], y[:test_size, :]
    a_tr, a_te = a[test_size:, :], a[:test_size, :]

    print(f'shape of x_tr, x_te: {x_tr.shape}, {x_te.shape}')
    print(f'shape of y_tr, y_te: {y_tr.shape}, {y_te.shape}')
    print(f'shape of a_tr, a_te: {a_tr.shape}, {a_te.shape}')

    kf = KFold(n_splits=5, shuffle=False, random_state=None)

    tr_idx, val_idx = list(kf.split(x_tr))[cv_index]
    it = InvokeTimes()
    data_list = [x_tr[tr_idx, :, :], a_tr[tr_idx, :],
                 x_tr[val_idx, :, :], a_tr[val_idx, :],
                 x_te, y_te]

    print(f'size of validation data: {data_list[-3].shape[0]}')
    exp_fp = Path(f'./Exps/{exp_name}/CV{cv_index}/')
    exp_fp.mkdir(parents=True, exist_ok=True)
    temp_fp = Path(f'./Exps/TEMP_CV{cv_index}_{exp_name}/')
    temp_fp.mkdir(parents=True, exist_ok=True)

    func = partial(hyperopt_score_estimation, data=data_list, counter=it, fp=temp_fp)

    config = {'repr_size': 16 * scope.int(hp.quniform('repr_size', 1, 8, 1)),
              'activation': hp.choice('activation', ['sigmoid', 'relu', 'tanh']),
              'l2_coef': np.power(10, scope.int(hp.quniform('l2_coef', -10, -1, 1))),
              'lr': np.power(10, scope.int(hp.quniform('lr', -10, -1, 1))),
              'batch_size': np.power(2, scope.int(hp.quniform('batch_size', 4, 7, 1))),
              }

    try:
        trials = pickle.load(open(exp_fp / 'trials.hyperopt', 'rb'))
        print('Resume from existing trials')
    except FileNotFoundError:
        print('Create new trials')
        trials = Trials()

    fmin(fn=func, space=config, algo=tpe.suggest, max_evals=100, trials=trials,
         rstate=np.random.RandomState(0), return_argmin=False, show_progressbar=True)

    df = pd.DataFrame(trials.results)
    df.to_csv(exp_fp / 'trials.csv')

    best_acc = df.loc[df.loss.idxmin(), 'test_acc']
    print(f'best in {cv_index}: {best_acc}')

    with open(exp_fp / 'trials.hyperopt', 'wb') as f:
        pickle.dump(trials, f)

    print(f'Information is saved in {exp_fp} for further analysis and training')

