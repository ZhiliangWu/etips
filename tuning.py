#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# etips
#
# Copyright (c) Siemens AG, 2020
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import gc
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import fmin, hp, rand, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope

from bandit import LoggingPolicy
from lstm import build_callbacks, build_lstm_classifier,  \
    build_direct_method_classifier, build_bandit_lstm_classifier
from utils import fix_random_seed, InvokeTimes, load_counting_data, load_mnist_data


@scope.define
def roundup(a, decimals=1):
    return np.around(a, decimals)


def hyperopt_supervised(space, data, counter, fp):
    """hyperopt for the supervised learning
    """
    x_tr, y_tr, x_val, y_val, x_te, y_te = data
    counter.add_one()
    print(f'============= TRIAL NUMBER {counter.number} =============')
    print(space)

    batch_size = space.pop('batch_size', 32)

    model = build_lstm_classifier(timesteps=x_tr.shape[1], feature_size=x_tr.shape[2],
                                  output_shape=y_tr.shape[1], **space)

    cbs = build_callbacks(monitor='val_loss', save=False)
    try:
        print('fitting the model')
        history = model.fit(x=x_tr, y=y_tr, epochs=100, verbose=0, batch_size=batch_size,
                            validation_data=(x_val, y_val), callbacks=cbs)
        model.save(fp / f'model_trial_{counter.number}.h5')
        print('model saved')
    except Exception as e:
        print(f'Exception: {e}')
        return {'status': STATUS_FAIL, 'exception': e, **space}
    else:
        loss = - history.history['val_acc'][-1]
        _, test_acc = model.evaluate(x_te, y_te, verbose=0)
        epoch_count = len(history.epoch)

        print(f'val_acc is {-loss}')
        print(f'test_acc: {test_acc}')
        print(f'number of epochs is {epoch_count}')

        # useful for runnining many experiments with tf
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return {'loss': loss,
                'test_acc': test_acc,
                'status': STATUS_OK,
                'epoch': epoch_count,
                'batch_size': batch_size,
                'trial_index': counter.number,
                **space
                }


def hyperopt_logging_policy(space, data, counter, fp):
    """hyperopt for the training on 5% data, use one of it as logging policy
    """
    x_tr, y_tr, x_te, y_te = data
    counter.add_one()
    print(f'============= TRIAL NUMBER {counter.number} =============')
    print(space)

    lp = LoggingPolicy(model_path=None, x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te, rate=0.05)

    try:
        history = lp.train_the_policy(**space)
        lp.model.save(fp / f'model_trial_{counter.number}.h5')
    except Exception as e:
        print(f'Exception: {e}')
        return {'status': STATUS_FAIL, 'exception': e, **space}
    else:
        loss = - history.history['val_acc'][-1]
        epoch_count = len(history.epoch)
        test_acc = lp.compute_performance()
        print(f'val_acc: {-loss}, test_acc: {test_acc}')
        print(f'number of epochs is {epoch_count}')

        tf.keras.backend.clear_session()
        gc.collect()
        del lp.model

        return {'loss': loss,
                'test_acc': test_acc,
                'status': STATUS_OK,
                'epoch': epoch_count,
                'trial_index': counter.number,
                **space}


def hyperopt_score_estimation(space, data, counter, fp):
    """hyperopt for estimating the propensity score
    """
    x_tr, a_tr, x_val, a_val, x_te, y_te = data
    counter.add_one()
    print(f'============= TRIAL NUMBER {counter.number} =============')
    print(space)

    batch_size = space.pop('batch_size', 32)

    model = build_lstm_classifier(timesteps=x_tr.shape[1], feature_size=x_tr.shape[2],
                                  output_shape=a_tr.shape[1], **space)

    cbs = build_callbacks(monitor='val_loss', save=False)
    try:
        print('fitting the model')
        history = model.fit(x=x_tr, y=a_tr, epochs=75, verbose=0, batch_size=batch_size,
                            validation_data=(x_val, a_val), callbacks=cbs)
        model.save(fp / f'model_trial_{counter.number}.h5')
        print('model saved')
    except Exception as e:
        print(f'Exception: {e}')
        return {'status': STATUS_FAIL, 'exception': e, **space}
    else:
        loss = - history.history['val_acc'][-1]
        _, test_acc = model.evaluate(x_te, y_te, verbose=0)
        epoch_count = len(history.epoch)

        print(f'val_acc is {-loss}')
        print(f'accuracy w.r.t ground-truth: {test_acc}')
        print(f'number of epochs is {epoch_count}')

        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return {'loss': loss,
                'test_acc': test_acc,
                'status': STATUS_OK,
                'epoch': epoch_count,
                'batch_size': batch_size,
                'trial_index': counter.number,
                **space
                }


def hyperopt_direct_method(space, data, counter, fp):
    """hyperopt for direct method
    """
    x_tr, a_tr, d_tr, x_val, a_val, d_val, x_te, y_te = data
    counter.add_one()
    print(f'============= TRIAL NUMBER {counter.number} =============')
    print(space)

    batch_size = space.pop('batch_size', 32)

    model = build_direct_method_classifier(timesteps=x_tr.shape[1], feature_size=x_tr.shape[2],
                                           action_size=a_tr.shape[1], **space)

    cbs = build_callbacks(monitor='val_loss', save=False)
    try:
        print('fitting the model')
        history = model.fit(x=[x_tr, a_tr], y=d_tr, epochs=60, verbose=0, batch_size=batch_size,
                            validation_data=([x_val, a_val], d_val), callbacks=cbs)
        model.save(fp / f'model_trial_{counter.number}.h5')
        print('model saved')
    except Exception as e:
        print(f'Exception: {e}')
        return {'status': STATUS_FAIL, 'exception': e, **space}
    else:
        loss = - history.history['val_acc'][-1]

        # compute the test accuracy with the model
        p_list = []
        for i in range(a_tr.shape[1]):
            a_i = np.zeros(y_te.shape)
            a_i[:, i] = 1
            p = model.predict([x_te, a_i])
            # p is a 1D-array, length is the same as x_te.shape[0]
            # each value is the probability of being one
            p_list.append(p)

        p_pred = np.concatenate(p_list, axis=1)
        y_pred = np.argmax(p_pred, axis=1)
        y_true = np.argmax(y_te, axis=1)
        test_acc = accuracy_score(y_true, y_pred)

        epoch_count = len(history.epoch)

        print(f'val_acc is {-loss}')
        print(f'test_acc: {test_acc}')
        print(f'number of epochs is {epoch_count}')

        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return {'loss': loss,
                'test_acc': test_acc,
                'status': STATUS_OK,
                'epoch': epoch_count,
                'batch_size': batch_size,
                'trial_index': counter.number,
                **space
                }


def hyperopt_ips(space, data, counter, fp, translation=False):
    """hyperopt for training with counterfactual risk minimization w/ or w/o
    translation
    """
    x_tr, a_tr, p_tr, d_tr, x_val, a_val, p_val, d_val, x_te, y_te = data
    counter.add_one()
    print(f'============= TRIAL NUMBER {counter.number} =============')
    print(space)

    batch_size = space.pop('batch_size', 32)

    model, m_test = build_bandit_lstm_classifier(timesteps=x_tr.shape[1], feature_size=x_tr.shape[2],
                                                 output_shape=a_tr.shape[1], inp_drop=0.0, re_drop=0.0, **space)

    cbs = build_callbacks(monitor='val_loss', save=False)
    try:
        print('fitting the model')
        history = model.fit(x=[x_tr, a_tr, p_tr, d_tr], y=None, epochs=60, verbose=0, batch_size=batch_size,
                            validation_data=([x_val, a_val, p_val, d_val], None), callbacks=cbs)
        model.save(fp / f'model_trial_{counter.number}.h5')
        m_test.save(fp / f'test_model_trial_{counter.number}.h5')
        print('model saved')
    except Exception as e:
        print(f'Exception: {e}')
        return {'status': STATUS_FAIL, 'exception': e, **space}
    else:
        val_loss = - history.history['val_loss'][-1]
        epoch_count = len(history.epoch)

        # quickly compute some important values
        predictions = m_test.predict(x_tr)
        pro_a = predictions[a_tr.nonzero()]
        imp_ratio = np.divide(pro_a, p_tr)
        average_imp_ratio = np.mean(imp_ratio)    #

        risks = np.multiply(d_tr, imp_ratio)
        ips_loss = np.mean(risks)
        # with translation, ips_loss will be very different from the loss during
        # training/validation
        if translation:
            sn_loss = ips_loss / average_imp_ratio  #
            loss = sn_loss
        # without translation, ips_loss will be similar to the training/validation loss
        else:
            loss = ips_loss

        _, test_acc = m_test.evaluate(x_te, y_te, verbose=0)

        print(f'test_acc: {test_acc}')
        print(f'number of epochs is {epoch_count}')

        tf.keras.backend.clear_session()
        del model, m_test
        gc.collect()

        return {'loss': loss,
                'val_loss': val_loss,
                'test_acc': test_acc,
                'average_imp_ratio': average_imp_ratio,
                'status': STATUS_OK,
                'epoch': epoch_count,
                'batch_size': batch_size,
                'trial_index': counter.number,
                **space
                }


def train_logging_policy(mpath, source_number=1):
    if source_number == 1:
        x, y = load_counting_data(fp=Path('./data'), fn='Dataset_10k.pickle')
    elif source_number == 2:
        x, y = load_mnist_data()
    else:
        raise ValueError('Source data is not found')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    data = X_train, y_train, X_test, y_test
    it = InvokeTimes()

    temp_fp = mpath / 'logging'
    temp_fp.mkdir(parents=True, exist_ok=True)

    func = partial(hyperopt_logging_policy, data=data, counter=it, fp=temp_fp)
    config = {'repr_size': 16 * scope.int(hp.quniform('repr_size', 1, 8, 1)),
              'activation': hp.choice('activation',
                                      ['sigmoid', 'relu', 'tanh']),
              # 'inp_drop': scope.roundup(hp.uniform('inp_drop', 0.1, 0.9)),
              # 're_drop': scope.roundup(hp.uniform('re_drop', 0.1, 0.9)),
              'l2_coef': np.power(10, scope.int(
                  hp.quniform('l2_coef', -10, -1, 1))),
              'lr': np.power(10, scope.int(hp.quniform('lr', -10, -1, 1))),
              }
    trials = Trials()
    fmin(fn=func, space=config, algo=rand.suggest, max_evals=20, trials=trials,
         rstate=np.random.RandomState(0), return_argmin=False,
         show_progressbar=True)
    df = pd.DataFrame(trials.results)
    df.to_csv(mpath / 'trials.csv')


if __name__ == '__main__':
    fix_random_seed(0)
    path = Path('./models')
    train_logging_policy(path)
