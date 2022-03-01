#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


import numpy as np

feature_space_range = [[-2., 2.], [-2., 2.]]

noise_level = 0.01
target_shift = 2.

def goldstein_price_wo_noise(x):
    '''
    Synthetic function to evaluate.
    :param x: points to evaluate. shape = n_points x features (2D in our case)
    :return: the value of the Rastringin func
    '''
    x_ = x[:, 0]
    y_ = x[:, 1]

    term_1 = 1 + ((x_ + y_ + 1) ** 2) * (19 - 14*x_ + 3*(x_ ** 2) - 14*y_ + 6*x_*y_ + 3*(y_ ** 2))
    term_2 = 30 + ((2*x_ - 3*y_) ** 2) * (18 - 32*x_ + 12*(x_ ** 2) + 48*y_ - 36*x_*y_ + 27*(y_ ** 2))

    return term_1 * term_2


def source(x):
    x_ = np.array(x, copy=True)
    return goldstein_price_wo_noise(x_) + np.random.normal(scale=noise_level, size=x_.shape[0])


def target(x, mean=target_shift):
    x_ = np.array(x, copy=True)
    x_ = x_ + mean * np.ones((x_.shape[0], x_.shape[1]))
    return goldstein_price_wo_noise(x_) + np.random.normal(scale=noise_level, size=x_.shape[0])


def source_bo(x):
    x_ = np.array(x, copy=True)[None, :2]
    return (goldstein_price_wo_noise(x_) + np.random.normal(scale=noise_level, size=x_.shape[0]))[0]


def target_bo(x):
    mean = 0
    x_ = np.array(x, copy=True)[None, :2] + mean * np.ones((1, 2))
    return (goldstein_price_wo_noise(x_) + np.random.normal(scale=noise_level, size=x_.shape[0]))[0]
