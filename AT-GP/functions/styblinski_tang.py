#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


import numpy as np

feature_space_range = [[-5.12, 5.12], [-5.12, 5.12]]

noise_level = 0.001
target_shift = 2

np.random.seed(20190917)

def styblinski_tang_wo_noise(x):
    '''
    Synthetic function to evaluate.
    :param x: points to evaluate. shape = n_points x features (2D in our case)
    :return: the value of the Rastringin func
    '''
    sum_x = np.zeros((x.shape[0]))

    for i in range(x.shape[1]):
        sum_x = sum_x + x[:, i]**4 - 16*x[:, i]**2 + 5*x[:, i]

    return 0.5 * sum_x


def source(x):
    x_ = np.array(x, copy=True)
    return styblinski_tang_wo_noise(x_) + np.random.normal(scale=noise_level, size=x_.shape[0])


def target(x, mean=target_shift):
    x_ = np.array(x, copy=True)
    x_ = x_ + mean * np.ones((x_.shape[0], x_.shape[1]))
    return styblinski_tang_wo_noise(x_) + np.random.normal(scale=noise_level, size=x_.shape[0])


def source_bo(x):
    x_ = np.array(x, copy=True)[None, :2]
    return (styblinski_tang_wo_noise(x_) + np.random.normal(scale=noise_level, size=x_.shape[0]))[0]


def target_bo(x):
    mean = 0
    x_ = np.array(x, copy=True)[None, :2] + mean * np.ones((1, 2))
    return (styblinski_tang_wo_noise(x_) + np.random.normal(scale=noise_level, size=x_.shape[0]))[0]
