#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


from skopt.space import Space
from skopt.sampler import Lhs
import numpy as np
from functions.simple_gaussian import source, target

def generate_data(
        feature_space_ranges=[[-10, 10]],
        source=source,
        target=target,
        source_size=20,
        target_size=5,
        test_size=40,
        mean_target=0):
    feature_spaces = []
    for feature_range in feature_space_ranges:
        feature_spaces.append((feature_range[0], feature_range[1]))

    space = Space(feature_spaces)
    sampler = Lhs(criterion=None, lhs_type="classic")

    X_train = np.array(sampler.generate(space.dimensions, source_size))
    Y_train = source(X_train)
    X_test = np.array(sampler.generate(space.dimensions, test_size))
    Y_test = target(X_test, mean=mean_target)
    X_target = np.array(sampler.generate(space.dimensions, target_size))
    Y_target = target(X_target, mean=mean_target)

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_target': X_target,
        'Y_target': Y_target,
        'X_test': X_test,
        'Y_test': Y_test
    }
