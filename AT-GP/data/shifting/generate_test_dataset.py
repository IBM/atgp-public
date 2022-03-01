#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


from functions.goldstein_price import source, target, feature_space_range
from sampler import uniform_sampler

import numpy as np
import warnings

warnings.filterwarnings('ignore')

shifts = np.arange(-2, 2.1, 0.2)
source_size = 400
target_size = 400
test_size = 400
number_datasets = 20
x_dims = len(feature_space_range)
print('feature dimensions: {}'.format(x_dims))

source_data = np.zeros((number_datasets, x_dims + 1, source_size)) # dataset x {x, y} x datapoint
target_data = np.zeros((number_datasets, x_dims + len(shifts), target_size)) # shift x dataset x {x, y} x datapoint
test_data = np.zeros((x_dims + len(shifts), test_size)) # (x_dims + shifts) x datapoint
print(target_data.shape)
print(test_data.shape)

mins = []
maxs = []
for feature_range in feature_space_range:
    mins.append(feature_range[0])
    maxs.append(feature_range[1])

test_data[:x_dims, :] = uniform_sampler(mins, maxs, test_size).T

for i in range(number_datasets):
    print('source dataset: {}'.format(i))
    samples = uniform_sampler(mins, maxs, source_size)
    source_data[i, :x_dims, :] = samples.T
    source_data[i, x_dims, :] = source(samples)
    samples = uniform_sampler(mins, maxs, target_size)
    target_data[i, :x_dims, :] = samples.T

for index, shift in enumerate(shifts):
    print('shift number {}'.format(index))

    for i in range(number_datasets):
        target_data[i, x_dims + index, :] = target(target_data[i, :x_dims, :].T, mean=shift)

    test_data[x_dims+index, :] = target(test_data[:x_dims, :].T, mean=shift)

np.savez('test_dataset_goldstein_400_target', source_data, target_data, test_data, shifts)
