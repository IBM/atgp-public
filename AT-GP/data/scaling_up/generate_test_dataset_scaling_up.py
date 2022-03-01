#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


from skopt.space import Space
from skopt.sampler import Lhs
from functions.styblinski_tang import source, feature_space_range
import numpy as np
import warnings

warnings.filterwarnings('ignore')

scales = [0.1, 0.5, 1, 2, 5, 10]
source_size = 200
target_size = 200
test_size = 400
number_datasets = 20
x_dims = len(feature_space_range)
print('feature dimensions: {}'.format(x_dims))

source_data = np.zeros((number_datasets, x_dims + 1, source_size)) # dataset x {x, y} x datapoint
target_data = np.zeros((number_datasets, x_dims + len(scales), target_size)) # shift x dataset x {x, y} x datapoint
test_data = np.zeros((x_dims + len(scales), test_size)) # (x_dims + shifts) x datapoint
print(target_data.shape)
print(test_data.shape)

space_dims = []
for feature_range in feature_space_range:
    space_dims.append((feature_range[0], feature_range[1]))

space = Space(space_dims)
sampler = Lhs(criterion=None, lhs_type="classic")
sample_noise_var = 0.1
test_data[:x_dims, :] = (np.array(sampler.generate(space.dimensions, test_size)).astype('float64')
                         + np.random.multivariate_normal([0, 0],
                                                         [[sample_noise_var, 0],
                                                          [0, sample_noise_var]],
                                                         size=test_size)).T

for i in range(number_datasets):

    print('source dataset: {}'.format(i))

    samples = np.array(sampler.generate(space.dimensions, source_size)).astype('float64')
    samples += np.random.multivariate_normal([0, 0],
                                             [[sample_noise_var, 0],
                                              [0, sample_noise_var]],
                                             size=source_size)
    source_data[i, :x_dims, :] = samples.T
    source_data[i, x_dims, :] = source(samples)
    samples = np.array(sampler.generate(space.dimensions, target_size)).astype('float64')
    samples += np.random.multivariate_normal([0, 0],
                                             [[sample_noise_var, 0],
                                              [0, sample_noise_var]],
                                             size=target_size)
    target_data[i, :x_dims, :] = samples.T

for index, scale in enumerate(scales):

    print('scaling factor {}'.format(index))

    for i in range(number_datasets):
        target_data[i, x_dims + index, :] = scale * source(target_data[i, :x_dims, :].T)

    test_data[x_dims+index, :] = scale * source(test_data[:x_dims, :].T)

np.savez('test_dataset_styblinski_scaling_up', source_data, target_data, test_data, scales)
