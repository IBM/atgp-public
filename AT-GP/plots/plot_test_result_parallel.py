#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


import matplotlib.pyplot as plt
import numpy as np

# test_result = np.load('../data/parallel_test_result_styblinski_200_source_1000_target.npy')
# test_result = np.load('../data/parallel_test_result_styblinski_2_source_1000_target_sampler.npy')
# test_result = np.load('../data/parallel_test_result_styblinski_200_source_1000_target_sampler_wo_norm.npy')
# test_result = np.load('../data/parallel_test_result_styblinski_200_source_1000_target_sampler_wo_norm_2.npy')
test_result = np.load('../data/shifting/parallel_test_result_styblinski_2_source_1000_target_sampler_wo_norm.npy')
# test_result = np.load('../data/parallel_test_result_different_functions.npy')
# test_result = np.load('../data/parallel_test_result_different_functions_with_normalization.npy')
print(test_result.shape)
# shifts = np.load('../data/shifting/parallel_test_dataset_styblinski.npz')['arr_3']
# nb_points = [25, 50, 75, 100, 125, 150, 175, 200]
nb_points = [50, 100, 200, 500, 1000]

# plt.figure(figsize=(10, 15))
for ind, shift in enumerate([-3., 0.]):
    plt.figure()
    only_target = test_result[ind, :, 0, :]
    ottertune = test_result[ind, :, 1, :]
    at_gpr_matern = test_result[ind, :, 2, :]
    lamb_matern = test_result[ind, :, 3, :]

    plt.errorbar(nb_points, (np.mean(only_target, axis=1)), np.std(only_target, axis=1), label='only target')
    plt.errorbar(nb_points, (np.mean(ottertune, axis=1)), np.std(ottertune, axis=1), label='ottertune')
    plt.errorbar(nb_points, (np.mean(at_gpr_matern, axis=1)), np.std(at_gpr_matern, axis=1), label='AT-GP')
    plt.title('shift = {:.2f}'.format(shift))
    plt.xlabel('number of points')
    plt.ylabel('MSE')
    plt.legend()

    plt.figure()
    plt.plot(nb_points, np.log(np.median(only_target, axis=1)), label='only target')
    plt.plot(nb_points, np.log(np.median(ottertune, axis=1)), label='ottertune')
    plt.plot(nb_points, np.log(np.median(at_gpr_matern, axis=1)), label='AT-GP')
    plt.title('shift = {:.2f}'.format(shift))
    plt.xlabel('number of points')
    plt.ylabel('MSE (log scale)')
    plt.legend()

    plt.figure()
    plt.errorbar(nb_points, np.mean(lamb_matern, axis=1), np.std(lamb_matern, axis=1), label='Lambda')
    plt.title('shift = {:.2f}'.format(shift))
    plt.legend()

plt.show()