#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


from functions.simple_gaussian import feature_space_range
import matplotlib.pyplot as plt
import numpy as np

test_result = np.load('../data/AT_Kernel_results/simple_gaussian/test_result_from_5_to_20.npy')
shifts = np.arange(*feature_space_range[0], 0.1)
shift_index = 100
shift = shifts[shift_index]
nb_points = np.array(range(5, 11))
print('data shape: {}'.format(test_result.shape))
print('shift = {}'.format(shift))

# only_target = test_result[:, shift_index, 0,:]
# ottertune = test_result[:, shift_index, 1, :]
# at_gpr_matern = test_result[:, shift_index, 2, :]
# lamb_matern = test_result[:, shift_index, 4, :]
#
# plt.errorbar(nb_points, np.median(only_target, axis=1), label='only target')
# plt.errorbar(nb_points+0.1, np.median(ottertune, axis=1), label='ottertune')
# plt.errorbar(nb_points+0.2, np.median(at_gpr_matern, axis=1), label='AT-GP')
# plt.title('Median of Mean Squared Error of 20 runs when shift = {:e}'.format(shift))
# plt.xlabel('Number of target train samples')
# plt.ylabel('Mean Squared Error')
# plt.legend()
# plt.show()

plt.figure(figsize=(12, 12))
plt.title('Median of Mean Squared Error using different number of target samples')
for index, shift_index in enumerate([0, 49, 100, 149, 199]):
    shift = shifts[shift_index]
    only_target = test_result[:, shift_index, 0, :]
    ottertune = test_result[:, shift_index, 1, :]
    at_gpr_matern = test_result[:, shift_index, 2, :]
    lamb_matern = test_result[:, shift_index, 3, :]

    plt.subplot(3, 2, index+1)
    plt.errorbar(nb_points, np.median(only_target[:6], axis=1), label='only target')
    plt.errorbar(nb_points, np.median(ottertune[:6], axis=1), label='ottertune')
    plt.errorbar(nb_points, np.median(at_gpr_matern[:6], axis=1), label='AT-GP')
    plt.title('shift = {:.2f}'.format(shift))
    plt.ylabel('Mean Squared Error')
    plt.legend()

plt.show()
