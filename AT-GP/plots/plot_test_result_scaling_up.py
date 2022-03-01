#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


import matplotlib.pyplot as plt
import numpy as np

suffix = ''
statistic = 'Median'
plot_std = False
log_scale = False
test_result = np.load('../data/scaling_up/test_result_styblinski_scaling_up_with_normalization'+suffix+'.npy')
scales = np.load('../data/scaling_up/test_dataset_styblinski_scaling_up'+suffix+'.npz')['arr_3']
print(scales.shape)

only_target = test_result[:, 0, :]
ottertune = test_result[:, 1, :]
at_gpr_matern = test_result[:, 2, :]
lamb_matern = test_result[:, 3, :]

data = []
if statistic == 'Median':
    data.append(np.median(only_target, axis=1))
    data.append(np.median(ottertune, axis=1))
    data.append(np.median(at_gpr_matern, axis=1))
else:
    data.append(np.mean(only_target, axis=1))
    data.append(np.mean(ottertune, axis=1))
    data.append(np.mean(at_gpr_matern, axis=1))

if log_scale:
    for i in range(len(data)):
        data[i] = np.log(data[i])

# median of the 20 runs for each shift
plt.plot(scales, data[0], label='only target')
if plot_std:
    plt.fill_between(scales,
                     np.mean(only_target, axis=1) - 2 * np.std(only_target, axis=1),
                     np.mean(only_target, axis=1) + 2 * np.std(only_target, axis=1),
                     alpha=0.2)
plt.plot(scales, data[1], label='ottertune')
if plot_std:
    plt.fill_between(scales,
                     np.mean(ottertune, axis=1) - 2 * np.std(ottertune, axis=1),
                     np.mean(ottertune, axis=1) + 2 * np.std(ottertune, axis=1),
                     alpha=0.2)
plt.plot(scales, data[2], label='AT-GP')
if plot_std:
    plt.fill_between(scales,
                     np.mean(at_gpr_matern, axis=1) - 2 * np.std(at_gpr_matern, axis=1),
                     np.mean(at_gpr_matern, axis=1) + 2 * np.std(at_gpr_matern, axis=1),
                     alpha=0.2)
plt.xlabel('Scaling factor of the target function respect to the source')
if log_scale:
    plt.ylabel('MSE (log scale)')
else:
    plt.ylabel('MSE')
plt.title('{} of {} runs for each scaling factor'.format(statistic, test_result.shape[-1]))
plt.legend()
plt.show()

# median of the lambda of the 20 runs for each shift
plt.plot(scales, np.median(lamb_matern, axis=1), label='lambda Matern Kernel')
if plot_std:
    plt.fill_between(scales,
                     np.mean(lamb_matern, axis=1) - 2 * np.std(lamb_matern, axis=1),
                     np.mean(lamb_matern, axis=1) + 2 * np.std(lamb_matern, axis=1),
                     alpha=0.2)
plt.xlabel('Scaling factor of the target function respect to the source')
plt.ylabel('lambda')
plt.title('{} of lambda of {} runs for each scaling factor'.format(statistic, test_result.shape[-1]))
plt.ylim(0, 1.1)
plt.legend()
plt.show()
