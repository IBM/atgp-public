#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


import matplotlib.pyplot as plt
import numpy as np

# suffix = '_50_target'
suffix = '_100_target'
statistic = 'Mean'
plot_std = False
log_scale = True
test_result = np.load('../data/test_result_goldstein'+suffix+'.npy')
shifts = np.load('../data/test_dataset_goldstein'+suffix+'.npz')['arr_3'][:]
print(shifts.shape)

only_target = test_result[:, 0, 1, :]
ottertune = test_result[:, 1, 1, :]
at_gpr_matern = test_result[:, 2, 1, :]
lamb_matern = test_result[:, 3, 0, :]

data = []
# if statistic == 'Median':
#     data.append(np.median(only_target, axis=1))
#     data.append(np.median(ottertune, axis=1))
#     data.append(np.median(at_gpr_matern, axis=1))
# else:
#     data.append(np.mean(only_target, axis=1))
#     data.append(np.mean(ottertune, axis=1))
#     data.append(np.mean(at_gpr_matern, axis=1))

data.append(np.max(only_target, axis=1))
data.append(np.max(ottertune, axis=1))
data.append(np.max(at_gpr_matern, axis=1))

if log_scale:
    for i in range(len(data)):
        data[i] = np.log(data[i])

# median of the 20 runs for each shift
plt.plot(shifts, data[0], label='only target')
if plot_std:
    plt.fill_between(shifts,
                     np.mean(only_target, axis=1) - 2 * np.std(only_target, axis=1),
                     np.mean(only_target, axis=1) + 2 * np.std(only_target, axis=1),
                     alpha=0.2)
plt.plot(shifts, data[1], label='ottertune')
if plot_std:
    plt.fill_between(shifts,
                     np.mean(ottertune, axis=1) - 2 * np.std(ottertune, axis=1),
                     np.mean(ottertune, axis=1) + 2 * np.std(ottertune, axis=1),
                     alpha=0.2)
plt.plot(shifts, data[2], label='AT-GP')
if plot_std:
    plt.fill_between(shifts,
                     np.mean(at_gpr_matern, axis=1) - 2 * np.std(at_gpr_matern, axis=1),
                     np.mean(at_gpr_matern, axis=1) + 2 * np.std(at_gpr_matern, axis=1),
                     alpha=0.2)
plt.xlabel('shift of the target')
if log_scale:
    plt.ylabel('RMSE (log scale)')
else:
    plt.ylabel('RMSE')
plt.title('{} of {} runs for each shift'.format(statistic, test_result.shape[3]))
plt.legend()
plt.show()

# median of the lambda of the 20 runs for each shift
plt.plot(shifts, np.median(lamb_matern, axis=1), label='lambda Matern Kernel')
if plot_std:
    plt.fill_between(shifts,
                     np.mean(lamb_matern, axis=1) - 2 * np.std(lamb_matern, axis=1),
                     np.mean(lamb_matern, axis=1) + 2 * np.std(lamb_matern, axis=1),
                     alpha=0.2)
plt.xlabel('shift of the target')
plt.ylabel('lambda')
plt.title('{} of lambda of {} runs for each shift'.format(statistic, test_result.shape[3]))
plt.legend()
plt.show()
