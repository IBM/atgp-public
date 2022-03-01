#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


from skopt.space import Space
from skopt.sampler import Lhs
from skopt.learning import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from goldstein_price import source, target, feature_space_range
from sampler import uniform_sampler
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

feature_spaces = []
for feature_range in feature_space_range:
    feature_spaces.append((feature_range[0], feature_range[1]))

mins = []
maxs = []
for feature_range in feature_space_range:
    mins.append(feature_range[0])
    maxs.append(feature_range[1])

space = Space(feature_spaces)
sampler = Lhs(criterion=None, lhs_type="classic")

train_size = 1600
test_size = 1600
nb_runs = 20
sample_noise_var = 0.05
shifts = np.arange(-2, 2.1, 0.5)
result = np.zeros((2, len(shifts), nb_runs))   # {RMSE, MAPE} x shifts x runs
print(result.shape)
X_test = np.array(sampler.generate(space.dimensions, test_size))\
         + np.random.multivariate_normal([0., 0.],
                                         [[sample_noise_var, 0.], [0., sample_noise_var]],
                                         size=test_size)
X_test = uniform_sampler(mins, maxs, size=test_size)
print(X_test.shape)
# sampler = Lhs(criterion=None, lhs_type="classic")
plt.scatter(X_test[:, 0], X_test[:, 1])
plt.title('Test sample set')
plt.show()

for index, shift in enumerate(shifts):
    print('shift = {}, index = {}'.format(shift, index))
    for i in range(nb_runs):
        print(' - run {}'.format(i))
        # X_train = np.array(sampler.generate(space.dimensions, train_size))\
        #           + np.random.multivariate_normal([0., 0.],
        #                                           [[sample_noise_var, 0.], [0., sample_noise_var]],
        #                                           size=train_size)
        X_train = uniform_sampler(mins, maxs, train_size)
        # plt.figure()
        # plt.scatter(X_train[:, 0], X_train[:, 1])
        # plt.title('Train sample set {}'.format(i))
        # plt.show()
        Y_train = target(X_train, mean=shift)
        Y_test = target(X_test, mean=shift)

        gpr = GaussianProcessRegressor(kernel=1**2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                                       n_restarts_optimizer=2,
                                       noise='gaussian',
                                       normalize_y=True).fit(X_train, Y_train)
        Y_pred = gpr.predict(X_test)
        rmse = mean_squared_error(Y_test, Y_pred, squared=False)
        mape = mean_absolute_percentage_error(Y_test, Y_pred)
        print('     - RMSE={}, MAE={}'.format(rmse, mape))
        result[0, index, i] = rmse
        result[1, index, i] = mape

np.save('result_'+str(train_size), result)

# result = np.load('result_200.npy')
# print('RMSE = {}, MAPE = {}'.format(np.median(result[0, 4, :]), np.median(result[1, 4, :])))

# print('RMSE: {}'.format(result[0, :, :]))
# print('MAPE: {}'.format(result[1, :, :]))

# plt.errorbar(shifts, np.median(result[0, :, :], axis=1), label='RMSE')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.errorbar(shifts, np.median(result[1, :, :], axis=1), label='MAPE')
# plt.legend()
# plt.show()
