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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from styblinski_tang import source, target, feature_space_range
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

feature_spaces = []
for feature_range in feature_space_range:
    feature_spaces.append((feature_range[0], feature_range[1]))

space = Space(feature_spaces)
sampler = Lhs(criterion=None, lhs_type="classic")

train_size = 400
test_size = 200
nb_runs = 20
shifts = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
result = np.zeros((2, len(shifts), nb_runs))   # {RMSE, MAPE} x shifts x runs
print(result.shape)
X_test = np.array(sampler.generate(space.dimensions, test_size))

for index, shift in enumerate(shifts):
    print('shift = {}, index = {}'.format(shift, index))
    for i in range(nb_runs):
        print(' - run {}'.format(i))
        sampler = Lhs(criterion=None, lhs_type="classic")
        X_train = np.array(sampler.generate(space.dimensions, train_size))
        Y_train = target(X_train, mean=shift)
        Y_test = target(X_test, mean=shift)

        gpr = GaussianProcessRegressor(kernel=1 ** 2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1),
                                       n_restarts_optimizer=2,
                                       noise='gaussian',
                                       normalize_y=True).fit(X_train, Y_train)
        Y_pred = gpr.predict(X_test)
        rmse = mean_squared_error(Y_test, Y_pred, squared=False)
        mae = mean_absolute_error(Y_test, Y_pred)
        print('     - RMSE={}, MAE={}'.format(rmse, mae))
        result[0, index, i] = rmse
        result[1, index, i] = mae

np.save('result_'+str(train_size), result)

# result = np.load('result_200.npy')
# print('RMSE = {}, MAPE = {}'.format(np.median(result[0, 4, :]), np.median(result[1, 4, :])))

# print('RMSE: {}'.format(result[0, :, :]))
# print('MAPE: {}'.format(result[1, :, :]))

plt.errorbar(shifts, np.median(result[0, :, :], axis=1), label='RMSE')
plt.legend()
plt.show()

plt.figure()
plt.errorbar(shifts, np.median(result[1, :, :], axis=1), label='MAE')
plt.legend()
plt.show()
