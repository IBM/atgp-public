#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


from skopt.learning import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from functions.goldstein_price import feature_space_range
from models.AdaptiveTransferKernel import AdaptiveTransferKernel
from multiprocessing import Process, Lock, Manager
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class generate_test_result_multiprocessing():
    def __init__(self, d):
        self.lock = Lock()
        self.dict = d
        test_dataset = np.load('./test_dataset_goldstein_100_target.npz')
        self.source_data = test_dataset['arr_0']
        self.target_data = test_dataset['arr_1']
        self.test_data = test_dataset['arr_2']
        self.x_dims = len(feature_space_range)
        self.X_test = np.array(self.test_data[:self.x_dims, :]).T
        self.at_test = np.hstack((self.X_test, np.ones((self.X_test.shape[0], 1))))
        self.shifts = test_dataset['arr_3']

    def proc(self, idx):
        print('start computing for scaling factor {}'.format(idx))

        res = np.zeros((4, 3, 20)) # models x metrics (RMSE, MAE, MAPE) x datasets
        nb_datapoints = 100

        for dataset in range(20):
            source_x = np.array(self.source_data[dataset, :self.x_dims, :]).T
            source_y = np.array(self.source_data[dataset, self.x_dims, :])
            target_x = np.array(self.target_data[dataset, :self.x_dims, :nb_datapoints]).T
            target_y = np.array(self.target_data[dataset, self.x_dims + idx, :nb_datapoints])

            at_source = np.hstack((source_x, np.zeros((source_x.shape[0], 1))))
            at_target = np.hstack((target_x, np.ones((target_x.shape[0], 1))))
            at_train = np.vstack((at_source, at_target))
            at_y = np.append(source_y, target_y)

            # AT-GP model with Matern 5/2 + WhiteKernel
            gpr_at_transfer = GaussianProcessRegressor(kernel=AdaptiveTransferKernel(),
                                                       n_restarts_optimizer=2,
                                                       noise='gaussian',
                                                       normalize_y=True).fit(at_train, at_y)

            # model built from scratch using only target data
            base_estimator = 1 ** 2 * Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1)
            gpr = GaussianProcessRegressor(kernel=base_estimator,
                                           n_restarts_optimizer=2,
                                           noise='gaussian',
                                           normalize_y=True).fit(target_x, target_y)

            # Ottertune approach
            X_train = np.vstack((source_x, target_x))
            Y_train = np.append(source_y, target_y)
            gpr_ottertune = GaussianProcessRegressor(kernel=base_estimator,
                                                     n_restarts_optimizer=2,
                                                     noise=0.2,
                                                     normalize_y=True).fit(X_train, Y_train)

            y_pred = gpr.predict(self.X_test)
            res[0, 0, dataset] = mean_squared_error(self.test_data[idx + self.x_dims, :],
                                                    y_pred,
                                                    squared=False)
            res[0, 1, dataset] = mean_absolute_error(self.test_data[idx + self.x_dims, :],
                                                     y_pred)
            res[0, 2, dataset] = mean_absolute_percentage_error(self.test_data[idx + self.x_dims, :],
                                                                y_pred)

            y_pred = gpr_ottertune.predict(self.X_test)
            res[1, 0, dataset] = mean_squared_error(self.test_data[idx + self.x_dims, :],
                                                    y_pred,
                                                    squared=False)
            res[1, 1, dataset] = mean_absolute_error(self.test_data[idx + self.x_dims, :],
                                                     y_pred)
            res[1, 2, dataset] = mean_absolute_percentage_error(self.test_data[idx + self.x_dims, :],
                                                                y_pred)

            y_pred = gpr_at_transfer.predict(self.at_test)
            rmse = mean_squared_error(self.test_data[idx + self.x_dims, :],
                                      y_pred,
                                      squared=False)
            mae = mean_absolute_error(self.test_data[idx + self.x_dims, :],
                                      y_pred)
            mape = mean_absolute_percentage_error(self.test_data[idx + self.x_dims, :],
                                                  y_pred)

            print(' shift {}, dataset {}, rmse = {}, mae = {}, mape = {}'
                  .format(idx, dataset, rmse, mae, mape))

            res[2, 0, dataset] = rmse
            res[2, 1, dataset] = mae
            res[2, 0, dataset] = mape
            res[3, 0, dataset] = gpr_at_transfer.kernel_.get_params()['k1__lamb'] - 2

        with self.lock:
            self.dict[idx] = res

    def main(self):

        jobs = []
        for idx in range(len(self.shifts)):
            p = Process(target=self.proc, args=(idx,))
            jobs.append(p)

        for job in jobs:
            job.start()
        for job in jobs:
            job.join()

        result = []
        for idx in range(len(self.shifts)):
            result.append(self.dict[idx])
        result = np.array(result)
        np.save('test_result_goldstein_100_target', result)

if __name__ == '__main__':
    manager = Manager()
    d = manager.dict()
    obj = generate_test_result_multiprocessing(d)
    obj.main()