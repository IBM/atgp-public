#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


from skopt.learning import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_squared_error
from functions.styblinski_tang import feature_space_range
from models.AdaptiveTransferKernel import AdaptiveTransferKernel
from multiprocessing import Process, Lock, Manager
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def generate_normalized_y(source_y, target_y):
    mean_source = np.mean(source_y, axis=0)
    std_source = np.std(source_y, axis=0)
    if std_source == 0:
        std_source = 1
    source_normalized = (source_y - mean_source) / std_source

    mean_target = np.mean(target_y, axis=0)
    std_target = np.std(target_y, axis=0)
    if std_target == 0:
        std_target = 1
    target_normalized = (target_y - mean_target) / std_target

    return np.append(source_normalized, target_normalized), mean_target, std_target

class generate_test_result_multiprocessing():
    def __init__(self, d):
        self.lock = Lock()
        self.dict = d
        test_dataset = np.load('./parallel_test_dataset_different_functions.npz')
        self.source_data = test_dataset['arr_0']
        self.target_data = test_dataset['arr_1']
        self.test_data = test_dataset['arr_2']
        self.x_dims = len(feature_space_range)
        self.X_test = np.array(self.test_data[:self.x_dims, :]).T
        self.at_test = np.hstack((self.X_test, np.ones((self.X_test.shape[0], 1))))
        self.shifts = test_dataset['arr_3']
        self.numbers = [50, 100, 200, 500, 1000]

    def proc(self, idx):
        print('start computing for shift with index {}'.format(idx))

        res = np.zeros((len(self.numbers), 4, 20))  # number_datapoints x model/lambda x dataset

        for idx_datapoints, nb_datapoints in enumerate(self.numbers):

            for dataset in range(20):
                source_x = np.array(self.source_data[dataset, :self.x_dims, :]).T
                source_y = np.array(self.source_data[dataset, self.x_dims, :])
                target_x = np.array(self.target_data[dataset, :self.x_dims, :nb_datapoints]).T
                target_y = np.array(self.target_data[dataset, self.x_dims + idx, :nb_datapoints])

                at_source = np.hstack((source_x, np.zeros((source_x.shape[0], 1))))
                at_target = np.hstack((target_x, np.ones((target_x.shape[0], 1))))
                at_train = np.vstack((at_source, at_target))
                at_y, mean_target, std_target = generate_normalized_y(source_y, target_y)

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
                Y_train = np.array(at_y, copy=True)
                gpr_ottertune = GaussianProcessRegressor(kernel=base_estimator,
                                                         n_restarts_optimizer=2,
                                                         noise=0.2,
                                                         normalize_y=True).fit(X_train, Y_train)

                y_pred = gpr.predict(self.X_test)
                res[idx_datapoints, 0, dataset] = mean_squared_error(self.test_data[idx + self.x_dims, :],
                                                                     y_pred,
                                                                     squared=False)
                y_pred = gpr_ottertune.predict(self.X_test)
                y_pred = y_pred * std_target + mean_target
                res[idx_datapoints, 1, dataset] = mean_squared_error(self.test_data[idx + self.x_dims, :],
                                                                     y_pred,
                                                                     squared=False)
                y_pred = gpr_at_transfer.predict(self.at_test)
                y_pred = y_pred * std_target + mean_target
                rmse = mean_squared_error(self.test_data[idx + self.x_dims, :],
                                          y_pred,
                                          squared=False)

                print(' shift index {}, with {} target points, dataset {}, rmse = {}'
                      .format(idx, nb_datapoints, dataset, rmse))

                res[idx_datapoints, 2, dataset] = rmse
                res[idx_datapoints, 3, dataset] = gpr_at_transfer.kernel_.get_params()['k1__lamb'] - 2

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
        np.save('parallel_test_result_different_functions_with_normalization', result)

if __name__ == '__main__':
    manager = Manager()
    d = manager.dict()
    obj = generate_test_result_multiprocessing(d)
    obj.main()
