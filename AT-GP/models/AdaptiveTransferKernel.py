#  Copyright (c) 2022 International Business Machines
#  All rights reserved.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  Authors: Hengxuan Ying (hengxuanying@gmail.com)
#


from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin
from skopt.learning.gaussian_process.kernels import Kernel, Hyperparameter, Matern
import numpy as np

class AdaptiveTransferKernel(NormalizedKernelMixin, StationaryKernelMixin, Kernel):

    def __init__(self,
                 kernel = 1**2 * Matern(length_scale=1., nu=2.5),
                 lamb=2.,                # the original value is from -1 to 1. Shifted because of the log of zero issue.
                 lamb_bounds=(1., 3.),
                 different_noises=False,
                 source_noise_level=1.,
                 source_noise_bounds=(1e-05, 0.1),
                 target_noise_level=1.,
                 target_noise_bounds=(1e-06, 1e-05)):   # original range from -1 to 1
        self.lamb = lamb
        self.kernel = kernel
        self.noise_source_level = source_noise_level
        self.noise_target_level = target_noise_level
        self.lamb_bounds = lamb_bounds
        self.source_noise_bounds = source_noise_bounds
        self.target_noise_bounds = target_noise_bounds
        self.different_noises=different_noises

    def _f(self, X, Xp, eval_gradient=False):
        X_ = X[:, :-1]      # the last column is the label of the domain: 0 for the source and 1 for the the target
        Xp_ = Xp[:, :-1]

        isFromSameDomain = np.array([[x == xp for xp in Xp[:, -1]] for x in X[:, -1]])
        fromDifferentDomains = (self.lamb - 2) * np.logical_not(isFromSameDomain)
        matrix_lamb = fromDifferentDomains + isFromSameDomain

        noise_matrix_source, noise_matrix_target = 0, 0
        if self.different_noises:
            noise_matrix_source = np.array([[np.array_equal(x, xp) and x[-1] == 0. for xp in Xp] for x in X]).astype(
                'float64')
            noise_matrix_source = self.noise_source_level * noise_matrix_source
            noise_matrix_target = np.array([[np.array_equal(x, xp) and x[-1] == 1. for xp in Xp] for x in X]).astype(
                'float64')
            noise_matrix_target = self.noise_target_level * noise_matrix_target

        if eval_gradient:
            k, g = self.kernel(X_, eval_gradient=True)
            if self.different_noises:
                g = np.dstack((g, noise_matrix_source))                 # add gradient of the matrix respect to the source noise
                g = np.dstack((g, noise_matrix_target))                 # add gradient of the matrix respect to the target noise
            g = np.dstack((g, np.logical_not(isFromSameDomain) * k))    # add gradient of the matrix respect to lambda
            k = matrix_lamb * k
            if self.different_noises:
                k = k + noise_matrix_source + noise_matrix_target

            return k, g

        k = self.kernel(X_, Xp_)
        k = matrix_lamb * k
        if self.different_noises:
            k = k + noise_matrix_source + noise_matrix_target
        return k

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.array(X)
        if Y is None:
            if eval_gradient:
                return self._f(X, X, True)
            Y = X
        else:
            Y = np.array(Y)

        if eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        return self._f(X, Y)

    # needs to define this. Otherwise the std prediction is not right.
    def diag(self, X):
        res = self.kernel.diag(X[:, :-1])
        if self.different_noises:
            isTarget = X[:, 1]
            diag_white_kernel = self.noise_target_level * isTarget + self.noise_source_level * np.logical_not(isTarget)
            res = res + diag_white_kernel
        return res

    def get_params(self, deep=True):
        params = dict()
        if deep:
            params.update((key, val) for key, val in self.kernel.get_params().items())
        params.update(source_noise_level=self.noise_source_level)
        params.update(target_noise_level=self.noise_target_level)
        params.update(lamb=self.lamb)
        return params

    @property
    def hyperparameters(self):
        res = self.kernel.hyperparameters
        if self.different_noises:
            res.append(Hyperparameter('noise_source_level', 'numeric', self.source_noise_bounds))
            res.append(Hyperparameter('noise_target_level', 'numeric', self.target_noise_bounds))
        res.append(Hyperparameter('lamb', 'numeric', self.lamb_bounds))

        return res

    @property
    def theta(self):
        res = self.kernel.theta
        if self.different_noises:
            res = np.append(res,
                            np.log(self.noise_source_level))
            res = np.append(res,
                            np.log(self.noise_target_level))
        res = np.append(res,
                        np.log(self.lamb))
        return res

    @theta.setter
    def theta(self, theta):
        if self.different_noises:
            self.kernel.theta = theta[:-3]
            self.noise_source_level = np.exp(theta[-3])
            self.noise_target_level = np.exp(theta[-2])
            self.lamb = np.exp(theta[-1])
        else:
            self.kernel.theta = theta[:-1]
            self.lamb = np.exp(theta[-1])

    def gradient_x(self, x, X_train):
        gradient_x = self.kernel.gradient_x(x, X_train)
        gradient_x[:, -1] = 0

        isSource = np.array([x == 0 for x in X_train[:, -1]]).reshape(-1, 1)
        factor = np.repeat(((self.lamb - 2) * isSource + np.logical_not(isSource)), 3, axis=1)
        gradient_x = factor * gradient_x

        return gradient_x