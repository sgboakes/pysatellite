# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:04:41 2021

@author: ben-o_000
"""

from pysatellite import Functions
import numpy as np
import scipy
from copy import deepcopy


def ekf(x_state, cov_state, measurement, state_trans_matrix, measure_matrix, measure_noise, process_noise):
    """
    Variable Information
    Function for using Extended Kalman Filter
    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    x_state: An nx1 vector containing the system state.

    cov_state: An nxn matrix containing the state covariance.

    measurement: An mx1 vector containing the new measurement.

    state_trans_matrix: An nxn matrix that transforms the state vector forward.

    measure_matrix: An mxn matrix that predicts the measurement forward.

    measure_noise : An mxn matrix containing the measurement noise.

    process_noise: An nxn matrix containing the process noise.

    stepLength: The length of each time step in seconds.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    x_state: The new state vector.

    cov_state: The new covariance matrix.
    """
    
    # Prediction
    x_state = Functions.kepler(x_state)
    cov_state = state_trans_matrix @ cov_state @ state_trans_matrix.T + process_noise
    
    # If no measurement made, can't calculate K
    if (not np.any(measurement)) or (np.isnan(measurement).all()):
        return x_state, cov_state
    
    # Measurement-Update
    updated_measurement = measure_matrix @ x_state
    k = cov_state @ measure_matrix.T @ np.linalg.inv(measure_matrix @ cov_state @ measure_matrix.T + measure_noise)

    cov_state = (np.eye(len(cov_state)) - k @ measure_matrix) @ cov_state
    x_state = x_state + k @ (np.reshape(measurement, (3, 1)) - updated_measurement)
    
    return x_state, cov_state


class UKF:
    # Adapted from: https://github.com/balghane/pyUKF

    def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function):
        self.n_dim = int(num_states)
        self.n_sig = 1 + num_states * 2
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function

        self.lambd = pow(self.alpha, 2) * (self.n_dim + self.k) - self.n_dim

        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)

        self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
        self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))

        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1 / (2 * (self.n_dim + self.lambd))
            self.mean_weights[i] = 1 / (2 * (self.n_dim + self.lambd))

        self.sigmas = self.__get_sigmas()

    def __get_sigmas(self):
        """generates sigma points"""
        ret = np.zeros((self.n_sig, self.n_dim))

        tmp_mat = (self.n_dim + self.lambd) * self.p

        # print spr_mat
        spr_mat = scipy.linalg.sqrtm(tmp_mat)

        ret[0] = self.x
        for i in range(self.n_dim):
            ret[i + 1] = self.x + spr_mat[i]
            ret[i + 1 + self.n_dim] = self.x - spr_mat[i]

        return ret.T

    def predict(self, timestep, inputs=[]):
        """
        performs a prediction step
        :param timestep: float, amount of time since last prediction
        """

        sigmas_out = np.array([self.iterate(x, timestep, inputs) for x in self.sigmas.T]).T

        x_out = np.zeros(self.n_dim)

        # for each variable in X
        for i in range(self.n_dim):
            # the mean of that variable is the sum of
            # the weighted values of that variable for each iterated sigma point
            x_out[i] = sum((self.mean_weights[j] * sigmas_out[i][j] for j in range(self.n_sig)))

        p_out = np.zeros((self.n_dim, self.n_dim))
        # for each sigma point
        for i in range(self.n_sig):
            # take the distance from the mean
            # make it a covariance by multiplying by the transpose
            # weight it using the calculated weighting factor
            # and sum
            diff = sigmas_out.T[i] - x_out
            diff = np.atleast_2d(diff)
            p_out += self.covar_weights[i] * np.dot(diff.T, diff)

        # add process noise
        p_out += timestep * self.q

        self.sigmas = sigmas_out
        self.x = x_out
        self.p = p_out

    def update(self, states, data, r_matrix):
        """
        performs a measurement update
        :param states: list of indices (zero-indexed) of which states were measured, that is, which are being updated
        :param data: list of the data corresponding to the values in states
        :param r_matrix: error matrix for the data, again corresponding to the values in states
        """

        num_states = len(states)

        # create y, sigmas of just the states that are being updated
        sigmas_split = np.split(self.sigmas, self.n_dim)
        y = np.concatenate([sigmas_split[i] for i in states])

        # create y_mean, the mean of just the states that are being updated
        x_split = np.split(self.x, self.n_dim)
        y_mean = np.concatenate([x_split[i] for i in states])

        # differences in y from y mean
        y_diff = deepcopy(y)
        x_diff = deepcopy(self.sigmas)
        for i in range(self.n_sig):
            for j in range(num_states):
                y_diff[j][i] -= y_mean[j]
            for j in range(self.n_dim):
                x_diff[j][i] -= self.x[j]

        # covariance of measurement
        p_yy = np.zeros((num_states, num_states))
        for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
            p_yy += self.covar_weights[i] * val.dot(val.T)

        # add measurement noise
        p_yy += r_matrix

        # covariance of measurement with states
        p_xy = np.zeros((self.n_dim, num_states))
        for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1), np.array_split(x_diff, self.n_sig, 1))):
            p_xy += self.covar_weights[i] * val[1].dot(val[0].T)

        k = np.dot(p_xy, np.linalg.inv(p_yy))

        y_actual = data

        self.x += np.dot(k, (y_actual - y_mean))
        self.p -= np.dot(k, np.dot(p_yy, k.T))
        self.sigmas = self.__get_sigmas()

