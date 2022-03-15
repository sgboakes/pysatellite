# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:04:41 2021

@author: ben-o_000
"""

from pysatellite import Functions
import numpy as np


def ekf_eci(x_state, cov_state, measurement, state_trans_matrix, measure_matrix, measure_noise, process_noise):
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
