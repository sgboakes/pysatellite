# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:04:41 2021

@author: ben-o_000
"""

import pysatellite
import numpy as np


def EKF_ECI(xState, covState, measurement, stateTransMatrix, measureMatrix, measurementNoise, processNoise):
    '''
    Variable Information
    Function for using Extended Kalman Filter
    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    xState: An nx1 vector containing the system state.
    
    covState: An nxn matrix containing the state covariance.
    
    measurement: An mx1 vector containing the new measurement.
    
    stateTransMatrix: An nxn matrix that transforms the state vector forward.
    
    measureMatrix: An mxn matrix that predicts the measurement forward.
    
    measurementNoise: An mxn matrix containing the measurement noise.
    
    processNoise: An nxn matrix containing the process noise.
    
    stepLength: The length of each time step in seconds.
    
    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    xState: The new state vector.
    
    covState: The new covariance matrix.
    '''
    
    # Prediction
    xState = pysatellite.Functions.kepler(xState)
    # covState = np.matmul(np.matmul(stateTransMatrix, covState), stateTransMatrix.T) + processNoise
    covState = stateTransMatrix @ covState @ stateTransMatrix.T + processNoise
    
    # If no measurement made, can't calculate K
    if (not np.any(measurement)) or (np.isnan(measurement).all()):
        return xState, covState
    
    # Measurement-Update
    # updatedMeasurement = np.matmul(measureMatrix, xState)
    updatedMeasurement = measureMatrix @ xState
    # K = np.dot(np.dot(covState, measureMatrix.T), (np.linalg.inv(np.dot(np.dot(measureMatrix, covState), measureMatrix.T) + measurementNoise)))
    K = covState @ measureMatrix.T @ np.linalg.inv(measureMatrix @ covState @ measureMatrix.T + measurementNoise)
    
    # covState = np.eye(len(covState)) - np.matmul(np.matmul(K, measureMatrix), covState)
    covState = (np.eye(len(covState)) - K @ measureMatrix) @ covState
    # xState = xState + np.matmul(K, np.reshape(measurement,(3,1)) - updatedMeasurement)
    xState = xState + K @ (np.reshape(measurement, (3, 1)) - updatedMeasurement)
    
    return xState, covState
