'''
Benedict Oakes
Created 10/06/2021
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
# import Transformations
from pysatellite import Transformations
from pysatellite.jacobian_finder import jacobian_finder as jf

if __name__ == "__main__":

    # ~~~~ Variables
    
    WGS = {
        "LengthUnit" : 'meter',
        "SemimajorAxis" : 6378137,
        "SemiminorAxis" : 6.356752314245179e+06,
        "InverseFlattening" : 2.982572235630000e+02,
        "Eccentricity" : 0.081819190842621,
        "Flattening" : 0.0034,
        "ThirdFlattening" : 0.0017,
        "MeanRadius" : 6.371008771415059e+06,
        "SurfaceArea" : 5.100656217240886e+14,
        "Volume" : 1.083207319801408e+21,
    }
    
    sin = np.sin
    cos = np.cos
    pi = np.pi
    
    
    sensLat = 28.300697; sensLon = -16.509675; sensAlt = 2390
    sensLLA = [sensLat * pi/180], [sensLon * pi/180], [sensAlt]
    sensECEF = Transformations.LLAtoECEF(sensLLA, WGS)
    sensECEF.shape = (3,1)

    global stepLength
    simLength = 2000
    stepLength = 60

    satRadius = 7e6
    global mu
    mu = 3.9860e14
    TOrbit = 2*pi * np.sqrt(satRadius**3/mu)
    omegaSat = 2*pi/TOrbit


    # ~~~~ Satellite Conversion 
    
    satECI = np.zeros((3,simLength))
    for count in range(simLength):
        satECI[:,count] = [satRadius*sin(omegaSat*count*stepLength), 0, satRadius*cos(omegaSat*count*stepLength)]
        
    satAER = np.zeros((3,simLength))
    for count in range(simLength):
        satAER[:,[count]] = Transformations.ECItoAER(satECI[:,count], stepLength, count, sensECEF, sensLLA[0], sensLLA[1])
        
        
    angMeasDev = 1e-6
    rangeMeasDev = 20
    satAERMes = np.zeros((3,simLength))
    satAERMes[0,:] = satAER[0,:] + (angMeasDev*np.random.randn(1,simLength))
    satAERMes[1,:] = satAER[1,:] + (angMeasDev*np.random.randn(1,simLength))
    satAERMes[2,:] = satAER[2,:] + (rangeMeasDev*np.random.randn(1,simLength))
    
    
    # ~~~~ Convert back to ECI
    
    satECIMes = np.zeros((3,simLength))
    for count in range(simLength):
        satECIMes[:,[count]] = Transformations.AERtoECI(satAERMes[:,1], stepLength, count, sensECEF, sensLLA[0], sensLLA[1])
    
    
    # ~~~~ KF Matrices
    
    # Initialise state vector
    # (x, y, z, v_x, v_y, v_z)
    xState = np.array([[0],
                       [0],
                       [satRadius],
                       [0],
                       [0],
                       [0]])
    
    G = 6.67e-11
    m_e = 5.972e24
    m_s = 20
    
    v = np.sqrt((G*m_e) / np.linalg.norm(xState[0:3])) * np.array([[1],[0],[0]])
    xState[3:6] = v
    
    # Process noise
    stdAng = 2e0
    coefA = 0.25 * stepLength**4 * stdAng**2
    coefB = stepLength**2 * stdAng**2
    coefC = 0.5 * stepLength**3 * stdAng**2
    
    stdRange = 2e1
    coefA2 = 0.25 * stepLength**4 * stdRange**2
    coefB2 = stepLength**2 * stdRange**2
    coefC2 = 0.5 * stepLength**3 * stdRange**2
    
    procNoise = np.array([[coefA, 0, 0, coefC, 0, 0],
                          [0, coefA2, 0, 0, coefC2, 0],
                          [0, 0, coefA, 0, 0, coefC],
                          [coefC, 0, 0, coefB, 0, 0],
                          [0, coefC2, 0, 0, coefB2, 0],
                          [0, 0, coefC, 0, 0, coefB]])
    
    covState = 1e10 * np.identity(6)
    
    covAER = np.array([[(angMeasDev * 180/2)**2, 0, 0],
                       [0, (angMeasDev * 180/2)**2, 0],
                       [0, 0, rangeMeasDev]])
    
    measureMatrix = np.append(np.identity(3), np.zeros((3,3)), axis=1)
    
    totalStates = np.zeros((6,simLength))
    err_X_ECI = np.zeros((1,simLength))
    err_Y_ECI = np.zeros((1,simLength))
    err_Z_ECI = np.zeros((1,simLength))
    
    # ~~~~ Using EKF
    
    delta = 1e-6
    for count in range(simLength):
        #Func params
        func_params = {
            "stepLength": stepLength,
            "count": count,
            "sensECEF": sensECEF,
            "sensLLA[0]": sensLLA[0],
            "sensLLA[1]": sensLLA[1]
            }
        jacobian = jf("AERtoECI", np.reshape(satAERMes[:,count], (3, 1)), func_params, delta)
        
        covECI = np.matmul(np.matmul(jacobian, covAER), jacobian.T)
        
        stateTransMatrix = jf("kepler", xState, [], delta)
        