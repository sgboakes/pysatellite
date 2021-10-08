# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:36:04 2021

@author: sgboakes
"""

import numpy as np
import matplotlib.pyplot as plt
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg

if __name__ == "__main__":
    
    
    # ~~~~ Variables
    
    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)
    
    
    sensLat = np.float64(28.300697); sensLon = np.float64(-16.509675); sensAlt = np.float64(2390)
    sensLLA = np.array([[sensLat * pi/180], [sensLon * pi/180], [sensAlt]], dtype='float64')
    # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
    sensECEF = Transformations.LLAtoECEF(sensLLA)
    sensECEF.shape = (3,1)

    simLength = cfg.simLength
    stepLength = cfg.stepLength

    mu = cfg.mu
    
    
    trans_earth = False

    # ~~~~ Satellite Conversion 
    
    # Define sat pos in ECI and convert to AER
    # radArr: radii for each sat metres
    # omegaArr: orbital rate for each sat rad/s
    # thetaArr: inclination angle for each sat rad
    # kArr: normal vector for each sat metres
    
    radArr = np.array([[7e6], [8e6], [6.8e6], [7.5e6]], dtype='float64')
    
    omegaArr = 1 / np.sqrt(radArr**3 / mu)
    
    thetaArr = np.array([[0], [2*pi/3], [3*pi/2],[3*pi/4]], dtype='float64')
    
    kArr = np.array([[0, 0, 0],
                     [0, 0, 1],
                     [1/np.sqrt(2), 1/np.sqrt(2), 0],
                     [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]],
                    dtype='float64')
    
    num_sats = len(radArr)
    
    # Make data structures
    satECI = {chr(i+97):np.zeros((3,simLength)) for i in range(num_sats)}
    satAER = {chr(i+97):np.zeros((3,simLength)) for i in range(num_sats)}
    
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            v = np.array([[radArr[i] * sin(omegaArr[i]*(j+1)*stepLength)],
                          [0],
                          [radArr[i] * cos(omegaArr[i]*(j+1)*stepLength)]],
                          dtype='float64')
            
            satECI[c][:,j] = (v @ cos(thetaArr[i])) + (np.cross(kArr[i,:].T,v.T) * sin(thetaArr[i])) + (kArr[i,:].T * np.dot(kArr[i,:].T,v) * (1-cos(thetaArr[i])))
            
            satAER[c][:,j:j+1] = Transformations.ECItoAER(satECI[c][:,j], stepLength, j+1, sensECEF, sensLLA[0], sensLLA[1])
            print(satAER[c][:,j])
        
    
    
    
    
    
    
    