'''
Benedict Oakes
Created 10/06/2021
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# import Transformations
from pysatellite import Transformations

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

    simLength = 2000
    stepLength = 60

    TOrbitHours = 2.5
    omegaSat = 2*pi/(TOrbitHours * 3600)
    satRadius = 7e6


    # ~~~~ Satellite Conversion 
    
    satECI = np.zeros((3,simLength))
    for count in range(simLength):
        satECI[:,count] = [satRadius*sin(omegaSat*count*stepLength), 0, satRadius*cos(omegaSat*count*stepLength)]
        
    satAER = np.zeros((3,simLength))
    for count in range(simLength):
        satAER[:,count] = Transformations.ECItoAER(satECI[:,count], stepLength, count, sensECEF, sensLLA[0], sensLLA[1])
        
        
    angMeasDev = 1e-6
    rangeMeasDev = 20
    satAERMes = np.zeros((3,simLength))
    satAERMes[0,:] = satAER[0,:] + (angMeasDev*np.random.randn(1,simLength))
    satAERMes[1,:] = satAER[1,:] + (angMeasDev*np.random.randn(1,simLength))
    satAERMes[2,:] = satAER[2,:] + (rangeMeasDev*np.random.randn(1,simLength))
    