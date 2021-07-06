'''
Benedict Oakes
Created 10/06/2021
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# import Transformations
import Transformations

if __name__ == "__main__":

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

    