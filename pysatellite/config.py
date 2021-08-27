# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:28:24 2021

@author: ben-o_000
"""

'''
Here we define some global variables for use across multpile pysatellite files.


stepLength: Length of each time step, in s
simLength: Number of timesteps
mu: Product of gravitational constant G and mass M, in m^3/s^2
'''

stepLength = 60
simLength = 2000
mu = 3.9860e14

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