# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:28:24 2021

@author: ben-o_000
"""

import numpy as np

'''
Here we define some global variables for use across multiple pysatellite files.


stepLength: Length of each time step, in s
simLength: Number of time-steps
mu: Product of gravitational constant G and mass M, in m^3/s^2
'''

stepLength = 30
simLength = 500
mu = np.float64(3.9860e14)

WGS = {
    "LengthUnit": 'meter',
    "SemimajorAxis": np.float64(6378137),
    "SemiminorAxis": np.float64(6.356752314245179e+06),
    "InverseFlattening": np.float64(2.982572235630000e+02),
    "Eccentricity": np.float64(0.081819190842621),
    "Flattening": np.float64(0.0034),
    "ThirdFlattening": np.float64(0.0017),
    "MeanRadius": np.float64(6.371008771415059e+06),
    "SurfaceArea": np.float64(5.100656217240886e+14),
    "Volume": np.float64(1.083207319801408e+21),
}
