# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:28:24 2021

@author: ben-o_000
"""

import numpy as np
from collections import namedtuple

'''
Here we define some global variables for use across multiple pysatellite files.


stepLength: Length of each time step, in s
simLength: Number of time-steps
mu: Gravitational parameter: product of gravitational constant G and mass M, in m^3/s^2
'''

stepLength = 30
simLength = 500
mu = np.float64(3.9860e14)

wgs84 = namedtuple('WGS84', ['lengthunit', 'semimajoraxis', 'semiminoraxis', 'inverseflattening',
                           'eccentricity', 'flattening', 'thirdflattening', 'meanradius', 'surfacearea', 'volume'])

WGS84 = wgs84(lengthunit='meter',
              semimajoraxis=6378137,
              semiminoraxis=6.356752314245179e+06,
              inverseflattening=2.982572235630000e+02,
              eccentricity=0.081819190842621,
              flattening=0.0034,
              thirdflattening=0.0017,
              meanradius=6.371008771415059e+06,
              surfacearea=5.100656217240886e+14,
              volume=1.083207319801408e+21)

wgs72 = namedtuple('WGS72', ['mu', 'radiusearthkm', 'xke', 'tumin', 'j2', 'j3', 'j4', 'j3oj2',])


wgs72_mu = 398600.8  # in km3 / s2
wgs72_radiusearthkm = 6378.135  # km
wgs72_xke = 60.0 / np.sqrt(wgs72_radiusearthkm**3 / wgs72_mu)
wgs72_tumin = 1.0 / wgs72_xke
wgs72_j2 = 0.001082616
wgs72_j3 = -0.00000253881
wgs72_j4 = -0.00000165597
wgs72_j3oj2 = wgs72_j3 / wgs72_j2

WGS72 = wgs72(mu=wgs72_mu,
              radiusearthkm=wgs72_radiusearthkm,
              xke=wgs72_xke,
              tumin=wgs72_tumin,
              j2=wgs72_j2,
              j3=wgs72_j3,
              j4=wgs72_j4,
              j3oj2=wgs72_j3oj2)