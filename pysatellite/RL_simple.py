# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:42:27 2021

@author: sgboakes
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg

if __name__ == "__main__":
    
    
    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)
    
    simLength = cfg.simLength
    stepLength = cfg.stepLength

    satRadius = np.float64(7e6)
    mu = cfg.mu
    TOrbit = 2*pi * np.sqrt(satRadius**3/mu)
    omegaSat = 2*pi/TOrbit
    
    # Generate equidistant points on circle
    num_points = 10
    t = np.linspace(0, 2*pi, num_points, endpoint=False)
    x = satRadius * sin(t)
    y = satRadius * cos(t)
    
    # Get sat pos at each time step
    satPos = np.zeros((2,simLength))
    for count in range(simLength):
        satPos[:,count:count+1] = np.array([[x[count % num_points]],
                                            [y[count % num_points]]],
                                           dtype='float64'
                                           )
    
    
    