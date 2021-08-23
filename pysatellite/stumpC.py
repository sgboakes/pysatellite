# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:35:39 2021

@author: ben-o_000
"""

import numpy as np

def stumpC(z):
    if z > 0:
        c = (1-np.cos(np.sqrt(z)))/z
    elif z < 0:
        c = (np.cosh(np.sqrt(-z)) - 1)/(-z)
    else:
        c = 1/2
        
    return c