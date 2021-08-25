# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:38:23 2021

@author: ben-o_000
"""

import numpy as np

def stumpS(z):
    
    if z > 0:
        s = (np.sqrt(z) - np.sin(np.sqrt(z)))/(np.sqrt(z))**3
    elif z < 0:
        s = (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/(np.sqrt(-z))**3
    else:
        s = 1/6
        
    return s