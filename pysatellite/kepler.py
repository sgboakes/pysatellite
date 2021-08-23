# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:39:24 2021

@author: ben-o_000
"""

import numpy as np
from pysatellite import stumpC, stumpS

def kepler(xState):
    
    global stepLength
    
    t = stepLength
    
    global mu
    
    r0 = np.linalg.norm(xState[0:2])
    v0 = np.linalg.norm(xState[3:5])
    
    pos0 = xState[0:2]
    vel0 = xState[3:5]
    
    # Initial radial velocity
    vr0 = np.dot(pos0, vel0) / r0
    
    # Reciprocal of the semimajor axis (from the energy equation)
    alpha = 2 / r0 - v0**2/mu
    
    
    error = 1e-8
    nMax = 1000
    
    x = np.sqrt(mu) * np.abs(alpha) * t
    
    n = 0
    ratio = 1
    while np.abs(ratio) > error and n <= nMax:
        n += 1
        C = stumpC(alpha * x**2)
        S = stumpS(alpha * x**2)
        F = r0*vr0/np.sqrt(mu)*x**2*C + (1 - alpha*r0)*x**3*S + r0*x - np.sqrt(mu)*t
        dFdx = r0*vr0/np.sqrt(mu)*x*(1 - alpha*x**2*S) + (1 - alpha*r0)*x**2*C + r0
        ratio = F/dFdx
        x -= ratio
        
    if n > nMax:
        print('\n No. Iterations of Kepler''s equation = %g',n)
        print('\n F/dFdx = %g',F/dFdx)
        
    z = alpha*x**2
    
    f = 1 - x**2/r0*stumpC(z)
    
    g = t - 1/np.sqrt(mu)*x**3*stumpS(z)
    
    R = f*pos0 + g*vel0
    
    r = np.linalg.norm(R)
    
    fdot = np.sqrt(mu)/r/r0*(z*stumpS(z) - 1)*x
    
    gdot = 1 - x^2/r*stumpC(z)
    
    V = fdot*pos0 + gdot*vel0
    
    xState = [[R],[V]]
    return xState
        