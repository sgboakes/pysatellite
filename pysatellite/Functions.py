# -*- coding: utf-8 -*-
"""
Created on Thu Jul 8 17:39:18 2021

@author: ben-o_000
"""

import numpy as np
import pysatellite
import pysatellite.config as cfg



def jacobian_finder(func_name, func_variable, func_params, delta):
    #func = getattr(pysatellite.Transformations, func_name)
    
 
    if func_name in dir(pysatellite.Transformations):
        func = getattr(pysatellite.Transformations, func_name)
    elif func_name in dir(pysatellite.Functions):
        func = getattr(pysatellite.Functions, func_name)
    else:
        raise Exception('Function not found in jf')
    
    # if hasattr(pysatellite.Transformations, func_name):
    #     func = getattr(pysatellite.Transformations, func_name)
    # elif hasattr(pysatellite.Functions, func_name):
    #     func = getattr(pysatellite.Functions, func_name)
        
    
    num_elements = len(func_variable)
    
    #deriv = np.zeros((1, len(func_variable)))
    
    jacobian = np.zeros((len(func_variable),len(func_variable)))
    for i in range(num_elements):
        deriv = []
        delta_mat = np.zeros((len(func_variable),1))
        delta_mat[i] = delta
        if func_params == []:
            deriv = np.reshape(((func(func_variable+delta_mat) - func(func_variable)) / delta), (num_elements))
        else:
            deriv = np.reshape(((func(func_variable+delta_mat, *list(func_params.values())[:]) - func(func_variable, *list(func_params.values())[:])) / delta),(num_elements))
        
        
    # for i in range(num_elements):
            jacobian[:,i] = deriv
            
    return jacobian


def kepler(xState):
    
    t = cfg.stepLength
    mu = cfg.mu
    
    r0 = np.linalg.norm(xState[0:3])
    v0 = np.linalg.norm(xState[3:6])
    
    pos0 = xState[0:3]
    vel0 = xState[3:6]
    
    # Initial radial velocity
    vr0 = np.dot(np.reshape(pos0, 3), np.reshape(vel0, 3)) / r0
    
    # Reciprocal of the semimajor axis (from the energy equation)
    alpha = 2.0 / r0 - v0**2/mu
    
    
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
    
    gdot = 1 - x**2/r*stumpC(z)
    
    V = fdot*pos0 + gdot*vel0
    
    xState = np.concatenate((R, V))
    return xState


def stumpC(z):
    if z > 0:
        c = (1.0-np.cos(np.sqrt(z)))/z
    elif z < 0:
        c = (np.cosh(np.sqrt(-z)) - 1)/(-z)
    else:
        c = 1/2
        
    return c


def stumpS(z):
    
    if z > 0:
        s = (np.sqrt(z) - np.sin(np.sqrt(z)))/(np.sqrt(z))**3
    elif z < 0:
        s = (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/(np.sqrt(-z))**3
    else:
        s = 1/6
        
    return s