# -*- coding: utf-8 -*-
"""
Created on Thu Jul 8 17:39:18 2021

@author: ben-o_000
"""

import numpy as np
from pysatellite import Transformations, Functions
import pysatellite.config as cfg


def jacobian_finder(func_name, func_variable, func_params, delta):

    if func_name in dir(Transformations):
        func = getattr(Transformations, func_name)
    elif func_name in dir(Functions):
        func = getattr(Functions, func_name)
    else:
        raise Exception('Function not found in jf')
    
    # if hasattr(pysatellite.Transformations, func_name):
    #     func = getattr(pysatellite.Transformations, func_name)
    # elif hasattr(pysatellite.Functions, func_name):
    #     func = getattr(pysatellite.Functions, func_name)

    num_elements = len(func_variable)
    
    # deriv = np.zeros((1, len(func_variable)))
    
    jacobian = np.zeros((len(func_variable), len(func_variable)))
    for i in range(num_elements):
        # deriv = []
        delta_mat = np.zeros((len(func_variable), 1))
        delta_mat[i] = delta
        if not func_params:
            deriv = np.reshape(((func(func_variable+delta_mat) - func(func_variable)) / delta), (num_elements, 1))
        else:
            deriv = np.reshape(((func(func_variable+delta_mat, *list(func_params.values())[:]) - func(func_variable,
                                 *list(func_params.values())[:])) / delta), (num_elements, 1))

    # for i in range(num_elements):
        jacobian[:, i:i+1] = deriv
            
    return jacobian


def kepler(x_state):
    
    t = cfg.stepLength
    mu = cfg.mu
    
    r0 = np.linalg.norm(x_state[0:3])
    v0 = np.linalg.norm(x_state[3:6])
    
    pos0 = x_state[0:3]
    vel0 = x_state[3:6]
    
    # Initial radial velocity
    vr0 = np.dot(np.reshape(pos0, 3), np.reshape(vel0, 3)) / r0
    
    # Reciprocal of the semi-major axis (from the energy equation)
    alpha = 2.0 / r0 - v0**2/mu

    error = 1e-8
    n_max = 1000
    
    x = np.sqrt(mu) * np.abs(alpha) * t
    
    n = 0
    ratio = 1
    while np.abs(ratio) > error and n <= n_max:
        n += 1
        c = stump_c(alpha * x**2)
        s = stump_s(alpha * x**2)
        f = r0*vr0/np.sqrt(mu)*x**2*c + (1 - alpha*r0)*x**3*s + r0*x - np.sqrt(mu)*t
        dfdx = r0*vr0/np.sqrt(mu)*x*(1 - alpha*x**2*s) + (1 - alpha*r0)*x**2*c + r0
        ratio = f/dfdx
        x -= ratio
        
    if n > n_max:
        print('\n No. Iterations of Kepler''s equation = %g', n)
        print('\n F/dFdx = %g', f/dfdx)
        
    z = alpha*x**2
    
    f = 1 - x**2/r0*stump_c(z)
    
    g = t - 1/np.sqrt(mu)*x**3*stump_s(z)
    
    r = f*pos0 + g*vel0
    
    r_norm = np.linalg.norm(r)
    
    f_dot = np.sqrt(mu)/r_norm/r0*(z*stump_s(z) - 1)*x
    
    gdot = 1 - x**2/r_norm*stump_c(z)
    
    v = f_dot*pos0 + gdot*vel0
    
    x_state = np.concatenate((r, v))
    return x_state


def stump_c(z):
    if z > 0:
        c = (1.0-np.cos(np.sqrt(z)))/z
    elif z < 0:
        c = (np.cosh(np.sqrt(-z)) - 1)/(-z)
    else:
        c = 1/2
        
    return c


def stump_s(z):
    
    if z > 0:
        s = (np.sqrt(z) - np.sin(np.sqrt(z)))/(np.sqrt(z))**3
    elif z < 0:
        s = (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/(np.sqrt(-z))**3
    else:
        s = 1/6
        
    return s
