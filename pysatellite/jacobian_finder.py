# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:39:18 2021

@author: ben-o_000
"""

import numpy as np
from pysatellite import Transformations

def jacobian_finder(func_name, func_variable, func_params, delta):
    func = getattr(Transformations, func_name)
    
    num_elements = len(func_variable)
    
    deriv = np.zeros((1, len(func_variable)))
    
    for i in range(num_elements):
        delta_mat = np.zeros((len(func_variable),1))
        delta_mat[i] = delta
        deriv[i,:] = (func(func_variable+delta_mat, *list(func_params.values())[:]) - func(func_variable, *list(func_params.values())[:])) / delta
        
        jacobian = np.zeros((len(func_variable),len(func_variable)))
        for i in range(num_elements):
            jacobian[:,i] = deriv[i,:]
            
    return jacobian