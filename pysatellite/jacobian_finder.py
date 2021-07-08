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
    
    