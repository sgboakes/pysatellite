# -*- coding: utf-8 -*-
"""
Created on Thu Jul 8 17:39:18 2021

@author: ben-o_000
"""
import types

import numpy as np
from pysatellite import transformations
import pysatellite.config as cfg
import datetime

from tlefit_coe_fd import test_tle_fit_normalized

# t = cfg.stepLength
# TODO: Docstrings in this file


def h_x(x_state):
    hx = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0]],
                  dtype=np.float64)

    hxr = hx @ x_state
    return hxr


def jacobian_finder(func, func_variable, func_params, delta=1e-6):
    if not callable(func):
        raise Exception('Input func must be function handle')

    num_elements = len(func_variable)

    jacobian = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        # deriv = []
        delta_mat = np.zeros((num_elements, 1))
        delta_mat[i] = delta
        if not func_params:
            deriv = np.reshape(((func(func_variable + delta_mat) - func(func_variable)) / delta), (num_elements, 1))
        else:
            deriv = np.reshape(((func(func_variable + delta_mat, *list(func_params.values())[:]) -
                                 func(func_variable, *list(func_params.values())[:])) / delta), (num_elements, 1))

        # for i in range(num_elements):
        jacobian[:, i:i + 1] = deriv

    return jacobian


def jacobian_tle(satellite, timestamp):

    num_elements = 3
    jacobian = np.zeros((num_elements, num_elements))

    # delta_t = datetime.timedelta(seconds=10)
    # dt_prev = timestamp - delta_t
    # dt_next = timestamp + delta_t
    #
    # rv_prev = satellite.at(dt_prev)
    # rv_next = satellite.at(dt_next)

    rv = satellite.at(timestamp)
    r = rv.position.m
    r_x = r + [1e-6, 0., 0.]
    r_y = r + [0., 1e-6, 0.]
    r_z = r + [0., 0., 1e-6]

    # Adapt tle-tailor code to work here

    for i in range(num_elements):
        # deriv = []
        delta_mat = np.zeros((num_elements, 1))
        delta_mat[i] = delta_t
        if not func_params:
            deriv = np.reshape(((func(func_variable + delta_mat) - func(func_variable)) / delta), (num_elements, 1))
        else:
            deriv = np.reshape(((func(func_variable + delta_mat, *list(func_params.values())[:]) -
                                 func(func_variable, *list(func_params.values())[:])) / delta), (num_elements, 1))

        # for i in range(num_elements):
        jacobian[:, i:i + 1] = deriv

    return jacobian


def state_to_tle(satellite,
    central_diff=True,
    fit_span=4,
    max_iter=35,
    lamda=1e-3,
    bstar=1e-6,
    rms_epsilon=0.002,
    percent_chg=0.001,
    delta_amt_chg=1e-7,
    debug=False,
    hermitian=True,
    dx_limit=False,
    coe_limit=True,
    lm_reg=True,
):

    """Calls on function from tle-tailor"""

    return test_tle_fit_normalized(satellite, central_diff, fit_span, max_iter, lamda, bstar, rms_epsilon, percent_chg,
                                   delta_amt_chg, debug, hermitian, dx_limit, coe_limit, lm_reg)

def stump_c(z):
    if z > 0:
        c = (1.0 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        c = (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        c = 1 / 2

    return c


def stump_s(z):
    if z > 0:
        s = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
    elif z < 0:
        s = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3
    else:
        s = 1 / 6

    return s
