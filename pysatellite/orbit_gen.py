# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:09:04 2022

@author: sgboakes
"""

import numpy as np
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg
from poliastro.core.propagation import markley

t = cfg.stepLength
sin = np.sin
cos = np.cos
pi = np.float64(np.pi)
sqrt = np.sqrt

mu = cfg.mu


def gen_measurements(sat_aer, num_sats, sat_vis_check, sim_length, step_length, sens):
    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range
    ang_mes_dev, range_mes_dev = 1e-6, 20

    sat_aer_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        if sat_vis_check[c]:
            sat_aer_mes[c][0, :] = sat_aer[c][0, :] + (ang_mes_dev * np.random.randn(1, sim_length))
            sat_aer_mes[c][1, :] = sat_aer[c][1, :] + (ang_mes_dev * np.random.randn(1, sim_length))
            sat_aer_mes[c][2, :] = sat_aer[c][2, :] + (range_mes_dev * np.random.randn(1, sim_length))

    sat_eci_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        if sat_vis_check[c]:
            for j in range(sim_length):
                sat_eci_mes[c][:, j:j + 1] = Transformations.aer_to_eci(sat_aer_mes[c][:, j], step_length, j + 1,
                                                                        sens.ECEF, sens.LLA[0], sens.LLA[1])

    return sat_eci_mes, sat_aer_mes


def circular_orbits(num_sats, sim_length, step_length, sens, trans_earth):
    rad_arr = 7e6 * np.ones((num_sats, 1), dtype='float64')
    omega_arr = 1 / np.sqrt(rad_arr ** 3 / mu)
    theta_arr = np.array((2 * pi * np.random.rand(num_sats, 1)), dtype='float64')
    k_arr = np.ones((num_sats, 3), dtype='float64')
    k_arr[:, :] = 1 / np.sqrt(3)

    # Make data structures
    sat_eci = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_aer = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_vis_check = {chr(i + 97): True for i in range(num_sats)}

    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(sim_length):
            v = np.array([[rad_arr[i] * sin(omega_arr[i] * (j + 1) * step_length)],
                          [0],
                          [rad_arr[i] * cos(omega_arr[i] * (j + 1) * step_length)]], dtype='float64')

            sat_eci[c][:, j] = (v @ cos(theta_arr[i])) + (np.cross(k_arr[i, :].T, v.T) * sin(theta_arr[i])) + (
                    k_arr[i, :].T * np.dot(k_arr[i, :].T, v) * (1 - cos(theta_arr[i])))

            sat_aer[c][:, j:j + 1] = Transformations.eci_to_aer(sat_eci[c][:, j], step_length, j + 1, sens.ECEF,
                                                                sens.LLA[0], sens.LLA[1])

            if not trans_earth:
                if sat_aer[c][1, j] < 0:
                    sat_aer[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

        if np.isnan(sat_aer[c]).all():
            print('Satellite {s} is not observable'.format(s=i))
            sat_vis_check[c] = False

    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range

    sat_eci_mes, sat_aer_mes = gen_measurements(sat_aer, num_sats, sat_vis_check, sim_length, step_length, sens)

    # ang_mes_dev, range_mes_dev = 1e-6, 20
    #
    # sat_aer_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    # for i in range(num_sats):
    #     c = chr(i + 97)
    #     if sat_vis_check[c]:
    #         sat_aer_mes[c][0, :] = sat_aer[c][0, :] + (ang_mes_dev * np.random.randn(1, sim_length))
    #         sat_aer_mes[c][1, :] = sat_aer[c][1, :] + (ang_mes_dev * np.random.randn(1, sim_length))
    #         sat_aer_mes[c][2, :] = sat_aer[c][2, :] + (range_mes_dev * np.random.randn(1, sim_length))
    #
    # sat_eci_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    # for i in range(num_sats):
    #     c = chr(i + 97)
    #     if sat_vis_check[c]:
    #         for j in range(sim_length):
    #             sat_eci_mes[c][:, j:j + 1] = Transformations.aer_to_eci(sat_aer_mes[c][:, j], step_length, j + 1,
    #                                                                     sens.ECEF, sens.LLA[0], sens.LLA[1])

    return sat_eci, sat_eci_mes, sat_aer, sat_aer_mes


def coe_orbits(num_sats, sim_length, step_length, trans_earth, sens):
    # From poliastro and ssa-gym

    random_state = np.random.RandomState()
    RE_eq = cfg.WGS['SemimajorAxis']
    k = mu

    # Expand for multiple satellites
    inc = np.radians(random_state.uniform(0, 180, num_sats))  # (rad) – Inclination
    raan = np.radians(random_state.uniform(0, 360, num_sats))  # (rad) – Right ascension of the ascending node.
    argp = np.radians(random_state.uniform(0, 360, num_sats))  # (rad) – Argument of the pericenter.
    nu = np.radians(random_state.uniform(0, 360, num_sats))  # (rad) – True anomaly.
    a = random_state.uniform(RE_eq + 300 * 1000, RE_eq + 2000 * 1000, num_sats)  # (m) – Semi-major axis.
    ecc = random_state.uniform(0, .25, num_sats)  # (Unit-less) – Eccentricity.
    b = a * np.sqrt(1 - ecc ** 2)
    p = a * (1 - ecc ** 2)  # (km) - Semi-latus rectum or parameter

    sat_eci = {chr(i + 97): np.zeros((6, sim_length)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        pqw = np.array([[cos(nu[i]), sin(nu[i]), 0], [-sin(nu[i]), ecc[i] + cos(nu[i]), 0]]) * \
              np.array([[p[i] / (1 + ecc[i] * cos(nu[i]))], [sqrt(k / p[i])]])

        r = rotation_matrix(raan[i], 2)
        r = r @ rotation_matrix(inc[i], 0)
        rm = r @ rotation_matrix(argp[i], 2)

        ijk = pqw @ rm.T
        sat_eci[c][:, 0] = np.reshape(ijk, (6,))

    # orbit propagation
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(sim_length - 1):
            sat_eci[c][:, j + 1] = fx(sat_eci[c][:, j], step_length)

        sat_eci[c] = sat_eci[c][:3, :]

    # Conversion
    sat_aer = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_vis_check = {chr(i + 97): True for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(sim_length):
            sat_aer[c][:, j:j + 1] = Transformations.eci_to_aer(sat_eci[c][:, j], step_length, j + 1, sens.ECEF,
                                                                sens.LLA[0], sens.LLA[1])

            if not trans_earth:
                if sat_aer[c][1, j] < 0:
                    sat_aer[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

        if np.isnan(sat_aer[c]).all():
            print('Satellite {s} is not observable'.format(s=i))
            sat_vis_check[c] = False

    sat_eci_mes, sat_aer_mes = gen_measurements(sat_aer, num_sats, sat_vis_check, sim_length, step_length, sens)
    # ang_mes_dev, range_mes_dev = 1e-6, 20
    #
    # sat_aer_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    # for i in range(num_sats):
    #     c = chr(i + 97)
    #     if sat_vis_check[c]:
    #         sat_aer_mes[c][0, :] = sat_aer[c][0, :] + (ang_mes_dev * np.random.randn(1, sim_length))
    #         sat_aer_mes[c][1, :] = sat_aer[c][1, :] + (ang_mes_dev * np.random.randn(1, sim_length))
    #         sat_aer_mes[c][2, :] = sat_aer[c][2, :] + (range_mes_dev * np.random.randn(1, sim_length))
    #
    # sat_eci_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    # for i in range(num_sats):
    #     c = chr(i + 97)
    #     if sat_vis_check[c]:
    #         for j in range(sim_length):
    #             sat_eci_mes[c][:, j:j + 1] = Transformations.aer_to_eci(sat_aer_mes[c][:, j], step_length, j + 1,
    #                                                                     sens.ECEF, sens.LLA[0], sens.LLA[1])

    return sat_eci, sat_aer, sat_eci_mes, sat_aer_mes, sat_vis_check


def rotation_matrix(angle, axis):
    assert axis in (0, 1, 2)
    angle = np.asarray(angle)
    c = cos(angle)
    s = sin(angle)

    a1 = (axis + 1) % 3
    a2 = (axis + 2) % 3
    R = np.zeros(angle.shape + (3, 3))
    R[..., axis, axis] = 1.0
    R[..., a1, a1] = c
    R[..., a1, a2] = -s
    R[..., a2, a1] = s
    R[..., a2, a2] = c
    return R


def fx(x, dt):
    r0 = x[:3]
    v0 = x[3:]
    tof = dt
    rv = markley(mu, r0, v0, tof)  # (m^3 / s^2), (m), (m/s), (s)
    x_post = np.zeros(6)
    x_post[:3] = rv[0]
    x_post[3:] = rv[1]
    return x_post
