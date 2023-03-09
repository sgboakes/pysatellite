# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:09:04 2022

@author: sgboakes
"""

import numpy as np
from pysatellite import transformations, functions, filters
import pysatellite.config as cfg
from poliastro.core.propagation import markley

t = cfg.stepLength
sin = np.sin
cos = np.cos
pi = np.float64(np.pi)
sqrt = np.sqrt

mu = cfg.mu


def gen_measurements(sat_aer, num_sats, sat_vis_check, sim_length, step_length, sens, trans_earth):
    """
    Add small deviations for measurements
    Using calculated max measurement deviations for LT:
    Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    sigma = 1/2 * 0.15" for it to be definitely on that pixel
    Add angle devs to Az/Elev, and range devs to Range

    Parameters
    ----------
    sat_aer : array, dict
        Array containing AER data to which measurement noise is added
    num_sats : int
        Number of satellites to generate
    sat_vis_check : array, dict
        satellite visibility array
    sim_length : int
        Number of time-steps
    step_length : int
        Length of each time-step in seconds
    sens : class
        sensor object used to generate measurements
    trans_earth : bool
        Boolean, transparent earth

    Returns
    -------
    sat_eci_mes, sat_aer_mes
    """

    sat_aer_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        if sat_vis_check[c]:
            sat_aer_mes[c][0, :] = sens.AngVar * np.random.randn(sim_length,) + sat_aer[c][0, :]
            sat_aer_mes[c][1, :] = sens.AngVar * np.random.randn(sim_length,) + sat_aer[c][1, :]
            sat_aer_mes[c][2, :] = sens.RngVar * np.random.randn(sim_length,) + sat_aer[c][2, :]

    sat_eci_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        if sat_vis_check[c]:
            for j in range(sim_length):
                sat_eci_mes[c][:, j:j + 1] = transformations.aer_to_eci(sat_aer_mes[c][:, j], step_length, j + 1,
                                                                        sens.ECEF, sens.LLA[0], sens.LLA[1])

    # Making NaN measurements where elevation < 0
    # Think of more efficient way to do this? Vectorisation at top is good, implement something similar here?
    if not trans_earth:
        for i in range(num_sats):
            c = chr(i+97)
            for j in range(sim_length):
                if sat_aer[c][1, j] < 0:
                    sat_aer_mes[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])
                    sat_eci_mes[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

    return sat_eci_mes, sat_aer_mes


def circular_orbits(num_sats, sim_length, step_length, sens, trans_earth=False):
    """
    Generates circular orbits using Kepler's equation

    Parameters
    ----------
    num_sats : int
        Number of satellites to generate
    sim_length : int
        Number of time-steps
    step_length : int
        Length of each time-step in seconds
    sens : class
        sensor object used to generate measurements
    trans_earth : bool
        Boolean, transparent earth

    Returns
    -------
    sat_eci, sat_aer, sat_eci_mes, sat_aer_mes, sat_visible
    """
    # Define sat pos in ECI and convert to AER
    # radArr: radii for each sat metres
    # omegaArr: orbital rate for each sat rad/s
    # thetaArr: inclination angle for each sat rad
    # kArr: normal vector for each sat metres
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

            sat_aer[c][:, j:j + 1] = transformations.eci_to_aer(sat_eci[c][:, j], step_length, j + 1, sens.ECEF,
                                                                sens.LLA[0], sens.LLA[1])

            # if not trans_earth:
            #     if sat_aer[c][1, j] < 0:
            #         sat_aer[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

        if np.isnan(sat_aer[c]).all():
            print('Satellite {s} is not observable'.format(s=i))
            sat_vis_check[c] = False

    sat_eci_mes, sat_aer_mes = gen_measurements(sat_aer, num_sats, sat_vis_check, sim_length, step_length, sens,
                                                trans_earth)

    return sat_eci, sat_aer, sat_eci_mes, sat_aer_mes, sat_vis_check


def coe_orbits(num_sats, sim_length, step_length, sens, trans_earth=False):
    """
    Generates orbits from orbital elements

    Parameters
    ----------
    num_sats : int
        Number of satellites to generate
    sim_length : int
        Number of time-steps
    step_length : int
        Length of each time-step in seconds
    sens : class
        sensor object used to generate measurements
    trans_earth : bool
        Boolean, transparent earth

    Returns
    -------
    sat_eci, sat_aer, sat_eci_mes, sat_aer_mes, sat_visible
    """
    # From poliastro and ssa-gym
    RE_eq = cfg.WGS['SemimajorAxis']
    k = mu

    complete = False
    sat_counter = 0
    reject_counter = 0
    sat_eci = {chr(i + 97): np.zeros((6, sim_length)) for i in range(num_sats)}
    sat_aer = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_eci_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_aer_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_vis_check = {chr(i + 97): True for i in range(num_sats)}
    elements = {chr(i + 97): [] for i in range(num_sats)}

    ang_mes_dev, range_mes_dev = 1e-6, 20

    # TODO: Keep velocity part of ECI?
    # Keep for now, need to adjust either function calls/function usage
    while not complete:
        # Expand for multiple satellites?
        inc = np.deg2rad(180 * np.random.rand())  # (rad) – Inclination
        raan = np.deg2rad(360 * np.random.rand())  # (rad) – Right ascension of the ascending node.
        ecc = 0.25 * np.random.rand()  # (Unit-less) – Eccentricity.
        argp = np.deg2rad(360 * np.random.rand())  # (rad) – Argument of the pericenter/periapsis/perigee.
        nu = np.deg2rad(360 * np.random.rand())  # (rad) – True anomaly.

        low, high = RE_eq + 300 * 1000, RE_eq + 2000 * 1000
        a = ((low - high)*np.random.rand() + high)  # (m) – Semi-major axis.
        b = a * np.sqrt(1 - ecc ** 2)
        if b > low:
            p = a * (1 - ecc ** 2)  # (km) - Semi-latus rectum or parameter
        else:
            # reject_counter += 1
            continue

        pqw = np.array([[cos(nu), sin(nu), 0], [-sin(nu), ecc + cos(nu), 0]]) * \
            np.array([[p / (1 + ecc * cos(nu))], [sqrt(k / p)]])

        r = rotation_matrix(raan, 2)
        r = r @ rotation_matrix(inc, 0)
        rm = r @ rotation_matrix(argp, 2)

        ijk = pqw @ rm.T
        eci = np.zeros((6, sim_length))
        lla = np.zeros((3, sim_length))
        aer = np.zeros((3, sim_length))
        eci[:, 0] = np.reshape(ijk, (6,))

        # orbit propagation
        for j in range(sim_length - 1):
            eci[:, j+1] = fx(eci[:, j], step_length)

        for j in range(sim_length):
            lla[:, j:j+1] = transformations.eci_to_lla(eci[0:3, j], step_length, j+1)
            aer[:, j:j+1] = transformations.eci_to_aer(eci[0:3, j], step_length, j+1, sens.ECEF, sens.LLA[0],
                                                       sens.LLA[1])

        # Check for orbit validity
        # if lla[2, :].all() > 300*1000:
        if np.all(lla[2, :] > 300*1000):
            if max(aer[1, :]) > np.deg2rad(15):
                c = chr(sat_counter + 97)
                # sat_eci[c] = eci[0:3, :]  # DO I WANT TO DO THIS
                sat_eci[c] = eci
                sat_aer[c] = aer
                if np.isnan(sat_aer[c]).all():
                    print('Satellite {s} is not observable'.format(s=c))
                    sat_vis_check[c] = False
                # print('Sat {s} orbit done'.format(s=sat_counter))
                sat_counter += 1
                elements[c] = [a, ecc, inc, raan, argp, nu]

            else:
                reject_counter += 1
                continue
        else:
            reject_counter += 1
            continue

        if sat_counter >= num_sats:
            complete = True
            # Conversion
            sat_eci_mes, sat_aer_mes = gen_measurements(sat_aer, num_sats, sat_vis_check, sim_length, step_length, sens,
                                                        trans_earth)
            # print("Created {s} satellites, rejected {n} satellites after propagating".format(s=num_sats,
            #                                                                                  n=reject_counter))

    return sat_eci, sat_aer, sat_eci_mes, sat_aer_mes, sat_vis_check  # , elements


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
    # import pdb; pdb.set_trace()
    rv = markley(mu, r0, v0, tof)  # (m^3 / s^2), (m), (m/s), (s)
    x_post = np.zeros(6)
    x_post[:3] = rv[0]
    x_post[3:] = rv[1]
    return x_post


def coe_orbits2(num_sats, sim_length, step_length, sens, trans_earth=False):
    """
        Generates orbits from orbital elements

        Parameters
        ----------
        num_sats : int
            Number of satellites to generate
        sim_length : int
            Number of time-steps
        step_length : int
            Length of each time-step in seconds
        sens : class
            sensor object used to generate measurements
        trans_earth : bool
            Boolean, transparent earth

        Returns
        -------
        sat_eci, sat_aer, sat_eci_mes, sat_aer_mes, sat_visible
        """
    # From poliastro and ssa-gym
    RE_eq = cfg.WGS['SemimajorAxis']
    k = mu

    complete = False
    sat_counter = 0
    reject_counter = 0
    sat_eci = {chr(i + 97): np.zeros((6, sim_length)) for i in range(num_sats)}
    sat_aer = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_eci_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_aer_mes = {chr(i + 97): np.zeros((3, sim_length)) for i in range(num_sats)}
    sat_vis_check = {chr(i + 97): True for i in range(num_sats)}
    elements = {chr(i + 97): [] for i in range(num_sats)}

    # TODO: Keep velocity part of ECI?
    # Keep for now, need to adjust either function calls/function usage
    while not complete:
        # Expand for multiple satellites?
        inc = np.deg2rad(180 * np.random.rand())  # (rad) – Inclination
        raan = np.deg2rad(360 * np.random.rand())  # (rad) – Right ascension of the ascending node.
        ecc = 0.25 * np.random.rand()  # (Unit-less) – Eccentricity.
        argp = np.deg2rad(360 * np.random.rand())  # (rad) – Argument of the pericenter/periapsis/perigee.
        nu = np.deg2rad(360 * np.random.rand())  # (rad) – True anomaly.

        low, high = RE_eq + 300 * 1000, RE_eq + 2000 * 1000
        a = ((low - high) * np.random.rand() + high)  # (m) – Semi-major axis.
        b = a * np.sqrt(1 - ecc ** 2)
        if b > low:
            p = a * (1 - ecc ** 2)  # (km) - Semi-latus rectum or parameter
        else:
            # reject_counter += 1
            continue

        pqw = np.array([[cos(nu), sin(nu), 0], [-sin(nu), ecc + cos(nu), 0]]) * \
              np.array([[p / (1 + ecc * cos(nu))], [sqrt(k / p)]])

        r = rotation_matrix(raan, 2)
        r = r @ rotation_matrix(inc, 0)
        rm = r @ rotation_matrix(argp, 2)

        ijk = pqw @ rm.T
        eci = np.zeros((6, sim_length))
        lla = np.zeros((3, sim_length))
        aer = np.zeros((3, sim_length))
        eci[:, 0] = np.reshape(ijk, (6,))

        # orbit propagation
        for j in range(sim_length - 1):
            eci[:, j + 1] = fx(eci[:, j], step_length)

        for j in range(sim_length):
            lla[:, j:j + 1] = transformations.eci_to_lla(eci[0:3, j], step_length, j + 1)
            aer[:, j:j + 1] = transformations.eci_to_aer(eci[0:3, j], step_length, j + 1, sens.ECEF, sens.LLA[0],
                                                         sens.LLA[1])

        # Check for orbit validity
        # if lla[2, :].all() > 300*1000:
        if np.all(lla[2, :] > 300 * 1000):
            if max(aer[1, :]) > np.deg2rad(15):
                c = chr(sat_counter + 97)
                # sat_eci[c] = eci[0:3, :]  # DO I WANT TO DO THIS
                sat_eci[c] = eci
                sat_aer[c] = aer
                if np.isnan(sat_aer[c]).all():
                    print('Satellite {s} is not observable'.format(s=c))
                    sat_vis_check[c] = False
                # print('Sat {s} orbit done'.format(s=sat_counter))
                sat_counter += 1
                elements[c] = [a, ecc, inc, raan, argp, nu]

            else:
                reject_counter += 1
                continue
        else:
            reject_counter += 1
            continue

        if sat_counter >= num_sats:
            complete = True
            # Conversion
            sat_eci_mes, sat_aer_mes = gen_measurements(sat_aer, num_sats, sat_vis_check, sim_length, step_length, sens,
                                                        trans_earth)
            # print("Created {s} satellites, rejected {n} satellites after propagating".format(s=num_sats,
            #                                                                                  n=reject_counter))

    return sat_eci, sat_aer, sat_eci_mes, sat_aer_mes, sat_vis_check
