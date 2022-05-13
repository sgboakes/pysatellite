# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:36:04 2021

@author: sgboakes
"""
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysatellite import Transformations, Functions as Funcs
import pysatellite.config as cfg
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time
from sgp4.api import Satrec, SGP4_ERRORS, jday

if __name__ == "__main__":

    plt.close('all')
    # np.random.seed(2)
    # ~~~~ Variables

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)

    sensLat = np.float64(28.300697)
    sensLon = np.float64(-16.509675)
    sensAlt = np.float64(2390)
    sensLLA = np.array([[sensLat * pi / 180], [sensLon * pi / 180], [sensAlt]], dtype='float64')
    # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
    sensECEF = Transformations.lla_to_ecef(sensLLA)
    sensECEF.shape = (3, 1)

    simLength = cfg.simLength
    stepLength = cfg.stepLength

    mu = cfg.mu
    trans_earth = False

    # ~~~~ Satellite Conversion

    # Define sat pos in ECI and convert to AER
    # radArr: radii for each sat metres
    # omegaArr: orbital rate for each sat rad/s
    # thetaArr: inclination angle for each sat rad
    # # kArr: normal vector for each sat metres
    # num_sats = 40
    # radArr = 7e6 * np.ones((num_sats, 1), dtype='float64')
    # omegaArr = 1 / np.sqrt(radArr ** 3 / mu)
    # thetaArr = np.array((2 * pi * np.random.rand(num_sats, 1)), dtype='float64')
    # kArr = np.ones((num_sats, 3), dtype='float64')
    # kArr[:, :] = 1 / np.sqrt(3)
    #
    # # Make data structures
    # satECI = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    # satAER = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    # satVisCheck = {chr(i + 97): True for i in range(num_sats)}
    #
    # for i in range(num_sats):
    #     c = chr(i + 97)
    #     for j in range(simLength):
    #         v = np.array([[radArr[i] * sin(omegaArr[i] * (j + 1) * stepLength)],
    #                       [0],
    #                       [radArr[i] * cos(omegaArr[i] * (j + 1) * stepLength)]], dtype='float64')
    #
    #         satECI[c][:, j] = (v @ cos(thetaArr[i])) + (np.cross(kArr[i, :].T, v.T) * sin(thetaArr[i])) + (
    #                            kArr[i, :].T * np.dot(kArr[i, :].T, v) * (1 - cos(thetaArr[i])))
    #
    #         satAER[c][:, j:j + 1] = Transformations.eci_to_aer(satECI[c][:, j], stepLength, j + 1, sensECEF,
    #                                                            sensLLA[0], sensLLA[1])
    #
    #         if not trans_earth:
    #             if satAER[c][1, j] < 0:
    #                 satAER[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])
    #
    #     if np.isnan(satAER[c]).all():
    #         print('Satellite {s} is not observable'.format(s=i))
    #         satVisCheck[c] = False
    #
    # # Add small deviations for measurements
    # # Using calculated max measurement deviations for LT:
    # # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # # Add angle devs to Az/Elev, and range devs to Range
    #
    # angMeasDev, rangeMeasDev = 1e-6, 20
    #
    # satAERMes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    # for i in range(num_sats):
    #     c = chr(i + 97)
    #     if satVisCheck[c]:
    #         satAERMes[c][0, :] = satAER[c][0, :] + (angMeasDev * np.random.randn(1, simLength))
    #         satAERMes[c][1, :] = satAER[c][1, :] + (angMeasDev * np.random.randn(1, simLength))
    #         satAERMes[c][2, :] = satAER[c][2, :] + (rangeMeasDev * np.random.randn(1, simLength))
    #
    # satECIMes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    # for i in range(num_sats):
    #     c = chr(i + 97)
    #     if satVisCheck[c]:
    #         for j in range(simLength):
    #             satECIMes[c][:, j:j + 1] = Transformations.aer_to_eci(satAERMes[c][:, j], stepLength, j+1, sensECEF,
    #                                                                   sensLLA[0], sensLLA[1])

    # ~~~~ Temp ECI measurements from MATLAB

    # satECIMes['a'] = pd.read_csv('ECI_mes.txt', delimiter=' ').to_numpy(dtype='float64')
    # #satECIMes.to_numpy(dtype='float64')
    # satECIMes['a'] = satECIMes['a'].T
    # np.reshape(satECIMes['a'], (3, simLength))

    # ~~~~~~ USING TLE DATA FROM CELESTRAK
    # with open('TLE_stations.txt') as f:
    #     lines = f.readlines()

    # Download direct from Celestrak
    lines = []
    for line in urllib.request.urlopen('https://celestrak.com/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle'):
        lines.append((line.decode('utf-8')))

    num_sats = int(len(lines) / 3)
    tle_data = {chr(i+97): [] for i in range(num_sats)}
    satVisCheck = {chr(i + 97): True for i in range(num_sats)}
    for i in range(0, len(lines), 3):
        c = chr(int(i/3)+97)
        n = lines[i]
        s = lines[i+1]
        t = lines[i+2]

        tle_data[c] = Satrec.twoline2rv(s, t)

    # Convert to TEME/ITRS?

    # Vector of times for ~1 day
    jd, fr = [], []

    # Array of 12 times over 1 day
    num_hours = 12
    for i in range(num_hours):
        jd_temp, fr_temp = jday(2022, 5, 5, i, 0, 0)
        jd.append(jd_temp)
        fr.append(fr_temp)

    e = {chr(i+97): np.zeros((num_hours, 1), dtype=np.float64) for i in range(num_sats)}  #
    r = {chr(i+97): np.zeros((num_hours, 3), dtype=np.float64) for i in range(num_sats)}  # pos (1x3 cartesian)
    v = {chr(i+97): np.zeros((num_hours, 3), dtype=np.float64) for i in range(num_sats)}  # vel (1x3 cartesian)

    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(num_hours):
            # Calc e, pos, vel for each sat at each t
            e[c][j], r[c][j], v[c][j] = tle_data[c].sgp4(jd[j], fr[j])


    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)
    kf = {chr(i+97): UKF(dim_x=6, dim_z=3, dt=stepLength, fx=Funcs.kepler, hx=Funcs.h_x, points=points)
          for i in range(num_sats)}

    for i in range(num_sats):
        c = chr(i + 97)
        kf[c].x = np.zeros((6, 1))
        if satVisCheck[c]:
            for j in range(simLength):
                if np.all(np.isnan(satECIMes[c][:, j])):
                    continue
                else:
                    kf[c].x[0:3] = np.reshape(satECIMes[c][:, j], (3, 1))
                    kf[c].x = kf[c].x.flat
                    break

    # Process noise
    stdAng = np.float64(1e-5)
    coefA = np.float64(0.25 * stepLength ** 4.0 * stdAng ** 2.0)
    coefB = np.float64(stepLength ** 2.0 * stdAng ** 2.0)
    coefC = np.float64(0.5 * stepLength ** 3.0 * stdAng ** 2.0)

    for i in range(num_sats):
        c = chr(i+97)
        if satVisCheck[c]:
            kf[c].Q = np.array([[coefA, 0, 0, coefC, 0, 0],
                                [0, coefA, 0, 0, coefC, 0],
                                [0, 0, coefA, 0, 0, coefC],
                                [coefC, 0, 0, coefB, 0, 0],
                                [0, coefC, 0, 0, coefB, 0],
                                [0, 0, coefC, 0, 0, coefB]],
                               dtype='float64')

            kf[c].P = np.float64(1e10) * np.identity(6)

    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range

    angMeasDev, rangeMeasDev = 1e-6, 20
    covAER = np.array([[(angMeasDev * 180 / pi) ** 2, 0, 0],
                       [0, (angMeasDev * 180 / pi) ** 2, 0],
                       [0, 0, rangeMeasDev ** 2]],
                      dtype='float64')

    totalStates = {chr(i + 97): np.zeros((6, simLength)) for i in range(num_sats)}
    diffState = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    err_X_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}
    err_Y_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}
    err_Z_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}

    # ~~~~~ Using UKF
    delta = 1e-6
    for i in range(num_sats):
        c = chr(i + 97)
        if satVisCheck[c]:
            mesCheck = False
            for j in range(simLength):
                while not mesCheck:
                    if np.all(np.isnan(satECIMes[c][:, j])):
                        break
                    else:
                        mesCheck = True
                        break

                if not mesCheck:
                    continue

                func_params = {
                    "stepLength": stepLength,
                    "count": j + 1,
                    "sensECEF": sensECEF,
                    "sensLLA[0]": sensLLA[0],
                    "sensLLA[1]": sensLLA[1]
                }

                jacobian = Funcs.jacobian_finder("aer_to_eci", np.reshape(satAERMes[c][:, j], (3, 1)),
                                                 func_params, delta)

                kf[c].R = jacobian @ covAER @ jacobian.T

                kf[c].predict()
                if not np.any(np.isnan(satECIMes[c][:, j])):
                    kf[c].update(satECIMes[c][:, j])

                totalStates[c][:, j] = np.reshape(kf[c].x, 6)
                err_X_ECI[c][j] = (np.sqrt(np.abs(kf[c].P[0, 0])))
                err_Y_ECI[c][j] = (np.sqrt(np.abs(kf[c].P[1, 1])))
                err_Z_ECI[c][j] = (np.sqrt(np.abs(kf[c].P[2, 2])))
                diffState[c][:, j] = totalStates[c][0:3, j] - satECI[c][:, j]
                # print(satState[c])

    # ~~~~~ Plotting
    for i in range(num_sats):
        c = chr(i + 97)
        if satVisCheck[c]:
            fig, (ax1, ax2, ax3) = plt.subplots(3)
            axs = [ax1, ax2, ax3]
            fig.suptitle('Satellite {sat}'.format(sat=i))
            ax1.plot(satECI[c][0, :])
            # ax1.plot(satECIMes[c][0,:], 'r.')
            ax1.plot(totalStates[c][0, :])
            ax1.set(ylabel='$X_{ECI}$, metres')

            ax2.plot(satECI[c][1, :])
            # ax2.plot(satECIMes[c][1,:], 'r.')
            ax2.plot(totalStates[c][1, :])
            ax2.set(ylabel='$Y_{ECI}$, metres')

            ax3.plot(satECI[c][2, :])
            # ax3.plot(satECIMes[c][2,:], 'r.')
            ax3.plot(totalStates[c][2, :])
            ax3.set(xlabel='Time Step', ylabel='$Z_{ECI}$, metres')

            plt.show()

    # ~~~~~ Error plots

    for i in range(num_sats):
        c = chr(i + 97)
        if satVisCheck[c]:
            fig, (ax1, ax2, ax3) = plt.subplots(3)
            axs = [ax1, ax2, ax3]
            fig.suptitle('Satellite {sat} Errors'.format(sat=i))
            ax1.plot(err_X_ECI[c])
            ax1.plot(np.abs(diffState[c][0, :]))
            ax1.set(ylabel='$X_{ECI}$, metres')

            ax2.plot(err_Y_ECI[c])
            ax2.plot(np.abs(diffState[c][1, :]))
            ax2.set(ylabel='$Y_{ECI}$, metres')

            ax3.plot(err_Z_ECI[c])
            ax3.plot(np.abs(diffState[c][2, :]))
            ax3.set(xlabel='Time Step', ylabel='$Z_{ECI}$, metres')

            plt.show()
