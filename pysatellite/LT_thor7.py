# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:36:04 2021

@author: sgboakes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints

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

    # simLength = cfg.simLength
    # stepLength = cfg.stepLength
    stepLength = 30
    simLength = int(4*24*3600 / stepLength)  # num of time steps 4 full days

    mu = cfg.mu
    trans_earth = False

    # ~~~~ Satellite Conversion

    # Import LT measurements and times

    measurements = np.array([[0.916437473381186, 2.598016609177409],
                             [0.916692348240909, 2.597809943546340],
                             [0.916644495464303, 2.598207261330070],
                             [0.916821874956585, 2.597943668726428],
                             [0.916749822362960, 2.598334544415291],
                             [0.916966061222676, 2.597952845068490],
                             [0.916849466590489, 2.598395180517596],
                             [0.916668403993601, 2.597859453021996],
                             [0.916549747408426, 2.597875687387380],
                             [0.916674583236358, 2.597925821841022],
                             [0.916574411334118, 2.598357663271107],
                             [0.916470129271193, 2.597881666161044],
                             [0.916721819617671, 2.598464595447797],
                             [0.916790170557235, 2.599368382238589],
                             [0.916950399783682, 2.598218479476475],
                             [0.916510396308655, 2.598761575429670],
                             [0.917363213133516, 2.597974146128391],
                             [0.916481489724773, 2.597678036278715],
                             [0.916536951367966, 2.598320441833727],
                             [0.916798084917192, 2.598181514589937],
                             [0.916741110841644, 2.598384372307093],
                             [0.916920230235471, 2.598674147168976],
                             [0.916757733896978, 2.598629981277147],
                             [0.916835766898041, 2.598769916390368],
                             [0.916810122548999, 2.598723631873395],
                             [0.916987970767142, 2.598466008225915],
                             [0.917033068250969, 2.598124141436498],
                             [0.916702793720732, 2.598039076617566],
                             [0.916748887225397, 2.598174613346324],
                             [0.916753870821731, 2.598234121771780],
                             [0.916913020323713, 2.598442060123055],
                             [0.916913372435922, 2.598424294610170],
                             [0.916930444504799, 2.598619169341027],
                             [0.916956654399077, 2.598616389767954],
                             [0.916957869415373, 2.598657812116999]],
                            dtype=np.float64)

    # day:hour:minute:second
    times_raw = ['00:00:36:37', '00:01:32:32', '00:02:04:05', '00:02:36:37', '00:03:05:06', '00:03:43:44',
                 '00:04:01:01', '00:04:33:33', '01:00:05:06', '01:00:56:57', '01:01:05:06', '01:01:33:34',
                 '01:02:37:38', '01:03:18:19', '01:03:36:37', '01:04:31:32', '01:04:33:33', '02:00:30:31',
                 '02:00:32:32', '02:01:21:21', '02:01:40:40', '02:02:26:26', '02:02:35:36', '02:03:25:25',
                 '02:03:34:35', '02:04:05:06', '02:04:36:36', '03:00:06:07', '03:00:46:47', '03:01:05:05',
                 '03:02:12:12', '03:02:34:34', '03:03:04:05', '03:03:37:37', '03:04:14:14']

    # convert times_raw to minute index
    times_ind = []
    for i in range(len(times_raw)):
        t = times_raw[i]
        t_sp = t.split(':')
        t_sec = float((t_sp[0]))*24*3600 + float(t_sp[1])*3600 + float(t_sp[2])*60 + float(t_sp[3])
        times_ind.append(round(t_sec/stepLength)-1)


    # Define sat pos in ECI and convert to AER
    # radArr: radii for each sat metres
    # omegaArr: orbital rate for each sat rad/s
    # thetaArr: inclination angle for each sat rad
    # kArr: normal vector for each sat metres
    num_sats = 1


    for i in range(num_sats):
        c = chr(i+97)
        satAERMes = np.empty((3, simLength))
        for j in range(len(times_ind)):
            satAERMes[:, times_ind[j]] = np.array([measurements[j, 1], measurements[j, 0], 35786*1000],
                                                  dtype=np.float64)  # range is estimated

    satAERMes[satAERMes == 0.] = np.nan
    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range

    angMeasDev, rangeMeasDev = 1e-4, 1000

    for i in range(num_sats):
        c = chr(i + 97)
        satECIMes = np.empty((3, simLength))
        for j in range(simLength):
            satECIMes[:, j:j + 1] = Transformations.aer_to_eci(satAERMes[:, j], stepLength, j+1, sensECEF,
                                                               sensLLA[0], sensLLA[1])

    # ~~~~ Temp ECI measurements from MATLAB

    # satECIMes['a'] = pd.read_csv('ECI_mes.txt', delimiter=' ').to_numpy(dtype='float64')
    # #satECIMes.to_numpy(dtype='float64')
    # satECIMes['a'] = satECIMes['a'].T
    # np.reshape(satECIMes['a'], (3, simLength))

    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)
    kf = {chr(i+97): UKF(dim_x=6, dim_z=3, dt=stepLength, fx=Functions.kepler, hx=Functions.h_x, points=points)
          for i in range(num_sats)}

    for i in range(num_sats):
        c = chr(i + 97)
        kf[c].x = np.zeros((6, 1))
        for j in range(simLength):
            if np.all(np.isnan(satECIMes[:, j])):
                continue
            else:
                kf[c].x[0:3] = np.reshape(satECIMes[:, j], (3, 1))
                # Need vel prior?
                kf[c].x = kf[c].x.flat
                break

    # Process noise
    stdAng = np.float64(1e-2)
    coefA = np.float64(0.25 * stepLength ** 4.0 * stdAng ** 2.0)
    coefB = np.float64(stepLength ** 2.0 * stdAng ** 2.0)
    coefC = np.float64(0.5 * stepLength ** 3.0 * stdAng ** 2.0)

    for i in range(num_sats):
        c = chr(i+97)
        kf[c].Q = np.array([[coefA, 0, 0, coefC, 0, 0],
                            [0, coefA, 0, 0, coefC, 0],
                            [0, 0, coefA, 0, 0, coefC],
                            [coefC, 0, 0, coefB, 0, 0],
                            [0, coefC, 0, 0, coefB, 0],
                            [0, 0, coefC, 0, 0, coefB]],
                           dtype='float64')

        kf[c].P = np.float64(1e10) * np.identity(6)

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
    delta = 1e-5
    for i in range(num_sats):
        c = chr(i + 97)
        mesCheck = False
        for j in range(simLength):
            while not mesCheck:
                if np.all(np.isnan(satECIMes[:, j])):
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

            jacobian = Functions.jacobian_finder("aer_to_eci", np.reshape(satAERMes[:, j], (3, 1)),
                                                 func_params, delta)

            kf[c].R = jacobian @ covAER @ jacobian.T

            kf[c].predict()
            if not np.any(np.isnan(satECIMes[:, j])):
                kf[c].update(satECIMes[:, j])

            totalStates[c][:, j] = np.reshape(kf[c].x, 6)
            err_X_ECI[c][j] = (np.sqrt(np.abs(kf[c].P[0, 0])))
            err_Y_ECI[c][j] = (np.sqrt(np.abs(kf[c].P[1, 1])))
            err_Z_ECI[c][j] = (np.sqrt(np.abs(kf[c].P[2, 2])))
            # diffState[c][:, j] = totalStates[c][0:3, j] - satECI[c][:, j]
            # print(satState[c])

    # ~~~~~ Plotting
    for i in range(num_sats):
        c = chr(i + 97)
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        axs = [ax1, ax2, ax3]
        fig.suptitle('Satellite {sat}'.format(sat=i))
        # ax1.plot(satECI[c][0, :])
        # ax1.plot(satECIMes[c][0,:], 'r.')
        ax1.plot(totalStates[c][0, :])
        ax1.plot(satECIMes[0, :], 'r.')
        ax1.set(ylabel='$X_{ECI}$, metres')

        # ax2.plot(satECI[c][1, :])
        # ax2.plot(satECIMes[c][1,:], 'r.')
        ax2.plot(totalStates[c][1, :])
        ax2.plot(satECIMes[1, :], 'r.')
        ax2.set(ylabel='$Y_{ECI}$, metres')

        # ax3.plot(satECI[c][2, :])
        # ax3.plot(satECIMes[c][2,:], 'r.')
        ax3.plot(totalStates[c][2, :])
        ax3.plot(satECIMes[2, :], 'r.')
        ax3.set(xlabel='Time Step', ylabel='$Z_{ECI}$, metres')

        plt.show()

    # ~~~~~ Error plots

    for i in range(num_sats):
        c = chr(i + 97)
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

    for i in range(num_sats):
        c = chr(i+97)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(totalStates[c][0, :], totalStates[c][1, :], totalStates[c][2, :])

        plt.show()
