# -*- coding: utf-8 -*-
"""
Created on Tue Jan  18 13:29:40 2022

@author: sgboakes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg

if __name__ == "__main__":

    plt.close('all')
    # ~~~~ Variables

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)

    sensLat = np.float64(28.300697)
    sensLon = np.float64(-16.509675)
    sensAlt = np.float64(2390)
    sensLLA = np.array([[sensLat * pi / 180], [sensLon * pi / 180], [sensAlt]], dtype='float64')
    # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
    sensECEF = Transformations.LLAtoECEF(sensLLA)
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
    # kArr: normal vector for each sat metres

    radArr = np.array([7e6, 7e6, 7e6, 7e6], dtype='float64')

    omegaArr = 1 / np.sqrt(radArr ** 3 / mu)

    thetaArr = np.array([[0.0], [0.1], [0.2], [0.3]], dtype='float64')

    kArr = np.array([[0, 0, 0],
                     [0, 0, 1],
                     [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                     [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]],
                    dtype='float64')

    num_sats = len(radArr)

    # Make data structures
    satECI = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    satAER = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}

    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            v = np.array([[radArr[i] * sin(omegaArr[i] * (j + 1) * stepLength)],
                          [0],
                          [radArr[i] * cos(omegaArr[i] * (j + 1) * stepLength)]], dtype='float64')

            satECI[c][:, j] = (v @ cos(thetaArr[i])) + (np.cross(kArr[i, :].T, v.T) * sin(thetaArr[i])) + (
                        kArr[i, :].T * np.dot(kArr[i, :].T, v) * (1 - cos(thetaArr[i])))

            satAER[c][:, j:j + 1] = Transformations.ECItoAER(satECI[c][:, j], stepLength, j + 1, sensECEF, sensLLA[0],
                                                             sensLLA[1])

            if not trans_earth:
                if satAER[c][1, j] < 0:
                    satAER[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

        if np.isnan(satAER[c]).all():
            print('Satellite {s} is not observable'.format(s=c))

    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range

    angMeasDev, rangeMeasDev = 1e-6, 20

    satAERMes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        satAERMes[c][0, :] = satAER[c][0, :] + (angMeasDev * np.random.randn(1, simLength))
        satAERMes[c][1, :] = satAER[c][1, :] + (angMeasDev * np.random.randn(1, simLength))
        satAERMes[c][2, :] = satAER[c][2, :] + (rangeMeasDev * np.random.randn(1, simLength))

    satECIMes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            satECIMes[c][:, j:j + 1] = Transformations.AERtoECI(satAERMes[c][:, j], stepLength, j+1, sensECEF,
                                                                sensLLA[0], sensLLA[1])

    # ~~~~ Temp ECI measurements from MATLAB

    # satECIMes['a'] = pd.read_csv('ECI_mes.txt', delimiter=' ').to_numpy(dtype='float64')
    # #satECIMes.to_numpy(dtype='float64')
    # satECIMes['a'] = satECIMes['a'].T
    # np.reshape(satECIMes['a'], (3, simLength))

    satState = {chr(i + 97): np.zeros((6, 1)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            if np.all(np.isnan(satECIMes[c][:, j])):
                continue
            else:
                satState[c][0:3] = np.reshape(satECIMes[c][:, j], (3, 1))
                break

    # Process noise
    stdAng = np.float64(1e-5)
    coefA = np.float64(0.25 * stepLength ** 4.0 * stdAng ** 2.0)
    coefB = np.float64(stepLength ** 2.0 * stdAng ** 2.0)
    coefC = np.float64(0.5 * stepLength ** 3.0 * stdAng ** 2.0)

    procNoise = np.array([[coefA, 0, 0, coefC, 0, 0],
                          [0, coefA, 0, 0, coefC, 0],
                          [0, 0, coefA, 0, 0, coefC],
                          [coefC, 0, 0, coefB, 0, 0],
                          [0, coefC, 0, 0, coefB, 0],
                          [0, 0, coefC, 0, 0, coefB]],
                         dtype='float64')

    covState = {chr(i + 97): np.zeros((6, 6)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        covState[c] = np.float64(1e10) * np.identity(6)

    covAER = np.array([[(angMeasDev * 180 / pi) ** 2, 0, 0],
                       [0, (angMeasDev * 180 / pi) ** 2, 0],
                       [0, 0, rangeMeasDev ** 2]],
                      dtype='float64'
                      )

    measureMatrix = np.append(np.identity(3), np.zeros((3, 3)), axis=1)

    totalStates = {chr(i + 97): np.zeros((6, simLength)) for i in range(num_sats)}
    diffState = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    err_X_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}
    err_Y_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}
    err_Z_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}

    # ~~~~~ Using EKF

    delta = 1e-6
    for i in range(num_sats):
        c = chr(i + 97)
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

            jacobian = Functions.jacobian_finder("AERtoECI", np.reshape(satAERMes[c][:, j], (3, 1)), func_params, delta)

            # covECI = np.matmul(np.matmul(jacobian, covAER), jacobian.T)
            covECI = jacobian @ covAER @ jacobian.T

            stateTransMatrix = Functions.jacobian_finder("kepler", satState[c], [], delta)

            satState[c], covState[c] = Filters.EKF_ECI(satState[c], covState[c], satECIMes[c][:, j], stateTransMatrix,
                                                       measureMatrix, covECI, procNoise)

            totalStates[c][:, j] = np.reshape(satState[c], 6)
            err_X_ECI[c][j] = (np.sqrt(np.abs(covState[c][0, 0])))
            err_Y_ECI[c][j] = (np.sqrt(np.abs(covState[c][1, 1])))
            err_Z_ECI[c][j] = (np.sqrt(np.abs(covState[c][2, 2])))
            diffState[c][:, j] = totalStates[c][0:3, j] - satECI[c][:, j]
            # print(satState[c])

    # ~~~~~ Plotting

    for i in range(num_sats):
        c = chr(i + 97)
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        axs = [ax1, ax2, ax3]
        fig.suptitle('Satellite {sat}'.format(sat=c))
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
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        axs = [ax1, ax2, ax3]
        fig.suptitle('Satellite {sat} Errors'.format(sat=c))
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

    # ~~~~~ Globe Plot

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(60, 35)
    ax.set_aspect('auto')

    for i in range(num_sats):
        c = chr(i + 97)
        ax.plot3D(satECI[c][0, :], satECI[c][1, :], satECI[c][2, :])

    plt.show()

    # ~~~~~ Polar Plot

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")  # theta=0 at the top
    ax.set_theta_direction(-1)  # theta increasing clockwise
    ax.set_rlim(90, 0, 1)

    for i in range(num_sats):
        c = chr(i + 97)
        ax.plot(satAER[c][0, :], np.rad2deg(satAER[c][1, :]), 'x-')

    plt.show()
