# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:36:04 2021

@author: sgboakes
"""

import numpy as np
import matplotlib.pyplot as plt
from pysatellite import transformations, functions
import pysatellite.config as cfg
# from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
# from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from pysatellite.filters import UKF
from pysatellite import orbit_gen

if __name__ == "__main__":

    plt.close('all')
    np.random.seed(3)
    # ~~~~ Variables

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)


    class Sensor:
        def __init__(self):
            self.LLA = np.array([[np.deg2rad(28.300697)], [np.deg2rad(-16.509675)], [2390]], dtype='float64')
            # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
            self.ECEF = transformations.lla_to_ecef(self.LLA)
            self.ECEF.shape = (3, 1)
            self.AngVar = 1e-6
            self.RngVar = 20


    sens = Sensor()

    simLength = cfg.simLength
    simLength = 50
    stepLength = cfg.stepLength

    num_sats = 100

    # ~~~~ Satellite Conversion METHOD 1
    satECI, satECIMes, satAER, satAERMes, satVisible = orbit_gen.circular_orbits(num_sats, simLength,
                                                                                 stepLength, sens)

    # ~~~~ Satellite Conversion METHOD 2
    # satECI, satAER, satECIMes, satAERMes, satVisible = orbit_gen.coe_orbits(num_sats, simLength, stepLength, sens)

    # for i, c in enumerate(satECI):
    #     satECI[c] = satECI[c][0:3, :]

    # Process noise
    stdAng = np.float64(1e-5)
    coefA = np.float64(0.25 * stepLength ** 4.0 * stdAng ** 2.0)
    coefB = np.float64(stepLength ** 2.0 * stdAng ** 2.0)
    coefC = np.float64(0.5 * stepLength ** 3.0 * stdAng ** 2.0)

    Q = np.array([[coefA, 0, 0, coefC, 0, 0],
                  [0, coefA, 0, 0, coefC, 0],
                  [0, 0, coefA, 0, 0, coefC],
                  [coefC, 0, 0, coefB, 0, 0],
                  [0, coefC, 0, 0, coefB, 0],
                  [0, 0, coefC, 0, 0, coefB]],
                 dtype='float64')

    P = np.float64(1e10) * np.identity(6)

    x_init = {'{i}'.format(i=i): np.zeros((6, 1)) for i in range(num_sats)}
    for i, c in enumerate(satVisible):
        if satVisible[c]:
            for j in range(simLength):
                if np.all(np.isnan(satECIMes[c][:, j])):
                    continue
                else:
                    x_init[c][0:3] = np.reshape(satECIMes[c][:, j], (3, 1))
                    # x_init[c] = np.reshape(satECI[c][:, j], (6, 1))
                    x_init[c] = np.reshape(x_init[c], (6,))
                    break

    kf = {'{i}'.format(i=i): UKF(num_states=6, process_noise=Q, initial_state=x_init[c], initial_covar=P, alpha=0.1,
                                 k=0., beta=2., iterate_function=functions.kepler) for i, c in enumerate(x_init)}

    angMeasDev, rangeMeasDev = 1e-6, 20
    covAER = np.array([[(angMeasDev * 180 / pi) ** 2, 0, 0],
                       [0, (angMeasDev * 180 / pi) ** 2, 0],
                       [0, 0, rangeMeasDev ** 2]],
                      dtype='float64')

    totalStates = {'{i}'.format(i=i): np.zeros((6, simLength)) for i in range(num_sats)}
    diffState = {'{i}'.format(i=i): np.zeros((3, simLength)) for i in range(num_sats)}
    err_X_ECI = {'{i}'.format(i=i): np.zeros(simLength) for i in range(num_sats)}
    err_Y_ECI = {'{i}'.format(i=i): np.zeros(simLength) for i in range(num_sats)}
    err_Z_ECI = {'{i}'.format(i=i): np.zeros(simLength) for i in range(num_sats)}

    # ~~~~~ Using UKF
    for i, c in enumerate(satECIMes):
        if satVisible[c]:
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
                    "sensECEF": sens.ECEF,
                    "sensLLA[0]": sens.LLA[0],
                    "sensLLA[1]": sens.LLA[1]
                }

                jacobian = functions.jacobian_finder(transformations.aer_to_eci, np.reshape(satAERMes[c][:, j], (3, 1)),
                                                     func_params)

                R = jacobian @ covAER @ jacobian.T

                kf[c].predict(stepLength)
                if not np.any(np.isnan(satAERMes[c][:, j])):
                    # try:
                    kf[c].update([0, 1, 2], satECIMes[c][:, j], R)
                    # except ValueError:
                    #     print("pause here")
                    #     continue

                totalStates[c][:, j] = kf[c].get_state()
                covar = kf[c].get_covar()
                err_X_ECI[c][j] = (np.sqrt(np.abs(covar[0, 0])))
                err_Y_ECI[c][j] = (np.sqrt(np.abs(covar[1, 1])))
                err_Z_ECI[c][j] = (np.sqrt(np.abs(covar[2, 2])))
                diffState[c][:, j] = totalStates[c][0:3, j] - satECI[c][0:3, j]
                # print(satState[c])

    # ~~~~~ Plotting
    # for i, c in enumerate(satECI):
    #     if satVisible[c]:
    #         fig, (ax1, ax2, ax3) = plt.subplots(3)
    #         axs = [ax1, ax2, ax3]
    #         fig.suptitle('Satellite {sat}'.format(sat=i))
    #         ax1.plot(satECI[c][0, :])
    #         # ax1.plot(satECIMes[c][0,:], 'r.')
    #         ax1.plot(totalStates[c][0, :])
    #         ax1.set(ylabel='$X_{ECI}$, metres')
    #
    #         ax2.plot(satECI[c][1, :])
    #         # ax2.plot(satECIMes[c][1,:], 'r.')
    #         ax2.plot(totalStates[c][1, :])
    #         ax2.set(ylabel='$Y_{ECI}$, metres')
    #
    #         ax3.plot(satECI[c][2, :])
    #         # ax3.plot(satECIMes[c][2,:], 'r.')
    #         ax3.plot(totalStates[c][2, :])
    #         ax3.set(xlabel='Time Step', ylabel='$Z_{ECI}$, metres')
    #
    #         plt.show()

    # ~~~~~ Error plots

    # for i, c in enumerate(diffState):
    #     if satVisible[c]:
    #         fig, (ax1, ax2, ax3) = plt.subplots(3)
    #         axs = [ax1, ax2, ax3]
    #         fig.suptitle('Satellite {sat} Errors'.format(sat=i))
    #         ax1.plot(err_X_ECI[c])
    #         ax1.plot(np.abs(diffState[c][0, :]))
    #         ax1.set(ylabel='$X_{ECI}$, metres')
    #
    #         ax2.plot(err_Y_ECI[c])
    #         ax2.plot(np.abs(diffState[c][1, :]))
    #         ax2.set(ylabel='$Y_{ECI}$, metres')
    #
    #         ax3.plot(err_Z_ECI[c])
    #         ax3.plot(np.abs(diffState[c][2, :]))
    #         ax3.set(xlabel='Time Step', ylabel='$Z_{ECI}$, metres')
    #
    #         plt.show()

    # ~~~~~ Combined error plot

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    axs = [ax1, ax2, ax3]
    fig.suptitle('Satellite Errors')
    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    ax3.set_xlim(0, 50)
    ax1.set(ylabel='$X_{ECI}$, metres')
    ax2.set(ylabel='$Y_{ECI}$, metres')
    ax3.set(xlabel='Time Step', ylabel='$Z_{ECI}$, metres')
    for i, c in enumerate(diffState):
        if satVisible[c]:
            ax1.plot(err_X_ECI[c], linewidth=0.5)
            ax2.plot(err_Y_ECI[c], linewidth=0.5)
            ax3.plot(err_Z_ECI[c], linewidth=0.5)

    plt.show()

    # ~~~~~ Combined ground truth diff plot

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    axs = [ax1, ax2, ax3]
    fig.suptitle('Satellite Errors')
    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    ax3.set_xlim(0, 50)
    ax1.set(ylabel='$X_{ECI}$, metres')
    ax2.set(ylabel='$Y_{ECI}$, metres')
    ax3.set(xlabel='Time Step', ylabel='$Z_{ECI}$, metres')
    for i, c in enumerate(diffState):
        if satVisible[c]:
            ax1.plot(diffState[c][0, :], linewidth=0.5)
            ax2.plot(diffState[c][1, :], linewidth=0.5)
            ax3.plot(diffState[c][2, :], linewidth=0.5)

    plt.show()
