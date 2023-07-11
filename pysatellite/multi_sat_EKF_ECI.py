# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:36:04 2021

@author: sgboakes
"""

import numpy as np
import matplotlib.pyplot as plt
from pysatellite import transformations, functions, filters
import pysatellite.config as cfg
import pysatellite.orbit_gen as orbit_gen

if __name__ == "__main__":

    plt.close('all')
    np.random.seed(3)
    # ~~~~ Variables

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)

    class Sensor:
        def __init__(self):
            # Using Liverpool Telescope as location
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

    num_sats = 25

    # ~~~~ Satellite Conversion METHOD 1
    # satECI, satECIMes, satAER, satAERMes, satVisible = orbit_gen.circular_orbits(num_sats, simLength, stepLength,
    #                                                                              sens)

    # ~~~~ Satellite Conversion METHOD 2
    satECI, satAER, satECIMes, satAERMes, satVisible = orbit_gen.coe_orbits(num_sats, simLength, stepLength,
                                                                            sens)

    for i, c in enumerate(satECI):
        satECI[c] = satECI[c][0:3, :]

    # ~~~~ Temp ECI measurements from MATLAB

    # satECIMes['a'] = pd.read_csv('ECI_mes.txt', delimiter=' ').to_numpy(dtype='float64')
    # #satECIMes.to_numpy(dtype='float64')
    # satECIMes['a'] = satECIMes['a'].T
    # np.reshape(satECIMes['a'], (3, simLength))

    # Initialising filtering states from first measurement
    satState = {'{i}'.format(i=i): np.zeros((6, 1)) for i in range(num_sats)}
    for i, c in enumerate(satState):
        if satVisible[c]:
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

    covState = {'{i}'.format(i=i): np.float64(1e10) * np.identity(6) for i in range(num_sats)}

    angMeasDev, rangeMeasDev = 1e-6, 20
    covAER = np.array([[(sens.AngVar * 180 / pi) ** 2, 0, 0],
                       [0, (sens.AngVar * 180 / pi) ** 2, 0],
                       [0, 0, sens.RngVar ** 2]],
                      dtype='float64')

    measureMatrix = np.append(np.identity(3), np.zeros((3, 3)), axis=1)

    totalStates = {'{i}'.format(i=i): np.zeros((6, simLength)) for i in range(num_sats)}
    diffState = {'{i}'.format(i=i): np.zeros((3, simLength)) for i in range(num_sats)}
    err_X_ECI = {'{i}'.format(i=i): np.zeros(simLength) for i in range(num_sats)}
    err_Y_ECI = {'{i}'.format(i=i): np.zeros(simLength) for i in range(num_sats)}
    err_Z_ECI = {'{i}'.format(i=i): np.zeros(simLength) for i in range(num_sats)}

    # ~~~~~ Using EKF

    delta = 1e-6
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

                jacobian = functions.jacobian_finder("aer_to_eci", np.reshape(satAERMes[c][:, j], (3, 1)), func_params,
                                                     delta)

                # covECI = np.matmul(np.matmul(jacobian, covAER), jacobian.T)
                covECI = jacobian @ covAER @ jacobian.T

                stateTransMatrix = functions.jacobian_finder("kepler", satState[c], [], delta)

                satState[c], covState[c] = filters.ekf(satState[c], covState[c], satECIMes[c][:, j], stateTransMatrix,
                                                       measureMatrix, covECI, procNoise)

                totalStates[c][:, j] = np.reshape(satState[c], 6)
                err_X_ECI[c][j] = (np.sqrt(np.abs(covState[c][0, 0])))
                err_Y_ECI[c][j] = (np.sqrt(np.abs(covState[c][1, 1])))
                err_Z_ECI[c][j] = (np.sqrt(np.abs(covState[c][2, 2])))
                diffState[c][:, j] = totalStates[c][0:3, j] - satECI[c][:, j]
                # print(satState[c])

    # ~~~~~ Plotting

    # for i, c in enumerate(satECIMes):
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

    for i, c in enumerate(diffState):
        if satVisible[c]:
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
