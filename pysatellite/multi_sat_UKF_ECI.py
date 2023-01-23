# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:36:04 2021

@author: sgboakes
"""

import numpy as np
import matplotlib.pyplot as plt
from pysatellite import transformations, functions
import pysatellite.config as cfg
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from pysatellite import orbit_gen

if __name__ == "__main__":

    plt.close('all')
    # np.random.seed(2)
    # ~~~~ Variables

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)


    class Sensor:
        def __init__(self):
            self.Lat = np.float64(28.300697)
            self.Lon = np.float64(-16.509675)
            self.Alt = np.float64(2390)
            self.LLA = np.array([[self.Lat * pi / 180], [self.Lon * pi / 180], [self.Alt]], dtype='float64')
            # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
            self.ECEF = transformations.lla_to_ecef(self.LLA)
            self.ECEF.shape = (3, 1)


    sens = Sensor()

    simLength = cfg.simLength
    stepLength = cfg.stepLength
    trans_earth = False

    num_sats = 100

    # ~~~~ Satellite Conversion METHOD 1
    # satECI, satECIMes, satAER, satAERMes, satVisible = orbit_gen.circular_orbits(num_sats, simLength, stepLength,
    #                                                                              sens, trans_earth)

    # ~~~~ Satellite Conversion METHOD 2
    satECI, satAER, satECIMes, satAERMes, satVisible = orbit_gen.coe_orbits(num_sats, simLength, stepLength,
                                                                            sens)

    # ~~~~ Temp ECI measurements from MATLAB

    # satECIMes['a'] = pd.read_csv('ECI_mes.txt', delimiter=' ').to_numpy(dtype='float64')
    # #satECIMes.to_numpy(dtype='float64')
    # satECIMes['a'] = satECIMes['a'].T
    # np.reshape(satECIMes['a'], (3, simLength))

    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)
    kf = {chr(i+97): UKF(dim_x=6, dim_z=3, dt=stepLength, fx=functions.kepler, hx=functions.h_x, points=points)
          for i in range(num_sats)}

    for i in range(num_sats):
        c = chr(i + 97)
        kf[c].x = np.zeros((6, 1))
        if satVisible[c]:
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
        if satVisible[c]:
            kf[c].Q = np.array([[coefA, 0, 0, coefC, 0, 0],
                                [0, coefA, 0, 0, coefC, 0],
                                [0, 0, coefA, 0, 0, coefC],
                                [coefC, 0, 0, coefB, 0, 0],
                                [0, coefC, 0, 0, coefB, 0],
                                [0, 0, coefC, 0, 0, coefB]],
                               dtype='float64')

            kf[c].P = np.float64(1e10) * np.identity(6)

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

                jacobian = functions.jacobian_finder("aer_to_eci", np.reshape(satAERMes[c][:, j], (3, 1)),
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
    # for i in range(num_sats):
    #     c = chr(i + 97)
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

    for i in range(num_sats):
        c = chr(i + 97)
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
