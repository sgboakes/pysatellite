import os

import numpy as np
import matplotlib.pyplot as plt
from pysatellite import transformations, functions as Funcs
import pysatellite.config as cfg
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from skyfield.api import EarthSatellite, load, wgs84

if __name__ == "__main__":

    plt.close('all')
    # np.random.seed(2)
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

    # ~~~~~~ USING TLE DATA FROM space-track
    file = os.getcwd() + '/space-track_leo_tles.txt'

    with open(file) as f:
        tle_lines = f.readlines()

    tles = {}
    satellites = {}
    ts = load.timescale()
    for i in range(0, len(tle_lines) - 1, 2):
        tles['{i}'.format(i=int(i / 2))] = [tle_lines[i], tle_lines[i + 1]]
        satellites['{i}'.format(i=int(i / 2))] = EarthSatellite(line1=tles['{i}'.format(i=int(i / 2))][0],
                                                                line2=tles['{i}'.format(i=int(i / 2))][1])

   # Get rough epoch using first satellite
    epoch = satellites['0'].epoch.utc_datetime()

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
