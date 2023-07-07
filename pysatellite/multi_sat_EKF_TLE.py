import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
from pysatellite import transformations, functions, filters, orbit_gen
import pysatellite.config as cfg

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
    bluffton = wgs84.latlon(28.300697, -16.509675, 2390)

    simLength = cfg.simLength
    simLength = 50
    stepLength = cfg.stepLength

    num_sats = 25

    # ~~~~~~ USING TLE DATA FROM space-track
    # file = os.getcwd() + '/space-track_leo_tles.txt'
    file = os.getcwd() + '/space-track_leo_tles_visible.txt'

    with open(file) as f:
        tle_lines = f.readlines()

    tles = {}
    satellites = {}
    ts = load.timescale()
    for i in range(0, len(tle_lines) - 1, 2):
        tles['{i}'.format(i=int(i / 2))] = [tle_lines[i], tle_lines[i + 1]]
        satellites['{i}'.format(i=int(i / 2))] = EarthSatellite(line1=tles['{i}'.format(i=int(i / 2))][0],
                                                                line2=tles['{i}'.format(i=int(i / 2))][1])

    num_sats = len(satellites)
    # Get rough epoch using first satellite
    epoch = satellites['0'].epoch.utc_datetime()

    # Need a ground truth somehow? Use TLE at each step, then measurements can be TLE plus some WGN?
    # But then propagator will be equal for both ground truth generation and filtering

    satAER = {'{i}'.format(i=i): np.zeros((3, simLength)) for i in range(num_sats)}
    satECI = {'{i}'.format(i=i): np.zeros((3, simLength)) for i in range(num_sats)}
    satVis = {'{i}'.format(i=i): True for i in range(num_sats)}
    for i, c in enumerate(satellites):
        for j in range(simLength):
            t = ts.from_datetime(epoch+datetime.timedelta(seconds=j*stepLength))
            diff = satellites[c] - bluffton
            topocentric = diff.at(t)
            alt, az, dist = topocentric.altaz()
            satAER[c][:, j] = [az.radians, alt.radians, dist.m]
            satECI[c][:, j] = np.reshape(transformations.aer_to_eci(satAER[c][:, j], stepLength, j, sens.ECEF,
                                                                    sens.LLA[0], sens.LLA[1]), (3,))

    satECIMes, satAERMes = orbit_gen.gen_measurements(satAER, num_sats, satVis, simLength, stepLength, sens)

    # satAERVisible = {}
    # for i, c in enumerate(satAER):
    #     if all(i > 0 for i in satAER[c][:, 1]):
    #         satAERVisible[c] = satAER[c]
    #
    # file_reduced = os.getcwd() + '/space-track_leo_tles_visible.txt'
    # with open(file_reduced, 'w') as f:
    #     for i, c in enumerate(satAERVisible):
    #         f.writelines(tles[c])

    # Initialising filtering states from first measurement
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

    covState = {chr(i + 97): np.float64(1e10) * np.identity(6) for i in range(num_sats)}

    angMeasDev, rangeMeasDev = 1e-6, 20
    covAER = np.array([[(sens.AngVar * 180 / pi) ** 2, 0, 0],
                       [0, (sens.AngVar * 180 / pi) ** 2, 0],
                       [0, 0, sens.RngVar ** 2]],
                      dtype='float64')

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

    # ~~~~~ Globe Plot

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(45, 35)
    ax.set_aspect('auto')

    for i, c in enumerate(satECI):
        ax.plot3D(satECI[c][0, :], satECI[c][1, :], satECI[c][2, :], linewidth=0.2)

    plt.show()

    # ~~~~~ Polar Plot

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")  # theta=0 at the top
    ax.set_theta_direction(-1)  # theta increasing clockwise
    ax.set_rlim(90, 0, 1)

    for i, c in enumerate(satAER):
        ax.plot(satAER[c][0, :], np.rad2deg(satAER[c][1, :]), 'x-', linewidth=0.2, markevery=[-1])

    plt.show()
