# -*- coding: utf-8 -*-
"""
Created on Mon Aug  22 12:51:40 2022

@author: sgboakes
"""

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
from pysatellite import transformations, functions, filters, orbit_gen
import pysatellite.config as cfg

if __name__ == "__main__":

    plt.close('all')
    np.random.seed(4)  # Will seeding work with acceptance model in orbit generation?
    # ~~~~ Variables

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)


    class Sensor:
        def __init__(self):
            self.LLA = np.array([[np.deg2rad(np.float64(28.300697))],
                                 [np.deg2rad(np.float64(-16.509675))],
                                 [(np.float64(2390))]],
                                dtype='float64')
            # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
            self.ECEF = transformations.lla_to_ecef(self.LLA)
            self.ECEF.shape = (3, 1)
            self.AngVar = 1e-6
            self.RngVar = 20


    sens = Sensor()

    simLength = cfg.simLength
    simLength = 200
    stepLength = cfg.stepLength

    num_sats = 10

    # ~~~~ Satellite Conversion METHOD 1
    satECI, satAER, satECIMes, satAERMes, satVisible = orbit_gen.circular_orbits(num_sats, simLength, stepLength,
                                                                                 sens)

    # ~~~~ Satellite Conversion METHOD 2
    # return orbital elements of satellites?
    # satECI, satAER, satECIMes, satAERMes, satVisible = orbit_gen.coe_orbits(num_sats, simLength, stepLength,
    #                                                                         sens)

    # ~~~~~ Globe Plot

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(45, 35)
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

    # relationship between different accepted orbital elements
    # could use to reduce number of rejected samples
    # or could remove check for if elev > 0 for each satellite and use KDE sampling?
    # for j in range(len(elements[0])):
    #     plt.figure()
    #     for i in range(num_sats):
    #         c = chr(i + 97)
    #         plt.plot(i, elements[c][j], 'x')
    #
    #     plt.show()

    # Check for if satellite visible at all time-steps
    # for i in range(10000):
    #     np.random.seed(i)
    #     ECI, AER, ECIMes, AERMes, Visible = orbit_gen.circular_orbits(num_sats, simLength, stepLength, sens)
    #
    #     if not np.isnan(ECIMes['a']).any():
    #         print(i)
