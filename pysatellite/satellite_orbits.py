# -*- coding: utf-8 -*-
"""
Created on Mon Aug  22 12:51:40 2022

@author: sgboakes
"""

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
from pysatellite import Transformations, Functions, Filters, orbit_gen
import pysatellite.config as cfg

if __name__ == "__main__":

    plt.close('all')
    np.random.seed(3)  # Will seeding work with acceptance model in orbit generation?
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
            self.ECEF = Transformations.lla_to_ecef(self.LLA)
            self.ECEF.shape = (3, 1)
            self.AngVar = 1e-6
            self.RngVar = 20


    sens = Sensor()

    simLength = cfg.simLength
    simLength = 20
    stepLength = cfg.stepLength
    trans_earth = False

    num_sats = 10

    # ~~~~ Satellite Conversion METHOD 1
    # satECI, satAER, satECIMes, satAERMes, satVisible = orbit_gen.circular_orbits(num_sats, simLength, stepLength,
    #                                                                              sens, trans_earth)

    # ~~~~ Satellite Conversion METHOD 2
    satECI, satAER, satECIMes, satAERMes, satVisible = orbit_gen.coe_orbits(num_sats, simLength, stepLength,
                                                                            sens, trans_earth)

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