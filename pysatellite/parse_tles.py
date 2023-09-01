import numpy as np
from pysatellite import transformations, functions, filters
import pysatellite.config as cfg
import sgp4
import os
from skyfield.api import EarthSatellite, load, wgs84
from sgp4.api import Satrec
import datetime
import matplotlib.pyplot as plt

sin = np.sin
cos = np.cos
pi = np.float64(np.pi)


class Sensor:
    def __init__(self):
        # Using Liverpool Telescope as location
        self.LLA = np.array([[np.deg2rad(28.300697)],
                             [np.deg2rad(-16.509675)],
                             [2390]],
                            dtype='float64')
        # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
        self.ECEF = transformations.lla_to_ecef(self.LLA)
        self.ECEF.shape = (3, 1)
        self.AngVar = 1e-6
        self.RngVar = 20


sens = Sensor()
bluffton = wgs84.latlon(28.300697, -16.509675, 2390)

simLength = cfg.simLength
simLength = 200
stepLength = cfg.stepLength

# All satellites
# file = os.getcwd() + '/space-track_leo_tles.txt'

# Only visible satellites (see end of script)
file = os.getcwd() + '/space-track_leo_tles_visible.txt'

with open(file) as f:
    tle_lines = f.readlines()

tles = {}
satellites = {}
ts = load.timescale()
for i in range(0, len(tle_lines)-1, 2):
    tles['{i}'.format(i=int(i/2))] = [tle_lines[i], tle_lines[i+1]]
    satellites['{i}'.format(i=int(i/2))] = EarthSatellite(line1=tles['{i}'.format(i=int(i/2))][0],
                                                          line2=tles['{i}'.format(i=int(i/2))][1])

# satellite = satellites['0']
# t = ts.now()
# # Uses sgp4 propagator?
# geocentric = satellite.at(t)
# print(geocentric.position.km)
#
# bluffton = wgs84.latlon(28.300697, -16.509675, 2390)
# diff = satellite - bluffton
# topocentric = diff.at(t)
# alt, az, distance = topocentric.altaz()
# print(alt, az, distance.m)
#
# # How to get proper times? Need to be in JD
# satellitesgp = Satrec.twoline2rv(tles['0'][0], tles['0'][1])
# jd, fr = 2458827, 0.362605
# e, r, v = satellitesgp.sgp4(jd, fr)

num_sats = len(satellites)
# Get rough epoch using first satellite
epoch = satellites['0'].epoch.utc_datetime()
# Generate timestamps for each step in simulation
timestamps = []
for i in range(simLength):
    timestamps.append(ts.from_datetime(epoch+datetime.timedelta(seconds=i*stepLength)))

# Need a ground truth somehow? Use TLE at each step, then measurements can be TLE plus some WGN?
# But then propagator will be equal for both ground truth generation and filtering
# Think this is fine

satAER = {'{i}'.format(i=i): np.zeros((3, simLength)) for i in range(num_sats)}
satECI = {'{i}'.format(i=i): np.zeros((3, simLength)) for i in range(num_sats)}
satTEME = {'{i}'.format(i=i): np.zeros((6, simLength)) for i in range(num_sats)}
satVis = {'{i}'.format(i=i): True for i in range(num_sats)}
for i, c in enumerate(satellites):
    for j in range(simLength):
        # t = ts.from_datetime(epoch+datetime.timedelta(seconds=j*stepLength))
        t = timestamps[j]
        diff = satellites[c] - bluffton
        topocentric = diff.at(t)
        alt, az, dist = topocentric.altaz()
        satAER[c][:, j] = [az.radians, alt.radians, dist.m]
        satECI[c][:, j] = np.reshape(transformations.aer_to_eci(satAER[c][:, j], stepLength, j, sens.ECEF,
                                                                sens.LLA[0], sens.LLA[1]), (3,))
        satTEME[c][:, j] = np.reshape(([topocentric.position.m], [topocentric.velocity.m_per_s]), (6,))

# count = 0
# maxp = 0
# for i, c in enumerate(satECI):
#     if max(satECI[c][:, 2]) > maxp:
#         count = c
#         maxp = max(satECI[c][:, 2])
#
# print(count)

# Plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(45, 35)
ax.set_aspect('auto')

# for i, c in enumerate(satECI):
#     ax.plot3D(satECI[c][0, :], satECI[c][1, :], satECI[c][2, :])

ax.plot3D(satECI['0'][0, :], satECI['0'][1, :], satECI['0'][2, :])
ax.plot3D(satTEME['0'][0, :], satTEME['0'][1, :], satTEME['0'][2, :])

plt.show()

# ~~~~~ Work out visible TLEs and save to file

# satAERVisible = {}
# for i, c in enumerate(satAER):
#     if all(i > 0 for i in satAER[c][:, 1]):
#         satAERVisible[c] = satAER[c]
#
# file_reduced = os.getcwd() + '/space-track_leo_tles_visible.txt'
# with open(file_reduced, 'w') as f:
#     for i, c in enumerate(satAERVisible):
#         f.writelines(tles[c])

# ~~~~~ Test to see if returned tle is same as original

