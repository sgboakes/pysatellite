import numpy as np
from pysatellite import transformations, functions, filters
import pysatellite.config as cfg
import sgp4
import os
from skyfield.api import EarthSatellite, load, wgs84
from sgp4.api import Satrec

file = os.getcwd() + '/space-track_leo_tles.txt'

with open(file) as f:
    tle_lines = f.readlines()

tles = {}
satellites = {}
ts = load.timescale()
for i in range(0, len(tle_lines)-1, 2):
    tles['{i}'.format(i=int(i/2))] = [tle_lines[i], tle_lines[i+1]]
    satellites['{i}'.format(i=int(i/2))] = EarthSatellite(line1=tles['{i}'.format(i=int(i/2))][0],
                                                          line2=tles['{i}'.format(i=int(i/2))][1])
satellite = satellites['0']
t = ts.now()
# Uses sgp4 propagator?
geocentric = satellite.at(t)
print(geocentric.position.km)

bluffton = wgs84.latlon(28.300697, -16.509675, 2390)
diff = satellite - bluffton
topocentric = diff.at(t)
alt, az, distance = topocentric.altaz()
print(alt, az, distance.m)

# How to get proper times? Need to be in JD
satellitesgp = Satrec.twoline2rv(tles['0'][0], tles['0'][1])
jd, fr = 2458827, 0.362605
e, r, v = satellitesgp.sgp4(jd, fr)
