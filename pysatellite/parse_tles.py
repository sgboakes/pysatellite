import numpy as np
from pysatellite import transformations, functions, filters
import pysatellite.config as cfg
import astropy
import sgp4
import pandas as pd
import os
import re
from skyfield.api import EarthSatellite, load, wgs84

# Load json containing TLEs
# Maybe need to sanitise first? (Ensure only one space between elements)
file = os.getcwd() + '/space-track_leo_tles.txt'

file_sanitised = os.getcwd() + '/space-track_leo_tles_sanitised.txt'

lines = []
with open(file) as f:
    for line in f.readlines():
        lines.append(re.sub(' +', ' ', line))

with open(file_sanitised, 'w') as f:
    f.writelines(lines)

df = pd.read_csv(file_sanitised, sep=' ', header=None, on_bad_lines='skip')

df_line1 = df.iloc[::2, :]
df_line1.columns = ['Line Number', 'Satellite Catalog Number', 'Epoch', 'd1 Mean Motion', 'd2 Mean Motion', 'B*',
                    'Ephemeris Type','Element Set No.']
df_line2 = df.iloc[1::2, :]
df_line2.columns = ['Line Number', 'Satellite Catalog Number', 'Inclination', 'Raan', 'Ecc', 'Arg perigee',
                    'Mean anomaly', 'Mean motion']

ts = load.timescale()
sat_test = EarthSatellite(line1=df_line1.iloc[0].to_string(), line2=df_line2.iloc[0].to_string())
satellites = {EarthSatellite(line1=df_line1.iloc[i].to_string(), line2=df_line2.iloc[i].to_string())
              for i in range(len(df_line1-1))}