# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:00:45 2021

@author: ben-o_000
"""

import numpy as np
import pysatellite.config as cfg
from poliastro.core.elements import coe2rv, rv2coe

# Define common variables
sin = np.sin
cos = np.cos
omega = np.float64(7.2921158553e-5)  # Earth rotation rate ~SIDEREAL
WGS = cfg.WGS84
a = WGS.semimajoraxis
b = WGS.semiminoraxis
e = WGS.eccentricity
ePrime = np.sqrt((a**2 - b**2) / b**2)  # Square of second eccentricity


def ae_to_ned(ae):
    psi = ae[0]
    theta = ae[1]
    ned = [cos(theta)*cos(psi),
          cos(theta)*sin(psi), 
          sin(theta)]

    return ned

def aer_to_eci(pos_aer, step_length, step_num, ori_ecef, ori_lat, ori_lon):
    """
    Function for converting Az/Elev/Range to Latitude/Longitude/Altitude

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_aer: A 1x3 or 3x1 vector containing the Azimuth, Elevation, and Range
            positions in radians and metres, respectively.

    ori_ecef: A 1x3 or 3x1 vector containing the origin of the local NED frame
            as x_ecef, y_ecef, and z_ecef respectively.

    ori_lat: The latitude of the origin of the local NED frame in radians.

    ori_lon: The longitude of the origin of the local NED frame in radians.

    WGS: The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_lla: A 3x1 vector containing the latitude, longitude, and altitude
        positions in radians and metres, respectively.
    """

    az = pos_aer[0]
    elev = pos_aer[1]
    ran = pos_aer[2]

    z_up = ran * sin(elev)
    r = ran * cos(elev)
    y_east = r * sin(az)
    x_north = r * cos(az)
    
    pos_ned = np.array([[x_north, y_east, -z_up]], dtype='float64').T
    
    rot_matrix = np.array([[(-sin(ori_lat)*cos(ori_lon)), -sin(ori_lon), (-cos(ori_lat) * cos(ori_lon))],
                          [(-sin(ori_lat) * sin(ori_lon)), cos(ori_lon), (-cos(ori_lat) * sin(ori_lon))],
                          [cos(ori_lat), 0.0, (-sin(ori_lat))]], dtype='float64')
    
    pos_ecef_delta = rot_matrix @ pos_ned
    
    pos_ecef = pos_ecef_delta + ori_ecef

    # Generate matrices for multiplication
    rotation_matrix = np.array([[cos(step_num*step_length*omega), -sin(step_num*step_length*omega), 0.0],
                                [sin(step_num*step_length*omega), cos(step_num*step_length*omega), 0.0],
                                [0.0, 0.0, 1.0]],
                               dtype='float64')
                
    pos_eci = rotation_matrix @ pos_ecef
    return pos_eci


def aer_to_lla(pos_aer, ori_ecef, ori_lat, ori_lon):
    """
    Function for converting Az/Elev/Range to Latitude/Longitude/Altitude

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_aer: A 1x3 or 3x1 vector containing the Azimuth, Elevation, and Range
            positions in radians and metres, respectively.

    ori_ecef: A 1x3 or 3x1 vector containing the origin of the local NED frame
            as x_ecef, y_ecef, and z_ecef respectively.

    ori_lat: The latitude of the origin of the local NED frame in radians.

    ori_lon: The longitude of the origin of the local NED frame in radians.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_lla: A 3x1 vector containing the latitude, longitude, and altitude
        positions in radians and metres, respectively.
    """

    az = pos_aer[0]
    elev = pos_aer[1]
    ran = pos_aer[2]

    z_up = ran * sin(elev)
    r = ran * cos(elev)
    y_east = r * sin(az)
    x_north = r * cos(az)

    cos_phi = cos(ori_lat)
    sin_phi = sin(ori_lat)
    cos_lambda = cos(ori_lon)
    sin_lambda = sin(ori_lon)

    z_down = -z_up

    t = cos_phi * -z_down - sin_phi * x_north
    dz = sin_phi * -z_down + cos_phi * x_north

    dx = cos_lambda * t - sin_lambda * y_east
    dy = sin_lambda * t + cos_lambda * y_east

    x_ecef = ori_ecef[0] + dx
    y_ecef = ori_ecef[1] + dy
    z_ecef = ori_ecef[2] + dz

    # Closed formula set
    p = np.sqrt(x_ecef**2+y_ecef**2)
    theta = np.arctan2((z_ecef * a), (p * b))

    lon = np.arctan2(y_ecef, x_ecef)
    # lon = mod(lon,2*pi)
    lat = np.arctan2((z_ecef + (ePrime**2 * b * (sin(theta))**3)), (p - (e**2 * a * (cos(theta))**3)))
    n = a / (np.sqrt(1 - e**2 * (sin(lat))**2))
    alt = (p / cos(lat)) - n

    pos_lla = [lat], [lon], [alt]
    return pos_lla


def aer_to_ned(pos_aer):
    """
    Function for converting Az/Elev/Range to local North/East/Down

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_aer: A 1x3 or 3x1 vector containing the Azimuth, Elevation, and Range
            positions in radians and metres, respectively.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_ned: A 3x1 vector containing the north, east, and down positions,
        respectively.
    """

    az = pos_aer[0]
    elev = pos_aer[1]
    ran = pos_aer[2]

    z_up = ran * sin(elev)
    r = ran * cos(elev)
    y_east = r * sin(az)
    x_north = r * cos(az)

    pos_ned = [x_north], [y_east], [-z_up]
    return pos_ned


def body_to_earth(body, psi, theta, phi):
    """
    Function for converting body coordinates to earth coordinates through
    rotations in azimuth, elevation, and roll

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    body: A 3x1 vector containing the x, y, and z body positions,
            respectively.

    psi: Angle for rotating in azimuth in radians.

    theta: Angle for rotating in elevation in radians.

    phi: Angle for rotating in roll in radians.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    earth: A 3x1 vector containing the position in earth coordinates.
    """
    # Psi = azimuth
    # Theta = elevation
    # Phi = roll

    rot_az = np.array([[cos(psi), sin(psi), 0],
                       [-sin(psi), cos(psi), 0],
                       [0, 0, 1]])

    rot_elev = np.array([[cos(theta), 0, sin(theta)],
                         [0, 1, 0],
                         [-sin(theta), 0, cos(theta)]])

    rot_roll = np.array([[1, 0, 0],
                         [0, cos(phi), sin(phi)],
                         [0, -sin(phi), cos(phi)]])

    # THIS IS EARTH -> BODY
    # rotated_scope = rot_roll @ rot_elev @ rot_az @ telescope

    # THIS IS BODY -> EARTH
    earth = rot_az.T @ rot_elev.T @ rot_roll.T @ body

    return earth


def coe_to_rv(k, p, ecc, inc, raan, argp, nu):
    """Function for converting from classical orbital elements to cartesian position and velocity
    Uses poliastro's coe2rv function
    If gravitational parameter k is in m^3/s^2 then ijk will be in m and m/s?
    """

    return coe2rv(k, p, ecc, inc, raan, argp, nu)


def ecef_to_aer(pos_ecef, ori_ecef, ori_lat, ori_lon):
    """
    Function for converting ECEF position to Azimuth/Elevation/Range

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_eci: A 3x1 vector containing the x, y, and z ECI positions,
            respectively.

    ori_ecef: A 1x3 or 3x1 vector containing the origin of the local NED frame
             as x_ecef, y_ecef, and z_ecef respectively.

    ori_lat: The latitude of the origin of the local NED frame in radians.

    ori_lon: The longitude of the origin of the local NED frame in radians.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_aer: A 3x1 vector containing the azimuth, elevation, and range
            positions in radians, respectively.
    """

    transform_matrix = np.array(
        [[(-sin(ori_lat) * cos(ori_lon)), (-sin(ori_lat) * sin(ori_lon)), cos(ori_lat)],
         [(-sin(ori_lon)), cos(ori_lon), 0.0],
         [(-cos(ori_lat) * cos(ori_lon)), (-cos(ori_lat) * sin(ori_lon)), (-sin(ori_lat))]],
        dtype='float64')

    pos_delta = pos_ecef - ori_ecef

    pos_ned = transform_matrix @ pos_delta

    # Convert enu vector to NED vector
    x_north = np.float64(pos_ned[0])
    y_east = np.float64(pos_ned[1])
    z_down = np.float64(pos_ned[2])

    r1 = np.hypot(x_north, y_east)
    ran = np.hypot(r1, z_down)
    elevation = np.arctan2(-z_down, r1)
    azimuth = np.mod(np.arctan2(y_east, x_north), 2 * np.float64(np.pi))

    pos_aer = np.array([[azimuth], [elevation], [ran]])
    return pos_aer


def ecef_to_eci(pos_ecef, step_length, step_num):
    """
    Function for converting ECEF coordinates to ECI coordinates

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_ecef: A 3x1 vector containing the x, y, and z ECEF positions,
             respectively.

    step_length: The length of each time step of the simulation.

    step_num: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_eci: A 3x1 vector containing the x, y, and z ECI positions,
            respectively.
    """

    # omega = 2*pi / (24*60*60) ~NOT SIDEREAL

    t = np.array([[cos(omega*step_length*step_num), -(sin(omega*step_length*step_num)), 0.0],
                  [sin(omega*step_length*step_num), cos(omega*step_length*step_num), 0.0],
                  [0.0, 0.0, 1.0]],
                 dtype='float64')
    
    pos_eci = t @ pos_ecef
    return pos_eci


def ecef_to_lla(pos_ecef):
    """
    Function for converting ECEF coordinates to latitude/longitude/altitude,
    using a closed formula set.

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_ecef: A 1x3 or 3x1 vector containing the x, y, and z ECEF positions,
             respectively.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_lla: A 3x1 vector containing the latitude, longitude, and altitude
            positions in radians, respectively.
    """

    x_ecef = pos_ecef[0]
    y_ecef = pos_ecef[1]
    z_ecef = pos_ecef[2]

    # Closed formula set
    p = np.sqrt(x_ecef**2+y_ecef**2)
    theta = np.arctan2((z_ecef * a), (p * b))

    lon = np.arctan2(y_ecef, x_ecef)
    # lon = mod(lon,2*pi)

    lat = np.arctan2((z_ecef + (ePrime**2 * b * (sin(theta))**3)), (p - (e**2 * a * (cos(theta))**3)))

    n = a / (np.sqrt(1 - e**2 * (sin(lat))**2))

    alt = (p / cos(lat)) - n

    pos_lla = [lat], [lon], [alt]
    return pos_lla


def ecef_to_ned(pos_ecef, ori_ecef, ori_lat, ori_lon):
    """
    Function for converting ECEF coordinates to local NED coordinates

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_ecef: A 1x3 or 3x1 vector containing the x, y, and z ECEF positions,
             respectively.

    ori_ecef: A 1x3 or 3x1 vector containing the origin of the local NED frame
             as x_ecef, y_ecef, and z_ecef respectively.

    ori_lat: The latitude of the origin of the local NED frame in radians.

    ori_lon: The longitude of the origin of the local NED frame in radians.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_ned: A 3x1 vector containing the north, east, and down positions,
            respectively.
    """

    x_obj = pos_ecef[0]
    y_obj = pos_ecef[1]
    z_obj = pos_ecef[2]

    # Generate matrices for multiplication
    rotation_matrix = np.array(
        [[-(sin(ori_lon)), cos(ori_lon), 0.0],
         [(-(sin(ori_lat))*cos(ori_lon)), (-(sin(ori_lat))*sin(ori_lon)), cos(ori_lat)],
         [(cos(ori_lat)*cos(ori_lon)), (cos(ori_lat)*sin(ori_lon)), sin(ori_lat)]],
        dtype='float64')

    coord_matrix = [x_obj - ori_ecef[0]], [y_obj - ori_ecef[1]], [z_obj - ori_ecef[2]]

    # Find enu vector
    enu = rotation_matrix @ coord_matrix

    # Convert enu vector to NED vector
    x_north = enu[1]
    y_east = enu[0]
    z_down = -enu[2]

    pos_ned = [x_north], [y_east], [z_down]
    return pos_ned


def eci_to_aer(pos_eci, step_length, step_num, ori_ecef, ori_lat, ori_lon):
    """
    Function for converting ECI position to Azimuth/Elevation/Range

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_eci: A 3x1 vector containing the x, y, and z ECI positions,
            respectively.

    step_length: The length of each time step of the simulation.

    step_num: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    ori_ecef: A 1x3 or 3x1 vector containing the origin of the local NED frame
             as x_ecef, y_ecef, and z_ecef respectively.

    ori_lat: The latitude of the origin of the local NED frame in radians.

    ori_lon: The longitude of the origin of the local NED frame in radians.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_aer: A 3x1 vector containing the azimuth, elevation, and range
            positions in radians, respectively.
    """

    # omega = 2*pi / (24*60*60) ~NOT SIDEREAL

    rotation_matrix = np.array([[cos(step_num*step_length*omega), sin(step_num*step_length*omega), 0.0],
                                [-(sin(step_num*step_length*omega)), cos(step_num*step_length*omega), 0.0],
                                [0.0, 0.0, 1.0]],
                               dtype='float64')
    
    pos_eci = np.reshape(pos_eci, (3, 1))

    pos_ecef = rotation_matrix @ pos_eci

    transform_matrix = np.array(
        [[(-sin(ori_lat)*cos(ori_lon)), (-sin(ori_lat)*sin(ori_lon)), cos(ori_lat)],
         [(-sin(ori_lon)), cos(ori_lon), 0.0],
         [(-cos(ori_lat)*cos(ori_lon)), (-cos(ori_lat)*sin(ori_lon)), (-sin(ori_lat))]],
        dtype='float64')

    pos_delta = pos_ecef - ori_ecef

    pos_ned = transform_matrix @ pos_delta

    # Convert enu vector to NED vector
    x_north = np.float64(pos_ned[0])
    y_east = np.float64(pos_ned[1])
    z_down = np.float64(pos_ned[2])

    r1 = np.hypot(x_north, y_east)
    ran = np.hypot(r1, z_down)
    elevation = np.arctan2(-z_down, r1)
    azimuth = np.mod(np.arctan2(y_east, x_north), 2*np.float64(np.pi))

    pos_aer = np.array([[azimuth], [elevation], [ran]])
    return pos_aer


def eci_to_ecef(pos_eci, step_length, step_num):
    """
    Function for converting ECI coordinates to ECEF coordinates

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_eci: A 3x1 vector containing the x, y, and z ECI positions,
            respectively.

    step_length: The length of each time step of the simulation.

    step_num: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_ecef: A 3x1 vector containing the x, y, and z ECEF positions,
             respectively.
    """

    # omega = 2*pi / (24*60*60) ~NOT SIDEREAL
    
    t = np.array(
        [[cos(omega*step_length*step_num), sin(omega*step_length*step_num), 0.0],
         [-(sin(omega*step_length*step_num)), cos(omega*step_length*step_num), 0.0],
         [0.0, 0.0, 1.0]],
        dtype='float64')

    pos_ecef = t @ pos_eci
    return pos_ecef


def eci_to_lla(pos_eci, step_length, step_num):
    """
    Function for converting ECI coordinates to latitude/longitude/altitude,
    using a closed formula set.

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_eci: A 1x3 or 3x1 vector containing the x, y, and z ECEF positions,
            respectively.

    step_length: The length of each time step of the simulation.

    step_num: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_lla: A 3x1 vector containing the latitude, longitude, and altitude
            positions in radians, respectively.
    """

    # omega = 2*pi / (24*60*60) ~NOT SIDEREAL

    rotation_matrix = np.array(
        [[cos(step_num*step_length*omega), sin(step_num*step_length*omega), 0.0],
         [-sin(step_num*step_length*omega), cos(step_num*step_length*omega), 0.0],
         [0.0, 0.0, 1.0]],
        dtype='float64')

    pos_ecef = rotation_matrix @ pos_eci
    
    x_ecef = pos_ecef[0]
    y_ecef = pos_ecef[1]
    z_ecef = pos_ecef[2]

    # Closed formula set
    p = np.sqrt(x_ecef**2+y_ecef**2)
    theta = np.arctan2((z_ecef * a), (p * b))

    lon = np.arctan2(y_ecef, x_ecef)
    # lon = mod(lon,2*pi)

    lat = np.arctan2((z_ecef + (ePrime**2 * b * (sin(theta))**3)), (p - (e**2 * a * (cos(theta))**3)))

    n = a / (np.sqrt(1 - e**2 * (sin(lat))**2))

    alt = (p / cos(lat)) - n

    pos_lla = [lat], [lon], [alt]
    return pos_lla


def lla_to_aer(pos_lla, ori_ecef, ori_lat, ori_lon):
    """
    Function for converting Latitude/Longitude/Altitude to Az/Elev/Range

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_lla: A 3x1 vector containing the latitude, longitude, and altitude
            positions in radians, respectively.

    ori_ecef: A 1x3 or 3x1 vector containing the origin of the local NED frame
            as x_ecef, y_ecef, and z_ecef respectively.

    ori_lat: The latitude of the origin of the local NED frame in radians.

    ori_lon: The longitude of the origin of the local NED frame in radians.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_aer: A 1x3 or 3x1 vector containing the Azimuth, Elevation, and Range
        positions in radians, respectively.
    """

    lat = pos_lla[0]
    lon = pos_lla[1]
    alt = pos_lla[2]

    # Prime vertical radius of curvature N(phi)
    # Formula if eccentricity not defined: n_phi = a**2 / (sqrt((a**2*(cos(lat)**2))+(b**2*(sin(lat)**2))))

    n_phi = a / (np.sqrt(1 - (e**2*(sin(lat))**2)))
    x_ecef = (n_phi + alt) * cos(lat) * cos(lon)
    y_ecef = (n_phi + alt) * cos(lat) * sin(lon)
    z_ecef = (((b**2/a**2)*n_phi) + alt)*sin(lat)

    # Generate matrices for multiplication
    rotation_matrix = np.array(
        [[-sin(ori_lon), cos(ori_lon), 0],
         [(-sin(ori_lat)*cos(ori_lon)), (-sin(ori_lat)*sin(ori_lon)), cos(ori_lat)],
         [(cos(ori_lat)*cos(ori_lon)), (cos(ori_lat)*sin(ori_lon)), sin(ori_lat)]],
        dtype='float64')

    coord_matrix = np.array(
        [[x_ecef - ori_ecef[0]],
         [y_ecef - ori_ecef[1]],
         [z_ecef - ori_ecef[2]]],
        dtype='float64')

    # Find enu vector
    enu = rotation_matrix @ coord_matrix

    # Convert enu vector to NED vector
    x_north = enu[1]
    y_east = enu[0]
    z_down = -enu[2]

    r1 = np.hypot(x_north, y_east)
    ran = np.hypot(r1, z_down)
    elevation = np.arctan2(-z_down, r1)
    azimuth = np.mod(np.arctan2(y_east, x_north), 2*np.pi)

    pos_aer = [azimuth], [elevation], [ran]
    return pos_aer


def lla_to_ecef(pos_lla):
    """
    Function for converting ECEF coordinates to latitude/longitude/altitude.

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_lla: A 1x3 or 3x1 vector containing the latitude, longitude, and
            altitude positions in radians, respectively.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_ecef: A  3x1 vector containing the x, y, and z ECEF positions,
             respectively.
    """

    lat = pos_lla[0]
    lon = pos_lla[1]
    alt = pos_lla[2]

    # Prime vertical radius of curvature N(phi)
    # Formula if eccentricity not defined: n_phi = a**2 / (sqrt((a**2*(cos(lat)**2))+(b**2*(sin(lat)**2))))

    n_phi = a / (np.sqrt(1 - (e**2*(sin(lat))**2)))
    x_ecef = (n_phi + alt) * cos(lat) * cos(lon)
    y_ecef = (n_phi + alt) * cos(lat) * sin(lon)
    z_ecef = (((b**2/a**2)*n_phi) + alt)*sin(lat)

    pos_ecef = np.array([[x_ecef],
                         [y_ecef],
                         [z_ecef]],
                        dtype='float64')
    return pos_ecef


def lla_to_eci(pos_lla, step_length, step_num):
    """
    Function for converting latitude/longitude/altitude to ECI coordinates.

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_lla: A 1x3 or 3x1 vector containing the latitude, longitude, and
            altitude positions in radians, respectively.

    step_length: The length of each time step of the simulation.

    step_num: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_eci: A  3x1 vector containing the x, y, and z ECI positions,
            respectively.
    """

    lat = pos_lla[0]
    lon = pos_lla[1]
    alt = pos_lla[2]

    # Prime vertical radius of curvature N(phi)
    # Formula if eccentricity not defined: n_phi = a**2 / (sqrt((a**2*(cos(lat)**2))+(b**2*(sin(lat)**2))))

    n_phi = a / (np.sqrt(1 - (e**2*(sin(lat))**2)))
    x_ecef = (n_phi + alt) * cos(lat) * cos(lon)
    y_ecef = (n_phi + alt) * cos(lat) * sin(lon)
    z_ecef = (((b**2/a**2)*n_phi) + alt)*sin(lat)

    pos_ecef = [x_ecef], [y_ecef], [z_ecef]

    # Generate matrices for multiplication
    rotation_matrix = np.array([[cos(step_num*step_length*omega), -sin(step_num*step_length*omega), 0.0],
                                [sin(step_num*step_length*omega), cos(step_num*step_length*omega), 0.0],
                                [0.0, 0.0, 1.0]],
                               dtype='float64')
                
    pos_eci = rotation_matrix @ pos_ecef
    return pos_eci


def ned_to_ae(ned):
    """
    Function for converting local North/East/Down to unit Az/Elev

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    ned: A 1x3 or 3x1 vector containing the north, east, and down positions,
            respectively.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    azel: A 2x1 vector containing the Azimuth and Elevation
        positions in radians, respectively.
    """
    psi_ae = np.arctan2(ned[1], ned[0])
    theta_ae = np.arcsin(ned[2])
    azel = [psi_ae, theta_ae]
    return azel

def ned_to_aer(pos_ned):
    """
    Function for converting local North/East/Down to Az/Elev/Range

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_ned: A 1x3 or 3x1 vector containing the north, east, and down positions,
            respectively.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_aer: A 3x1 vector containing the Azimuth, Elevation, and Range
        positions in radians, respectively.
    """
    
    x_north = pos_ned[0]
    y_east = pos_ned[1]
    z_down = pos_ned[2]

    r1 = np.hypot(x_north, y_east)
    ran = np.hypot(r1, z_down)
    elevation = np.arctan2(-z_down, r1)
    azimuth = np.mod(np.arctan2(y_east, x_north), 2*np.pi)

    pos_aer = [azimuth], [elevation], [ran]
    return pos_aer


def ned_to_ecef(pos_ned, ori_ecef, ori_lat, ori_lon):
    """
    Function for converting ECEF coordinates to local NED coordinates

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    pos_ned: A 1x3 or 3x1 vector containing the north, east, and down positions,
            respectively.

    ori_ecef: A 1x3 or 3x1 vector containing the origin of the local NED frame
             as x_ecef, y_ecef, and z_ecef respectively.

    ori_lat: The latitude of the origin of the local NED frame in radians.

    ori_lon: The longitude of the origin of the local NED frame in radians.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    pos_ecef: A 3x1 vector containing the x, y, and z ECEF positions,
             respectively.
    """

    cos_phi = cos(ori_lat)
    sin_phi = sin(ori_lat)
    cos_lambda = cos(ori_lon)
    sin_lambda = sin(ori_lon)

    x_north = pos_ned[0]
    y_east = pos_ned[1]
    z_down = pos_ned[2]

    t = cos_phi * -z_down - sin_phi * x_north
    dz = sin_phi * -z_down + cos_phi * x_north

    dx = cos_lambda * t - sin_lambda * y_east
    dy = sin_lambda * t + cos_lambda * y_east

    x_ecef = ori_ecef[0] + dx
    y_ecef = ori_ecef[1] + dy
    z_ecef = ori_ecef[2] + dz

    pos_ecef = [x_ecef], [y_ecef], [z_ecef]
    return pos_ecef


def rv_to_coe(r, v, mu=cfg.mu):
    """Function for converting from cartesian position and velocity to classical orbital elements
    Uses poliastro's rv2coe function
    If gravitational parameter k is in m^3/s^2 then ijk will be in m and m/s?
    """

    return rv2coe(mu, r, v)
