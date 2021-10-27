# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:00:45 2021

@author: ben-o_000
"""

import numpy as np
import pysatellite.config as cfg


def AERtoECI(posAER, stepLength, stepNum, OriECEF, latOri, lonOri):
    """
    Function for converting Az/Elev/Range to Latitude/Longitude/Altitude

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posAER: A 1x3 or 3x1 vector containing the Azimuth, Elevation, and Range
            positions in radians and metres, respectively.

    OriECEF: A 1x3 or 3x1 vector containing the origin of the local NED frame
            as xECEF, yECEF, and zECEF respectively.

    latOri: The latitude of the origin of the local NED frame in radians.

    lonOri: The longitude of the origin of the local NED frame in radians.

    WGS: The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posLLA: A 3x1 vector containing the latitude, longitude, and altitude
        positions in radians and metres, respectively.
    """
    omega = np.float64(7.2921158553e-5)  # Earth rotation rate (radians/sec) ~SIDEREAL
    sin = np.sin
    cos = np.cos

    az = posAER[0]
    elev = posAER[1]
    ran = posAER[2]

    zUp = ran * sin(elev)
    r = ran * cos(elev)
    yEast  = r * sin(az)
    xNorth = r * cos(az)
    
    posNED = np.array([[xNorth, yEast, -zUp]], dtype='float64').T
    
    # rotMatrix = [[(-sin(latOri)*cos(lonOri)), -sin(lonOri), (-cos(latOri) * cos(lonOri))], [(-sin(latOri) * sin(lonOri)), cos(lonOri), (-cos(latOri) * sin(lonOri))], [cos(latOri), 0, (-sin(lonOri))]]
    # rotMatrix = np.array(rotMatrix)
    
    rotMatrix = np.array([[(-sin(latOri)*cos(lonOri)), -sin(lonOri), (-cos(latOri) * cos(lonOri))],
                          [(-sin(latOri) * sin(lonOri)), cos(lonOri), (-cos(latOri) * sin(lonOri))],
                          [cos(latOri), 0.0, (-sin(latOri))]], dtype='float64')
    
    posECEFDelta = rotMatrix @ posNED
    
    posECEF = posECEFDelta + OriECEF

    # Generate matrices for multiplication
    rotationMatrix = np.array([[cos(stepNum*stepLength*omega), -sin(stepNum*stepLength*omega), 0.0],
                               [sin(stepNum*stepLength*omega), cos(stepNum*stepLength*omega), 0.0],
                               [0.0, 0.0, 1.0]], dtype='float64')
                
    posECI = rotationMatrix @ posECEF
    return posECI


def AERtoLLA(posAER, OriECEF, latOri, lonOri):
    '''
    Function for converting Az/Elev/Range to Latitude/Longitude/Altitude

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posAER: A 1x3 or 3x1 vector containing the Azimuth, Elevation, and Range
            positions in radians and metres, respectively.

    OriECEF: A 1x3 or 3x1 vector containing the origin of the local NED frame
            as xECEF, yECEF, and zECEF respectively.

    latOri: The latitude of the origin of the local NED frame in radians.

    lonOri: The longitude of the origin of the local NED frame in radians.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posLLA: A 3x1 vector containing the latitude, longitude, and altitude
        positions in radians and metres, respectively.
    '''
    sin = np.sin
    cos = np.cos
    WGS = cfg.WGS

    az = posAER[0]
    elev = posAER[1]
    ran = posAER[2]

    zUp = ran * sin(elev)
    r   = ran * cos(elev)
    yEast  = r * sin(az)
    xNorth = r * cos(az)

    cosPhi = cos(latOri)
    sinPhi = sin(latOri)
    cosLambda = cos(lonOri)
    sinLambda = sin(lonOri)

    zDown = -zUp

    t = cosPhi * -zDown - sinPhi * xNorth
    dz = sinPhi * -zDown + cosPhi * xNorth

    dx = cosLambda * t - sinLambda * yEast
    dy = sinLambda * t + cosLambda * yEast

    xECEF = OriECEF[0] + dx
    yECEF = OriECEF[1] + dy
    zECEF = OriECEF[2] + dz


    # Ellipsoid properties
    a = WGS["SemimajorAxis"] # Semimajor axis
    b = WGS["SemiminorAxis"] # Semiminor axis
    f = WGS["Flattening"]    # Flattening
    e2 = f * (2 - f)      # Square of (first) eccentricity
    ep2 = e2 / (1 - e2)   # Square of second eccentricity
    e = np.sqrt((a**2 - b**2) / a**2)
    ePrime = np.sqrt((a**2 - b**2) / b**2)


    #Closed formula set
    p = np.sqrt(xECEF**2+yECEF**2)
    theta = np.arctan2((zECEF * a), (p * b))

    lon = np.arctan2(yECEF,xECEF)
    #lon = mod(lon,2*pi)
    lat = np.arctan2((zECEF + (ePrime**2 * b * (sin(theta))**3)), (p - (e**2 * a * (cos(theta))**3)))
    N = a / (np.sqrt(1 - e**2 * (sin(lat))**2))
    alt = (p / cos(lat)) - N

    posLLA = [lat], [lon], [alt]
    return posLLA


def AERtoNED(posAER):
    '''
    Function for converting Az/Elev/Range to local North/East/Down

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posAER: A 1x3 or 3x1 vector containing the Azimuth, Elevation, and Range
            positions in radians and metres, respectively.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posNED: A 3x1 vector containing the north, east, and down positions,
        respectively.
    '''
    sin = np.sin
    cos = np.cos

    az = posAER[0]
    elev = posAER[1]
    ran = posAER[2]

    zUp = ran * sin(elev)
    r   = ran * cos(elev)
    yEast  = r * sin(az)
    xNorth = r * cos(az)

    posNED = [xNorth], [yEast], [-zUp]
    return posNED


def ECEFtoECI(posECEF, stepLength, stepNum):
    '''
    Function for converting ECEF coordinates to ECI coordinates

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posECEF: A 3x1 vector containing the x, y, and z ECEF positions,
             respectively.

    stepLength: The length of each time step of the simulation.

    stepNum: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posECI: A 3x1 vector containing the x, y, and z ECI positions,
            respectively.
    '''
    sin = np.sin
    cos = np.cos
    omega = np.float64(7.2921158553e-5) #Earth rotation rate (radians/sec) ~SIDEREAL
    #omega = 2*pi / (24*60*60)

    T = np.array(
        [[cos(omega*stepLength*stepNum), -(sin(omega*stepLength*stepNum)), 0.0],
         [sin(omega*stepLength*stepNum), cos(omega*stepLength*stepNum), 0.0],
         [0.0, 0.0, 1.0]],
        dtype='float64'
        )
    
    posECI = T @ posECEF
    return posECI


def ECEFtoLLA(posECEF):
    '''
    Function for converting ECEF coordinates to latitude/longitude/altitude,
    using a closed formula set.

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posECEF: A 1x3 or 3x1 vector containing the x, y, and z ECEF positions,
             respectively.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posLLA: A 3x1 vector containing the latitude, longitude, and altitude
            positions in radians, respectively.
    '''
    sin = np.sin
    cos = np.cos
    WGS = cfg.WGS

    # Ellipsoid properties
    a = WGS["SemimajorAxis"]            # Semimajor axis
    b = WGS["SemiminorAxis"]            # Semiminor axis
    e = np.sqrt((a**2 - b**2) / a**2)      # Square of (first) eccentricity
    ePrime = np.sqrt((a**2 - b**2) / b**2) # Square of second eccentricity

    xECEF = posECEF[0]
    yECEF = posECEF[1]
    zECEF = posECEF[2]


    #Closed formula set
    p = np.sqrt(xECEF**2+yECEF**2)
    theta = np.arctan2((zECEF * a), (p * b))

    lon = np.arctan2(yECEF,xECEF)
    #lon = mod(lon,2*pi)

    lat = np.arctan2((zECEF + (ePrime**2 * b * (sin(theta))**3)), (p - (e**2 * a * (cos(theta))**3)))

    N = a / (np.sqrt(1 - e**2 * (sin(lat))**2))

    alt = (p / cos(lat)) - N

    posLLA = [lat], [lon], [alt]
    return posLLA


def ECEFtoNED(posECEF, OriECEF, latOri, lonOri):
    '''
    Function for converting ECEF coordinates to local NED coordinates

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posECEF: A 1x3 or 3x1 vector containing the x, y, and z ECEF positions,
             respectively.

    OriECEF: A 1x3 or 3x1 vector containing the origin of the local NED frame
             as xECEF, yECEF, and zECEF respectively.

    latOri: The latitude of the origin of the local NED frame in radians.

    lonOri: The longitude of the origin of the local NED frame in radians.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posNED: A 3x1 vector containing the north, east, and down positions,
            respectively.
    '''
    sin = np.sin
    cos = np.cos

    xObj = posECEF[0]
    yObj = posECEF[1]
    zObj = posECEF[2]


    #Generate matrices for multiplication
    rotationMatrix = np.array(
        [[-(sin(lonOri)), cos(lonOri), 0.0],
         [(-(sin(latOri))*cos(lonOri)), (-(sin(latOri))*sin(lonOri)), cos(latOri)],
         [(cos(latOri)*cos(lonOri)), (cos(latOri)*sin(lonOri)), sin(latOri)]],
        dtype='float64'
        )

    coordMatrix = [xObj - OriECEF[0]], [yObj - OriECEF[1]], [zObj - OriECEF[2]]

    #Find ENU vector
    ENU = rotationMatrix @ coordMatrix

    #Convert ENU vector to NED vector
    xNorth = ENU[1]
    yEast = ENU[0]
    zDown = -ENU[2]

    posNED = [xNorth], [yEast], [zDown]
    return posNED


def ECItoAER(posECI, stepLength, stepNum, OriECEF, latOri, lonOri):
    '''
    Function for converting ECI position to Azimuth/Elevation/Range 

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posECI: A 3x1 vector containing the x, y, and z ECI positions,
            respectively.

    stepLength: The length of each time step of the simulation.

    stepNum: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    OriECEF: A 1x3 or 3x1 vector containing the origin of the local NED frame
             as xECEF, yECEF, and zECEF respectively.

    latOri: The latitude of the origin of the local NED frame in radians.

    lonOri: The longitude of the origin of the local NED frame in radians.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posAER: A 3x1 vector containing the azimuth, elevation, and range
            positions in radians, respectively.
    '''
    sin = np.sin
    cos = np.cos
    omega = np.float64(7.2921158553e-5) #Earth rotation rate (radians/sec) SIDEREAL
    #omega = 2*pi / (24*60*60)
    

    rotationMatrix = np.array(
        [[cos(stepNum*stepLength*omega), sin(stepNum*stepLength*omega), 0.0],
        [-(sin(stepNum*stepLength*omega)), cos(stepNum*stepLength*omega), 0.0],
        [0.0, 0.0, 1.0]],
        dtype='float64'
        )
    
    posECI = np.reshape(posECI, (3,1))

    #posECEF = np.matmul(rotationMatrix, posECI)
    posECEF = rotationMatrix @ posECI

    transformMatrix = np.array(
        [[(-sin(latOri)*cos(lonOri)), (-sin(latOri)*sin(lonOri)), cos(latOri)],
        [(-sin(lonOri)), cos(lonOri), 0.0],
        [(-cos(latOri)*cos(lonOri)), (-cos(latOri)*sin(lonOri)), (-sin(latOri))]],
        dtype='float64'
        )
    
    # transformMatrix = [(-sin(latOri)*cos(lonOri)), (-sin(latOri)*sin(lonOri)), cos(latOri)], [(-sin(lonOri)), cos(lonOri), 0.0], [(-cos(latOri)*cos(lonOri)), (-cos(latOri)*sin(lonOri)), (-sin(latOri))]
    
    posDelta = posECEF - OriECEF
    
    # posNED = np.matmul(transformMatrix, posDelta)
    posNED = transformMatrix @ posDelta

    #Convert ENU vector to NED vector
    xNorth = np.float64(posNED[0])
    yEast = np.float64(posNED[1])
    zDown = np.float64(posNED[2])


    r1 = np.hypot(xNorth, yEast)
    ran = np.hypot(r1,zDown)
    elevation = np.arctan2(-zDown,r1)
    azimuth = np.mod(np.arctan2(yEast, xNorth),2*np.float64(np.pi))

    posAER = np.array([[azimuth], [elevation], [ran]])
    return posAER


def ECItoECEF(posECI, stepLength, stepNum):
    '''
    Function for converting ECI coordinates to ECEF coordinates

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posECI: A 3x1 vector containing the x, y, and z ECI positions,
            respectively.

    stepLength: The length of each time step of the simulation.

    stepNum: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posECEF: A 3x1 vector containing the x, y, and z ECEF positions,
             respectively.
    '''
    sin = np.sin
    cos = np.cos
    omega = np.float64(7.2921158553e-5) #Earth rotation rate (radians/sec) ~SIDEREAL
    #omega = 2*pi / (24*60*60)
    
    T = np.array(
        [[cos(omega*stepLength*stepNum), sin(omega*stepLength*stepNum), 0.0],
         [-(sin(omega*stepLength*stepNum)), cos(omega*stepLength*stepNum), 0.0],
         [0.0, 0.0, 1.0]],
         dtype='float64'
        )
    posECEF = np.matmul(T, posECI)
    return posECEF


def ECItoLLA(posECI, stepLength, stepNum):
    '''
    Function for converting ECI coordinates to latitude/longitude/altitude,
    using a closed formula set.

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posECI: A 1x3 or 3x1 vector containing the x, y, and z ECEF positions,
            respectively.

    stepLength: The length of each time step of the simulation.

    stepNum: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posLLA: A 3x1 vector containing the latitude, longitude, and altitude
            positions in radians, respectively.
    '''
    sin = np.sin
    cos = np.cos
    WGS = cfg.WGS
    omega = np.float64(7.2921158553e-5) #Earth rotation rate (radians/sec) ~SIDEREAL
    #omega = 2*pi / (24*60*60)

    # Ellipsoid properties
    a = WGS["SemimajorAxis"]            # Semimajor axis
    b = WGS["SemiminorAxis"]            # Semiminor axis
    e = np.sqrt((a**2 - b**2) / a**2)      # Square of (first) eccentricity
    ePrime = np.sqrt((a**2 - b**2) / b**2) # Square of second eccentricity

    rotationMatrix = np.array(
        [[cos(stepNum*stepLength*omega), sin(stepNum*stepLength*omega), 0.0],
         [-sin(stepNum*stepLength*omega), cos(stepNum*stepLength*omega), 0.0],
         [0.0, 0.0, 1.0]],
        dtype='float64'
        )

    posECEF = rotationMatrix @ posECI
    
    xECEF = posECEF[1]
    yECEF = posECEF[2]
    zECEF = posECEF(3)

    #Closed formula set
    p = np.sqrt(xECEF**2+yECEF**2)
    theta = np.arctan2((zECEF * a), (p * b))

    lon = np.arctan2(yECEF,xECEF)
    #lon = mod(lon,2*pi)

    lat = np.arctan2((zECEF + (ePrime**2 * b * (sin(theta))**3)), (p - (e**2 * a * (cos(theta))**3)))

    N = a / (np.sqrt(1 - e**2 * (sin(lat))**2))

    alt = (p / cos(lat)) - N

    posLLA = [lat], [lon], [alt]
    return posLLA


def LLAtoAER(posLLA, OriECEF, latOri, lonOri):
    '''
    Function for converting Az/Elev/Range to Latitude/Longitude/Altitude

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posLLA: A 3x1 vector containing the latitude, longitude, and altitude
            positions in radians, respectively.

    OriECEF: A 1x3 or 3x1 vector containing the origin of the local NED frame
            as xECEF, yECEF, and zECEF respectively.

    latOri: The latitude of the origin of the local NED frame in radians.

    lonOri: The longitude of the origin of the local NED frame in radians.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posAER: A 1x3 or 3x1 vector containing the Azimuth, Elevation, and Range
        positions in radians, respectively.
    '''
    sin = np.sin
    cos = np.cos
    WGS = cfg.WGS

    #Ellipsoid parameters
    a = WGS["SemimajorAxis"]
    b = WGS["SemiminorAxis"]
    e = WGS["Eccentricity"]


    lat = posLLA[0]
    lon = posLLA[1]
    alt = posLLA[2]

    #Prime vertical radius of curvature N(phi)
    #Formula if eccentricty not defined: NPhi = a**2 / (sqrt((a**2*(cos(lat)**2))+(b**2*(sin(lat)**2))))

    NPhi = a / (np.sqrt(1 - (e**2*(sin(lat))**2)))
    xECEF = (NPhi + alt) * cos(lat) * cos(lon)
    yECEF = (NPhi + alt) * cos(lat) * sin(lon)
    zECEF = (((b**2/a**2)*NPhi) + alt)*sin(lat)

    #Generate matrices for multiplication
    rotationMatrix = np.array(
        [[-sin(lonOri), cos(lonOri), 0],
         [(-sin(latOri)*cos(lonOri)), (-sin(latOri)*sin(lonOri)), cos(latOri)],
         [(cos(latOri)*cos(lonOri)), (cos(latOri)*sin(lonOri)), sin(latOri)]],
        dtype='float64'
        )

    coordMatrix = np.array(
        [[xECEF - OriECEF[0]],
         [yECEF - OriECEF[1]],
         [zECEF - OriECEF[2]]],
        dtype='float64'
        )

    #Find ENU vector
    ENU = rotationMatrix @ coordMatrix

    #Convert ENU vector to NED vector
    xNorth = ENU[1]
    yEast = ENU[0]
    zDown = -ENU[2]


    r1 = np.hypot(xNorth, yEast)
    ran = np.hypot(r1,zDown)
    elevation = np.arctan2(-zDown,r1)
    azimuth = np.mod(np.arctan2(yEast, xNorth),2*np.pi)

    posAER = [azimuth], [elevation], [ran]
    return posAER

def LLAtoECEF(posLLA):
    '''
    Function for converting ECEF coordinates to latitude/longitude/altitude.

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posLLA: A 1x3 or 3x1 vector containing the latitude, longitude, and 
            altitude positions in radians, respectively.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posECEF: A  3x1 vector containing the x, y, and z ECEF positions,
             respectively.
    '''
    sin = np.sin
    cos = np.cos
    WGS = cfg.WGS

    #Ellipsoid parameters
    a = WGS["SemimajorAxis"]
    b = WGS["SemiminorAxis"]
    e = WGS["Eccentricity"]


    lat = posLLA[0]
    lon = posLLA[1]
    alt = posLLA[2]

    #Prime vertical radius of curvature N(phi)
    #Formula if eccentricty not defined: NPhi = a**2 / (sqrt((a**2*(cos(lat)**2))+(b**2*(sin(lat)**2))))

    NPhi = a / (np.sqrt(1 - (e**2*(sin(lat))**2)))
    xECEF = (NPhi + alt) * cos(lat) * cos(lon)
    yECEF = (NPhi + alt) * cos(lat) * sin(lon)
    zECEF = (((b**2/a**2)*NPhi) + alt)*sin(lat)

    posECEF = np.array([[xECEF],
                        [yECEF],
                        [zECEF]])
    return posECEF


def LLAtoECI(posLLA, stepLength, stepNum):
    '''
    Function for converting latitude/longitude/altitude to ECI coordinates.

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posLLA: A 1x3 or 3x1 vector containing the latitude, longitude, and 
            altitude positions in radians, respectively.

    stepLength: The length of each time step of the simulation.

    stepNum: The current step number of the simulation. This works with step
             length to convert increasing steps through the simulation.

    WGS:  The WGS84 reference ellipsoid

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posECI: A  3x1 vector containing the x, y, and z ECI positions,
            respectively.
    '''
    sin = np.sin
    cos = np.cos
    WGS = cfg.WGS
    omega = np.float64(7.2921158553e-5) #Earth rotation rate (radians/sec) ~SIDEREAL

    #Ellipsoid parameters
    a = WGS["SemimajorAxis"]
    b = WGS["SemiminorAxis"]
    e = WGS["Eccentricity"]


    lat = posLLA[0]
    lon = posLLA[1]
    alt = posLLA[2]

    #Prime vertical radius of curvature N(phi)
    #Formula if eccentricty not defined: NPhi = a**2 / (sqrt((a**2*(cos(lat)**2))+(b**2*(sin(lat)**2))))

    NPhi = a / (np.sqrt(1 - (e**2*(sin(lat))**2)))
    xECEF = (NPhi + alt) * cos(lat) * cos(lon)
    yECEF = (NPhi + alt) * cos(lat) * sin(lon)
    zECEF = (((b**2/a**2)*NPhi) + alt)*sin(lat)

    posECEF = [xECEF], [yECEF], [zECEF]

    #Generate matrices for multiplication
    rotationMatrix = np.array(
        [[cos(stepNum*stepLength*omega), -sin(stepNum*stepLength*omega), 0.0],
         [sin(stepNum*stepLength*omega), cos(stepNum*stepLength*omega), 0.0],
         [0.0, 0.0, 1.0]],
        dtype='float64'
        )
                
    posECI = rotationMatrix @ posECEF
    
    return posECI


def NEDtoAER(posNED):
    '''
    Function for converting local North/East/Down to Az/Elev/Range

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posNED: A 1x3 or 3x1 vector containing the north, east, and down positions,
            respectively.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posAER: A 3x1 vector containing the Azimuth, Elevation, and Range
        positions in radians, respectively.
    '''
    
    xNorth = posNED[0]
    yEast = posNED[1]
    zDown = posNED[2]

    r1 = np.hypot(xNorth, yEast)
    ran = np.hypot(r1,zDown)
    elevation = np.arctan2(-zDown,r1)
    azimuth = np.mod(np.arctan2(yEast, xNorth),2*np.pi)

    posAER = [azimuth], [elevation], [ran]
    return posAER

def NEDtoECEF(posNED, OriECEF, latOri, lonOri):
    '''
    Function for converting ECEF coordinates to local NED coordinates

    ~~~~~~~~~~~~~~~~~INPUTS~~~~~~~~~~~~
    posNED: A 1x3 or 3x1 vector containing the north, east, and down positions,
            respectively.

    OriECEF: A 1x3 or 3x1 vector containing the origin of the local NED frame
             as xECEF, yECEF, and zECEF respectively.

    latOri: The latitude of the origin of the local NED frame in radians.

    lonOri: The longitude of the origin of the local NED frame in radians.

    ~~~~~~~~~~~~~~~OUTPUTS~~~~~~~~~~~~
    posECEF: A 3x1 vector containing the x, y, and z ECEF positions,
             respectively.
    '''
    sin = np.sin
    cos = np.cos

    cosPhi = cos(latOri)
    sinPhi = sin(latOri)
    cosLambda = cos(lonOri)
    sinLambda = sin(lonOri)

    xNorth = posNED[0]
    yEast = posNED[1]
    zDown = posNED[2]

    t = cosPhi * -zDown - sinPhi * xNorth
    dz = sinPhi * -zDown + cosPhi * xNorth

    dx = cosLambda * t - sinLambda * yEast
    dy = sinLambda * t + cosLambda * yEast

    xECEF = OriECEF[0] + dx
    yECEF = OriECEF[1] + dy
    zECEF = OriECEF[2] + dz

    posECEF = [xECEF], [yECEF], [zECEF]
    return posECEF