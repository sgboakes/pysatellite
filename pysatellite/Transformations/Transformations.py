# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:00:45 2021

@author: ben-o_000
"""

import numpy as np


def AERtoECI(posAER, latOri, lonOri, OriECEF, stepLength, stepNum):
    '''
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
    '''
    omega = np.float64(7.2921158553e-5) #Earth rotation rate (radians/sec) ~SIDEREAL
    sin = np.sin
    cos = np.cos

    az = posAER(0)
    elev = posAER(1)
    range = posAER(2)

    # if abs(az-2*pi) < 1e-5 || abs(az-pi) < 1e-5 || abs(pi) < 1e-5
    #     posLLA = [NaN NaN NaN]
    #     return
    # elseif abs(elev-2*pi) < 1e-5 || abs(elev-pi) < 1e-5 || abs(pi) < 1e-5
    #     posLLA = [NaN NaN NaN]
    #     return
    # end

    zUp = range * sin(elev)
    r   = range * cos(elev)
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

    xECEF = OriECEF(0) + dx
    yECEF = OriECEF(1) + dy
    zECEF = OriECEF(2) + dz

    posECEF = [xECEF], [yECEF], [zECEF]

    

    #Generate matrices for multiplication
    rotationMatrix = [cos(stepNum*stepLength*omega), -sin(stepNum*stepLength*omega), 0], [sin(stepNum*stepLength*omega), cos(stepNum*stepLength*omega), 0], [0, 0, 1 ]
                
    posECI = np.matmul(rotationMatrix,posECEF)
    return posECI


def AERtoLLA(posAER, OriECEF, latOri, lonOri, WGS):
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

    az = posAER(0)
    elev = posAER(1)
    range = posAER(2)

    zUp = range * sin(elev)
    r   = range * cos(elev)
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

    xECEF = OriECEF(0) + dx
    yECEF = OriECEF(1) + dy
    zECEF = OriECEF(2) + dz


    # Ellipsoid properties
    a = WGS.SemimajorAxis # Semimajor axis
    b = WGS.SemiminorAxis # Semiminor axis
    f = WGS.Flattening    # Flattening
    e2 = f * (2 - f)      # Square of (first) eccentricity
    ep2 = e2 / (1 - e2)   # Square of second eccentricity
    e = np.sqrt((a^2 - b^2) / a^2)
    ePrime = np.sqrt((a^2 - b^2) / b^2)


    #Closed formula set
    p = np.sqrt(xECEF^2+yECEF^2)
    theta = np.arctan2((zECEF * a), (p * b))

    lon = np.arctan2(yECEF,xECEF)
    #lon = mod(lon,2*pi)
    lat = np.arctan2((zECEF + (ePrime^2 * b * (sin(theta))^3)), (p - (e^2 * a * (cos(theta))^3)))
    N = a / (np.sqrt(1 - e^2 * (sin(lat))^2))
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

    az = posAER(0)
    elev = posAER(1)
    range = posAER(2)

    zUp = range * sin(elev)
    r   = range * cos(elev)
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

    T = [cos(omega*stepLength*stepNum), -(sin(omega*stepLength*stepNum)), 0], [sin(omega*stepLength*stepNum), cos(omega*stepLength*stepNum), 0], [0, 0, 1]
    
    posECI = np.matmul(T, posECEF)
    return posECI


def ECEFtoLLA(posECEF, WGS):
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

    # Ellipsoid properties
    a = WGS.SemimajorAxis            # Semimajor axis
    b = WGS.SemiminorAxis            # Semiminor axis
    e = np.sqrt((a^2 - b^2) / a^2)      # Square of (first) eccentricity
    ePrime = np.sqrt((a^2 - b^2) / b^2) # Square of second eccentricity

    xECEF = posECEF(0)
    yECEF = posECEF(1)
    zECEF = posECEF(2)


    #Closed formula set
    p = np.sqrt(xECEF^2+yECEF^2)
    theta = np.arctan2((zECEF * a), (p * b))

    lon = np.arctan2(yECEF,xECEF)
    #lon = mod(lon,2*pi)

    lat = np.arctan2((zECEF + (ePrime^2 * b * (sin(theta))^3)), (p - (e^2 * a * (cos(theta))^3)))

    N = a / (np.sqrt(1 - e^2 * (sin(lat))^2))

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

    xObj = posECEF(0)
    yObj = posECEF(1)
    zObj = posECEF(2)


    #Generate matrices for multiplication
    rotationMatrix = [-(sin(lonOri)), cos(lonOri), 0], [(-(sin(latOri))*cos(lonOri)), (-(sin(latOri))*sin(lonOri)), cos(latOri)], [(cos(latOri)*cos(lonOri)), (cos(latOri)*sin(lonOri)), sin(latOri)]

    coordMatrix = [xObj - OriECEF(0)], [yObj - OriECEF(1)], [zObj - OriECEF(2)]

    #Find ENU vector
    ENU = np.matmul(rotationMatrix, coordMatrix)

    #Convert ENU vector to NED vector
    xNorth = ENU(1)
    yEast = ENU(0)
    zDown = -ENU(2)

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
    

    rotationMatrix = [cos(stepNum*stepLength*omega), sin(stepNum*stepLength*omega), 0], [-(sin(stepNum*stepLength*omega)), cos(stepNum*stepLength*omega), 0], [0, 0, 1 ]

    posECEF = np.matmul(rotationMatrix, posECI)

    #~~Possible check for rounding errors at posistions close to ECI axes
    # if abs(posECEF(1)) < 1e-5
    #     posAER = [NaN NaN NaN]
    #     return
    # elseif abs(posECEF(2)) < 1e-5
    #     posAER = [NaN NaN NaN]
    #     return
    # end


    xObj = posECEF(0)
    yObj = posECEF(1)
    zObj = posECEF(2)

    transformMatrix = [-sin(lonOri), cos(lonOri), 0], [(-sin(latOri)*cos(lonOri)), (-sin(latOri)*sin(lonOri)), cos(latOri)], [(cos(latOri)*cos(lonOri)), (cos(latOri)*sin(lonOri)), sin(latOri)]


    posECEFMatrix = [xObj - OriECEF(0)], [yObj - OriECEF(1)], [zObj - OriECEF(2)]

    #Find ENU vector
    ENU = transformMatrix * posECEFMatrix
    ENU = np.matmul(transformMatrix, posECEFMatrix)

    #Convert ENU vector to NED vector
    xNorth = ENU(1)
    yEast = ENU(0)
    zDown = -ENU(2)


    r1 = np.hypot(xNorth, yEast)
    range = np.hypot(r1,zDown)
    elevation = np.arctan2(-zDown,r1)
    azimuth = np.mod(np.arctan2(yEast, xNorth),2*np.pi)

    posAER = [azimuth], [elevation], [range]
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
    
    T = [cos(omega*stepLength*stepNum), sin(omega*stepLength*stepNum), 0], [-(sin(omega*stepLength*stepNum)), cos(omega*stepLength*stepNum), 0], [0, 0, 1]
    posECEF = np.matmul(T, posECI)
    return posECEF


def ECItoLLA(posECI, stepLength, stepNum, WGS):
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
    omega = np.float64(7.2921158553e-5) #Earth rotation rate (radians/sec) ~SIDEREAL
    #omega = 2*pi / (24*60*60)

    # Ellipsoid properties
    a = WGS.SemimajorAxis            # Semimajor axis
    b = WGS.SemiminorAxis            # Semiminor axis
    e = np.sqrt((a^2 - b^2) / a^2)      # Square of (first) eccentricity
    ePrime = np.sqrt((a^2 - b^2) / b^2) # Square of second eccentricity

    rotationMatrix = [cos(stepNum*stepLength*omega), sin(stepNum*stepLength*omega), 0], [-sin(stepNum*stepLength*omega), cos(stepNum*stepLength*omega), 0], [0, 0, 1 ]

    posECEF = np.matmul(rotationMatrix,posECI)
    xECEF = posECEF(1)
    yECEF = posECEF(2)
    zECEF = posECEF(3)

    #Closed formula set
    p = np.sqrt(xECEF^2+yECEF^2)
    theta = np.arctan2((zECEF * a), (p * b))

    lon = np.arctan2(yECEF,xECEF)
    #lon = mod(lon,2*pi)

    lat = np.arctan2((zECEF + (ePrime^2 * b * (sin(theta))^3)), (p - (e^2 * a * (cos(theta))^3)))

    N = a / (np.sqrt(1 - e^2 * (sin(lat))^2))

    alt = (p / cos(lat)) - N

    posLLA = [lat], [lon], [alt]
    return posLLA


def LLAtoAER(posLLA, OriECEF, latOri, lonOri, WGS):
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

    #Ellipsoid parameters
    a = WGS.SemimajorAxis
    b = WGS.SemiminorAxis
    e = WGS.Eccentricity


    lat = posLLA(0)
    lon = posLLA(1)
    alt = posLLA(2)

    #Prime vertical radius of curvature N(phi)
    #Formula if eccentricty not defined: NPhi = a^2 / (sqrt((a^2*(cos(lat)^2))+(b^2*(sin(lat)^2))))

    NPhi = a / (np.sqrt(1 - (e^2*(sin(lat))^2)))
    xECEF = (NPhi + alt) * cos(lat) * cos(lon)
    yECEF = (NPhi + alt) * cos(lat) * sin(lon)
    zECEF = (((b^2/a^2)*NPhi) + alt)*sin(lat)

    #Generate matrices for multiplication
    rotationMatrix = [-sin(lonOri), cos(lonOri), 0], [(-sin(latOri)*cos(lonOri)), (-sin(latOri)*sin(lonOri)), cos(latOri)], [(cos(latOri)*cos(lonOri)), (cos(latOri)*sin(lonOri)), sin(latOri)]

    coordMatrix = [xECEF - OriECEF(0)], [yECEF - OriECEF(1)], [zECEF - OriECEF(2)]

    #Find ENU vector
    ENU = rotationMatrix * coordMatrix

    #Convert ENU vector to NED vector
    xNorth = ENU(1)
    yEast = ENU(0)
    zDown = -ENU(2)


    r1 = np.hypot(xNorth, yEast)
    range = np.hypot(r1,zDown)
    elevation = np.arctan2(-zDown,r1)
    azimuth = np.mod(np.arctan2(yEast, xNorth),2*np.pi)

    posAER = [azimuth], [elevation], [range]
    return posAER

def LLAtoECEF(posLLA, WGS):
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

    #Ellipsoid parameters
    a = WGS.SemimajorAxis
    b = WGS.SemiminorAxis
    e = WGS.Eccentricity


    lat = posLLA(0)
    lon = posLLA(1)
    alt = posLLA(2)

    #Prime vertical radius of curvature N(phi)
    #Formula if eccentricty not defined: NPhi = a^2 / (sqrt((a^2*(cos(lat)^2))+(b^2*(sin(lat)^2))))

    NPhi = a / (np.sqrt(1 - (e^2*(sin(lat))^2)))
    xECEF = (NPhi + alt) * cos(lat) * cos(lon)
    yECEF = (NPhi + alt) * cos(lat) * sin(lon)
    zECEF = (((b^2/a^2)*NPhi) + alt)*sin(lat)

    posECEF = [xECEF], [yECEF], [zECEF]
    return posECEF


def LLAtoECI(posLLA, stepLength, stepNum, WGS):
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
    omega = np.float64(7.2921158553e-5) #Earth rotation rate (radians/sec) ~SIDEREAL

    #Ellipsoid parameters
    a = WGS.SemimajorAxis
    b = WGS.SemiminorAxis
    e = WGS.Eccentricity


    lat = posLLA(0)
    lon = posLLA(1)
    alt = posLLA(2)

    #Prime vertical radius of curvature N(phi)
    #Formula if eccentricty not defined: NPhi = a^2 / (sqrt((a^2*(cos(lat)^2))+(b^2*(sin(lat)^2))))

    NPhi = a / (np.sqrt(1 - (e^2*(sin(lat))^2)))
    xECEF = (NPhi + alt) * cos(lat) * cos(lon)
    yECEF = (NPhi + alt) * cos(lat) * sin(lon)
    zECEF = (((b^2/a^2)*NPhi) + alt)*sin(lat)

    posECEF = [xECEF], [yECEF], [zECEF]

    #Generate matrices for multiplication
    rotationMatrix = [cos(stepNum*stepLength*omega), -sin(stepNum*stepLength*omega), 0], [sin(stepNum*stepLength*omega), cos(stepNum*stepLength*omega), 0], [0, 0, 1 ]
                
    posECI = np.matmul(rotationMatrix,posECEF)
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
    
    xNorth = posNED(0)
    yEast = posNED(1)
    zDown = posNED(2)

    r1 = np.hypot(xNorth, yEast)
    range = np.hypot(r1,zDown)
    elevation = np.arctan2(-zDown,r1)
    azimuth = np.mod(np.arctan2(yEast, xNorth),2*np.pi)

    posAER = [azimuth], [elevation], [range]
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

    xNorth = posNED(0)
    yEast = posNED(1)
    zDown = posNED(2)

    t = cosPhi * -zDown - sinPhi * xNorth
    dz = sinPhi * -zDown + cosPhi * xNorth

    dx = cosLambda * t - sinLambda * yEast
    dy = sinLambda * t + cosLambda * yEast

    xECEF = OriECEF(0) + dx
    yECEF = OriECEF(1) + dy
    zECEF = OriECEF(2) + dz

    posECEF = [xECEF], [yECEF], [zECEF]
    return posECEF