"""
Benedict Oakes
Created 10/06/2021
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg

if __name__ == "__main__":

    start_nonKF = time.time()
    # ~~~~ Variables
    
    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)

    sensLat = np.float64(28.300697)
    sensLon = np.float64(-16.509675)
    sensAlt = np.float64(2390)
    sensLLA = np.array([[sensLat * pi/180], [sensLon * pi/180], [sensAlt]], dtype='float64')
    # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
    sensECEF = Transformations.lla_to_ecef(sensLLA)
    sensECEF.shape = (3, 1)

    simLength = cfg.simLength
    stepLength = cfg.stepLength

    satRadius = np.float64(7e6)
    mu = cfg.mu
    TOrbit = 2*pi * np.sqrt(satRadius**3/mu)
    omegaSat = 2*pi/TOrbit

    # ~~~~ Satellite Conversion 
    
    satECI = np.zeros((3, simLength))
    for count in range(simLength):
        satECI[:, count:count+1] = np.array([[satRadius*sin(omegaSat*(count+1)*stepLength)],
                                            [0.0],
                                            [satRadius*cos(omegaSat*(count+1)*stepLength)]],
                                            dtype='float64'
                                            )
        
    satAER = np.zeros((3, simLength))
    for count in range(simLength):
        satAER[:, count:count+1] = Transformations.eci_to_aer(satECI[:, count], stepLength, count+1, sensECEF,
                                                              sensLLA[0], sensLLA[1])

    angMeasDev = np.float64(1e-6)
    rangeMeasDev = np.float64(20)
    # angMeasDev = 0
    # rangeMeasDev = 0
    satAERMes = np.zeros((3, simLength))
    satAERMes[0, :] = satAER[0, :] + (angMeasDev*np.random.randn(1, simLength))
    satAERMes[1, :] = satAER[1, :] + (angMeasDev*np.random.randn(1, simLength))
    satAERMes[2, :] = satAER[2, :] + (rangeMeasDev*np.random.randn(1, simLength))

    # ~~~~ Convert back to ECI
    
    satECIMes = np.zeros((3, simLength))
    for count in range(simLength):
        satECIMes[:, [count]] = Transformations.aer_to_eci(satAERMes[:, count], stepLength, count+1, sensECEF,
                                                           sensLLA[0], sensLLA[1])

    # ~~~~ Temp ECI measurements from MATLAB
    
    #  satECIMes = pd.read_csv('ECI_mes.txt', delimiter=' ').to_numpy(dtype='float64')
    # # satECIMes.to_numpy(dtype='float64')
    #  satECIMes = satECIMes.T
    # # np.reshape(satECIMes, (3,simLength))

    # ~~~~ KF Matrices
    
    # Initialise state vector
    # (x, y, z, v_x, v_y, v_z)
    xState = np.array([[0.0],
                       [0.0],
                       [satRadius],
                       [0.0],
                       [0.0],
                       [0.0]],
                      dtype='float64')
    
    G = np.float64(6.67e-11)
    m_e = np.float64(5.972e24)
    
    xState[3:6] = np.sqrt((G*m_e) / np.linalg.norm(xState[0:3])) * np.array([[1.0], [0.0], [0.0]], dtype='float64')
    
    # Process noise
    stdAng = np.float64(1e-5)
    coefA = np.float64(0.25 * stepLength**4.0 * stdAng**2.0)
    coefB = np.float64(stepLength**2.0 * stdAng**2.0)
    coefC = np.float64(0.5 * stepLength**3.0 * stdAng**2.0)
    
    procNoise = np.array([[coefA, 0, 0, coefC, 0, 0],
                          [0, coefA, 0, 0, coefC, 0],
                          [0, 0, coefA, 0, 0, coefC],
                          [coefC, 0, 0, coefB, 0, 0],
                          [0, coefC, 0, 0, coefB, 0],
                          [0, 0, coefC, 0, 0, coefB]],
                         dtype='float64'
                         )
    
    covState = np.float64(1e10) * np.identity(6)
    
    covAER = np.array([[(angMeasDev * 180/pi)**2, 0, 0],
                       [0, (angMeasDev * 180/pi)**2, 0],
                       [0, 0, rangeMeasDev**2]],
                      dtype='float64'
                      )
    
    measureMatrix = np.append(np.identity(3), np.zeros((3, 3)), axis=1)
    
    totalStates = np.zeros((6, simLength), dtype='float64')
    diffStates = np.zeros((3, simLength), dtype='float64')
    err_X_ECI = np.zeros((1, simLength), dtype='float64')
    err_Y_ECI = np.zeros((1, simLength), dtype='float64')
    err_Z_ECI = np.zeros((1, simLength), dtype='float64')
    # err_X_ECI = []
    # err_Y_ECI = []
    # err_Z_ECI = []
    
    end_nonKF = time.time()
    nonKFtime = end_nonKF - start_nonKF
    
    # ~~~~ Using EKF
    
    start_KF = time.time()
    
    delta = np.float64(1e-6)
    for count in range(simLength):
        # Func params
        func_params = {
            "stepLength": stepLength,
            "count": count+1,
            "sensECEF": sensECEF,
            "sensLLA[0]": sensLLA[0],
            "sensLLA[1]": sensLLA[1]
            }
        
        jacobian = Functions.jacobian_finder("AERtoECI", np.reshape(satAERMes[:, count], (3, 1)), func_params, delta)
        
        # covECI = np.matmul(np.matmul(jacobian, covAER), jacobian.T)
        covECI = jacobian @ covAER @ jacobian.T
        
        stateTransMatrix = Functions.jacobian_finder("kepler", xState, [], delta)
        
        xState, covState = Filters.ekf_eci(xState, covState, satECIMes[:, count], stateTransMatrix, measureMatrix,
                                           covECI, procNoise)
        
        totalStates[:, count] = np.reshape(xState, 6)
        err_X_ECI[0, count] = (np.sqrt(np.abs(covState[0, 0])))
        err_Y_ECI[0, count] = (np.sqrt(np.abs(covState[1, 1])))
        err_Z_ECI[0, count] = (np.sqrt(np.abs(covState[2, 2])))
        diffStates[:, count] = totalStates[0:3, count] - satECIMes[:, count]
        
    end_KF = time.time()
    KFtime = end_KF - start_KF
        
    # ~~~~~ Plots
    start_plots = time.time()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    axs = [ax1, ax2, ax3]
    fig.suptitle('Satellite Position')
    ax1.plot(satECI[0, :])
    # ax1.plot(satECIMes[0,:], 'r.')
    ax1.plot(totalStates[0, :])
    ax1.set(ylabel='$X_{ECI}$, metres')
    
    ax2.plot(satECI[1, :])
    # ax2.plot(satECIMes[1,:], 'r.')
    ax2.plot(totalStates[1, :])
    ax2.set(ylabel='$Y_{ECI}$, metres')
    
    ax3.plot(satECI[2, :])
    # ax3.plot(satECIMes[2,:], 'r.')
    ax3.plot(totalStates[2, :])
    ax3.set(xlabel='Time Step', ylabel='$Z_{ECI}$, metres')
    
    plt.show()
    
    # ~~~~~ Error Plots
    plt.figure()
    plt.plot(err_X_ECI[0, :])
    plt.plot(np.abs(diffStates[0, :]))
    plt.xlabel('Time Step'), plt.ylabel('$X_{ECI}$, metres')
    plt.show()
    
    plt.figure()
    plt.plot(err_Y_ECI[0, :])
    plt.plot(np.abs(diffStates[1, :]))
    plt.xlabel('Time Step'), plt.ylabel('$Y_{ECI}$, metres')
    plt.show()
    
    plt.figure()
    plt.plot(err_Z_ECI[0, :])
    plt.plot(np.abs(diffStates[2, :]))
    plt.xlabel('Time Step'), plt.ylabel('$Z_{ECI}$, metres')
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(satECI[0, :], satECI[1, :], satECI[2, :])
    ax.plot(totalStates[0, :], totalStates[1, :], totalStates[2, :])
    
    end_plots = time.time()
    plots_time = end_plots - start_plots
    
    total_time = nonKFtime + KFtime + plots_time
    
    print("Non KF Time: {:.4f}".format(nonKFtime))
    print('KF Time: {:.4f}'.format(KFtime))
    print('Plot Time: {:.4f}'.format(plots_time))
    print('Total Time: {:.4f}'.format(total_time))
    
    
    