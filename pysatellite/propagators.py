import numpy as np
from pysatellite import functions, config as cfg

stump_c = functions.stump_c
stump_s = functions.stump_s

def kepler(x_state, t=cfg.stepLength, *args):
    # t = cfg.stepLength
    mu = cfg.mu

    r0 = np.linalg.norm(x_state[0:3])
    v0 = np.linalg.norm(x_state[3:6])

    pos0 = x_state[0:3]
    vel0 = x_state[3:6]

    # Initial radial velocity
    vr0 = np.dot(np.reshape(pos0, 3), np.reshape(vel0, 3)) / r0

    # Reciprocal of the semi-major axis (from the energy equation)
    alpha = 2.0 / r0 - v0 ** 2 / mu

    error = 1e-8
    n_max = 1000

    x = np.sqrt(mu) * np.abs(alpha) * t

    n = 0
    ratio = 1
    while np.abs(ratio) > error and n <= n_max:
        n += 1
        c = stump_c(alpha * x ** 2)
        s = stump_s(alpha * x ** 2)
        f = r0 * vr0 / np.sqrt(mu) * x ** 2 * c + (1 - alpha * r0) * x ** 3 * s + r0 * x - np.sqrt(mu) * t
        dfdx = r0 * vr0 / np.sqrt(mu) * x * (1 - alpha * x ** 2 * s) + (1 - alpha * r0) * x ** 2 * c + r0
        ratio = f / dfdx
        x -= ratio

    # if n > n_max:
    #     print('\n No. Iterations of Kepler''s equation = %g', n)
    #     print('\n F/dFdx = %g', f/dfdx)

    z = alpha * x ** 2

    f = 1 - x ** 2 / r0 * stump_c(z)

    g = t - 1 / np.sqrt(mu) * x ** 3 * stump_s(z)

    r = f * pos0 + g * vel0

    r_norm = np.linalg.norm(r)

    f_dot = np.sqrt(mu) / r_norm / r0 * (z * stump_s(z) - 1) * x

    gdot = 1 - x ** 2 / r_norm * stump_c(z)

    v = f_dot * pos0 + gdot * vel0

    x_state = np.concatenate((r, v))
    return x_state