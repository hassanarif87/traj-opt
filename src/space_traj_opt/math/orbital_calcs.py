import numpy as np

from space_traj_opt.math.constants import MU_EARTH


def rv_to_orbital_elements(r_vec, v_vec, mu = MU_EARTH):
    """
    Convert position and velocity vectors to classical orbital elements.

    Parameters
    ----------
    r_vec : array_like, shape (3,)
        Position vector [m]
    v_vec : array_like, shape (3,)
        Velocity vector [m/s]
    mu : float
        Gravitational parameter [m^3/s^2]

    Returns
    -------
        a      : semi-major axis [m]
        e      : eccentricity
        i      : inclination [rad]
        raan   : right ascension of ascending node [rad]
        argp   : argument of periapsis [rad]
        nu     : true anomaly [rad]
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Eccentricity vector
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)

    # Inclination
    i = np.arccos(h_vec[2] / h)

    # Node vector
    k_vec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)

    # Specific orbital energy
    energy = 0.5 * v**2 - mu / r

    # Semi-major axis
    if abs(e - 1.0) > 1e-12:
        a = -mu / (2.0 * energy)
    else:
        a = np.inf

    # RAAN
    if n > 1e-12:
        raan = np.arctan2(n_vec[1], n_vec[0]) % (2.0 * np.pi)
    else:
        raan = 0.0

    # Argument of periapsis
    if n > 1e-12 and e > 1e-12:
        argp = np.arctan2(
            np.dot(np.cross(n_vec, e_vec), h_vec) / (n * e * h),
            np.dot(n_vec, e_vec) / (n * e)
        ) % (2.0 * np.pi)
    else:
        argp = 0.0

    # True anomaly
    if e > 1e-12:
        nu = np.arctan2(
            np.dot(np.cross(e_vec, r_vec), h_vec) / (e * h * r),
            np.dot(e_vec, r_vec) / (e * r)
        ) % (2.0 * np.pi)
    else:
        nu = 0.0

    return a, e, i, raan, argp, nu


def rv_to_aei(r_vec, v_vec, mu=MU_EARTH):
    """
    Convert position and velocity vectors to classical orbital elements.

    Parameters
    ----------
    r_vec : array_like, shape (3,)
        Position vector [m]
    v_vec : array_like, shape (3,)
        Velocity vector [m/s]
    mu : float
        Gravitational parameter [m^3/s^2]

    Returns
    -------
        a      : semi-major axis [m]
        e      : eccentricity
        i      : inclination [rad]
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Eccentricity vector
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)

    # Inclination
    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
    
    # Specific orbital energy
    energy = 0.5 * v**2 - mu / r

    # Semi-major axis
    a = -mu / (2.0 * energy)

    return a,e ,i 
