import numpy as np

from space_traj_opt.math.constants import MU_EARTH


def rv_to_orbital_elements(r_vec, v_vec, mu = MU_EARTH):
    """
    Convert position and velocity vectors to classical orbital elements.

    Parameters
    ----------
    r_vec : array_like, shape (..., 3)
        Position vector or batch of position vectors [m]
    v_vec : array_like, shape (..., 3)
        Velocity vector or batch of velocity vectors [m/s]
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

    r = np.linalg.norm(r_vec, axis=-1)
    v = np.linalg.norm(v_vec, axis=-1)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec, axis=-1)

    # Eccentricity vector
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r[..., np.newaxis]
    e = np.linalg.norm(e_vec, axis=-1)

    # Inclination
    i = np.arccos(h_vec[..., 2] / h)

    # Node vector
    k_vec = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec, axis=-1)

    # Specific orbital energy
    energy = 0.5 * v**2 - mu / r

    # Semi-major axis
    non_parabolic = np.abs(e - 1.0) > 1e-12
    a = np.full(np.shape(energy), np.inf, dtype=float)
    np.divide(-mu, 2.0 * energy, out=a, where=non_parabolic)

    # RAAN
    has_node = n > 1e-12
    raan = np.where(
        has_node,
        np.arctan2(n_vec[..., 1], n_vec[..., 0]) % (2.0 * np.pi),
        0.0,
    )

    # Argument of periapsis
    has_periapsis = has_node & (e > 1e-12)
    argp_y = np.zeros_like(e, dtype=float)
    argp_x = np.zeros_like(e, dtype=float)
    np.divide(
        np.sum(np.cross(n_vec, e_vec) * h_vec, axis=-1),
        n * e * h,
        out=argp_y,
        where=has_periapsis,
    )
    np.divide(
        np.sum(n_vec * e_vec, axis=-1),
        n * e,
        out=argp_x,
        where=has_periapsis,
    )
    argp = np.where(
        has_periapsis,
        np.arctan2(argp_y, argp_x) % (2.0 * np.pi),
        0.0,
    )

    # True anomaly
    has_eccentricity = e > 1e-12
    nu_y = np.zeros_like(e, dtype=float)
    nu_x = np.zeros_like(e, dtype=float)
    np.divide(
        np.sum(np.cross(e_vec, r_vec) * h_vec, axis=-1),
        e * h * r,
        out=nu_y,
        where=has_eccentricity,
    )
    np.divide(
        np.sum(e_vec * r_vec, axis=-1),
        e * r,
        out=nu_x,
        where=has_eccentricity,
    )
    nu = np.where(
        has_eccentricity,
        np.arctan2(nu_y, nu_x) % (2.0 * np.pi),
        0.0,
    )

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
