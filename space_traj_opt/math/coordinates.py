import numpy as np
from .constants import OMEGA_EARTH, SMA_EARTH, ECCEN_EARTH_SQ
from .quaternion import quat_conj, q_from_axisangle, q_mult

def eci2ecef(r_eci, t):
    """Convert from ECI to ECEF coordinates

    Args:
        r_eci : position vector in ECI frame
        t : time in seconds since epoch sim t =0
    Returns:
        r_ecef : position vector in ECEF frame
    """
    theta = OMEGA_EARTH * t
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    r_ecef = np.array([
        cos_theta * r_eci[0] + sin_theta * r_eci[1],
        -sin_theta * r_eci[0] + cos_theta * r_eci[1],
        r_eci[2]
    ])
    return r_ecef

def ecef2eci(r_ecef, t):
    """Convert from ECEF to ECI coordinates

    Args:
        r_ecef : position vector in ECEF frame
        t : time in seconds since epoch sim t =0
    Returns:
        r_eci : position vector in ECI frame
    """
    theta = OMEGA_EARTH * t
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    r_eci = np.array([
        cos_theta * r_ecef[0] - sin_theta * r_ecef[1],
        sin_theta * r_ecef[0] + cos_theta * r_ecef[1],
        r_ecef[2]
    ])
    return r_eci

def ecef2lla(r_ecef):
    """Convert from ECEF to Latitude, Longitude, Altitude

    References
    ----------
    .. Jekeli, C.,"Inertial Navigation Systems With Geodetic
       Applications", Walter de Gruyter, New York, 2001, pp. 24

    Args:
        r_ecef : position vector in ECEF frame
    Returns:
        lat : latitude in radians
        lon : longitude in radians
        alt : altitude in meters
    """ 
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    # Horizontal distance from the z-axis
    p = np.sqrt(x**2 + y**2)
    # Initial Lat guess
    lat = np.arctan2(z, p * (1 - ECCEN_EARTH_SQ))
    err = 1.0
    alt = 0.0

    while abs(err) > 1e-10:
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)

        N = SMA_EARTH / np.sqrt(1 - ECCEN_EARTH_SQ*sin_lat*sin_lat)

        # Two altitude formulas
        h1 = p / cos_lat - N
        # Use alternate formula near the poles to avoid division by cos(lat) ~ 0
        h2 = z / sin_lat - (1 - ECCEN_EARTH_SQ)*N

        # Branchless selector: 1 for non-pole, 0 for near-pole
        use_h1 = float(abs(np.pi/2 - abs(lat)) > 1e-3)

        alt = use_h1*h1 + (1 - use_h1)*h2
        new_lat = np.arctan2(z + ECCEN_EARTH_SQ*N*sin_lat, p)
        err = new_lat - lat
        lat = new_lat
    return lat, lon, alt

def lla2ecef(lat, lon, alt):
    """Convert from Latitude, Longitude, Altitude to ECEF
    
    Ref: https://github.com/NavPy/NavPy/blob/master/navpy/core/navpy.py
    Args:
        lat : latitude in radians
        lon : longitude in radians
        alt : altitude in meters
    Returns:
        r_ecef : position vector in ECEF frame
    """ 
    a = SMA_EARTH
    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - ECCEN_EARTH_SQ * np.sin(lat)**2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - ECCEN_EARTH_SQ) + alt) * np.sin(lat)
    return np.array([x, y, z])

def quat_eci2ecef(t):
    """Get the quaternion representing the rotation from ECI to ECEF frame at time t

    Args:
        t : time in seconds since epoch sim t = 0
    Returns:
        q_eci2ecef : quaternion representing rotation from ECI to ECEF frame
    """    
    theta = OMEGA_EARTH * t
    cos_half_theta = np.cos(theta / 2)
    sin_half_theta = np.sin(theta / 2)
    return np.array([cos_half_theta, 0, 0, sin_half_theta])

def quat_ecef2eci(t):
    """Get the quaternion representing the rotation from ECEF to ECI frame at time t

    Args:
        t : time in seconds since epoch sim t =0
    Returns:
        q_ecef2eci : quaternion representing rotation from ECEF to ECI frame
    """   
    return quat_conj(quat_eci2ecef(t))

def quat_ecef2ned(lat, lon):
    """Get the quaternion representing the rotation from ECEF to NED frame at given latitude and longitude

    Args:
        lat : latitude in radians
        lon : longitude in radians
    Returns:
        q_ecef2ned : quaternion representing rotation from ECEF to NED frame
    """    
    q1 = q_from_axisangle(-np.pi/2, np.array([0, 1, 0]))
    
    q2 = q_from_axisangle(-lon, np.array([0, 0, 1]))

    q3 = q_from_axisangle(lat, np.array([1, 0, 0]))
    return q_mult(q1, q_mult(q2, q3))

def quat_eci2ned(t, x):
    """Get the quaternion representing the rotation from NED to ECI frame at time t and position x

    Args:
        t : time in seconds since epoch sim t =0
        x : position vector in ECI frame
    Returns:
        q_eci2ned : quaternion representing rotation from ECI to NED frame
    """    
    r_ecef = eci2ecef(x, t)
    lat, lon, _ = ecef2lla(r_ecef)
    q_ecef2ned = quat_ecef2ned(lat, lon)
    q_eci2ecef = quat_eci2ecef(t)
    return q_mult(q_eci2ecef, q_ecef2ned)

def quat_ned2eci(t, x):
    """Get the quaternion representing the rotation from NED to ECI frame at time t and position x

    Args:
        t : time in seconds since epoch sim t =0
        x : position vector in ECI frame
    Returns:
        q_ned2eci : quaternion representing rotation from NED to ECI frame
    """    
    return quat_conj(quat_eci2ned(t, x))


def dcm_rsw2eci(r_vec, v_vec):
    """Get the quaternion representing the rotation from RSW to ECI frame given position and velocity vectors

    Args:
        r_vec : position vector in ECI frame
        v_vec : velocity vector in ECI frame
    Returns:
        q_rsw2eci : quaternion representing rotation from RSW to ECI frame
    """     
    # Radial unit vector
    r_hat = r_vec / np.linalg.norm(r_vec)
    # Cross-track unit vector
    w_vec = np.cross(r_vec, v_vec)
    w_hat = w_vec / np.linalg.norm(w_vec)
    #  Along-track unit vector
    s_vec = np.cross(w_hat, r_hat)
    s_hat = s_vec / np.linalg.norm(s_vec)
    # DCM RSW -> ECI
    return np.column_stack((r_hat, s_hat, w_hat))
    