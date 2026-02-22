import numpy as np
from .constants import OMEGA_EARTH, R_EARTH, ECCEN_EARTH
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

# TODO: Add more accurate models for lla
def ecef2lla(r_ecef):
    """Convert from ECEF to Latitude, Longitude, Altitude

    Args:
        r_ecef : position vector in ECEF frame
    Returns:
        lat : latitude in radians
        lon : longitude in radians
        alt : altitude in meters
    """ 
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - 0.081819190842622))
    alt = p / np.cos(lat) - R_EARTH
    return lat, lon, alt

def lla2ecef(lat, lon, alt):
    """Convert from Latitude, Longitude, Altitude to ECEF

    Args:
        lat : latitude in radians
        lon : longitude in radians
        alt : altitude in meters
    Returns:
        r_ecef : position vector in ECEF frame
    """ 
    a = R_EARTH
    e2 = ECCEN_EARTH*ECCEN_EARTH
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    return np.array([x, y, z])

def quat_eci2ecef(t):
    """Get the quaternion representing the rotation from ECI to ECEF frame at time t

    Args:
        t : time in seconds since epoch sim t =0
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