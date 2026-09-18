import numpy as np
import numpy.typing as npt
import enum

from space_traj_opt.math.coordinates import  quat_ned2eci, dcm_rsw2eci, quat_ecef2eci
from space_traj_opt.math.quaternion import q_rotate_frame
from space_traj_opt.aero_model import get_wind_relative_velocity


def normalize(v):
    return v / np.linalg.norm(v)

# @numba.njit
class CtrlMode(enum.Enum):
    """Enum class defining control mode"""

    ANGLE_STEER = 1
    AERO_STEERING = 2
    LTS = 3
    POLYNOMIAL = 4

def flight_path_angle(r_vec, v_vec):
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Radial velocity
    v_r = np.dot(r_vec, v_vec) / r

    # Transverse velocity
    v_t = np.sqrt(v**2 - v_r**2)

    # Flight path angle
    gamma = np.arctan2(v_r, v_t)

    return gamma

    
def dir_from_pitch_yaw(pitch, yaw):
    # Convert pitch and yaw to a unit vector in the rsw frame
    r = np.cos(pitch) * np.cos(yaw)
    s = np.sin(pitch) * np.cos(yaw)
    w = np.sin(yaw)

    return np.array([r, s, w])

def lts_control(t: float, x: npt.ArrayLike, params) -> npt.ArrayLike:
    """LTS control law

    Args:
        t : time
        x : state vector
        params: Control parameters (e.g. LTS gains)

    Returns:
        Desired unit thrust vector in ECI frame
    """

    a,b,c,d = params  # LTS gain
    pitch = np.arctan(a * t + b)
    yaw = np.arctan(c * t + d)
    thrust_rsw = dir_from_pitch_yaw(np.pi/2 - pitch, yaw)
    thrust_eci = dcm_rsw2eci(x[0:3], x[3:6])  @ thrust_rsw
    return normalize(thrust_eci)

def ned_steering(t: float, x: npt.ArrayLike, params, t_phase_start=0) -> npt.ArrayLike:

    """NED steering control law, which steers the thrust vector towards a desired NED direction,
    NED is fixed at the start of the phase.

    Args:
        t : time
        x : state vector
        params: Control parameters (e.g. desired NED direction)

    Returns:
        Desired unit thrust unit vector in ECI frame
    """
    # Placeholder implementation, replace with actual NED steering logic
    ned_dir = params[0]  # Desired NED direction as a unit vector

    q = quat_ned2eci(t_phase_start, x)  # Get current NED to ECI rotation
    return q_rotate_frame(q, ned_dir)  # Rotate by desired NED direction




def aero_steering(t, x, params, t_phase_start=0):
    """
    Aerodynamic Alpha-Beta steering law.

    params[0] = alpha (angle of attack, radians)
    params[1] = beta  (sideslip angle, radians)

    Returns:
        Desired unit thrust vector in ECI frame
    """

    alpha = params[0]   # Angle of attack
    beta  = params[1]   # Sideslip angle

    # Position in ECEF
    r_ecef = x[0:3]

    # Wind-relative velocity in ECEF
    v_rel = get_wind_relative_velocity(t, x, t_phase_start)
    v_hat = normalize(v_rel)

    # TODO: should i do this in bodyframe
    # ---- Construct wind frame axes ----

    # x_w: points into the wind (opposite velocity direction)
    x_w = -v_hat

    # z_w: normal to the flight plane (orbit normal)
    h = np.cross(r_ecef, v_rel)

    if np.linalg.norm(h) < 1e-10:
        # Degenerate case: nearly radial flight
        # Choose an arbitrary perpendicular direction
        east = normalize(np.cross([0, 0, 1], r_ecef))
        z_w = normalize(np.cross(v_hat, east))
    else:
        z_w = normalize(h)

    # y_w: completes right-handed wind frame
    y_w = np.cross(z_w, x_w)

    # ---- Thrust direction in wind frame ----
    # Standard aircraft convention
    T_w = np.array([
        np.cos(alpha) * np.cos(beta),  # forward component
        np.sin(beta),                  # lateral component
        np.sin(alpha) * np.cos(beta)   # vertical component
    ])

    # ---- Convert wind-frame vector to ECEF ----
    # C = np.column_stack((x_w, y_w, z_w))
    # thrust_ecef = C @ T_w
    thrust_ecef = (
        T_w[0] * x_w +
        T_w[1] * y_w +
        T_w[2] * z_w
    )
    q = quat_ecef2eci(t_phase_start)
    thrust_eci = q_rotate_frame(q, thrust_ecef)
    return normalize(thrust_eci)

def control(t: float, x: npt.ArrayLike, params: tuple, t_phase_start: float = 0) -> float:    
    """Functions selects the parameterized control scheme and returns the desired pitch angle
    Args:
        t : time
        x : Vehicle state
        params : Tuple of parameters containing the control type and control law parameters
    Returns:
        Desired unit thrust vector in ECI frame
    """
    ctrl_mode, ctrl_param = params
    match ctrl_mode:
        case CtrlMode.ANGLE_STEER:
            thrust_eci = ned_steering(t, x, ctrl_param, t_phase_start)
        case CtrlMode.AERO_STEERING:
            thrust_eci = aero_steering(t, x, ctrl_param, t_phase_start)
        case CtrlMode.LTS:
            thrust_eci = lts_control(t, x, ctrl_param)  
        # case CtrlMode.POLYNOMIAL:
        #     thrust_eci = polynomial_steering(t, x, ctrl_param)
        case _:
            thrust_eci = np.array([1.0, 0.0, 0.0])  # Default thrust vector (zero)
            print("3d Control mode not defined")

    return thrust_eci