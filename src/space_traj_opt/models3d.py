import numba
import numpy as np
import numpy.typing as npt
from space_traj_opt.math.constants import MU_EARTH, STANDARD_GRAV
from space_traj_opt.controller3d import control

def dynamics(t: float, x: npt.ArrayLike, params) -> npt.ArrayLike:
    """Full system dynamics

    Args:
        t : integration time
        x : state vector, pos x, pos y, vel x, vel y, mass
        params: Control and vehicle parameters

    Returns:
        Derivative vector
    """

    u = control(t, x, params[1])
    thrust, Isp = params[0]

    return dynamics_plant(t, x, u, (thrust, Isp))


@numba.njit
def dynamics_plant(
    t: float, x: npt.ArrayLike, u: npt.ArrayLike, params: tuple
) -> npt.ArrayLike:
    """3D ECI dynamics of the plant

    Args:
        t : integration time
        x : state vector, [rx, ry, rz, vx, vy, vz, mass]
        u :quaternion attitude
        params: Control and vehicle parameters

    Returns:
        Derivative vector
    """
    S_ref = 1.2
    thrust, Isp = params

    # Unpack state
    rx, ry, rz, vx, vy, vz, m = x
    r_vec = np.array([rx, ry, rz])
    v_vec = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v_vec)
    r_mag = np.linalg.norm(r_vec)

    # Gravity (point mass, central body at origin)
    g_vec = -MU_EARTH * r_vec / (r_mag**3)

    # Get pitch and yaw from u
    unit_thrust_eci = u

    thrust_eci = unit_thrust_eci *thrust  # Thrust along x-axis in body frame

    # Atmospheric properties (altitude = r_mag - Earth's radius)
    # altitude = r_mag - R_EARTH
    # speed_of_sound, rho = get_atm(altitude)

    # Drag
    if False: #v_mag > 1e-5:
        pass
        # CdA, CdN = S_ref * get_drag_coeff(v_mag / speed_of_sound)
        # axis_mag = 0.5 * rho * CdA[0] * v_mag**2
        # normal_mag = 0.5 * rho * CdN[1] * v_mag**2
        
    else:
        aero_forces = np.zeros(3)

    # Accelerations
    accel_vec = (thrust_eci + aero_forces) / m + g_vec

    # Mass flow
    mdot = -thrust / STANDARD_GRAV / Isp

    # State derivative
    dx = np.zeros(7)
    dx[0:3] = v_vec
    dx[3:6] = accel_vec
    dx[6] = mdot
    return dx
