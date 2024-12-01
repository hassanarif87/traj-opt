import numba
import numpy as np
import numpy.typing as npt
import enum

STANDARD_GRAV = 9.80665


def dynamics(t: float, x: npt.ArrayLike, params) -> npt.ArrayLike:
    """Full syustem dynamics

    Args:
        t : integration time
        x : state vector, pos x, pos y, vel x, vel y, mass
        params: Control and vechicle parameters

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
    """Dynamics of the plant

    Args:
        t : integration time
        x : state vector, pos x, pos y, vel x, vel y, mass
        u : pitch command
        params: Control and vechicle parameters

    Returns:
        Derivative vector
    """
    S_ref = 1.2
    thrust, Isp = params
    cos_theta = np.cos(u)
    sin_theta = np.sin(u)
    # dx = np.zeros_like(x, dtype=np.float64)
    v_sq = x[2] * x[2] + x[3] * x[3]
    v_mag = np.sqrt(v_sq)
    speed_of_sound, rho = get_atm(x[1])
    CdA = S_ref * get_drag_coeff(v_mag / speed_of_sound)
    drag = 0.5 * rho * CdA * v_sq
    xdot = x[2]
    ydot = x[3]
    # dx[2] = (
    #     thrust * cos_theta - drag * np.divide(x[2], v_mag, where=v_mag != 0.0)
    # ) / x[4]  #
    # dx[3] = (
    #     thrust * sin_theta - drag * np.divide(x[3], v_mag, where=v_mag != 0.0)
    # ) / x[4] - STANDARD_GRAV  #
    dx = 0.
    dy = 0.
    if v_mag > 0.:
        dx = drag * x[2]/ v_mag
        dy = drag * x[3]/ v_mag
    xddot = (thrust * cos_theta - dx) / x[4]  #
    yddot = (thrust * sin_theta - dy) / x[4] - STANDARD_GRAV  #
    mdot= -thrust / STANDARD_GRAV / Isp
    return np.array([xdot, ydot, xddot, yddot, mdot])


@numba.jit
def get_drag_coeff(mach: float) -> float:
    """Ballpark aerodynamic drag

    Args:
        mach : mach number

    Returns:
        drag coefficient
    """
    # Define the different conditions
    if mach <= 0.6:
        return 0.2083333 * mach**2 - 0.25 * mach + 0.46
    elif mach <= 0.8:
        return 1.25 * mach**3 - 2.125 * mach**2 + 1.2 * mach + 0.16
    elif mach <= 0.95:
        return 10.37037 * mach**3 - 22.88889 * mach**2 + 16.91111 * mach - 3.78963
    elif mach <= 1.05:
        return -30.0 * mach**3 + 88.5 * mach**2 - 85.425 * mach + 27.51375
    elif mach <= 1.15:
        return -20.0 * mach**3 + 60.0 * mach**2 - 58.65 * mach + 19.245
    elif mach <= 1.3:
        return 11.85185 * mach**3 - 44.88889 * mach**2 + 56.22222 * mach - 22.58519
    elif mach <= 2.0:
        return -0.04373178 * mach**3 + 0.3236152 * mach**2 - 1.019679 * mach + 1.554752
    elif mach <= 3.25:
        return 0.01024 * mach**3 - 0.00864 * mach**2 - 0.33832 * mach + 1.08928
    elif mach <= 4.5:
        return -0.01408 * mach**3 + 0.19168 * mach**2 - 0.86976 * mach + 1.53544
    else:
        return 0.22


@numba.njit
def get_atm(altitude: float) -> tuple[float, float]:
    air_den = 1.225 * np.exp(-altitude / 8.44e3)
    speed_of_sound = 300.0
    return (speed_of_sound, air_den)


### Controls


@numba.njit
def angle_steering(t: float, x: npt.ArrayLike, params: tuple) -> float:
    """Constant desired pitch

    Args:
        t : time
        x : Vehicle state
        params : control law params, pitch

    Returns:
        Desired pitch
    """
    u = params[0]
    return u


@numba.njit
def zero_alpha(t: float, x: npt.ArrayLike, params: tuple) -> float:
    """For zero alpha pitch = flight path angle
    Args:
        t : time
        x : Vehicle state
        params : control law params, this law has no parameters
    Returns:
        Desired pitch
    """
    vx, vy = x[2:4]
    return np.arctan2(vy, vx)


@numba.njit
def polynomial_steering(t: float, x: npt.ArrayLike, params: tuple) -> float:
    a, b, c = params
    return a * t**2 + b * t + c


@numba.njit
def lts_control(t: float, x: npt.ArrayLike, params: tuple) -> float:
    a, b = params
    return np.arctan(a * t + b)


# @numba.njit
class CtrlMode(enum.Enum):
    """Enum class defining control mode"""

    ANGLE_STEER = 1
    ZERO_ALPHA = 2
    LTS = 3
    POLYNOMIAL = 4

# @numba.njit
def control(t: float, x: npt.ArrayLike, params: tuple) -> float:
    """Functions selects the parameterized control scheme and returns the desired pitch angle
    Args:
        t : time
        x : Vehicle state
        params : Tuple of parameters containing the control type and control law parameters
    Returns:
        Desired pitch
    """
    ctrl_mode, ctrl_param = params
    if ctrl_mode == CtrlMode.ANGLE_STEER:
        u_pitch = angle_steering(t, x, ctrl_param)
    elif ctrl_mode == CtrlMode.ZERO_ALPHA:
        u_pitch = zero_alpha(t, x, ctrl_param)
    elif ctrl_mode == CtrlMode.LTS:
        u_pitch = lts_control(t, x, ctrl_param)
    elif ctrl_mode == CtrlMode.POLYNOMIAL:
        u_pitch = polynomial_steering(t, x, ctrl_param)
    else:
        u_pitch = 0
        print("Control mode not define")

    return u_pitch