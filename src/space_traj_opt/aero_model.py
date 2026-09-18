import numba
import numpy as np
from space_traj_opt.math.coordinates import eci2ecef, quat_ned2eci, dcm_rsw2eci
from space_traj_opt.math.constants import OMEGA_EARTH


@numba.jit
def get_aero_coeff(mach: float) -> float:
    """
    Ballbark Aero represent the approximate drag of a launch vehicle, the implemetation here is an average of the Saturn-Apollo
    and the Mercury Atlas vehicles

    "For most medium to large expendable launch vehicles on nominal trajectories,
    the velocity losses due to drag are about 50 to 150 m/s. Therefore,
    an error of 10% in the drag coefficient might result in an error of only 10 m/s
    in the estimate of the total Δv required to attain orbit. Unless high precision is needed,
    this is often close enough."

    # Reference
    * <http://www.braeunig.us/space/aerodyn_wip.html>

        Args:
        mach : mach number

    Returns:
        drag coefficient and normal force coefficient
    """
    # Define the different conditions
    if mach <= 0.6:
        Cd =  0.2083333 * mach**2 - 0.25 * mach + 0.46
    elif mach <= 0.8:
        Cd =  1.25 * mach**3 - 2.125 * mach**2 + 1.2 * mach + 0.16
    elif mach <= 0.95:
        Cd =  10.37037 * mach**3 - 22.88889 * mach**2 + 16.91111 * mach - 3.78963
    elif mach <= 1.05:
        Cd =  -30.0 * mach**3 + 88.5 * mach**2 - 85.425 * mach + 27.51375
    elif mach <= 1.15:
        Cd =  -20.0 * mach**3 + 60.0 * mach**2 - 58.65 * mach + 19.245
    elif mach <= 1.3:
        Cd =  11.85185 * mach**3 - 44.88889 * mach**2 + 56.22222 * mach - 22.58519
    elif mach <= 2.0:
        Cd =  -0.04373178 * mach**3 + 0.3236152 * mach**2 - 1.019679 * mach + 1.554752
    elif mach <= 3.25:
        Cd =  0.01024 * mach**3 - 0.00864 * mach**2 - 0.33832 * mach + 1.08928
    elif mach <= 4.5:
        Cd = -0.01408 * mach**3 + 0.19168 * mach**2 - 0.86976 * mach + 1.53544
    else:
        Cd = 0.22
    
    Cn_alpha = 3.5
    return Cd , Cn_alpha


def get_wind_relative_velocity(t: float, state: np.array, t_phase_start:float) -> np.array:
    """
    Returns wind-relative velocity in ECEF

    Args:        
        t : time
        state: state vector in ECI frame
        t_phase_start: start time of the phase, used to compute the ECEF frame      

    Returns:        
        Wind-relative velocity in ECEF frame    
    """
    r_eci = state[0:3]
    v_eci = state[3:6]
    t_ = t_phase_start + t

    r_ecef = eci2ecef(r_eci, t_)

    omega = np.array([0, 0, OMEGA_EARTH])
    v_ecef = eci2ecef(v_eci, t_) - np.cross(omega, r_ecef)

    v_wind_ecef = np.array([0, 0, 0])

    return v_ecef - v_wind_ecef