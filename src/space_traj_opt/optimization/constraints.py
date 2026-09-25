import enum

import numpy as np

from space_traj_opt.math.constants import MU_EARTH, SMA_EARTH
from space_traj_opt.math.orbital_calcs import rv_to_aei


class ConstraintType(enum.Enum):
    """Enum class defining Dynamics mode"""

    ON_STATE = 1
    ON_ORBIT = 2

def terminal_constraint(kind: ConstraintType, decision_var, sol, num_state):
    """Functions selects ConstraintType dynamics scheme and returns the derivative of the state
    Args:
        t : time
        x : Vehicle state
        num_state : Number of terminal states
    Returns:
        dx
    """
    match kind:
        case ConstraintType.ON_STATE:
            return terminal_constraint_on_state(decision_var, sol, num_state)
        case ConstraintType.ON_ORBIT:
            return terminal_constraint_on_orbit(decision_var, sol, num_state)
        case _:
            raise print("ConstraintType not defined")

def terminal_constraint_on_state(decision_var, sol, num_state):
        terminal_state = decision_var[-num_state:]
        terminal_defect = terminal_state - sol.y[:, -1]
        terminal_defect /= np.array([10000, 10000, 8000, 5000, 1000])

        return terminal_defect

def terminal_constraint_on_orbit(decision_var, sol, num_state):
    # Terminal Defect
    # calculate orbital elements here
    a_desired, e_desired, v_mag_des, i_desired, m_desired = decision_var[-num_state:]
    e_scale = 1.0

    r = sol.y[:,-1][0:3]
    v = sol.y[:,-1][3:6]
    m = sol.y[:,-1][6]
    a, e, i = rv_to_aei(r, v, MU_EARTH)
    v_mag = np.linalg.norm(v)
    terminal_defect = np.array([
        (a   - a_desired) / SMA_EARTH,
        (e   - e_desired) / e_scale,
        (v_mag   - v_mag_des) / 1000,  # Normalize velocity
        (i - i_desired) / np.pi /2,
        (m   - m_desired) / 100,
    ])

    return terminal_defect