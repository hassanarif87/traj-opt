import enum

from space_traj_opt.models.models import dynamics as dynamics2d
from space_traj_opt.models.models3d import dynamics as dynamics3d


class DynEnum(enum.Enum):
    """Enum class defining Dynamics mode"""

    DYNAMICS_2D = 1
    DYNAMICS_3D = 2

def dynamics(dyn_type: DynEnum):
    """Functions selects the dynamics scheme and returns the derivative of the state
    Args:
        t : time
        x : Vehicle state
        params : Tuple of parameters containing the control type and control law parameters
    Returns:
        dx
    """
    match dyn_type:
        case DynEnum.DYNAMICS_2D:
            func = dynamics2d
        case DynEnum.DYNAMICS_3D:
            func = dynamics3d
        case _:
            raise print("Dynamics mode not defined")

    return func