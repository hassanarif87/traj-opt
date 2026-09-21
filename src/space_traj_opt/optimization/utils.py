from functools import lru_cache

import numpy as np

from space_traj_opt.math.integrator import OdeResult, integrate
from space_traj_opt.models.dynamics import dynamics


@lru_cache(maxsize=128, typed=True) 
def traj_rollout(t_terminal:float, x0: np.array, params: tuple) -> OdeResult:
    """Integrates a phase of the trajectory.
    The trajectory is evaluated at a set time points using t_eval, this greatly improves convergance and stability of the gradients 
    lru_cache decerases the time required to calculate the jac, since scipy uses forward diff the cached f(x) is used instead of a re-compute
    Args:
        t_terminal : Terminal time of the phase
        x0 : Initial state of the phase
        params : Phase Parameter

    Returns:
        OdeResult: The solution of the phase
    """
    dyn_type, model_params , control_params = params
    dyn_func = dynamics(dyn_type)

    ode_params = (model_params , control_params)

    t_sol, y_sol = integrate(
        dyn_func, 
        t_span=[0.0, t_terminal], 
        t_eval= np.linspace(0.0, t_terminal,50),
        y0=x0,    
        args=(ode_params,)
    )
    return OdeResult(t_sol, y_sol)  

def normalize_decision_vec(decision_vector, bounds, normalization_vector, offset_vector=None):
    """
    Normalize a decision vector and its bounds using a scaling normalization vector 
    and an optional offset vector.

    Args:
        decision_vector: The original decision vector to normalize.
        bounds: List of tuples representing (lower, upper) bounds for the decision variables.
        normalization_vector: Array or list of scaling factors for normalization.
        offset_vector: Array or list of offsets for normalization. Defaults to None.

    Returns:
        normalized_vector: The normalized decision vector.
        normalized_bounds: The normalized bounds as a list of (lower, upper) tuples.
    """

    # Ensure the normalization vector matches the length of the decision vector
    if len(decision_vector) != len(normalization_vector):
        raise ValueError("Normalization vector must match the length of the decision vector.")

    # Default offset vector to zeros if not provided
    if offset_vector is None:
        offset_vector = np.zeros_like(decision_vector)

    # Ensure the offset vector matches the length of the decision vector
    if len(decision_vector) != len(offset_vector):
        raise ValueError("Offset vector must match the length of the decision vector.")

    # Normalize the decision vector
    normalized_vector = (decision_vector - offset_vector) / normalization_vector

    # Normalize the bounds
    normalized_bounds = [
        (
            (lb - offset) / scale if lb is not None else None,
            (ub - offset) / scale if ub is not None else None
        )
        for (lb, ub), scale, offset in zip(bounds, normalization_vector, offset_vector)
    ]

    return normalized_vector, normalized_bounds

def denormalize_decision_vec(normalized_vector, normalization_vector, offset_vector=None):
    """
    Denormalize a decision vector a scaling normalization vector and an optional offset vector.

    Args:
        normalized_vector: The normalized decision vector to denormalize.
        normalized_bounds: List of tuples representing (lower, upper) bounds in the normalized space.
        offset_vector: Array or list of offsets used for normalization. Defaults to None.

    Returns:
        denormalized_vector: The denormalized decision vector.
    """
    # Default offset vector to zeros if not provided
    if offset_vector is None:
        offset_vector = np.zeros_like(normalized_vector)

    # Denormalize the decision vector
    denormalized_vector = normalized_vector * normalization_vector + offset_vector

    # Denormalize the bounds
    return denormalized_vector

