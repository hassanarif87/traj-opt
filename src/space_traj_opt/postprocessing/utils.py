from functools import wraps
from pathlib import Path
from timeit import default_timer

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from space_traj_opt import OUT_DIR
from space_traj_opt.math.integrator import ODEResult
from space_traj_opt.optimization.problem import Problem


def _json_ready(value):
    """Convert NumPy and pathlib values to JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def vocalTimeit(*args, **kwargs):
    ''' provides the decorator @vocalTime which will print the name of the function as well as the
        execution time in seconds '''

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            start = default_timer()
            results = function(*args, **kwargs)
            end = default_timer()
            print(f'{function.__name__} execution time: {end-start} s')
            return results
        return wrapper
    return decorator

def unpack_sol_list(sol_list_in: list[ODEResult] , state_index: list[int])-> tuple[list, list]:
    """Helper function to add time offsets to each phase

    Args:
        sol_list_in : OdeSoluion for each phase
        state_index (_type_): Index of the state to unpack

    Returns:
        tuple of lists of time arrays and lists of the state arrays for each phase S
    """
    t_offsets = 0
    x_list = []
    y_list = []
    for sol in sol_list_in:
        x_list.append(sol.t + t_offsets )
        y_list.append(sol.y[state_index] )
        t_offsets += sol.t[-1]

    return x_list, y_list


def _sol_to_dataframe(sol_list, header):
    """Convert a list of solver solutions into a single DataFrame."""
    t_offset = 0.0
    frames = []

    for phase, sol in enumerate(sol_list):
        df = pd.DataFrame(
            sol.y.T,
            columns=header,
        )

        df.insert(0, "phase", phase)
        df.insert(0, "time", sol.t + t_offset)

        frames.append(df)

        t_offset += sol.t[-1]

    return pd.concat(frames, ignore_index=True)


def sol_to_csv(sol_list: list, header: list[str], out_name: str):
    """Convert a list of solver solutions into a single CSV file."""
    out: Path = OUT_DIR / (out_name + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = _sol_to_dataframe(sol_list, header)
    df.to_csv(out, index=False)



def save_metadata(d_opt: ArrayLike, problem: Problem, out_name: str, result=None) -> dict:
    """Create the prescribed metadata for the solved problem and save it to JSON."""
    data = {
        "x_opt": _json_ready(np.asarray(d_opt, dtype=float)),
        "num_states": int(problem.num_states),
        "num_terminal_states": int(problem.num_terminal_states),
    }

    out: Path = OUT_DIR / (out_name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
