import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from space_traj_opt import OUT_DIR
from space_traj_opt.optimization.problem import Problem


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


def problem_results(d_opt: ArrayLike, problem: Problem, out_name: str, result=None) -> dict:
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
