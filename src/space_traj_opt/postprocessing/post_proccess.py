import pandas as pd
from pathlib import Path
from space_traj_opt import OUT_DIR

def sol_to_dataframe(sol_list, header):
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

def sol_to_csv(sol_list: list, header: list[str], out_name : str):
    """Convert a list of solver solutions into a single CSV file."""
    out : Path = OUT_DIR / (out_name + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df : pd.DataFrame = sol_to_dataframe(sol_list, header)
    df.to_csv(out, index=False)