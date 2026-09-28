import numpy as np
import pandas as pd

from space_traj_opt.postprocessing.data_client import CSVClient
from space_traj_opt.postprocessing.pp_functions import process_3dtrajectory
from space_traj_opt.sims.config import load_config


def postprocess_trajectory(sim_output: str, config_path: str) -> pd.DataFrame:
    """Load, process, and rewrite a scenario's trajectory CSV."""
    client = CSVClient(sim_output)
    if client.metadata is None or "x_opt" not in client.metadata:
        raise ValueError(f"Optimization metadata with x_opt is required for {sim_output!r}.")

    config = load_config(config_path)
    client.df = process_3dtrajectory(
        client.df,
        np.asarray(client.metadata["x_opt"]),
        config,
    )
    client.df.to_csv(client.out, index=False)
    return client.df


if __name__ == "__main__":
    postprocess_trajectory("second_stage_ascent", "scenarios/second_stage_ascent.yaml")

