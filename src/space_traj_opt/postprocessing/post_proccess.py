from pathlib import Path

import numpy as np
import pandas as pd

from space_traj_opt.postprocessing.data_client import CSVClient
from space_traj_opt.postprocessing.pp_functions import process_3dtrajectory
from space_traj_opt.sims.config import ScenarioConfig, load_config


POSTPROCESSORS = {
    "process_3dtrajectory": process_3dtrajectory,
}


def postprocess_trajectory(
    sim_output: str,
    config: ScenarioConfig | str | Path,
) -> pd.DataFrame | None:
    """Load, process, and rewrite a scenario's trajectory CSV."""
    if isinstance(config, (str, Path)):
        config = load_config(config)

    try:
        processors = [POSTPROCESSORS[name] for name in config.post_process]
    except KeyError as error:
        available = ", ".join(sorted(POSTPROCESSORS))
        raise ValueError(
            f"Unknown postprocessor {error.args[0]!r}; available: {available}"
        ) from error

    if not processors:
        return None

    client = CSVClient(sim_output)
    if client.metadata is None or "x_opt" not in client.metadata:
        raise ValueError(f"Optimization metadata with x_opt is required for {sim_output!r}.")

    decision_vector = np.asarray(client.metadata["x_opt"])
    for processor in processors:
        client.df = processor(client.df, decision_vector, config)

    client.df.to_csv(client.out, index=False)
    return client.df


if __name__ == "__main__":
    postprocess_trajectory("second_stage_ascent", "scenarios/second_stage_ascent.yaml")

