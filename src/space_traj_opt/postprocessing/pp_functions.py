## Post proessing
import numpy as np
import pandas as pd

from space_traj_opt.math.constants import EARTH_R
from space_traj_opt.math.orbital_calcs import rv_to_orbital_elements
from space_traj_opt.models.controller3d import (
    dcm_rsw2eci,
    flight_path_angle,
    lts_control,
)
from space_traj_opt.models.models3d import dynamics_plant
from space_traj_opt.postprocessing.data_client import CSVClient
from space_traj_opt.sims.config import ScenarioConfig, load_config


def process_phase(
    phase_df: pd.DataFrame,
    ctrl_params: np.ndarray,
    model_params: tuple,
    state_headers: list[str],
) -> pd.DataFrame:
    """Calculate derived trajectory channels for one phase."""
    time = phase_df["time"].to_numpy()
    state = phase_df[state_headers].to_numpy()
    r_eci = state[:, :3]
    v_eci = state[:, 3:6]

    a, e, i, raan, argp, nu = rv_to_orbital_elements(r_eci, v_eci)
    fpa = flight_path_angle(r_eci, v_eci)
    r_periapsis = a * (1.0 - e)
    r_apoapsis = a * (1.0 + e)

    u_values = []
    dx_values = []
    pitch_values = []
    yaw_values = []
    for time_value, current_state in zip(time, state):
        thrust_hat_eci = lts_control(time_value, current_state, ctrl_params)
        u_values.append(thrust_hat_eci)
        thrust_hat_rsw = (
            dcm_rsw2eci(current_state[:3], current_state[3:6]).T
            @ thrust_hat_eci
        )
        pitch_values.append(np.arctan2(thrust_hat_rsw[1], thrust_hat_rsw[0]))
        yaw_values.append(
            np.arctan2(
                thrust_hat_rsw[2],
                np.hypot(thrust_hat_rsw[0], thrust_hat_rsw[1]),
            )
        )
        dx_values.append(
            dynamics_plant(time_value, current_state, thrust_hat_eci, model_params)
        )

    u = np.asarray(u_values)
    dx = np.asarray(dx_values)
    return pd.DataFrame(
        {
            "a": a,
            "e": e,
            "i": i,
            "raan": raan,
            "argp": argp,
            "nu": nu,
            "fpa": fpa,
            "v_eci_mag": np.linalg.norm(v_eci, axis=1),
            "pitch": pitch_values,
            "yaw": yaw_values,
            "r_periapsis": r_periapsis,
            "r_apoapsis": r_apoapsis,
            "h_periapsis": r_periapsis - EARTH_R,
            "h_apoapsis": r_apoapsis - EARTH_R,
            "u_x": u[:, 0],
            "u_y": u[:, 1],
            "u_z": u[:, 2],
            "ax": dx[:, 3],
            "ay": dx[:, 4],
            "az": dx[:, 5],
            "m_dot": dx[:, 6],
        },
        index=phase_df.index,
    )


def process_trajectory(
    trajectory_df: pd.DataFrame,
    d_opt: np.ndarray,
    config: ScenarioConfig,
) -> pd.DataFrame:
    """Process each phase using its own control and model parameters."""
    results = trajectory_df.copy()
    control_start = 0

    for phase_index, phase_config in enumerate(config.phases):
        control_count = len(phase_config.controller.guess)
        ctrl_params = d_opt[control_start : control_start + control_count]
        control_start += control_count + config.num_states + 1

        phase_rows = results["phase"] == phase_index
        if not phase_rows.any():
            continue

        phase_data = process_phase(
            results.loc[phase_rows],
            ctrl_params,
            tuple(phase_config.model_params),
            config.state_headers,
        )
        results.loc[phase_rows, phase_data.columns] = phase_data

    return results


def process_file(sim_output: str, config_path: str) -> pd.DataFrame:
    """Load, process, and rewrite a scenario's trajectory CSV."""
    client = CSVClient(sim_output)
    if client.metadata is None or "x_opt" not in client.metadata:
        raise ValueError(f"Optimization metadata with x_opt is required for {sim_output!r}.")

    config = load_config(config_path)
    client.df = process_trajectory(
        client.df,
        np.asarray(client.metadata["x_opt"]),
        config,
    )
    client.df.to_csv(client.out, index=False)
    return client.df


if __name__ == "__main__":
    process_file("second_stage_ascent", "scenarios/second_stage_ascent.yaml")

