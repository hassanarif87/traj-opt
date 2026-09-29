from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# State ordering
# ---------------------------------------------------------------------------

STATE_NAMES = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
)

STATE_SIZE = len(STATE_NAMES)

CONTROL_NAMES = (
    "pitch",
    "yaw",
)


# ---------------------------------------------------------------------------
# Trajectory container
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """
    Optimized trajectory used to construct the neighborhood-optimal guidance
    table.

    All arrays have shape (N,).

    acceleration:
        Scheduling variable used for the guidance table.

    state:
        State history with columns:
            [x, y, z, vx, vy, vz]

    control:
        Guidance history with columns:
            [pitch, yaw]
    """

    acceleration: np.ndarray
    state: np.ndarray
    control: np.ndarray

    def __post_init__(self):
        self.acceleration = np.asarray(self.acceleration)
        self.state = np.asarray(self.state)
        self.control = np.asarray(self.control)

        if self.state.ndim != 2 or self.state.shape[1] != 6:
            raise ValueError(
                "state must have shape (N, 6): "
                "[x, y, z, vx, vy, vz]"
            )

        if self.control.ndim != 2 or self.control.shape[1] != 2:
            raise ValueError(
                "control must have shape (N, 2): [pitch, yaw]"
            )

        if not (
            len(self.acceleration)
            == len(self.state)
            == len(self.control)
        ):
            raise ValueError("Trajectory arrays must have equal length.")


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_trajectory(
    trajectory: Trajectory,
    acceleration_grid: np.ndarray,
) -> Trajectory:
    """
    Interpolate a trajectory onto a common acceleration grid.

    Assumes acceleration is monotonic.
    """

    a = trajectory.acceleration

    # np.interp expects increasing x.
    if a[0] > a[-1]:
        a = a[::-1]
        state = trajectory.state[::-1]
        control = trajectory.control[::-1]
    else:
        state = trajectory.state
        control = trajectory.control

    if np.any(np.diff(a) <= 0):
        raise ValueError(
            "Acceleration must be strictly monotonic for interpolation."
        )

    state_interp = np.column_stack([
        np.interp(acceleration_grid, a, state[:, i])
        for i in range(6)
    ])

    control_interp = np.column_stack([
        np.interp(acceleration_grid, a, control[:, i])
        for i in range(2)
    ])

    return Trajectory(
        acceleration=acceleration_grid,
        state=state_interp,
        control=control_interp,
    )


# ---------------------------------------------------------------------------
# Gain table
# ---------------------------------------------------------------------------

@dataclass
class GainTable:
    """
    Neighborhood-optimal guidance table.

    acceleration:
        Shape (M,)

    nominal_state:
        Shape (M, 6)

    nominal_control:
        Shape (M, 2)

    gain:
        Shape (M, 2, 6)

        gain[:, 0, :] = pitch gains
        gain[:, 1, :] = yaw gains
    """

    acceleration: np.ndarray
    nominal_state: np.ndarray
    nominal_control: np.ndarray
    gain: np.ndarray

    def guidance(
        self,
        acceleration: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate [pitch, yaw] for the current state.

        Parameters
        ----------
        acceleration:
            Current acceleration.

        state:
            Current [x, y, z, vx, vy, vz].

        Returns
        -------
        np.ndarray
            [pitch, yaw]
        """

        state = np.asarray(state)

        if state.shape != (6,):
            raise ValueError("state must have shape (6,)")

        # Interpolate nominal state/control.
        nominal_state = np.array([
            np.interp(
                acceleration,
                self.acceleration,
                self.nominal_state[:, i],
            )
            for i in range(6)
        ])

        nominal_control = np.array([
            np.interp(
                acceleration,
                self.acceleration,
                self.nominal_control[:, i],
            )
            for i in range(2)
        ])

        # Interpolate each gain element.
        K = np.empty((2, 6))

        for i in range(2):
            for j in range(6):
                K[i, j] = np.interp(
                    acceleration,
                    self.acceleration,
                    self.gain[:, i, j],
                )

        dx = state - nominal_state

        return nominal_control + K @ dx


# ---------------------------------------------------------------------------
# Gain construction
# ---------------------------------------------------------------------------

def build_gain_table(
    nominal: Trajectory,
    perturbations: dict[str, Trajectory],
    deltas: dict[str, float],
    acceleration_grid: np.ndarray,
) -> GainTable:
    """
    Construct a neighborhood-optimal gain table.

    Parameters
    ----------
    nominal:
        Nominal optimized trajectory.

    perturbations:
        Dictionary containing 12 trajectories:

            "+x", "-x"
            "+y", "-y"
            "+z", "-z"
            "+vx", "-vx"
            "+vy", "-vy"
            "+vz", "-vz"

    deltas:
        Initial-state perturbation magnitudes:

            {
                "x": dx,
                "y": dy,
                "z": dz,
                "vx": dvx,
                "vy": dvy,
                "vz": dvz,
            }

    acceleration_grid:
        Common acceleration grid.

    Returns
    -------
    GainTable
    """

    # Interpolate nominal.
    nominal_i = interpolate_trajectory(
        nominal,
        acceleration_grid,
    )

    # Interpolate all neighboring trajectories.
    perturbed_i = {
        name: interpolate_trajectory(
            trajectory,
            acceleration_grid,
        )
        for name, trajectory in perturbations.items()
    }

    n = len(acceleration_grid)

    gain = np.empty((n, 2, 6))

    state_order = [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
    ]

    for k in range(n):

        # ---------------------------------------------------------------
        # Build dX / d(initial state)
        #
        # Shape:
        #     DX = 6 x 6
        # ---------------------------------------------------------------

        DX = np.empty((6, 6))

        # ---------------------------------------------------------------
        # Build dU / d(initial state)
        #
        # Shape:
        #     DU = 2 x 6
        # ---------------------------------------------------------------

        DU = np.empty((2, 6))

        for j, state_name in enumerate(state_order):

            plus = perturbed_i[f"+{state_name}"]
            minus = perturbed_i[f"-{state_name}"]

            delta = deltas[state_name]

            # Central finite difference.
            DX[:, j] = (
                plus.state[k] - minus.state[k]
            ) / (2.0 * delta)

            DU[:, j] = (
                plus.control[k] - minus.control[k]
            ) / (2.0 * delta)

        # ---------------------------------------------------------------
        # DU = K DX
        #
        # Therefore:
        #
        # K = DU DX^-1
        #
        # Rather than explicitly computing inverse(DX), solve:
        #
        #     DX.T K.T = DU.T
        # ---------------------------------------------------------------

        try:
            K = np.linalg.solve(DX.T, DU.T).T

        except np.linalg.LinAlgError:
            # If the neighborhood becomes locally singular, use a
            # pseudoinverse rather than allowing the entire table to fail.
            K = DU @ np.linalg.pinv(DX)

        gain[k] = K

    return GainTable(
        acceleration=acceleration_grid,
        nominal_state=nominal_i.state,
        nominal_control=nominal_i.control,
        gain=gain,
    )


# ---------------------------------------------------------------------------
# Save/load
# ---------------------------------------------------------------------------

def save_gain_table(
    table: GainTable,
    filename: str | Path,
) -> None:
    """Save the guidance table as a NumPy .npz file."""

    np.savez_compressed(
        filename,
        acceleration=table.acceleration,
        nominal_state=table.nominal_state,
        nominal_control=table.nominal_control,
        gain=table.gain,
    )


def load_gain_table(
    filename: str | Path,
) -> GainTable:
    """Load a previously generated guidance table."""

    data = np.load(filename)

    return GainTable(
        acceleration=data["acceleration"],
        nominal_state=data["nominal_state"],
        nominal_control=data["nominal_control"],
        gain=data["gain"],
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("None")
   