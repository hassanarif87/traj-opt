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

    var:
        Scheduling variable used for the guidance table.

    state:
        State history with columns:
            [x, y, z, vx, vy, vz]

    control:
        Guidance history with columns:
            [pitch, yaw]
    """

    var: np.ndarray
    state: np.ndarray
    state_deriv: np.ndarray
    control: np.ndarray
    t_terminal: float

    def __post_init__(self):
        self.var = np.asarray(self.var)
        self.state = np.asarray(self.state)
        self.state_deriv = np.asarray(self.state_deriv)
        self.control = np.asarray(self.control)

        if self.state_deriv.ndim != 2 or self.state_deriv.shape[1] != 6:
            raise ValueError(
                "state_deriv must have shape (N, 6): "
                "[x, y, z, vx, vy, vz]"
            )
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
            len(self.var)
            == len(self.state)
            == len(self.control)
        ):
            raise ValueError("Trajectory arrays must have equal length.")


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_trajectory(
    trajectory: Trajectory,
    var_grid: np.ndarray,
) -> Trajectory:
    """
    Interpolate a trajectory onto a common var grid.

    Assumes var is monotonic.
    """

    a = trajectory.var

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
            "Lookup var must be strictly monotonic for interpolation."
        )

    state_interp = np.column_stack([
        np.interp(var_grid, a, state[:, i])
        for i in range(6)
    ])

    control_interp = np.column_stack([
        np.interp(var_grid, a, control[:, i])
        for i in range(2)
    ])

    return Trajectory(
        var=var_grid,
        state=state_interp,
        control=control_interp,
        t_terminal=trajectory.t_terminal
    )


# ---------------------------------------------------------------------------
# Gain table
# ---------------------------------------------------------------------------

@dataclass
class GainTable:
    """
    Neighborhood-optimal guidance table.

    var:
        Shape (M,)

    nominal_state:
        Shape (M, 6)

    nominal_control:
        Shape (M, 2)

    control_gain:
        Shape (M, 2, 6)

        control_gain[:, 0, :] = pitch gains
        control_gain[:, 1, :] = yaw gains
    """

    var: np.ndarray
    nominal_state: np.ndarray
    nominal_control: np.ndarray
    control_gain: np.ndarray
    terminal_time_gain: np.ndarray
    nominal_state_deriv: np.ndarray
    

    def guidance(
        self,
        var: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate [pitch, yaw] for the current state.

        Parameters
        ----------
        var:
            Current var.

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
        
        # Actual clock time
        t = var

        # Start by assuming no index-time correction.
        t_index = t

        # Estimate index time.
        for _ in range(2):

            x_nom = self._interpolate_nominal_state(t_index)
            xdot_nom = self._interpolate_nominal_state_deriv(t_index)
            m = self._interpolate_terminal_time_gain(t_index)

            # Eq. corresponding to fixed dpsi = 0
            dx = (
                state
                - x_nom
                - xdot_nom * (t - t_index)
            )

            denom = 1.0 + m @ xdot_nom

            epsilon = (m @ dx) / denom

            t_index -= epsilon

        # ------------------------------------------------------------
        # Normal neighboring-optimal feedback, but indexed by t_index
        # ------------------------------------------------------------

        x_nom = self._interpolate_nominal_state(t_index)
        u_nom = self._interpolate_nominal_control(t_index)
        K = self._interpolate_control_gain(t_index)

        dx = state - x_nom

        control = u_nom + K @ dx

        return control

    def _interpolate_terminal_time_gain(
        self,
        var: float,
    ) -> np.ndarray:

        return np.array([
            np.interp(
                var,
                self.var,
                self.terminal_time_gain[:, i],
            )
            for i in range(6)
        ])

    def _interpolate_nominal_state(self, var: float) -> np.ndarray:
        """
        Interpolate nominal state for a given var.
        """ 
        return np.array([
            np.interp(
                var,
                self.var,
                self.nominal_state[:, i],
            )
            for i in range(6)
        ])
    def _interpolate_nominal_control(self, var: float) -> np.ndarray:
        """
        Interpolate nominal control for a given var.
        """ 

        nominal_control = np.array([
            np.interp(
                var,
                self.var,
                self.nominal_control[:, i],
            )
            for i in range(2)
        ])
        return nominal_control
    
    def _interpolate_control_gain(self, var: float) -> np.ndarray:
        # Interpolate each control_gain element.
        K = np.empty((2, 6))

        for i in range(2):
            for j in range(6):
                K[i, j] = np.interp(
                    var,
                    self.var,
                    self.control_gain[:, i, j],
                )
        return K


# ---------------------------------------------------------------------------
# Gain construction
# ---------------------------------------------------------------------------

def build_gain_table(
    nominal: Trajectory,
    perturbations: dict[str, Trajectory],
    deltas: dict[str, float],
    var_grid: np.ndarray,
) -> GainTable:
    """
    Construct a neighborhood-optimal control_gain table.

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

    var_grid:
        Common var grid. Independent loopup variable.

    Returns
    -------
    GainTable
    """

    # Interpolate nominal.
    nominal_i = interpolate_trajectory(
        nominal,
        var_grid,
    )

    # Interpolate all neighboring trajectories.
    perturbed_i = {
        name: interpolate_trajectory(
            trajectory,
            var_grid,
        )
        for name, trajectory in perturbations.items()
    }

    n = len(var_grid)

    control_gain = np.empty((n, 2, 6))
    terminal_time_gain = np.empty((n, 6))
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

        # ---------------------------------------------------------------
        # Build dT / d(initial state)
        #
        # Shape:
        #     DT = 1 x 6
        # ---------------------------------------------------------------

        DT = np.empty((1, 6))


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

            DT[0, j] = (
                plus.t_terminal - minus.t_terminal
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
            KT = np.linalg.solve(DX.T, DT.T).T

        except np.linalg.LinAlgError:
            print(f"Warning: Singular DX at index {k}. Using pseudoinverse. Condition number: {np.linalg.cond(DX)}")
            # If the neighborhood becomes locally singular, use a
            # pseudoinverse rather than allowing the entire table to fail.
            K = DU @ np.linalg.pinv(DX)
            KT = DT @ np.linalg.pinv(DX)

        control_gain[k] = K
        terminal_time_gain[k] = KT[0, :]
    return GainTable(
        var=var_grid,
        nominal_state=nominal_i.state,
        nominal_control=nominal_i.control,
        control_gain=control_gain,
        terminal_time_gain=terminal_time_gain,
        nominal_state_deriv=nominal_i.state_deriv,
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
        var=table.var,
        nominal_state=table.nominal_state,
        nominal_control=table.nominal_control,
        control_gain=table.control_gain,
        terminal_time_gain=table.terminal_time_gain,
    )


def load_gain_table(
    filename: str | Path,
) -> GainTable:
    """Load a previously generated guidance table."""

    data = np.load(filename)

    return GainTable(
        var=data["var"],
        nominal_state=data["nominal_state"],
        nominal_control=data["nominal_control"],
        control_gain=data["control_gain"],
        terminal_time_gain=data["terminal_time_gain"],
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("None")
   