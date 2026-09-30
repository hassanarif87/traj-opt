from dataclasses import dataclass, field

import numpy as np

from space_traj_opt.models.controller3d import dcm_rsw2eci, dir_from_pitch_yaw
from space_traj_opt.models.models3d import dynamics_plant
from space_traj_opt.noc.noc_gain_generation import GainTable, load_gain_table


@dataclass
class NOCGuidance:
    table: GainTable
    control_gradient_alpha: float = 0.01
    previous_control: np.ndarray | None = field(default=None, init=False)
    control_gradient: np.ndarray = field(
        default_factory=lambda: np.zeros(2), init=False
    )

    def __post_init__(self):
        if not 0.0 <= self.control_gradient_alpha < 1.0:
            raise ValueError("control_gradient_alpha must be in [0, 1)")

    @classmethod
    def from_gain_table(cls, path: str):
        table = load_gain_table(path)
        return cls(table)

    def update(self, t, x):
        mass = x[6]
        if mass < 700 and self.previous_control is not None:
            control = self.previous_control + self.control_gradient
            self.previous_control = control.copy()
        else:
            control = np.asarray(
                self.table.guidance(-mass, x[0:6]), dtype=float
            )
            if self.previous_control is not None:
                control_delta = control - self.previous_control
                alpha = self.control_gradient_alpha
                self.control_gradient = (
                    alpha * self.control_gradient
                    + (1.0 - alpha) * control_delta
                )
            self.previous_control = control.copy()

        pitch, yaw = control
        thrust_hat_rsw = dir_from_pitch_yaw(pitch, yaw)

        # Pitch yaw to thrust vector
        return dcm_rsw2eci(x[0:3], x[3:6]) @ thrust_hat_rsw

@dataclass
class Vehicle:
    controller: NOCGuidance
    plant_params: tuple

    def run(self, t, x):
        u = self.controller.update(t,x)
        dx = dynamics_plant(t,x,u,self.plant_params)
        return dx




