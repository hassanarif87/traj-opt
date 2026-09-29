from dataclasses import dataclass

import numpy as np

from space_traj_opt.models.models3d import dynamics_plant
from space_traj_opt.noc.noc_gain_generation import GainTable, load_gain_table


@dataclass
class NOCGuidance:
    table: GainTable

    @classmethod
    def from_gain_table(cls, path: str):
        cls.table = load_gain_table(path)

    def run(self, t, x):

        # Recalc  accel from plant, in a real alg ths would be from the IMU.
        dx = dynamics_plant(t, x)
        sensed_accel = np.linalg.norm( dx[3:6]) 
        ## Pitch and yaw are parametrized in the rsw frame.
        pitch, yaw = self.table.guidance(sensed_accel, x[0:6] )

        # Pitch yaw to thrust vector
        return thrust_hat_eci