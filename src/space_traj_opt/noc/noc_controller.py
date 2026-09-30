from dataclasses import dataclass

import numpy as np

from space_traj_opt.models.controller3d import dcm_rsw2eci, dir_from_pitch_yaw
from space_traj_opt.models.models3d import dynamics_plant
from space_traj_opt.noc.noc_gain_generation import GainTable, load_gain_table


@dataclass
class NOCGuidance:
    table: GainTable

    @classmethod
    def from_gain_table(cls, path: str):
        table = load_gain_table(path)
        return cls(table)

    def update(self, t, x):

        # Recalc  accel from plant, in a real alg ths would be from the IMU.
        mass = x[6] 
        # Pitch and yaw are parametrized in the rsw frame.
        pitch, yaw = self.table.guidance(-mass, x[0:6] )
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




