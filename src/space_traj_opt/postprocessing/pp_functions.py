## Post proessing
import numpy as np
import pandas as pd

from space_traj_opt.math.constants import EARTH_R
from space_traj_opt.math.orbital_calcs import rv_to_aei
from space_traj_opt.models.controller3d import flight_path_angle, lts_control
from space_traj_opt.models.models3d import dynamics_plant
from space_traj_opt.sims.config import load_config

from space_traj_opt.postprocessing.data_client import CSVClient

clients = [CSVClient("second_stage_ascent")]
plant_params = tuple(load_config("scenarios/second_stage_ascent.yaml").phases[0].model_params)

d_opt = np.asarray(clients[0].metadata["x_opt"])

time = clients[0].get_time()
r_eci = clients[0].get_channels(["r<x,y,z>"])
v_eci = clients[0].get_channels(["v<x,y,z>"])
m = np.asarray(clients[0].get_channels(["m"])).reshape(-1, 1)
v_eci_mag = np.linalg.norm(v_eci, axis=1)

state = np.hstack([r_eci, v_eci, m])
ctrl_params = d_opt[0:4]

a_values = []
e_values = []
i_values = []
fpa_values = []
u_values = []
dx_values = []
pitch_values = []
yaw_values = []
for p, v in zip(r_eci, v_eci):
    a, e, i = rv_to_aei(p, v)
    a_values.append(a)
    e_values.append(e)
    i_values.append(i)
    fpa_values.append(flight_path_angle(p, v))

for t, cur_x in zip(time, state):
    thrust_hat = lts_control(t, cur_x, ctrl_params)
    u_values.append(thrust_hat)
    pitch_values.append(np.atan2(thrust_hat[1], thrust_hat[0]))
    yaw_values.append(np.arctan2(thrust_hat[2], np.hypot(thrust_hat[0], thrust_hat[1])))

    dx_values.append(dynamics_plant(t, cur_x, thrust_hat, plant_params))

a = np.asarray(a_values)
e = np.asarray(e_values)
i = np.asarray(i_values)
fpa = np.asarray(fpa_values)
u = np.asarray(u_values)
dx = np.asarray(dx_values)
pitch = np.asarray(pitch_values)
yaw = np.asarray(yaw_values)
r_periapsis = a * (1.0 - e)
r_apoapsis = a * (1.0 + e)

h_periapsis = r_periapsis - EARTH_R
h_apoapsis = r_apoapsis - EARTH_R

results = {
    "a": a,
    "e": e,
    "i": i,
    "fpa": fpa,
    "v_eci_mag": v_eci_mag,
    "pitch": pitch,
    "yaw": yaw,
    "r_periapsis": r_periapsis,
    "r_apoapsis": r_apoapsis,
    "h_periapsis": h_periapsis,
    "h_apoapsis": h_apoapsis,
    "u_x": u[:, 0],
    "u_y": u[:, 1],
    "u_z": u[:, 2],
    "ax": dx[:, 3],
    "ay": dx[:, 4],
    "az": dx[:, 5],
    "m_dot": dx[:, 6],
}

clients[0].df = pd.concat(
    [clients[0].df.reset_index(drop=True), pd.DataFrame(results)],
    axis=1,
)
clients[0].df.to_csv(clients[0].out, index=False)

