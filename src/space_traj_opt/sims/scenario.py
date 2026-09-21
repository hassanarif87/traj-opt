import numpy as np
from numpy import array as arr
from numpy import deg2rad as d2r
from scipy.optimize import minimize

from space_traj_opt.models.models import CtrlMode
from space_traj_opt.optimization.phases import (
    DynEnum,
    Phase,
    PhaseDefect,
    TerminalConditions,
)
from space_traj_opt.optimization.problem import Problem
from space_traj_opt.optimization.transcription import MultiShootingTranscription
from space_traj_opt.optimization.utils import (
    denormalize_decision_vec,
    normalize_decision_vec,
)

s1_vch_params  = (224190.36000000002, 311.0)
s2_vch_params  = (25000.0, 343.0)


x0 = arr([
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.29495370e+04],
    [7.50000000e+00, 3.90000000e+02, 1.50000000e+00, 8.00000000e+01, 1.22000000e+04],
    [3.00000000e+04, 6.00000000e+04, 2.00000000e+03, 5.00000000e+02, 2.90245491e+03],
    [4.50000000e+04, 8.00000000e+04, 2.80000000e+03, 1.00000000e+03, 2.68894357e+03]
    ])

x_f = arr([1.00000000e+06, 2.00000000e+05, 7.78434281e+03, 0.00000000e+00, 5.07900937e+02])
x0_n_vec = arr([[200000., 200000.,   5000.,   1000.,   5000.]])



state_bounds = [(0, None), (0, None), (0, None), (0, None), (100, None)]

phase0 = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER ,s1_vch_params)
phase0.set_state(DynEnum.DYNAMICS_2D, x0[0], bounds=x0[0], norm_vec=x0_n_vec)
phase0.set_controller(
    CtrlMode.ANGLE_STEER,
    u0=d2r(89.5),
    bounds=[(d2r(85), d2r(89.8))],
    norm_vec=[np.pi],
)
phase0.set_time(10, bounds=10)

phase1 = Phase("phase1", DynEnum.DYNAMICS_2D, CtrlMode.ZERO_ALPHA, s1_vch_params)
phase1.set_state(DynEnum.DYNAMICS_2D, x0[1], bounds=state_bounds, norm_vec=x0_n_vec)
phase1.set_controller(CtrlMode.ZERO_ALPHA, u0=[], norm_vec=[])
phase1.set_time(100, bounds=(60, 180))

s2_sep_mass = 2902.454913676189
sep_bound = [(0, None), (0, None), (0, None), (0, None), (s2_sep_mass, s2_sep_mass)]
phase2 = Phase("phase2", DynEnum.DYNAMICS_2D, CtrlMode.LTS, s2_vch_params)
phase2.set_state(DynEnum.DYNAMICS_2D, x0[2], bounds=sep_bound, norm_vec=x0_n_vec)
phase2.set_controller(
    CtrlMode.LTS,
    u0=arr([-0.001, 1]),
    bounds=[(-0.1, 0.1), (-3, 3)],
    norm_vec=[0.1, np.pi / 2],
)


phase2.set_time(22, bounds=22)

phase3 = Phase("phase3", DynEnum.DYNAMICS_2D, CtrlMode.LTS, s2_vch_params)
phase3.set_state(DynEnum.DYNAMICS_2D, x0[3], norm_vec=x0_n_vec)
phase3.set_controller(
    CtrlMode.LTS,
    u0=arr([-0.001, 1]),
    bounds=[(-0.1, 0.1), (-3, 3)],
    norm_vec=[0.1, np.pi / 2],
)
phase3.set_time(320)


problem_builder = MultiShootingTranscription(["phase0", "phase1", "phase2", "phase3"], 5)

for phase in (phase0, phase1, phase2, phase3):
    problem_builder.add_phase(phase.name, phase)

fairing_mass = 50.0
s1_dry_mass = 1076.47308279

problem_builder.add_defect(
    "stage_1_separation",
    ("phase1", "phase2"),
    PhaseDefect(
        "stage_1_separation", 
        arr([0, 0, 0, 0, s1_dry_mass]), 
        arr([100000, 100000, 8000, 5000, 1000])
        ),
)

problem_builder.add_defect(
    "fairing_separation",
    ("phase2", "phase3"),
    PhaseDefect(
        "fairing_separation", 
        arr([0, 0, 0, 0, fairing_mass]), 
        arr([100000, 100000, 8000, 5000, 1000])
        ),
)

v_circ = 7784.342809549733
circ_orbit_alt = 200_000.0 
problem_builder.add_terminal(
    TerminalConditions.set_terminal(
        x_final=x_f,
        bounds=arr([ None, circ_orbit_alt, v_circ,0 , None]),
        norm_vec=x0_n_vec,
    )
)


d0, d_bounds, normalization_vec, full_params = problem_builder.build()
d0_norm, d_bounds_norm = normalize_decision_vec(
    d0,
    d_bounds,
    normalization_vec,
)

problem = Problem(
    d0_norm, 
    d_bounds_norm, 
    normalization_vec,
    5,5,4)

constraints = [{'type': 'eq', 'fun': problem.dynamics_knot_constrant, 'args':(full_params,) },]


result = minimize(
    problem.objective, 
    problem.d0_guess_normalized, 
    jac= problem.jac_objective,
    method='SLSQP', 
    bounds=problem.d_bounds_norm, 
    constraints=constraints,
    options = {"maxiter": 500, "disp": True},
    args=(full_params,)
)

x_opt = denormalize_decision_vec(result.x, normalization_vec)

sol_list = problem.full_traj_rollout(x_opt, full_params)

from space_traj_opt.postprocessing.post_proccess import sol_to_csv

name = "traj_opt_sol"
sol_to_csv(sol_list, ["x", "y", "vx", "vy", "m"], name)
