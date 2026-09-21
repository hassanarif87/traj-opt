import numpy as np

from space_traj_opt.models.models import CtrlMode
from space_traj_opt.optimization.phases import DynEnum, Phase, PhaseDefect, TerminalConditions


def test_set_state():
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    bounds = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]
    norm_vec = [1.0, 1.0, 1.0, 1.0]

    phase.set_state(DynEnum.DYNAMICS_2D, x0, bounds, norm_vec)

    assert np.array_equal(phase.x_guess, x0)
    assert phase.x_bounds == bounds
    assert phase.x_normalize == norm_vec
    assert phase.dynamics_type is DynEnum.DYNAMICS_2D


def test_set_controller():
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())
    u0 = np.array([1.0, 2.0])
    bounds = [(0.0, 2.0), (1.0, 3.0)]
    norm_vec = [1.0, 1.0]

    phase.set_controller(CtrlMode.ANGLE_STEER, u0, bounds, norm_vec)

    assert np.array_equal(phase.u_guess, u0)
    assert phase.u_bounds == bounds
    assert phase.u_normalize == norm_vec
    assert phase.control_type is CtrlMode.ANGLE_STEER


def test_set_time():
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())

    phase.set_time(10.0, (5.0, 15.0))

    assert phase.t_guess == 10.0
    assert phase.t_bounds == (5.0, 15.0)

def test_set_time_nobounds():
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())

    phase.set_time(10.0)

    assert phase.t_guess == 10.0
    assert phase.t_bounds == (0.0, None)

def test_set_time_fixed_bound():
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())

    phase.set_time(10.0, 10.0)

    assert phase.t_guess == 10.0
    assert phase.t_bounds == (10.0, 10.0)

def test_terminal_conditions():
    x_final = np.array([1.0, 2.0, 3.0, 4.0])
    bounds = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]
    norm_vec = [1.0, 1.0, 1.0, 1.0]
    terminal = TerminalConditions(x_final, bounds, norm_vec)

    assert np.array_equal(terminal.x_final, x_final)
    assert terminal.bounds == bounds
    assert terminal.norm_vec == norm_vec


def test_phase_defect():
    defect = PhaseDefect("defect", np.array([0.1, 0.2]), np.array([1.0, 1.0]))

    assert defect.name == "defect"
    np.testing.assert_allclose(defect.defect, [0.1, 0.2])
    np.testing.assert_allclose(defect.defect_norm_vec, [1.0, 1.0])
