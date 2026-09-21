import numpy as np

from space_traj_opt.models.models import CtrlMode
from space_traj_opt.optimization.problem import Problem


def test_unpack_decision_var():
    problem = Problem(np.array([]), np.array([]), np.array([]), num_states=4, num_terminal_states=1, num_phases=1)
    decision_var = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    config = [CtrlMode.ANGLE_STEER, (0, 2), (2, 6), (6, 7)]

    u, x, t_terminal, control_law = problem.unpack_decision_var(decision_var, config)

    assert np.array_equal(u, decision_var[0:2])
    assert np.array_equal(x, decision_var[2:6])
    assert t_terminal == decision_var[6]
    assert control_law == CtrlMode.ANGLE_STEER
