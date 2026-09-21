import numpy as np

from space_traj_opt.models.models import CtrlMode
from space_traj_opt.optimization.phases import DynEnum, Phase, TerminalConditions
from space_traj_opt.optimization.problem import Problem
from space_traj_opt.optimization.transcription import MultiShootingTranscription
from space_traj_opt.optimization.utils import normalize_decision_vec


def test_unpack_decision_var():
    phase = Phase(
        name="phase0",
        dynamics_type=DynEnum.DYNAMICS_2D,
        control_type=CtrlMode.ANGLE_STEER,
        model_params=(),
    )
    phase.set_controller(CtrlMode.ANGLE_STEER, np.array([1.0, 2.0]), [(0.0, 3.0)] * 2, [1.0] * 2)
    phase.set_state(DynEnum.DYNAMICS_2D, np.array([3.0, 4.0]), [(0.0, 5.0)] * 2, [1.0] * 2)
    phase.set_time(5.0, (1.0, 10.0))

    builder = MultiShootingTranscription(["phase0"], num_states=2)
    builder.add_phase("phase0", phase)
    builder.add_terminal(TerminalConditions(np.array([6.0, 7.0]), [(0.0, 8.0)] * 2, [1.0] * 2))

    decision_var, bounds, normalization_vec, configs = builder.build()
    decision_var_norm, bounds_norm = normalize_decision_vec(decision_var, bounds, normalization_vec)

    problem = Problem(
        decision_var_norm,
        bounds_norm,
        normalization_vec,
        num_states=2,
        num_terminal_states=2,
        num_phases=1,
    )

    u, x, t_terminal, control_law = problem.unpack_decision_var(decision_var, configs[0])

    assert np.array_equal(u, decision_var[0:2])
    assert np.array_equal(x, decision_var[2:4])
    assert t_terminal == decision_var[4]
    assert control_law == CtrlMode.ANGLE_STEER
