import numpy as np
import pytest
from space_traj_opt.transcription import MultiShootingTranscription
from space_traj_opt.models import CtrlMode
from space_traj_opt.phases import DynEnum, Phase, PhaseDefect, TerminalConditions

def test_multishooting_construction():
    problem = MultiShootingTranscription(["phase0", "phase1", "phase2"], 5)
    assert ["phase0", "phase1", "phase2"] == list(problem.phase_names)

def test_repr_uses_dataclass_representations():
    problem = MultiShootingTranscription(["phase0"], num_states=2)
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER)
    terminal = TerminalConditions(np.array([1.0, 2.0]), [(0.0, 3.0)] * 2, [1.0] * 2)

    problem.add_phase("phase0", phase)
    problem.add_terminal(terminal)

    output = repr(problem)

    assert "phase_dict={'phase0': Phase(" in output
    assert "terminal_conditons=TerminalConditions(" in output

def test_build_uses_registered_phase_data():
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER)
    phase.set_controller(CtrlMode.ANGLE_STEER, np.array([1.0, 2.0]), [(0.0, 3.0)] * 2, [1.0] * 2)
    phase.set_state(DynEnum.DYNAMICS_2D, np.array([3.0, 4.0]), [(0.0, 5.0)] * 2, [1.0] * 2)
    phase.set_time(5.0, (1.0, 10.0))
    problem = MultiShootingTranscription(["phase0"], num_states=2)
    problem.add_phase("phase0", phase)
    problem.add_terminal(TerminalConditions(np.array([6.0, 7.0]), [(0.0, 8.0)] * 2, [1.0] * 2))

    d0, bounds, normalization, configs = problem.build()

    np.testing.assert_allclose(d0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert len(bounds) == len(d0)
    assert len(normalization) == len(d0)
    assert configs == [(CtrlMode.ANGLE_STEER, (0, 2), None, DynEnum.DYNAMICS_2D)]


def test_registers_adjacent_defect():
    phase0 = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER)
    phase1 = Phase("phase1", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER)
    problem = MultiShootingTranscription(["phase0", "phase1"], num_states=2)
    problem.add_phase("phase0", phase0)
    problem.add_phase("phase1", phase1)
    defect = PhaseDefect("defect", np.array([0.1, 0.2]), np.array([1.0, 1.0]))

    problem.add_defect("defect", (phase0, phase1), defect)

    assert problem.defect_dict["defect"] == ((phase0, phase1), defect)

def test_unpack_decision_var():
    problem = MultiShootingTranscription(["phase0"], num_states=4)
    decision_var = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    config = [CtrlMode.ANGLE_STEER, (0, 2), (2, 6), (6, 7)]

    u, x, t_terminal, control_law = problem.unpack_decision_var(decision_var, config)

    assert np.array_equal(u, decision_var[0:2])
    assert np.array_equal(x, decision_var[2:6])
    assert t_terminal == decision_var[6]
    assert control_law == CtrlMode.ANGLE_STEER

def test_normalize_decision_vec():
    decision_vector = np.array([1.0, 2.0, 3.0])
    bounds = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]
    normalization_vector = np.array([1.0, 2.0, 3.0])
    offset_vector = np.array([0.0, 1.0, 2.0])

    normalized_vector, normalized_bounds = MultiShootingTranscription.normalize_decision_vec(
        decision_vector, bounds, normalization_vector, offset_vector
    )

    expected_normalized_vector = np.array([1.0, 0.5, 0.33333333])
    expected_normalized_bounds = [(0.0, 2.0), (0.0, 1.0), (0.0, 0.66666667)]

    np.testing.assert_allclose(normalized_vector, expected_normalized_vector)
    np.testing.assert_allclose(normalized_bounds, expected_normalized_bounds)

def test_denormalize_decision_vec():
    normalized_vector = np.array([1.0, 0.5, 0.33333333])
    normalization_vector = np.array([1.0, 2.0, 3.0])
    offset_vector = np.array([0.0, 1.0, 2.0])

    denormalized_vector = MultiShootingTranscription.denormalize_decision_vec(
        normalized_vector, normalization_vector, offset_vector
    )

    expected_denormalized_vector = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(denormalized_vector, expected_denormalized_vector)

def test_traj_rollout():
    t_terminal = 1.0
    x0 = (0.0, 0.0, 0.0, 0.0, 10000.0)
    params = ((1000.0, 100.0), (CtrlMode.ANGLE_STEER, (0.5,)))

    solution = MultiShootingTranscription.traj_rollout(t_terminal, x0, params)

    assert solution.t.shape == (50,)
    assert solution.y.shape == (5, 50)
    np.testing.assert_allclose(solution.t, np.linspace(0.0, t_terminal, 50))
    np.testing.assert_allclose(solution.y[:, 0], x0)

# Run the tests
if __name__ == "__main__":
    pytest.main()
