import numpy as np
import pytest
from space_traj_opt.transcription import MultiShootingTranscription
from space_traj_opt.models import CtrlMode

def test_multishooting_construction():
    problem = MultiShootingTranscription(["phase0", "phase1", "phase2"], 5)
    assert ["phase0", "phase1", "phase2"] == list(problem.phase_names)

def test_set_dynamics_params():
    problem = MultiShootingTranscription(["phase0", "phase1"], num_states=4)
    params_phase0 = (1.0, 2.0, 3.0)
    params_phase1 = (4.0, 5.0, 6.0)

    problem.set_dynamics_params("phase0", params_phase0)
    problem.set_dynamics_params("phase1", params_phase1)

    assert problem.params["phase0"] == params_phase0
    assert problem.params["phase1"] == params_phase1

def test_set_phase_init_x():
    problem = MultiShootingTranscription(["phase0"], num_states=4)
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    bounds = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]
    norm_vec = [1.0, 1.0, 1.0, 1.0]

    problem.set_phase_init_x("phase0", x0, bounds, norm_vec)

    assert np.array_equal(problem.x0_array["phase0"], x0)
    assert problem.x0_array["phase0_bnds"] == bounds
    assert problem.x0_array["phase0_normvec"] == norm_vec

def test_set_phase_control():
    problem = MultiShootingTranscription(["phase0"], num_states=4)
    u0 = np.array([1.0, 2.0])
    bounds = [(0.0, 2.0), (1.0, 3.0)]
    norm_vec = [1.0, 1.0]

    problem.set_phase_control("phase0", CtrlMode.ANGLE_STEER, u0, bounds, norm_vec)

    assert np.array_equal(problem.u0_array["phase0"], u0)
    assert problem.u0_array["phase0_bnds"] == bounds
    assert problem.u0_array["phase0_normvec"] == norm_vec
    assert problem.phase_configs["phase0"] == [CtrlMode.ANGLE_STEER]

def test_set_phase_time():
    problem = MultiShootingTranscription(["phase0"], num_states=4)
    t0 = 10.0
    bounds = (5.0, 15.0)

    problem.set_phase_time("phase0", t0, bounds)

    assert problem.t0_array["phase0"] == t0
    assert problem.t0_array["phase0_bnds"] == bounds

def test_set_non_zero_defect():
    problem = MultiShootingTranscription(["phase0", "phase1"], num_states=4)
    defect_vec = np.array([0.1, 0.2, 0.3, 0.4])

    problem.set_non_zero_defect(("phase0", "phase1"), defect_vec)

    assert np.array_equal(problem.defects["phase1"], defect_vec)

def test_set_terminal_state():
    problem = MultiShootingTranscription(["phase0"], num_states=4)
    x_final = np.array([1.0, 2.0, 3.0, 4.0])
    bounds = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]
    norm_vec = [1.0, 1.0, 1.0, 1.0]

    problem.set_terminal_state(x_final, bounds, norm_vec)

    assert np.array_equal(problem.terminal_state, x_final)
    assert problem.terminal_bounds == bounds
    assert problem.terminal_normvec == norm_vec

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


# Run the tests
if __name__ == "__main__":
    pytest.main()
