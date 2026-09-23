import numpy as np

from space_traj_opt.models.models import CtrlMode
from space_traj_opt.optimization.phases import DynEnum
from space_traj_opt.optimization.utils import denormalize_decision_vec, normalize_decision_vec, traj_rollout


def test_normalize_decision_vec():
    decision_vector = np.array([1.0, 2.0, 3.0])
    bounds = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]
    normalization_vector = np.array([1.0, 2.0, 3.0])
    offset_vector = np.array([0.0, 1.0, 2.0])

    normalized_vector, normalized_bounds = normalize_decision_vec(
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

    denormalized_vector = denormalize_decision_vec(
        normalized_vector, normalization_vector, offset_vector
    )

    expected_denormalized_vector = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(denormalized_vector, expected_denormalized_vector)


def test_traj_rollout():
    t_terminal = 1000.0
    x0 = (0.0, 0.0, 0.0, 0.0, 1000.0)
    params = (DynEnum.DYNAMICS_2D, (1000.0, 100.0), (CtrlMode.ANGLE_STEER, (0.5,)))

    solution = traj_rollout(t_terminal, tuple(x0), params)
    print(solution.t.shape)
    assert solution.t.shape == (50,) 
    assert solution.y.shape == (5, 50)
    np.testing.assert_allclose(solution.t, np.linspace(0.0, t_terminal, 50))
    np.testing.assert_allclose(solution.y[:, 0], x0)
