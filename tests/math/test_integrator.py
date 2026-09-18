import numpy as np

from space_traj_opt.math.integrator import integrate


def test_integrate_constant_derivative():
    def rhs(t, y):
        return np.array([1.0])

    t_span = (0.0, 1.0)
    y0 = np.array([0.0])
    t_eval = np.linspace(0.0, 1.0, 5)

    t, y = integrate(rhs, t_span, y0, t_eval=t_eval)

    expected = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    np.testing.assert_allclose(y, expected)
    np.testing.assert_allclose(t, t_eval)


def test_integrate_linear_ode_matches_exponential_solution():
    def rhs(t, y):
        return np.array([-y[0]])

    t_span = (0.0, 1.0)
    y0 = np.array([1.0])
    t_eval = np.linspace(0.0, 1.0, 11)

    t, y = integrate(rhs, t_span, y0, t_eval=t_eval)

    expected = np.exp(-t_eval)
    np.testing.assert_allclose(y[0], expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(t, t_eval)


def test_integrate_uses_default_t_eval_length():
    def rhs(t, y):
        return np.array([2.0])

    t_span = (0.0, 1.0)
    y0 = np.array([0.0])

    t, y = integrate(rhs, t_span, y0)

    assert len(t) == 100
    assert y.shape == (1, 100)
    np.testing.assert_allclose(y[:, 0], [0.0])
    np.testing.assert_allclose(y[:, -1], [2.0])


def test_integrate_passes_args_to_rhs():
    def rhs(t, y, scale):
        return np.array([scale * y[0]])

    t_span = (0.0, 1.0)
    y0 = np.array([1.0])
    t_eval = np.linspace(0.0, 1.0, 5)

    t, y = integrate(rhs, t_span, y0, t_eval=t_eval, args=(2.0,))

    expected = np.array([[1.0, np.exp(0.5), np.exp(1.0), np.exp(1.5), np.exp(2.0)]])
    np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-8)