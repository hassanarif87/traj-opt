import numpy as np

class OdeResult:
    def __init__(self,t,  y):
        self.y = y
        self.t = t


def integrate(fun, t_span, y0, t_eval=None, args=()):
    """
    Fixed-step RK4 integrator with a solve_ivp-like interface.

    Parameters
    ----------
    fun : callable
        RHS function: fun(t, y, *args) -> dydt
    t_span : tuple
        (t0, tf)
    y0 : array_like
        Initial state.
    t_eval : array_like, optional
        Times at which to return the solution.
        If None, uses 100 evenly spaced points.
    args : tuple
        Additional arguments passed to fun.

    Returns
    -------
    t_eval : array_like
        Times at which the solution is evaluated.
    y : array_like
        Solution values at each time in t_eval.
    """
    t0, tf = t_span
    y0 = np.asarray(y0, dtype=float)

    if t_eval is None:
        t_eval = np.linspace(t0, tf, 100)
    else:
        t_eval = np.asarray(t_eval, dtype=float)

    y = np.empty((y0.size, len(t_eval)))
    y[:, 0] = y0

    for i in range(1, len(t_eval)):
        t = t_eval[i - 1]
        h = t_eval[i] - t
        yn = y[:, i - 1]

        k1 = fun(t,          yn,             *args)
        k2 = fun(t + h/2,    yn + h*k1/2,    *args)
        k3 = fun(t + h/2,    yn + h*k2/2,    *args)
        k4 = fun(t + h,      yn + h*k3,      *args)

        y[:, i] = yn + h * (k1 + 2*k2 + 2*k3 + k4) / 6



    return t_eval, y