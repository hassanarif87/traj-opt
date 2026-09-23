import numpy as np
import pytest

from space_traj_opt.models.controller3d import CtrlMode
from space_traj_opt.models.models3d import (
    dynamics,
    dynamics_plant,
)
from space_traj_opt.reports.orbit_plot import EARTH_RADIUS


def test_dynamics():
    t = 0.0
    x = np.array([
        EARTH_RADIUS+1e3, 0.0, 0.0, 
        1000.0, 0.0, 0.0,
        1000])
    params = ((1000.0, 100.0, False), (CtrlMode.ANGLE_STEER, (0,0,-1)))
    
    res_dx = dynamics(t, x, params)
    assert len(res_dx) == 7
    print(res_dx)
    np.testing.assert_allclose(
        res_dx, 
        np.array([ 1000,  0.0,  0.0, -8.79521374e+00, 0.0, 0.0, -1.01971621]), 
        atol = 1e-9)

def test_dynamics_plant():
    t = 0.0
    x = np.array([EARTH_RADIUS, 0.0, 0.0, 1000, 10.0, 0.0, 10000.0])
    u = np.array([100,0,0])
    params = (1000.0, 100.0, False)
    
    res_dx = dynamics_plant(t, x, u, params)
    assert len(res_dx) == 7

    # Hand calculation:
    #   dr/dt = v = [1000, 10, 0] m/s
    #   g_x = -MU_EARTH / EARTH_RADIUS**2 = -9.798285479187 m/s^2
    #   thrust_x / m = (1000 N * 100) / 10000 kg = 10 m/s^2
    #   a_x = g_x + thrust_x / m = 0.201714520813 m/s^2
    #   mdot = -thrust / (STANDARD_GRAV * Isp)
    #        = -1000 / (9.80665 * 100) = -1.019716212978 kg/s
    np.testing.assert_allclose(
        res_dx,
        np.array([
            1000.0,
            10.0,
            0.0,
            0.201714520813,
            0.0,
            0.0,
            -1.019716212978,
        ]),
        atol=1e-12,
    )

# Run the tests
if __name__ == "__main__":
    pytest.main()