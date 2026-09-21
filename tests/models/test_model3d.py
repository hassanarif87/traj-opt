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
    x = np.array([0.0, 0.0, 0.0, 0.0, 10000.0])
    params = ((1000.0, 100.0), (CtrlMode.ANGLE_STEER, (0.5,)))
    
    res_dx = dynamics(t, x, params)
    assert len(res_dx) == 5
    print(res_dx)
    np.testing.assert_allclose(res_dx, np.array([ 0., 0., 0.08775826, -9.75870745, -1.01971621]))

def test_dynamics_plant():
    t = 0.0
    x = np.array([EARTH_RADIUS, 0.0, 0.0, 1000, 10.0, 0.0, 10000.0])
    u = np.array([100,0,0])
    params = (1000.0, 100.0)
    
    res_dx = dynamics_plant(t, x, u, params)
    assert len(res_dx) == 7
    np.testing.assert_allclose(res_dx, np.array([ 0., 0., 0.08775826, -9.75870745, -1.01971621]))

# Run the tests
if __name__ == "__main__":
    pytest.main()