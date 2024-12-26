import numpy as np
import pytest

from space_traj_opt.models import (
    dynamics,
    dynamics_plant,
    get_drag_coeff,
    get_atm,
    angle_steering,
    zero_alpha,
    polynomial_steering,
    lts_control,
    CtrlMode,
    control
)

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
    x = np.array([0.0, 0.0, 0.0, 0.0, 10000.0])
    u = 0.5
    params = (1000.0, 100.0)
    
    res_dx = dynamics_plant(t, x, u, params)
    assert len(res_dx) == 5
    np.testing.assert_allclose(res_dx, np.array([ 0., 0., 0.08775826, -9.75870745, -1.01971621]))

def test_get_drag_coeff():
    mach = 0.5
    result = get_drag_coeff(mach)
    assert result == 0.38708332500000003

def test_get_atm():
    altitude = 1000.0
    speed_of_sound, air_den = get_atm(altitude)
    assert speed_of_sound == 300
    assert air_den == 1.088126523521983

def test_angle_steering():
    t = 0.0
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    params = (0.5,)
    
    result = angle_steering(t, x, params)
    expected = 0.5
    
    assert result == expected

def test_zero_alpha():
    t = 0.0
    x = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    params = ()
    
    result = zero_alpha(t, x, params)
    expected = np.arctan2(1.0, 1.0)
    
    assert result == expected

def test_polynomial_steering():
    t = 2.0
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    params = (1.0, 2.0, 3.0)
    
    result = polynomial_steering(t, x, params)
    expected = 1.0 * t**2 + 2.0 * t + 3.0
    
    assert result == expected

def test_lts_control():
    t = 2.0
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    params = (1.0, 2.0)
    
    result = lts_control(t, x, params)
    expected = np.arctan(1.0 * t + 2.0)
    
    assert result == expected

def test_control():
    t = 0.0
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    params = (CtrlMode.ANGLE_STEER, (0.5,))
    
    result = control(t, x, params)
    expected = 0.5
    
    assert result == expected

# Run the tests
if __name__ == "__main__":
    pytest.main()