import numpy as np

from space_traj_opt.models.controller3d import (
    CtrlMode,
    aero_steering,
    control,
    dir_from_pitch_yaw,
    flight_path_angle,
    lts_control,
    ned_steering,
)
from space_traj_opt.math.coordinates import dcm_rsw2eci, quat_ned2eci
from space_traj_opt.math.quaternion import q_rotate_frame


def test_flight_path_angle_for_oblique_velocity_vector():
    r_vec = np.array([1.0, 0.0, 0.0])
    v_vec = np.array([3.0, 4.0, 0.0])

    gamma = flight_path_angle(r_vec, v_vec)

    assert np.isclose(gamma, np.arctan2(3.0, 4.0))


def test_dir_from_pitch_yaw_matches_expected_rsw_vector():
    pitch = 0.3
    yaw = -0.7

    result = dir_from_pitch_yaw(pitch, yaw)

    expected = np.array([
        np.cos(pitch) * np.cos(yaw),
        np.sin(pitch) * np.cos(yaw),
        np.sin(yaw),
    ])
    np.testing.assert_allclose(result, expected)


def test_lts_control_returns_unit_thrust_vector():
    t = 2.0
    x = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    params = (1.0, 0.5, 0.25, -0.3)

    result = lts_control(t, x, params)

    pitch = np.arctan(1.0 * t + 0.5)
    yaw = np.arctan(0.25 * t + -0.3)
    expected_rsw = dir_from_pitch_yaw(np.pi / 2 - pitch, yaw)
    expected_eci = dcm_rsw2eci(x[0:3], x[3:6]) @ expected_rsw
    expected = expected_eci / np.linalg.norm(expected_eci)

    np.testing.assert_allclose(result, expected)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_ned_steering_rotates_fixed_ned_direction_into_eci():
    t = 0.0
    x = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    params = (0.0, 0.0, -1.0)

    result = ned_steering(t, x, params)
   
    np.testing.assert_allclose(result, np.array([1., 0., 0.]), atol=1e-10)


# TODO: properly test phase
# def test_aero_steering_returns_unit_vector_for_zero_alpha_beta():
#     t = 0.0
#     x = np.array([
#         1.0e7, 0.0, 0.0,
#         0.0, 7.5e3, 0.0,
#     ])
#     params = (0.0, 0.0)
# 
#     result = aero_steering(t, x, params)
# 
#     assert np.isclose(np.linalg.norm(result), 1.0)
#     assert np.all(np.isfinite(result))


def test_control_dispatches_angle_steer_mode():
    t = 0.0
    x = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    params = (CtrlMode.ANGLE_STEER, (0.0, 0.0, 1.0))

    result = control(t, x, params)
    expected = ned_steering(t, x, params[1])

    np.testing.assert_allclose(result, expected)


def test_control_dispatches_lts_mode():
    t = 2.0
    x = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    params = (CtrlMode.LTS, (1.0, 0.5, 0.25, -0.3))

    result = control(t, x, params)
    expected = lts_control(t, x, params[1])

    np.testing.assert_allclose(result, expected)


def test_control_falls_back_to_default_vector_for_unknown_mode():
    t = 0.0
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    params = (99, ())

    result = control(t, x, params)

    np.testing.assert_allclose(result, np.array([1.0, 0.0, 0.0]))
