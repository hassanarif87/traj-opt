import pytest
import numpy as np
from space_traj_opt.math.constants import OMEGA_EARTH, SMA_EARTH, SEMI_MINOR_AXIS_EARTH

from space_traj_opt.math.coordinates import (
    eci2ecef, ecef2eci, ecef2lla, lla2ecef,
    quat_eci2ecef, quat_ecef2eci, quat_ecef2ned,
    quat_eci2ned, quat_ned2eci, dcm_rsw2eci
)
from space_traj_opt.math.quaternion import q_from_axisangle, quat2dcm


class TestECI2ECEF:
    def test_zero_time(self):
        r_eci = np.array([SMA_EARTH, 0.0, 0.0])
        r_ecef = eci2ecef(r_eci, 0)
        np.testing.assert_array_almost_equal(r_ecef, r_eci)

    def test_quarter_rotation(self):
        r_eci = np.array([SMA_EARTH, 0.0, 0.0])
        t = np.pi / 2 / OMEGA_EARTH
        r_ecef = eci2ecef(r_eci, t)
        np.testing.assert_array_almost_equal(r_ecef, np.array([0.0, -SMA_EARTH, 0.0]), decimal=5)

    def test_z_component_unchanged(self):
        r_eci = np.array([1.0, 2.0, 5.0])
        r_ecef = eci2ecef(r_eci, 100)
        assert r_ecef[2] == r_eci[2]


class TestECEF2ECI:
    def test_zero_time(self):
        r_ecef = np.array([1.0, 0.0, 0.0])
        r_eci = ecef2eci(r_ecef, 0)
        np.testing.assert_array_almost_equal(r_eci, r_ecef)

    def test_roundtrip_conversion(self):
        r_eci_orig = np.array([6.378e6, 1e6, 2e6])
        t = 1000
        r_ecef = eci2ecef(r_eci_orig, t)
        r_eci_back = ecef2eci(r_ecef, t)
        np.testing.assert_array_almost_equal(r_eci_orig, r_eci_back, decimal=5)


class TestECEF2LLA:
    def test_equator_prime_meridian(self):
        r_ecef = np.array([SMA_EARTH, 0.0, 0.0])
        lat, lon, alt = ecef2lla(r_ecef)
        assert abs(lat) < 1e-10
        assert abs(lon) < 1e-10
        assert abs(alt) < 1e-5

    def test_north_pole(self):
        # At the north pole, x=y=0 and z is approximately SEMI_MINOR_AXIS_EARTH
        r_ecef = np.array([0.0, 0.0, SEMI_MINOR_AXIS_EARTH])
        lat, lon, alt = ecef2lla(r_ecef)
        assert abs(lat - np.pi/2) < 1e-10
        assert abs(alt) < 1e-5

    def test_longitude_range(self):
        r_ecef = np.array([0.0, SMA_EARTH, 0.0])
        lat, lon, alt = ecef2lla(r_ecef)
        assert -np.pi <= lon <= np.pi


class TestLLA2ECEF:
    def test_equator_prime_meridian(self):
        lat, lon, alt = 0.0, 0.0, 0.0
        r_ecef = lla2ecef(lat, lon, alt)
        expected = np.array([SMA_EARTH, 0.0, 0.0])
        np.testing.assert_array_almost_equal(r_ecef, expected, decimal=5)

    def test_north_pole(self):
        lat, lon, alt = np.pi/2, 0.0, 0.0
        r_ecef = lla2ecef(lat, lon, alt)
        expected = np.array([0.0, 0.0, SEMI_MINOR_AXIS_EARTH])
        np.testing.assert_array_almost_equal(r_ecef, expected, decimal=5)


    def test_lla_ecef_roundtrip(self):
        # Number of random test points
        N = 1000

        # Random latitude in [-90, 90] degrees
        lat = np.deg2rad(np.random.uniform(-90.0, 90.0, size=N))

        # Random longitude in [-180, 180] degrees
        lon = np.deg2rad(np.random.uniform(-180.0, 180.0, size=N))

        # Random altitude in meters (-500m to 50 km)
        alt = np.random.uniform(-500.0, 50_000.0, size=N)

        # Convert to ECEF
        ecef = np.zeros((N,3))
        for i in range(N):
            ecef[i] = lla2ecef(lat[i], lon[i], alt[i])

        # Convert back
        lat2 = np.zeros(N)
        lon2 = np.zeros(N)
        alt2 = np.zeros(N)

        for i in range(N):
            lat2[i], lon2[i], alt2[i] = ecef2lla(ecef[i])

        # Normalize longitude wrap-around
        lon_diff = (lon2 - lon + 180) % 360 - 180

        # Assertions (tolerances depend on algorithm)
        assert np.max(np.abs(lat - lat2)) < 1e-6      # ~0.1 meter
        assert np.max(np.abs(lon_diff)) < 1e-6
        assert np.max(np.abs(alt - alt2)) < 0.01     # 10 mm-level


class TestQuatECI2ECEF:
    def test_zero_time(self):
        q = quat_eci2ecef(0)
        np.testing.assert_array_almost_equal(q, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_eci2ecef_quater_rotation(self):
        t = np.pi / 4 / OMEGA_EARTH
        quat_ecef2eci = quat_eci2ecef(t)
        expected = np.array([np.cos(np.pi/8), 0.0, 0.0, np.sin(np.pi/8)])
        np.testing.assert_array_almost_equal(quat_ecef2eci, expected, decimal=0.001)


class TestQuatECEF2ECI:
    def test_conjugate_relationship(self):
        t = 50
        q_eci2ecef = quat_eci2ecef(t)
        q_ecef2eci = quat_ecef2eci(t)
        # q_ecef2eci should be conjugate of q_eci2ecef
        np.testing.assert_array_almost_equal(q_ecef2eci, np.array([q_eci2ecef[0], -q_eci2ecef[1], -q_eci2ecef[2], -q_eci2ecef[3]]))

class TestNED: 
    # def test_north_pole(self):
    #     q = quat_ecef2ned([0.0, 0.0, SEMI_MINOR_AXIS_EARTH])
    #     print(q)
    #     norm = np.linalg.norm(q)
    #     assert abs(norm - 1.0) < 1e-10

    def test_zero_latitude_longitude(self):
        q = quat_ecef2ned(0,0)
        print(q)
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 1e-10

        # 90 rotation about y
        q_expected =  q_from_axisangle(-np.pi/2, np.array([0, 1, 0]))

        np.testing.assert_array_almost_equal(q, q_expected)
    

    def test_zero_latitude_45_longitude(self):
        q = quat_ecef2ned(0, np.pi/4)
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 1e-10

        dcm = quat2dcm(q)

        one_by_sqr_2 = 1/np.sqrt(2)
        # Visual sketch
        x_basis = np.array([ 0.,  0., 1.]) 
        y_basis = np.array([ -one_by_sqr_2, one_by_sqr_2, 0.])
        z_basis = np.array([ -one_by_sqr_2,  -one_by_sqr_2,  0.])
        np.testing.assert_array_almost_equal(dcm[0,:], x_basis,decimal=9)
        np.testing.assert_array_almost_equal(dcm[1,:], y_basis,decimal=9)
        np.testing.assert_array_almost_equal(dcm[2,:], z_basis,decimal=9)    
    def test_45_latitude_zero_longitude(self):
        q = quat_ecef2ned(np.pi/4, 0)
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 1e-10

        dcm = quat2dcm(q)
        one_by_sqr_2 = 1/np.sqrt(2)

        # Visual sketch
        x_basis = np.array([ -one_by_sqr_2,  0., one_by_sqr_2]) 
        y_basis = np.array([ 0., 1., 0.])
        z_basis = np.array([ -one_by_sqr_2,  0.,  -one_by_sqr_2])

        np.testing.assert_array_almost_equal(dcm[0,:], x_basis,decimal=9)
        np.testing.assert_array_almost_equal(dcm[1,:], y_basis,decimal=9)
        np.testing.assert_array_almost_equal(dcm[2,:], z_basis,decimal=9)


class TestQuatNED2ECI:
    def test_conjugate_relationship(self):
        x = np.array([6.378e6, 1e6, 2e6])
        t = 100
        q_eci2ned = quat_eci2ned(t, x)
        q_ned2eci = quat_ned2eci(t, x)
        np.testing.assert_array_almost_equal(q_ned2eci, np.array([q_eci2ned[0], -q_eci2ned[1], -q_eci2ned[2], -q_eci2ned[3]]))


class TestDCMRSW2ECI:
    def test_orthogonal_matrix(self):
        r_vec = np.array([6.378e6, 6.378e6, 0.0])
        v_vec = np.array([1000.0, 7.546e3, 0.0])
        dcm = dcm_rsw2eci(r_vec, v_vec)
        # Check orthogonality: DCM^T * DCM = I
        identity = dcm.T @ dcm
        np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=10)

    def test_direction(self):
        r_vec = np.array([6.378e6, 0.0, 0.0])
        v_vec = np.array([0.0, 7.546e3, 0.0])
        dcm = dcm_rsw2eci(r_vec, v_vec)
        np.testing.assert_array_almost_equal(dcm, np.eye(3), decimal=10)



# Run the tests
if __name__ == "__main__":
    pytest.main()
