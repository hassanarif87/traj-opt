import pytest
from scipy.integrate import OdeSolution
import numpy as np
from space_traj_opt.utils import unpack_sol_list

def test_unpack_sol_list():
    # Create mock OdeSolution objects
    t1 = np.array([0, 1, 2])
    y1 = np.array([[8, 9, 10], [11, 12, 13]])
    sol1 = OdeSolution(t1, y1)
    sol1.t = t1
    sol1.y = y1

    t2 = np.array([0, 1, 2])
    y2 = np.array([[8.1, 9.1, 10.1], [11.1, 12.1, 13.1]])
    sol2 = OdeSolution(t2, y2)
    sol2.t = t2
    sol2.y = y2

    sol_list_in = [sol1, sol2]

    x_list, y_list = unpack_sol_list(sol_list_in, 0)


    # Expected results
    expected_x_list = [np.array([0, 1, 2]), np.array([2, 3, 4])]
    expected_y_list = [np.array([8, 9, 10]), np.array([8.1, 9.1, 10.1])]
    # Check if the results match the expected values
    for x, expected_x in zip(x_list, expected_x_list):
        np.testing.assert_array_equal(x, expected_x)

    for y, expected_y in zip(y_list, expected_y_list):
        np.testing.assert_array_equal(y, expected_y)

if __name__ == "__main__":
    pytest.main()