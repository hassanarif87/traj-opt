import numpy as np

from space_traj_opt.math.constants import MU_EARTH
from space_traj_opt.math.orbital_calcs import rv_to_orbital_elements


def test_rv_to_orbital_elements_vectorizes_over_orbit_batches():
    r_vecs = np.array(
        [
            [7.0e6, 0.0, 0.0],
            [0.0, 7.1e6, 0.0],
            [8.0e6, 0.0, 0.0],
        ]
    )
    v_vecs = np.array(
        [
            [0.0, np.sqrt(MU_EARTH / 7.0e6), 0.0],
            [-7.4e3, 0.0, 0.8e3],
            [0.0, np.sqrt(2.0 * MU_EARTH / 8.0e6), 0.0],
        ]
    )

    batched = rv_to_orbital_elements(r_vecs, v_vecs)
    scalar_results = np.array(
        [
            rv_to_orbital_elements(r_vec, v_vec)
            for r_vec, v_vec in zip(r_vecs, v_vecs)
        ]
    )

    assert all(element.shape == (3,) for element in batched)
    np.testing.assert_allclose(np.asarray(batched).T, scalar_results)
    assert batched[3][0] == 0.0
    assert batched[4][0] == 0.0
    assert batched[5][0] == 0.0
    assert np.isinf(batched[0][2])