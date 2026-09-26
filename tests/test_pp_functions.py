from types import SimpleNamespace

import numpy as np
import pandas as pd

from space_traj_opt.postprocessing import pp_functions


def test_process_trajectory_dispatches_each_phase_with_its_own_parameters(monkeypatch):
    config = SimpleNamespace(
        phases=[
            SimpleNamespace(
                controller=SimpleNamespace(guess=[0.0, 0.0]),
                model_params=[10.0],
            ),
            SimpleNamespace(
                controller=SimpleNamespace(guess=[0.0]),
                model_params=[20.0, 21.0],
            ),
        ],
        num_states=2,
        state_headers=["x", "y"],
    )
    trajectory = pd.DataFrame(
        {"phase": [0, 1], "time": [0.0, 1.0], "x": [1.0, 2.0], "y": [3.0, 4.0]}
    )
    decision_vector = np.array([1, 2, 30, 31, 32, 3, 4, 40, 41, 42, 99])
    calls = []

    def fake_process_phase(phase_df, ctrl_params, model_params, state_headers):
        calls.append((phase_df.index.tolist(), ctrl_params.copy(), model_params))
        return pd.DataFrame(
            {"processed": [model_params[0]]}, index=phase_df.index
        )

    monkeypatch.setattr(pp_functions, "process_phase", fake_process_phase)

    result = pp_functions.process_trajectory(trajectory, decision_vector, config)

    assert [call[0] for call in calls] == [[0], [1]]
    np.testing.assert_array_equal(calls[0][1], [1, 2])
    np.testing.assert_array_equal(calls[1][1], [3])
    assert [call[2] for call in calls] == [(10.0,), (20.0, 21.0)]
    assert result["processed"].tolist() == [10.0, 20.0]