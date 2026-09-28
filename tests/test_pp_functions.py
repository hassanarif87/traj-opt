from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from space_traj_opt.postprocessing import post_proccess, pp_functions


def test_postprocess_trajectory_runs_configured_processors(monkeypatch, tmp_path):
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
        post_process=["process_3dtrajectory", "mark_processed"],
    )
    trajectory = pd.DataFrame(
        {"phase": [0, 1], "time": [0.0, 1.0], "x": [1.0, 2.0], "y": [3.0, 4.0]}
    )
    decision_vector = np.array([1, 2, 30, 31, 32, 3, 4, 40, 41, 42, 99])
    phase_calls = []
    processor_calls = []
    client = SimpleNamespace(
        metadata={"x_opt": decision_vector.tolist()},
        df=trajectory,
        out=tmp_path / "trajectory.csv",
    )

    def fake_process_phase(phase_df, ctrl_params, model_params, state_headers):
        phase_calls.append(
            (phase_df.index.tolist(), ctrl_params.copy(), model_params)
        )
        return pd.DataFrame({"processed": [model_params[0]]}, index=phase_df.index)

    def mark_processed(phase_df, decision_vector, scenario_config):
        assert "processed" in phase_df
        processor_calls.append("mark_processed")
        return phase_df.assign(marked=True)

    monkeypatch.setattr(pp_functions, "process_phase", fake_process_phase)
    monkeypatch.setattr(post_proccess, "CSVClient", lambda _: client)
    monkeypatch.setattr(
        post_proccess,
        "POSTPROCESSORS",
        {
            "process_3dtrajectory": pp_functions.process_3dtrajectory,
            "mark_processed": mark_processed,
        },
    )

    result = post_proccess.postprocess_trajectory("trajectory", config)

    assert [call[0] for call in phase_calls] == [[0], [1]]
    np.testing.assert_array_equal(phase_calls[0][1], [1, 2])
    np.testing.assert_array_equal(phase_calls[1][1], [3])
    assert [call[2] for call in phase_calls] == [(10.0,), (20.0, 21.0)]
    assert result["processed"].tolist() == [10.0, 20.0]
    assert processor_calls == ["mark_processed"]
    assert result["marked"].tolist() == [True, True]
    assert client.out.exists()


def test_empty_or_unknown_processors_skip_csv_load(monkeypatch):
    def fail_if_csv_loaded(_):
        raise AssertionError("CSV should not be loaded without a valid processor")

    monkeypatch.setattr(post_proccess, "CSVClient", fail_if_csv_loaded)

    assert post_proccess.postprocess_trajectory(
        "trajectory",
        SimpleNamespace(post_process=[]),
    ) is None

    with pytest.raises(ValueError, match="Unknown postprocessor 'missing'"):
        post_proccess.postprocess_trajectory(
            "trajectory",
            SimpleNamespace(post_process=["missing"]),
        )