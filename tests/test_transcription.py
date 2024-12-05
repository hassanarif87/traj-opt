import numpy as np
import pytest
from space_traj_opt.transcription import MultiShootingTranscription

def test_multishooting_construction():
    problem = MultiShootingTranscription(["phase0", "phase1", "phase2"])
    print(problem.t0_array.keys())
    assert ["phase0", "phase1", "phase2"] == list(problem.t0_array.keys())

    #    np.testing.assert_allclose(result, expected_result)

# Run the test
if __name__ == "__main__":
    pytest.main()
