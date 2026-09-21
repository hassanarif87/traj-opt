import numpy as np
import pytest
from space_traj_opt.optimization.transcription import MultiShootingTranscription
from space_traj_opt.models.models import CtrlMode
from space_traj_opt.optimization.phases import DynEnum, Phase, PhaseDefect, TerminalConditions

def test_multishooting_construction():
    problem = MultiShootingTranscription(["phase0", "phase1", "phase2"], 5)
    assert ["phase0", "phase1", "phase2"] == list(problem.phase_names)

def test_repr_uses_dataclass_representations():
    problem = MultiShootingTranscription(["phase0"], num_states=2)
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())
    terminal = TerminalConditions(np.array([1.0, 2.0]), [(0.0, 3.0)] * 2, [1.0] * 2)

    problem.add_phase("phase0", phase)
    problem.add_terminal(terminal)

    output = repr(problem)

    assert "phase_dict={'phase0': Phase(" in output
    assert "terminal_conditons=TerminalConditions(" in output

def test_build_uses_registered_phase_data():
    phase = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())
    phase.set_controller(CtrlMode.ANGLE_STEER, np.array([1.0, 2.0]), [(0.0, 3.0)] * 2, [1.0] * 2)
    phase.set_state(DynEnum.DYNAMICS_2D, np.array([3.0, 4.0]), [(0.0, 5.0)] * 2, [1.0] * 2)
    phase.set_time(5.0, (1.0, 10.0))
    problem = MultiShootingTranscription(["phase0"], num_states=2)
    problem.add_phase("phase0", phase)
    problem.add_terminal(TerminalConditions(np.array([6.0, 7.0]), [(0.0, 8.0)] * 2, [1.0] * 2))

    d0, bounds, normalization, configs = problem.build()

    np.testing.assert_allclose(d0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert len(bounds) == len(d0)
    assert len(normalization) == len(d0)
    assert len(configs) == 1
    assert configs[0][0] == CtrlMode.ANGLE_STEER
    assert configs[0][1] == (0, 2)
    np.testing.assert_allclose(configs[0][2], np.zeros(2))
    assert configs[0][3] == DynEnum.DYNAMICS_2D
    assert configs[0][4] == ()


def test_registers_adjacent_defect():
    phase0 = Phase("phase0", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())
    phase1 = Phase("phase1", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER, ())
    problem = MultiShootingTranscription(["phase0", "phase1"], num_states=2)
    problem.add_phase("phase0", phase0)
    problem.add_phase("phase1", phase1)
    defect = PhaseDefect("defect", np.array([0.1, 0.2]), np.array([1.0, 1.0]))

    problem.add_defect("defect", ("phase0", "phase1"), defect)

    assert problem.defect_dict["defect"] == (("phase0", "phase1"), defect)

# Run the tests
if __name__ == "__main__":
    pytest.main()
