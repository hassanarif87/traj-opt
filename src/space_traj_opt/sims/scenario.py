from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import deg2rad as d2r
from scipy.optimize import minimize

from space_traj_opt.models.controller3d import CtrlMode as CtrlMode3D
from space_traj_opt.models.models import CtrlMode as CtrlMode2D
from space_traj_opt.optimization.constraints import ConstraintType
from space_traj_opt.optimization.phases import (
    DynEnum,
    Phase,
    PhaseDefect,
    TerminalConditions,
)
from space_traj_opt.optimization.problem import Problem
from space_traj_opt.optimization.transcription import MultiShootingTranscription
from space_traj_opt.optimization.utils import (
    denormalize_decision_vec,
    normalize_decision_vec,
)
from space_traj_opt.postprocessing.post_proccess import sol_to_csv
from space_traj_opt.sims.config import PhaseConfig, ScenarioConfig, load_config


@dataclass
class BuiltScenario:
    problem: Problem
    full_params: dict
    normalization_vec: np.ndarray
    num_states: int


def _enum_value(enum_type, value: str):
    try:
        return enum_type[value.upper()]
    except KeyError as error:
        valid_values = ", ".join(member.name for member in enum_type)
        raise ValueError(
            f"Unknown {enum_type.__name__} value {value!r}; expected one of {valid_values}"
        ) from error


def _as_bounds(value):
    if value is None:
        return None
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [tuple(bound) for bound in value]
    return value


def _convert_angles(values: list[float], unit: str) -> np.ndarray:
    if unit.lower() == "rad":
        return np.asarray(values, dtype=float)
    if unit.lower() == "deg":
        return d2r(values)
    raise ValueError(f"Unknown angle unit {unit!r}; expected 'rad' or 'deg'")


def _build_phase(config: PhaseConfig) -> Phase:
    dynamics = _enum_value(DynEnum, config.dynamics)
    if dynamics == DynEnum.DYNAMICS_2D:
        control = _enum_value(CtrlMode2D, config.control)
    elif dynamics == DynEnum.DYNAMICS_3D:
        control = _enum_value(CtrlMode3D, config.control)

    phase = Phase(config.name, dynamics, control, tuple(config.model_params))

    state_guess = np.asarray(config.state.guess, dtype=float)
    phase.set_state(
        state_guess,
        bounds=_as_bounds(config.state.bounds),
        norm_vec=config.state.normalize,
    )

    controller = config.controller
    control_guess = _convert_angles(controller.guess, controller.angle_unit)
    control_bounds = _as_bounds(controller.bounds)
    if controller.angle_unit.lower() == "deg" and control_bounds is not None:
        if isinstance(control_bounds, list) and control_bounds and isinstance(
            control_bounds[0], tuple
        ):
            control_bounds = [tuple(d2r(bound)) for bound in control_bounds]
        else:
            control_bounds = d2r(control_bounds)
    phase.set_controller(
        control,
        control_guess,
        bounds=control_bounds,
        norm_vec=controller.normalize,
    )
    phase.set_time(config.time.guess, bounds=config.time.bounds)
    return phase


def build_scenario(config: ScenarioConfig) -> BuiltScenario:
    """Build the optimization problem represented by a validated config."""
    phase_names = [phase.name for phase in config.phases]
    builder = MultiShootingTranscription(phase_names, config.num_states)
    for phase_config in config.phases:
        builder.add_phase(phase_config.name, _build_phase(phase_config))

    for defect_config in config.defects:
        builder.add_defect(
            defect_config.name,
            defect_config.phases,
            PhaseDefect(
                defect_config.name,
                np.asarray(defect_config.values, dtype=float),
                np.asarray(defect_config.normalize, dtype=float),
            ),
        )

    terminal = config.terminal
    builder.add_terminal(
        TerminalConditions.set_terminal(
            x_final=np.asarray(terminal.final, dtype=float),
            bounds=_as_bounds(terminal.bounds),
            norm_vec=terminal.normalize,
        )
    )

    d0, d_bounds, normalization_vec, full_params = builder.build()
    d0_norm, d_bounds_norm = normalize_decision_vec(
        d0,
        d_bounds,
        normalization_vec,
    )
    problem = Problem(
        d0_norm,
        d_bounds_norm,
        normalization_vec,
        ConstraintType[terminal.kind.name],
        config.num_states,
        len(terminal.final),
        len(config.phases),
    )
    return BuiltScenario(problem, full_params, normalization_vec, config.num_states)


def run_scenario(config: ScenarioConfig):
    """Run optimization and write the configured trajectory output."""
    built = build_scenario(config)
    constraints = [
        {
            "type": "eq",
            "fun": built.problem.dynamics_knot_constrant,
            "args": (built.full_params,),
        }
    ]
    result = minimize(
        built.problem.objective,
        built.problem.d0_guess_normalized,
        jac=built.problem.jac_objective,
        method=config.solver.method,
        bounds=built.problem.d_bounds_norm,
        constraints=constraints,
        options={"maxiter": config.solver.maxiter, "disp": config.solver.display},
        args=(built.full_params,),
    )
    x_opt = denormalize_decision_vec(result.x, built.normalization_vec)
    sol_list = built.problem.full_traj_rollout(x_opt, built.full_params)
    sol_to_csv(sol_list, config.state_headers, config.output_name)
    return result


def run_scenario_file(path: str | Path):
    return run_scenario(load_config(path))

if __name__ == "__main__":
    run_scenario_file("scenarios/second_stage_ascent.yaml")