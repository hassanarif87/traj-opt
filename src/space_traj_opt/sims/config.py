from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TerminalKind(str, Enum):
    ON_STATE = "ON_STATE"
    ON_ORBIT = "ON_ORBIT"


class StateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guess: list[float]
    bounds: Any = None
    normalize: list[float] | None = None


class ControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guess: list[float]
    bounds: Any = None
    normalize: list[float] | None = None
    angle_unit: str = "rad"


class TimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guess: float
    bounds: Any = None


class PhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dynamics: str
    control: str
    model_params: list[float]
    state: StateConfig
    controller: ControllerConfig
    time: TimeConfig


class DefectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    phases: tuple[str, str]
    values: list[float]
    normalize: list[float]


class TerminalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final: list[float]
    bounds: Any = None
    normalize: list[float] | None = None
    kind: TerminalKind = TerminalKind.ON_STATE


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = "SLSQP"
    maxiter: int = Field(default=500, gt=0)
    display: bool = True


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    num_states: int = Field(gt=0)
    phases: list[PhaseConfig] = Field(min_length=1)
    defects: list[DefectConfig] = []
    terminal: TerminalConfig
    output_name: str = "traj_opt_sol"
    state_headers: list[str]
    solver: SolverConfig = SolverConfig()


def load_config(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario configuration from YAML."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    if not isinstance(data, dict):
        raise ValueError("Scenario YAML must contain a top-level mapping")
    return ScenarioConfig.model_validate(data)