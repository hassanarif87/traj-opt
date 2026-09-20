import numpy.typing as npt
import numpy as np
from copy import deepcopy
from space_traj_opt.models import CtrlMode
from functools import lru_cache
import enum

from concurrent.futures import ThreadPoolExecutor
from space_traj_opt.math.integrator import integrate, OdeResult
from space_traj_opt.phases import Phase, PhaseDefect, TerminalConditions, dynamics


class MultiShootingTranscription:
    """A class to transcribe a multi-phase trajectory optimization problem into a Nonlinear Programming (NLP) problem using multiple shooting.

    The class collects trajectory data, including initial state gusses, control inputs, and time spans for each phase,
    and converts them into decision variables and bounds for the NLP problem.

    Parameters
    ----------
    phase_names : list of str
        Names of the phases in the trajectory optimization problem.

    Attributes
    ----------
    phase_names : list of str
        Names of the phases provided during initialization.
    phase_dict : dict[str, Phase]
        Registered phases keyed by name.
    defect_dict : dict[str, tuple[tuple[Phase, Phase], PhaseDefect]]
        Registered defects keyed by name.
    terminal_conditons : TerminalConditions or None
        Terminal conditions registered for the transcription.

    Methods
    -------
    build():
        Constructs the decision variable vector (`d0`) and the corresponding bounds for the NLP problem.
    set_phase_init_x(phase_name, x0, bounds=None):
        Sets the initial state for a given phase and its bounds.
    set_phase_control(phase_name, ctrl_mode, u0, bounds=None):
        Sets the control inputs and control mode for a given phase and their bounds.
    set_phase_time(phase_name, t0, bounds=None):
        Sets the time span for a given phase and its bounds.
    Examples
    --------
    ```python
    import numpy as np

    from space_traj_opt.models import CtrlMode
    from space_traj_opt.phases import DynEnum, Phase, TerminalConditions

    phase = Phase("ascent", DynEnum.DYNAMICS_2D, CtrlMode.ANGLE_STEER)
    phase.set_state(
        DynEnum.DYNAMICS_2D,
        np.array([1.0, 0.0, 0.0, 1.0]),
        bounds=None,
        norm_vec=[1.0, 1.0, 1.0, 1.0],
    )
    phase.set_controller(
        CtrlMode.ANGLE_STEER,
        np.array([0.0]),
        bounds=None,
        norm_vec=[1.0],
    )
    phase.set_time(10.0)

    transcription = MultiShootingTranscription(["ascent"], num_states=4)
    transcription.add_phase("ascent", phase)
    transcription.add_terminal(
        TerminalConditions(
            x_final=np.array([2.0, 0.0, 0.0, 1.0]),
            bounds=[(None, None)] * 4,
            norm_vec=[1.0] * 4,
        )
    )
    d0, d0_bounds, normalization, phase_configs = transcription.build()
    ```
    """

    def __init__(self, phase_names, num_states ):
        self.phase_names = phase_names
        self.num_states = num_states

        self.phase_dict = {}
        self.defect_dict = {}
        self.terminal_conditons = None

        self.params = {}

        # Initialize to ero defect between phases
        for phase in phase_names:
            self.defects[phase] = np.zeros(num_states)
        self.defects[phase_names[0]] = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  phase_dict={self.phase_dict!r},\n"
            f"  defect_dict={self.defect_dict!r},\n"
            f"  terminal_conditons={self.terminal_conditons!r},\n"
            f")"
        )

    def add_phase(self, name: str, phase: Phase):
        """Register a phase with this transcription."""
        if not isinstance(name, str) or not name:
            raise ValueError("Phase name must be a non-empty string")
        if not isinstance(phase, Phase):
            raise TypeError("phase must be a Phase instance")
        if name in self.phase_dict:
            raise ValueError(f"Phase already registered: {name}")

        phase.name = name
        self.phase_dict[name] = phase

    def add_defect(
        self,
        name: str,
        phases: tuple[Phase, Phase],
        defect: PhaseDefect,
    ):
        """Register a defect between two adjacent registered phases."""
        if not isinstance(name, str) or not name:
            raise ValueError("Defect name must be a non-empty string")
        if name in self.defect_dict:
            raise ValueError(f"Defect already registered: {name}")
        if len(phases) != 2 or not all(isinstance(phase, Phase) for phase in phases):
            raise TypeError("phases must contain exactly two Phase instances")
        if not isinstance(defect, PhaseDefect):
            raise TypeError("defect must be a PhaseDefect instance")
        if any(phase.name not in self.phase_dict for phase in phases):
            raise ValueError("Both defect phases must be registered first")

        phase_names = list(self.phase_dict)
        first_idx = phase_names.index(phases[0].name)
        second_idx = phase_names.index(phases[1].name)
        if second_idx != first_idx + 1:
            raise ValueError("Defect phases must be adjacent")

        self.defect_dict[name] = (phases, defect)

    def add_terminal(self, terminal: TerminalConditions):
        """Register the terminal conditions for this transcription."""
        if not isinstance(terminal, TerminalConditions):
            raise TypeError("terminal must be a TerminalConditions instance")
        if self.terminal_conditons is not None:
            raise ValueError("Terminal conditions already registered")

        self.terminal_conditons = terminal
    
    def build(self):
        """
        Constructs the decision variable vector (`d0`) and bounds (`d0_bounds`)
        for the NLP problem, and returns the phase configurations.

        Returns
        -------
        tuple
            - d0 : numpy.ndarray
                The concatenated decision variable vector containing controls,
                initial states, and time spans.
            - d0_bounds : list of tuple
                The corresponding bounds for the decision variables.
            - phase_configs : dict
                The phase configuration dictionary.
        """
        # Initialize lists for decision variables and bounds
        d0 = []
        d0_bounds = []
        normalization_vec = []
        ctrl_idx = 0

        phase_configs_built = {}
        phase_configs_tuple = []
        for phase_name in self.phase_names:
            phase = self.phase_dict[phase_name]

            # Append controls and their bounds
            control_guess = np.atleast_1d(phase.u_guess)
            d0.extend(control_guess)
            d0_bounds.extend(phase.u_bounds)
            normalization_vec.extend(phase.u_normalize)

            # Append states and their bounds
            
            d0.extend(np.asarray(phase.x_guess).flatten())
            d0_bounds.extend(phase.x_bounds)
            normalization_vec.extend(phase.x_normalize)

            # Append time spans and their bounds
            d0.append(phase.t_guess)  # Time is a scalar
            d0_bounds.append(phase.t_bounds)
            normalization_vec.append(phase.t_guess)

            end_idx = ctrl_idx + len(np.atleast_1d(phase.u_guess))
            # Check length of state vector and add t to the ctrl start idx # TODOL 
            ctrl_range = (ctrl_idx, end_idx)
            phase_defect = next(
                (
                    defect.defect
                    for phases, defect in self.defect_dict.values()
                    if phases[1].name == phase_name
                ),
                None,
            )
            phase_configs_built[phase_name] = [
                phase.control_type,
                ctrl_range,
                phase_defect,
                phase.dynamics_type,
            ]
            ctrl_idx = len(d0)

        # Terminal Conditions
        d0.extend(self.terminal_conditons.x_final)
        d0_bounds.extend(self.terminal_conditons.bounds)
        normalization_vec.extend(self.terminal_conditons.norm_vec)

        for _, value in phase_configs_built.items():
            phase_configs_tuple.append(tuple(value))

        # Convert decision variables to numpy array for consistency
        d0 = np.array(d0, dtype=float)
        normalization_vec = np.array(normalization_vec, dtype=float)

        return d0, d0_bounds, normalization_vec, phase_configs_tuple


    def unpack_decision_var(self,decision_var, config):
        """Converts the decision 

        Args:
            decision_var : Optimzation decission vector
            config : Config for this phase
        Returns:
            tuple: control, state, terminal time, control_law

        """
        control_law = config[0]
        ctrl_idx_range = list(config[1])
        u = decision_var[range(*config[1])]
        x = decision_var[ctrl_idx_range[-1]: ctrl_idx_range[-1]+self.num_states]
        t_terminal = decision_var[ctrl_idx_range[-1]+self.num_states]

        return (u, x, t_terminal, control_law)

    @staticmethod
    def normalize_decision_vec(decision_vector, bounds, normalization_vector, offset_vector=None):
        """
        Normalize a decision vector and its bounds using a scaling normalization vector 
        and an optional offset vector.

        Args:
            decision_vector: The original decision vector to normalize.
            bounds: List of tuples representing (lower, upper) bounds for the decision variables.
            normalization_vector: Array or list of scaling factors for normalization.
            offset_vector: Array or list of offsets for normalization. Defaults to None.

        Returns:
            normalized_vector: The normalized decision vector.
            normalized_bounds: The normalized bounds as a list of (lower, upper) tuples.
        """

        # Ensure the normalization vector matches the length of the decision vector
        if len(decision_vector) != len(normalization_vector):
            raise ValueError("Normalization vector must match the length of the decision vector.")

        # Default offset vector to zeros if not provided
        if offset_vector is None:
            offset_vector = np.zeros_like(decision_vector)

        # Ensure the offset vector matches the length of the decision vector
        if len(decision_vector) != len(offset_vector):
            raise ValueError("Offset vector must match the length of the decision vector.")

        # Normalize the decision vector
        normalized_vector = (decision_vector - offset_vector) / normalization_vector

        # Normalize the bounds
        normalized_bounds = [
            (
                (lb - offset) / scale if lb is not None else None,
                (ub - offset) / scale if ub is not None else None
            )
            for (lb, ub), scale, offset in zip(bounds, normalization_vector, offset_vector)
        ]

        return normalized_vector, normalized_bounds

    @staticmethod
    def denormalize_decision_vec(normalized_vector, normalization_vector, offset_vector=None):
        """
        Denormalize a decision vector a scaling normalization vector and an optional offset vector.

        Args:
            normalized_vector: The normalized decision vector to denormalize.
            normalized_bounds: List of tuples representing (lower, upper) bounds in the normalized space.
            offset_vector: Array or list of offsets used for normalization. Defaults to None.

        Returns:
            denormalized_vector: The denormalized decision vector.
        """
        # Default offset vector to zeros if not provided
        if offset_vector is None:
            offset_vector = np.zeros_like(normalized_vector)

        # Denormalize the decision vector
        denormalized_vector = normalized_vector * normalization_vector + offset_vector

        # Denormalize the bounds
        return denormalized_vector
    
    @staticmethod
    @lru_cache(maxsize=128, typed=True) 
    def traj_rollout(t_terminal:float, x0: np.array, params: tuple) -> OdeResult:
        """Integrates a phase of the trajectory.
        The trajectory is evaluated at a set time points using t_eval, this greatly improves convergance and stability of the gradients 
        lru_cache decerases the time required to calculate the jac, since scipy uses forward diff the cached f(x) is used instead of a re-compute
        Args:
            t_terminal : Terminal time of the phase
            x0 : Initial state of the phase
            params : Phase Parameter

        Returns:
            OdeResult: The solution of the phase
        """

        t_sol, y_sol = integrate(
            dynamics, 
            t_span=[0.0, t_terminal], 
            t_eval= np.linspace(0.0, t_terminal,50),
            y0=x0,    
            args=(params,)
        )
        return OdeResult(t_sol, y_sol)  
    
    def full_traj_rollout(self, decision_var, config_list)->list[OdeResult]:
        """Rolls out all the trajectory segments. Each segment is rolled out in parallel using ThreadPoolExecutor.
        Args:
            decision_var : Optimzation decission vector
            config_list : Configs for each phase
    
        Returns:
            list of ode solutions for each segment
        """
        def process_phase(config):
            u, x, t_terminal, control_law = self.unpack_decision_var(decision_var, config)
            # make inputs hashable, needed for lru cache, the copy is cheaper than a second f(x) eval
            u_ = tuple(u.tolist())
            x_ = tuple(x.tolist())
            t_ = float(t_terminal)
            vch_params = (config[3], (control_law, u_))
            return self.traj_rollout(t_, x_, vch_params)

        with ThreadPoolExecutor() as executor:
            sol_list = list(executor.map(process_phase, config_list))
        return sol_list
