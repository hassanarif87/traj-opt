import numpy as np

from space_traj_opt.optimization.phases import Phase, PhaseDefect, TerminalConditions


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
        self.num_phases = len(phase_names)
        
        self.phase_dict = {}
        self.defect_dict = {}
        self.terminal_conditons = None
        self.num_terminal_states = 0
        self.params = {}


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
        phases: tuple[str, str],
        defect: PhaseDefect,
    ):
        """Register a defect between two adjacent registered phases.
        Args:
            name: Name of the defect.
            phases: Names of phases in between the defect exists.
            defect: defect struct.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Defect name must be a non-empty string")
        if name in self.defect_dict:
            raise ValueError(f"Defect already registered: {name}")
        if len(phases) != 2: # TODO: move to defect builder
            raise TypeError("phases must contain exactly two Phase instances")
        if not isinstance(defect, PhaseDefect):
            raise TypeError("defect must be a PhaseDefect instance")
        if any(phase not in self.phase_dict for phase in phases):
            raise ValueError("Both defect phases must be registered first")

        phase_names = list(self.phase_dict)
        first_idx = phase_names.index(phases[0])
        second_idx = phase_names.index(phases[1])
        if second_idx != first_idx + 1:
            raise ValueError("Defect phases must be adjacent")

        self.defect_dict[name] = (phases, defect)

    def add_terminal(self, terminal: TerminalConditions):
        """Register the terminal conditions for this transcription."""
        if not isinstance(terminal, TerminalConditions):
            raise TypeError("terminal must be a TerminalConditions instance")
        if self.terminal_conditons is not None:
            raise ValueError("Terminal conditions already registered")

        self.num_terminal_states = len(terminal.x_final)
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
        for idx, phase_name in enumerate(self.phase_names):
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
                    if phases[1] == phase_name
                ),
                np.zeros(self.num_states),
            )
            phase_configs_built[phase_name] = [
                phase.control_type,
                ctrl_range,
                phase_defect,
                phase.dynamics_type,
                phase.model_params,
            ]
            ctrl_idx = len(d0)

        # Terminal Conditions
        d0.extend(self.terminal_conditons.x_final)
        d0_bounds.extend(self.terminal_conditons.bounds)
        normalization_vec.extend(self.terminal_conditons.norm_vec)

        for value in phase_configs_built.values():
            phase_configs_tuple.append(tuple(value))

        # Convert decision variables to numpy array for consistency
        d0 = np.array(d0, dtype=float)
        normalization_vec = np.hstack(normalization_vec, dtype=float)

        return d0, d0_bounds, normalization_vec, phase_configs_tuple
