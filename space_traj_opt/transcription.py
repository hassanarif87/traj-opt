import numpy.typing as npt
import numpy as np
from copy import deepcopy

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
    phase_configs : dict
        A dictionary storing phase configurations (e.g., control modes).
    x0_array : dict
        A dictionary storing the initial states for each phase and their associated bounds.
    u0_array : dict
        A dictionary storing the control inputs for each phase and their associated bounds.
    t0_array : dict
        A dictionary storing the time spans for each phase and their associated bounds.

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
    set_non_zero_defect(defect_phases, 
    """

    def __init__(self, phase_names, num_states):
        self.phase_names = phase_names
        self.num_states = num_states
        self.num_phases = len(phase_names)
        self.phase_configs = {}
        self.x0_array = {}
        self.u0_array = {}
        self.t0_array = {}
        self.defects = {}
        self.terminal_state = None
        self.terninal_bounds = None
        self.params = None
        for phase in phase_names:
            self.defects[phase]= np.zeros(num_states)
        self.defects[phase_names[0]]= None
    def __repr__(self):
        def _format_dict(d):
            """Helper to format a dictionary as a string."""
            formatted_items = [f"    {repr(k)}: {repr(v)}" for k, v in d.items()]
            return "{\n" + ",\n".join(formatted_items) + "\n  }"

        return (
            f"{self.__class__.__name__}(\n"
            f"  phase_configs={_format_dict(self.phase_configs)},\n"
            f"  x0_array={_format_dict(self.x0_array)},\n"
            f"  u0_array={_format_dict(self.u0_array)},\n"
            f"  t0_array={_format_dict(self.t0_array)}\n"
            f"  defects={_format_dict(self.defects)}\n"
            f")"
        )

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
        ctrl_idx = 0 

        # Deepcopy to keep member variables unaltered
        phase_configs_built = deepcopy(self.phase_configs)
        phase_configs_tuple = []
        # Loop through each phase
        for phase_name in self.phase_names:
            # Append controls and their bounds
            if phase_name in self.u0_array:
                if isinstance(self.u0_array[phase_name], float):
                    self.u0_array[phase_name] = np.array([self.u0_array[phase_name]])
                d0.extend(self.u0_array[phase_name])
                d0_bounds.extend(self.u0_array.get(f"{phase_name}_bnds", []))
        
        for phase_name in self.phase_names:
            # Append states and their bounds
            if phase_name in self.x0_array:
                d0.extend(
                    self.x0_array[phase_name].flatten()
                )  # Flatten the state array
                d0_bounds.extend(self.x0_array.get(f"{phase_name}_bnds", []))
        
        # Terminal Conditions 
        d0.extend(self.terminal_state)
        d0_bounds.extend(self.terminal_bounds)
        for phase_name in self.phase_names:
            # Append time spans and their bounds
            if phase_name in self.t0_array:
                d0.append(self.t0_array[phase_name])  # Time is a scalar
                d0_bounds.append(self.t0_array.get(f"{phase_name}_bnds", (None, None)))
            
            # Update phase config
            end_idx = ctrl_idx + len(self.u0_array[phase_name])
            ctrl_range = (ctrl_idx, end_idx)
            phase_configs_built[phase_name].append(ctrl_range)
            phase_configs_built[phase_name].append(self.defects[phase_name])
            phase_configs_built[phase_name].append(self.params)
            ctrl_idx= end_idx
        for _ , value in phase_configs_built.items():
            phase_configs_tuple.append(tuple(value))
        # Convert decision variables to numpy array for consistency
        d0 = np.array(d0, dtype=float)

        return d0, d0_bounds, phase_configs_tuple

    def set_phase_init_x(
        self,
        phase_name: str,
        x0: npt.ArrayLike,
        bounds: None | npt.ArrayLike | tuple = None,
    ):
        """Sets the initial state for a given phase and its bounds.

        Parameters
        ----------
        phase_name : The name of the phase.
        x0 : The initial state for the phase.
        bounds : The bounds for the initial state. If None, no bounds are applied. \
            If equal to `x0`, the bounds are fixed at `x0` values. Default is None.
        """
        self.x0_array[phase_name] = x0
        if bounds is None:
            bounds = [(None, None) for _ in x0]
        elif (bounds == x0).all():
            bounds = [(val, val) for val in x0]
        self.x0_array[phase_name + "_bnds"] = bounds

    def set_phase_control(
        self,
        phase_name: str,
        ctrl_mode,
        u0: npt.ArrayLike,
        bounds: None | npt.ArrayLike | tuple = None,
    ):
        """
        Sets the control inputs and control mode for a given phase and their bounds.

        Parameters
        ----------
        phase_name : The name of the phase.
        ctrl_mode : The control mode for the phase.
        u0 : The control inputs for the phase.
        bounds : The bounds for the control inputs. If None, no bounds are applied. \
            If equal to `u0`, the bounds are fixed at `u0` values. Default is None.
        """
        self.u0_array[phase_name] = u0
        if bounds is None:
            bounds = [(None, None) for _ in u0]
        elif (bounds == u0).all():
            bounds = [(val, val) for val in u0]
        self.u0_array[phase_name + "_bnds"] = bounds
        self.phase_configs[phase_name] = [ctrl_mode]

    def set_phase_time(self, phase_name, t0, bounds=None):
        """
        Sets the terminal time guess for a given phase and its bounds.

        Parameters
        ----------
        phase_name : The name of the phase.
        t0 :The terminal time guess guess for the phase.
        bounds : The bounds for the time span. If None, no bounds are applied. \
            If equal to `t0`, the bounds are fixed at `t0` values. Default is None.
        """
        self.t0_array[phase_name] = t0
        if bounds is None:
            bounds = (None, None)
        elif bounds == t0:
            bounds = (t0, t0)
        self.t0_array[phase_name + "_bnds"] = bounds

    def set_non_zero_defect(self, defect_phases: tuple[str, str], defect_vec: npt.ArrayLike):
        """Set a non zero defect at the knot point of the trajectory phases. 

        Parameters
        ----------
        defect_phases : the 2 phases between which the defect is set
        defect_vec : The defect vector, this should have the same dimentions as the state of the shooting dynamics 
        """
        assert(len(defect_vec) == self.num_states), "Defect vector lenght not equal to state vector "
        for idx, phase in enumerate(self.phase_names):
            if phase == defect_phases[1]:
                self.defects[phase] = defect_vec
                assert(self.phase_names[idx-1] == defect_phases[0]), "Phases are not adjacent"

    def set_terminal_state(self, x_final: npt.ArrayLike, bounds: tuple | npt.ArrayLike | None =None):
        """
        Sets the terminal state for the trajectory and its bounds.

        Parameters
        ----------
        x_final : The desired terminal state as a 1D array.
        bounds : The bounds for the terminal state. If None, no bounds are applied. \
            If specified, it should be a list of tuples (lower_bound, upper_bound) \
            for each state variable. Default is None.

        Example
        -------
        ```
        obj.set_terminal_state(
            x_final=np.array([200_000, 200_000, 0.0, 7500, 500]),
            bounds=[(None, 200_000), (None, 200_000), (0.0, 0.0), (7500, 7500), (None, None)]
        )
        ```
        """
        # Store the terminal state
        self.terminal_state = x_final

        # Set bounds if not provided
        if bounds is None:
            bounds = [(None, None) for _ in x_final]
        
        # Set bounds if not provided if an array like bound is provided set elememts as upper and lower bound 
        bounds_arr = np.array(bounds)
        if np.shape(bounds_arr) == np.shape(x_final):
            bounds = [(x, x) for x in bounds]
        # Ensure bounds match the terminal state dimensions
        assert len(bounds) == len(x_final), "Bounds must match the size of the terminal state."

        # Store the bounds
        self.terminal_bounds = bounds
    
    def set_params(self, params: tuple):
        self.params = params


