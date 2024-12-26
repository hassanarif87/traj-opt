import numpy.typing as npt
import numpy as np
from copy import deepcopy
from space_traj_opt.models import CtrlMode
from scipy.integrate import solve_ivp, OdeSolution
from functools import lru_cache
from space_traj_opt.models import dynamics
from concurrent.futures import ThreadPoolExecutor

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
        self.terminal_bounds = None
        self.terminal_normvec = None

        self.params = {}
        for phase in phase_names:
            self.defects[phase] = np.zeros(num_states)
        self.defects[phase_names[0]] = None

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
            f"  terminal_state={self.terminal_state}\n"
            f"  terminal_bounds={self.terminal_bounds}\n"
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
        normalization_vec = []
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
                normalization_vec.extend(self.u0_array.get(f"{phase_name}_normvec", []))
            # Append states and their bounds
            if phase_name in self.x0_array:
                d0.extend(
                    self.x0_array[phase_name].flatten()
                )  # Flatten the state array
                d0_bounds.extend(self.x0_array.get(f"{phase_name}_bnds", []))
                normalization_vec.extend(self.x0_array.get(f"{phase_name}_normvec", []))

            # Append time spans and their bounds
            if phase_name in self.t0_array:
                d0.append(self.t0_array[phase_name])  # Time is a scalar
                d0_bounds.append(self.t0_array.get(f"{phase_name}_bnds", (None, None)))
                normalization_vec.append(self.t0_array[phase_name])

            end_idx = ctrl_idx + len(self.u0_array[phase_name])
            # Check length of state vector and add t to the ctrl start idx # TODOL 
            ctrl_range = (ctrl_idx, end_idx)
            phase_configs_built[phase_name].append(ctrl_range)
            phase_configs_built[phase_name].append(self.defects[phase_name])
            phase_configs_built[phase_name].append(self.params[phase_name])
            ctrl_idx = len(d0)
        # Terminal Conditions
        d0.extend(self.terminal_state)
        d0_bounds.extend(self.terminal_bounds)
        normalization_vec.extend(self.terminal_normvec)
        for _, value in phase_configs_built.items():
            phase_configs_tuple.append(tuple(value))
        # Convert decision variables to numpy array for consistency
        d0 = np.array(d0, dtype=float)
        normalization_vec = np.array(normalization_vec, dtype=float)
        return d0, d0_bounds, normalization_vec, phase_configs_tuple

    def set_phase_init_x(
        self,
        phase_name: str,
        x0: npt.ArrayLike,
        bounds: None | npt.ArrayLike | tuple = None,
        norm_vec: None | npt.ArrayLike |list= None,

    ):
        """Sets the initial state for a given phase and its bounds.

        Parameters
        ----------
        phase_name : The name of the phase.
        x0 : The initial state for the phase.
        bounds : The bounds for the initial state. If None, no bounds are applied. \
            If equal to `x0`, the bounds are fixed at `x0` values. Default is None.
        norm_vec : Vector used to normalize the states
        """
        self.x0_array[phase_name] = x0
        if bounds is None:
            bounds = [(0., None) for _ in x0]
        elif np.shape(x0) == np.shape(bounds) and (bounds == x0).all():
            bounds = [(val, val) for val in x0]
        self.x0_array[phase_name + "_bnds"] = bounds
        self.x0_array[phase_name + "_normvec"] = norm_vec

    def set_phase_control(
        self,
        phase_name: str,
        ctrl_mode: CtrlMode,
        u0: npt.ArrayLike,
        bounds: None | npt.ArrayLike | tuple = None,
        norm_vec: None | npt.ArrayLike |list= None,
    ):
        """
        Sets the control inputs and control mode for a given phase and their bounds.

        Parameters
        ----------
        phase_name : The name of the phase.
        ctrl_mode : Enum defining the control mode for the phase.
        u0 : The control inputs for the phase.
        bounds : The bounds for the control inputs. If None, no bounds are applied. \
            If equal to `u0`, the bounds are fixed at `u0` values. Default is None.
        norm_vec : Vector used to normalize the controls
        """
        self.u0_array[phase_name] = u0
        if bounds is None:
            bounds = [(None, None) for _ in u0]
        elif (bounds == u0).all():
            bounds = [(val, val) for val in u0]
        self.u0_array[phase_name + "_bnds"] = bounds
        self.u0_array[phase_name + "_normvec"] = norm_vec

        self.phase_configs[phase_name] = [ctrl_mode]

    def set_phase_time(self, phase_name: str, t0: float, bounds=None):
        """
        Sets the terminal time guess for a given phase and its bounds.
        1 sec is the min length allowed to avoid collapsing trajectory
        
        Parameters
        ----------
        phase_name : The name of the phase.
        t0 :The terminal time guess guess for the phase.
        bounds : The bounds for the time span. If None, (1., None) is applied as a bound. \
            If equal to `t0`, the bounds are fixed at `t0` values. Default is None.
        """
        self.t0_array[phase_name] = t0
        if bounds is None:
            bounds = (0., None)
        elif bounds == t0:
            bounds = (t0, t0)
        self.t0_array[phase_name + "_bnds"] = bounds

    def set_non_zero_defect(
        self, defect_phases: tuple[str, str], defect_vec: npt.ArrayLike
    ):
        """Set a non zero defect at the knot point of the trajectory pShases.

        Parameters
        ----------
        defect_phases : the 2 phases between which the defect is set
        defect_vec : The defect vector, this should have the same dimentions as the state of the shooting dynamics
        """
        assert (
            len(defect_vec) == self.num_states
        ), "Defect vector length not equal to state vector "
        for idx, phase in enumerate(self.phase_names):
            if phase == defect_phases[1]:
                self.defects[phase] = defect_vec
                assert (
                    self.phase_names[idx - 1] == defect_phases[0]
                ), "Phases are not adjacent"

    def set_terminal_state(
        self, x_final: npt.ArrayLike, 
        bounds: tuple | npt.ArrayLike | None = None,         
        norm_vec: None | npt.ArrayLike |list= None,
    ):
        """
        Sets the terminal state for the trajectory and its bounds.

        Parameters
        ----------
        x_final : The desired terminal state as a 1D array.
        bounds : The bounds for the terminal state. If None, no bounds are applied. \
            If specified, it should be a list of tuples (lower_bound, upper_bound) \
            for each state variable. Default is None.
        norm_vec : Vector used to normalize the terminal state

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
        self.terminal_normvec = norm_vec
        # Set bounds if not provided
        if bounds is None:
            bounds = [(None, None) for _ in x_final]

        # Set bounds if not provided if an array like bound is provided set elements as upper and lower bound
        bounds_arr = np.array(bounds)
        if np.shape(bounds_arr) == np.shape(x_final):
            bounds_out = [(x, x) for x in bounds]
        else:
            bounds_out = bounds

        # Ensure bounds match the terminal state dimensions
        assert len(bounds_out) == len(
            x_final
        ), "Bounds must match the size of the terminal state."

        # Store the bounds
        self.terminal_bounds = bounds_out

    def set_dynamics_params(self, phase_name: str, params: tuple):
        """Params needed for the dynamics function in the trajectory rollout

        Args:
            phase_name: The name of the phase.
            params: Parameters needed in the dynamics function 
        """
        self.params[phase_name] = params

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
    def traj_rollout(t_terminal:float, x0: np.array, params: tuple) -> OdeSolution:
        """Integrates a phase of the trajectory.
        The trajectory is evaluated at a set time points using t_eval, this greatly improves convergance and stability of the gradients 
        lru_cache decerases the time required to calculate the jac, since scipy uses forward diff the cached f(x) is used instead of a re-compute
        Args:
            t_terminal : Terminal time of the phase
            x0 : Initial state of the phase
            params : Phase Parameter
    
        Returns:
            OdeSolution: The solution of the phase
        """
        sol = solve_ivp(
            dynamics, 
            t_span=[0.0, t_terminal], 
            t_eval= np.linspace(0.0, t_terminal,50), # This greatly improves convergance and stability of the jac
            y0=x0,    
            args=(params,)
        )
        return sol  
    
    def full_traj_rollout(self, decision_var, config_list)->list[OdeSolution]:
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
