from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from space_traj_opt.models.dynamics import DynEnum
from space_traj_opt.models.models import CtrlMode


@dataclass
class Phase:
    name: str
    dynamics_type: DynEnum
    control_type: CtrlMode
    model_params : tuple
    x_guess: None | npt.ArrayLike = None
    x_normalize: None | npt.ArrayLike |list= None
    x_bounds: None | npt.ArrayLike | tuple = None,
    u_guess:  None | npt.ArrayLike |list= None
    u_normalize: None | npt.ArrayLike |list= None
    u_bounds: None | npt.ArrayLike | tuple = None,
    t_guess: float = 0.0
    t_bounds: float = 0.0


    def set_state(
            self,  
            x0: npt.ArrayLike,
            bounds: None | npt.ArrayLike | tuple = None,
            norm_vec: None | npt.ArrayLike |list= None
            ):
        """Sets the initial state for a given phase and its bounds.

        Parameters
        ----------
        x0 : The initial state for the phase.
        bounds : The bounds for the initial state. If None, no bounds are applied. \
            If equal to `x0`, the bounds are fixed at `x0` values. Default is None.
        norm_vec : Vector used to normalize the states
        """
        self.x_guess = x0
        if bounds is None:
            bounds = [(0., None) for _ in x0]
        elif np.shape(x0) == np.shape(bounds) and (bounds == x0).all():
            bounds = [(val, val) for val in x0]
        self.x_bounds = bounds
        self.x_normalize = norm_vec

    def set_controller(
            self,
            ctrl_mode: CtrlMode,
            u0: npt.ArrayLike,
            bounds: None | npt.ArrayLike | tuple = None,
            norm_vec: None | npt.ArrayLike |list= None,
            ):
        """
        Sets the control inputs and control mode for a given phase and their bounds.

        Parameters
        ----------
        ctrl_mode : Enum defining the control mode for the phase.
        u0 : The control inputs for the phase.
        bounds : The bounds for the control inputs. If None, no bounds are applied. \
            If equal to `u0`, the bounds are fixed at `u0` values. Default is None.
        norm_vec : Vector used to normalize the controls
        """
        self.u_guess = u0
        bounds_checker = np.array(bounds)
        if bounds is None:
            bounds = [(None, None) for _ in u0]
        elif (bounds_checker.shape == u0.shape) and (bounds_checker == u0).all():
            bounds = [(val, val) for val in u0]
        self.u_bounds = bounds
        self.u_normalize = norm_vec

        self.control_type = ctrl_mode

    def set_time(self,t0: float, bounds=None ):
        """
        Sets the terminal time guess for a given phase and its bounds.
        1 sec is the min length allowed to avoid collapsing trajectory
        
        Parameters
        ----------
        t0 :The terminal time guess guess for the phase.
        bounds : The bounds for the time span. If None, (1., None) is applied as a bound. \
            If equal to `t0`, the bounds are fixed at `t0` values. Default is None.
        """
        self.t_guess = t0
        if bounds is None:
            bounds = (0., None)
        elif bounds == t0:
            bounds = (t0, t0)
        self.t_bounds = bounds
    

@dataclass
class PhaseDefect:
    name: str
    defect: float = 0.0
    defect_norm_vec: float =0.0

@dataclass
class TerminalConditions:
    x_final: npt.ArrayLike
    bounds: tuple
    norm_vec: npt.ArrayLike

    @classmethod
    def set_terminal(
        cls, 
        x_final: npt.ArrayLike, 
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

        return cls(
            x_final,
            bounds_out,
            norm_vec
        )
