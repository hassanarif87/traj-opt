
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from space_traj_opt.math.integrator import OdeResult
from space_traj_opt.optimization.constraints import ConstraintType, terminal_constraint
from space_traj_opt.optimization.utils import denormalize_decision_vec, traj_rollout


@dataclass
class Problem:
    d0_guess_normalized: ArrayLike
    d_bounds_norm: ArrayLike
    normalization_vec: ArrayLike
    terminal_con_kind: ConstraintType
    num_states: int
    num_terminal_states: int
    num_phases: int


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

    def full_traj_rollout(self, decision_var, config_list)->list[OdeResult]:
        """Rolls out all the trajectory segments. Each segment is rolled out in parallel using ThreadPoolExecutor.
        Args:
            decision_var : Optimzation decission vector
            config_list : Configs for each phase
    
        Returns:
            list of ode solutions for each segment
        """
        def process_phase(config):
            model_params = config[-1]
            dyn_type = config[-2]

            u, x, t_terminal, control_law = self.unpack_decision_var(decision_var, config)
            # make inputs hashable, needed for lru cache, the copy is cheaper than a second f(x) eval
            u_ = tuple(u.tolist())
            x_ = tuple(x.tolist())
            t_ = float(t_terminal)
            vch_params = (dyn_type, model_params, (control_law, u_))
            return traj_rollout(t_, x_, vch_params)

        with ThreadPoolExecutor() as executor:
            sol_list = list(executor.map(process_phase, config_list))
        return sol_list

    def dynamics_knot_constrant(self, decision_var, config_list):
        """Calculate the defect between phase knot points."""
        d0 = denormalize_decision_vec(decision_var, self.normalization_vec)
        defect_vector_list = []
        sol_list = self.full_traj_rollout(d0, config_list)
        for idx in range(1, self.num_phases):
            _, _, knot_defect, _, _ = config_list[idx]
            defect_sub_vector = sol_list[idx].y[:, 0] - sol_list[idx - 1].y[:, -1] + knot_defect
            defect_sub_vector /= np.array([100000, 100000, 8000, 5000, 1000])
            defect_vector_list.append(defect_sub_vector)


        terminal_defect = terminal_constraint(
            self.terminal_con_kind, 
            d0, 
            sol_list[-1], 
            self.num_terminal_states 
            )
        
        defect_vector_list.append(terminal_defect)
        return np.array(defect_vector_list).flatten()

    @staticmethod
    def objective(decision_var: tuple, params: tuple) -> float:
        """Objective function for min prop

        Args:
            decision_var : Optimization problem decision vector
            params : 

        Returns:
            Cost to minimize
        """
        terminal_mass= decision_var[-1]
        return -terminal_mass*terminal_mass*10000

    @staticmethod
    def jac_objective(decision_var: tuple, params: tuple):
        """Jac of the decision vector wrt the cost."""
        
        jac = np.zeros_like(decision_var)
        val = -decision_var[-1] - decision_var[-1]
        jac[-1]= val*10000
        return jac