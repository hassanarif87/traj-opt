def dynamics_knot_constrant(decision_var, config_list): 
    """Integrate the dynamics of each segment. Calcculate the defect  between the knot points.
    This vector is used as the equality constraint for the optimization problem.
    The defect is calculated as the difference between the final state of the previous segment and the initial state of the next segment.

    Args:
        decision_var : Optimzation decission vector
        config_list : List of configs for each phase

    Returns:
        Knot defect vector
    """
    d0 = problem.denormalize_decision_vec(decision_var, normalization_vec)
    defect_vector_list = []
    sol_list=  problem.full_traj_rollout(d0, config_list)
    for idx in range(1,NUM_PHASE):
        _,_, knot_defect,_ = config_list[idx]
        defect_sub_vector = sol_list[idx].y[:,0] - sol_list[idx-1].y[:,-1] + knot_defect
        defect_sub_vector /= arr([100000, 100000, 8000, 5000, 1000])
        defect_vector_list.append(defect_sub_vector)
    # Terminal Defect
    terminal_state = d0[-NUM_X:] # pull from slice 
    terminal_defect = terminal_state - sol_list[-1].y[:,-1]
    terminal_defect /= arr([10000, 10000, 8000, 5000, 1000])
    defect_vector_list.append(terminal_defect)
    defect_vec = arr(defect_vector_list).flatten()
    return defect_vec



