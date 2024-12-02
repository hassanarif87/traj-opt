from timeit import default_timer
from functools import wraps
from scipy.integrate import  OdeSolution

def vocalTimeit(*args, **kwargs):
    ''' provides the decorator @vocalTime which will print the name of the function as well as the
        execution time in seconds '''

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            start = default_timer()
            results = function(*args, **kwargs)
            end = default_timer()
            print('{} execution time: {} s'.format(function.__name__, end-start))
            return results
        return wrapper
    return decorator

def unpack_sol_list(sol_list_in: list[OdeSolution] , state_index: list[int])-> tuple[list, list]:
    """Helper function to add time offsets to each phase

    Args:
        sol_list_in : OdeSoluion for each phase
        state_index (_type_): Index of the state to unpack

    Returns:
        tuple of lists of time arrays and lists of the state arrays for each phase S
    """
    t_offsets = 0
    x_list = []
    y_list = []
    for sol in sol_list_in:
        x_list.append(sol.t + t_offsets )
        y_list.append(sol.y[state_index] )
        t_offsets += sol.t[-1]

    return x_list, y_list