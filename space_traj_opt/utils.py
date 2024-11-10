from timeit import default_timer
from functools import wraps

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