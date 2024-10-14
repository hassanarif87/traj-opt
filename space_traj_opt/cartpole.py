import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numba



# Define the dynamics of the cartpole system
@numba.njit
def dynamics(x, u):
    l = 0.5
    m1 = 1
    m2 = 0.3
    g = 9.81
    dx = np.zeros_like(x)
    q2 = x[:,1]
    q2_d = x[:,3]
    U = np.append(u, 0)
    dx[:,0] = x[:,2]
    dx[:,1] = x[:,3]
    dx[:,2] = ((l*m2*np.sin(q2)*q2_d**2)+U+(m2*g*np.cos(q2)*np.sin(q2)))/(m1 + m2*(1-(np.cos(q2))**2))
    dx[:,3] = -((l*m2*np.cos(q2)*np.sin(q2)*q2_d**2)+(U*np.cos(q2)) + ((m1+m2)*g*np.sin(q2)))/(l*m1 + l*m2*(1-(np.cos(q2))**2))
    return dx

# Define the objective function to minimize control effort
@numba.njit
def objective(u):
    return np.sum(u**2)

# Define the dynamics defects
@numba.njit
def state_defects(decision_variables, args):
    N = args[0] 
    states_dim = args[1] 
    dt = args[2]
    # Index [0 N )in the decision_variables vector contains the control inputs 
    u = decision_variables[:N]
    # Index [N -1] in the decision_variables vector contains the state  
    x = decision_variables[N:].reshape((N+1, states_dim))
    # Calculate the dynamics
    x_dot = dynamics(x, u)
    # Calculate the approximation of integral using trapezoidal quadrature
    integral = ((x_dot[:-1] + x_dot[1:])) / 2 * dt
    # Calculate the state defects
    defects = x[1:] - x[:-1]  - integral

    return defects.transpose().flatten()

# Define the direct collocation optimization problem
def optimization_problem(x0, xf, params):

    N = params[0] 
    states_dim = params[1] 
    dt = params[2]
    u_b = params[3]
    l_b = params[4]
    u_init = np.zeros(N)
    u_init = u_init

    # Initial guess for states
    x_init = np.zeros((N+1, states_dim))
    x_init[:, 0] = np.linspace(x0[0], xf[0], N+1)
    x_init[:, 1] = np.linspace(0, np.pi, N+1)
    # Concatenate control inputs and states into a single decision variable
    initial_guess = np.concatenate([u_init, x_init.flatten()])

    # Define the optimization problem
    def problem(decision_variables, args):
        N = args[0] 
        u = decision_variables[:N]

        obj_value = objective(u)

        return obj_value

    # Define the bounds for the decision variables
    bounds = [(-20, 20)] * N 
    num_state_bounds = states_dim*(N+1)

    state_bounds = [(None, None)] * num_state_bounds

    for i in range(0,N+1):
        state_bounds[states_dim*i] = (l_b,u_b)

    bounds = bounds + state_bounds    
    #Enforcing Bound constraint on initial and final states
    bounds[N] = (0.0,0.0)
    bounds[N+1] = (0.0,0.0)
    bounds[N+2] = (0.0, 0.0)
    bounds[N +3] = (0.0, 0.0)
    # Final Bounds
    bounds[N+ num_state_bounds - states_dim + 0] = (1.0,1.0)
    bounds[N+ num_state_bounds - states_dim + 1] = (np.pi,np.pi)
    bounds[N+ num_state_bounds - states_dim + 2] = (0.0,0.0)
    bounds[N+ num_state_bounds - states_dim + 3] = (0.0,0.0)
    arguments = (params,)
    # Define the constraints
    constraints = [{'type': 'eq', 'fun': state_defects, 'args':arguments },]
    # Solve the optimization problem
    result = minimize(problem, initial_guess, method='SLSQP', bounds=bounds, args=arguments, constraints=constraints)

    return result
