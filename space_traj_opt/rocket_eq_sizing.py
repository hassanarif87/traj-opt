import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
STANDARD_GRAV = 9.80665 # m/s/s https://en.wikipedia.org/wiki/Standard_gravity
# Size a 2 stage Rocket
def rocket_equation(I_sp, m_0, m_f):
    delta_v = I_sp * STANDARD_GRAV * np.log(m_0 / m_f)
    return delta_v

# Electron specs
# https://www.rocketlabusa.com/assets/Uploads/Electron-Payload-User-Guide-7.0.pdf
# Settings
n_engines_s1 = 9
n_engines_s2 = 1
isp_s1 = 311
engine_thrust_s1 = n_engines_s1*24910.04  # N Average between sl and vac
isp_s2 = 343
engine_thrust_s2 = n_engines_s2* 25_000  # N
payload = 250
mass_fraction_s1 = 0.12
mass_fraction_s2 = 0.11
fairing_mass = 50
farinig_timing = 184 -162 # sec
target_deltaV = 7600

deltav_loses = 1800 # WAG 2200 + 400 - 274.32
dry_mass_s1_guess = 1000 
dry_mass_s2_guess = 300

# Constraints 
min_thurst2_weight_s1 = 1.2
min_thurst2_weight_s2 = 0.6

def calculated_vehicle_deltav(dry_mass_s1, dry_mass_s2, fairing_mass, fairing_timing):

    wetmass_s1 = dry_mass_s1 / mass_fraction_s1 + dry_mass_s1
     
    init_prop_mass_pre_fairing = dry_mass_s2 / mass_fraction_s2 
    init_wetmass_s2_pre_fairing = init_prop_mass_pre_fairing + dry_mass_s2 

    mass_burnt_s2_fairing = fairing_timing *engine_thrust_s2 / STANDARD_GRAV / isp_s2
    wetmass_s2_at_fairing_sep = init_prop_mass_pre_fairing - mass_burnt_s2_fairing + dry_mass_s2

    # Stacked 
    initial_mass_stacked = wetmass_s1 + init_wetmass_s2_pre_fairing + payload + fairing_mass
    final_stacked = init_wetmass_s2_pre_fairing + dry_mass_s1 + payload + fairing_mass

    # Pre fairing sep
    s2_initial_pre_fairing = init_wetmass_s2_pre_fairing + payload + fairing_mass
    s2_final_pre_fairing = wetmass_s2_at_fairing_sep + payload + fairing_mass
    
    # Post Faring fairing sep
    s2_initial_post_fairing = wetmass_s2_at_fairing_sep + payload
    s2_final_post_fairing = dry_mass_s2 + payload
    
    t2w_stacked = engine_thrust_s1/ initial_mass_stacked/STANDARD_GRAV
    t2w_s2 = engine_thrust_s2/ s2_initial_pre_fairing /STANDARD_GRAV

    # Get total deltav
    deltav_stacked = rocket_equation(isp_s1, initial_mass_stacked, final_stacked)
    deltav_pre_fairing = rocket_equation(isp_s2, s2_initial_pre_fairing, s2_final_pre_fairing)
    deltav_post_fairing = rocket_equation(isp_s2, s2_initial_post_fairing, s2_final_post_fairing)
    
    
    total_dv = deltav_stacked + deltav_pre_fairing + deltav_post_fairing - deltav_loses
    return total_dv, t2w_stacked, t2w_s2
initial_guess = [dry_mass_s1_guess, dry_mass_s2_guess]

def constraint_func(z):
    _, t2w_stacked, t2w_s2 = calculated_vehicle_deltav(*z)
    return np.array([t2w_stacked, t2w_s2 ])

non_linear_constr = NonlinearConstraint(
    constraint_func, [min_thurst2_weight_s1, min_thurst2_weight_s2], [np.inf, np.inf])

bounds = [
    (0.0, None),
    (0.0, None),
    (fairing_mass, fairing_mass),
    (farinig_timing,farinig_timing)
]
def obj_func(z):
    dv_total, _, _ = calculated_vehicle_deltav(*z)
    return (target_deltaV - dv_total)**2

initial_guess = [dry_mass_s1_guess, dry_mass_s2_guess, fairing_mass, farinig_timing]
result = minimize(obj_func, x0=initial_guess, bounds=bounds, constraints=non_linear_constr)
print(f"Res = {result}")

print(f"Stage masses = {result.x}")
print(f"  Minimum = {result.fun}")
dv_total, t2w_stacked, t2w_s2 = calculated_vehicle_deltav(*result.x)
print(f"Change in velocity (\u0394V): {dv_total:.2f} m/s")
print(f"Stacked thrust to Weight = {t2w_stacked}")
print(f"S2 thrust to Weight = {t2w_s2}")

dry_mass_s1, dry_mass_s2,_,_ = result.x
wetmass_s1 = dry_mass_s1 / mass_fraction_s1 + dry_mass_s1
wetmass_s2 = dry_mass_s2 / mass_fraction_s2 + dry_mass_s2

initial_mass_stacked = wetmass_s1 + wetmass_s2 + payload + fairing_mass
print(f"S1 wet Mass = {wetmass_s1}")
print(f"s2 wet Mass = {wetmass_s2}")
print(f"Total Mass = {initial_mass_stacked}")
print(f"S1 Thrust = {engine_thrust_s1}")

print(f"sensitivity = {result.jac}")


