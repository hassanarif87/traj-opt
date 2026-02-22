
import numpy as np
OMEGA_EARTH=7.2921150e-5  # Earth's rotation rate in radians per second
R_EARTH=6371e3  # Earth's radius in meters
STANDARD_GRAV = 9.80665
SEMI_MAJOR_AXIS_EARTH = 6378137.0  # in meters
SEMI_MINOR_AXIS_EARTH = 6356752.3142  # in meters
ECCEN_EARTH = np.sqrt(1 - (SEMI_MINOR_AXIS_EARTH**2 / SEMI_MAJOR_AXIS_EARTH**2))
