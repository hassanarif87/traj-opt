

import numpy as np
def skew_sym_mat(vec):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param x: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -vec[2], vec[1]], 
                     [vec[2], 0, -vec[0]], 
                     [-vec[1], vec[0], 0]])

def identiy_mul(x):
    return np.identity(3) * x

def quat_conj(q):
    return np.array([q[0] , -q[1], -q[2], -q[3]])
                    
# q · p = qm(q, p) = Q(q)p = Q¯(p)q (106)
def quat_mat(q):
    """Fundimentals of Attitude Dynamics and Control pg 38 Eq 3.86"""

    return np.array([
        [q[0] , -q[1], -q[2], -q[3]],
        [q[1] , q[0], -q[3], q[2]],
        [q[2] , q[3], q[0], -q[1]],
        [q[3] , -q[2], q[1], q[0]]
    ])


def q_mult(q1, q2):
    """Multiplies two quaternions together using the quat_mat function
        qa2c = qa2b * qb2c
    Args:
        q1 : first quaternion
        q2 : second quaternion
    Returns:
        q_mult : product of the two quaternions
    """
    return quat_mat(q1) @ q2

def quat2mat(q):
    """
    Convert a Unit quaternion representing a rotation to a  Transformation matrix
    https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf pg. 15 eq 125

    Args:
        q : Unit Quaternion
    Returns:
        mat : 3x3 direct cosine matrix
    """
    q2 = q**2
    q1q2 = q[1]*q[2]
    q0q3 = q[0]*q[3]
    q1q3 = q[1]*q[3] 
    q0q2 = q[0]*q[2]
    q2q3 = q[2]*q[3] 
    q0q1 = q[0]*q[1]
    return np.array([
        [q2[0] + q2[1] - q2[2] - q2[3], 2*(q1q2 + q0q3), 2*(q1q3- q0q2)],
        [2*(q1q2 - q0q3), q2[0] - q2[1] + q2[2] - q2[3], 2*(q2q3+ q0q1)],
        [2*(q1q3+ q0q2), 2*(q2q3- q0q1), q2[0] - q2[1] - q2[2] + q2[3]]
             ])


def quat_rate_mat(q):
    """Fundimentals of Attitude Dynamics and Control pg 71 Eq 3.20
    """
    return np.array([
        [-q[1], -q[2], -q[3]],
        [q[0], -q[3], q[2]],
        [q[3], q[0], -q[1]],
        [-q[2], q[1], q[0]]

    ])

def quat_deriv(q,omega_b):

    """Fundimentals of Attitude Dynamics and Control pg 71 Eq 3.20
    0.5 * q_xi(q) * omega_b
    """
    return 0.5*quat_rate_mat(q)*omega_b

def q_rotate_frame(q, v):
    """Rotate a vector from one frame to another using a quaternion
    """
    return quat2mat(q) @ v

def q_rotate_vector(q, v):
    """Rotate a vector in the same frame using a quaternion
    """
    q_conj = quat_conj(q)
    return quat2mat(q_conj) @ v

def q_from_axisangle(angle: float, axis: np.array):
    """Create quaternion from and angle and axis

    Args:
        angle: Rotation angle
        axis: Unit axis about which rotation occures  
    """
    axis = axis / np.linalg.norm(axis)
    half_angle = angle * 0.5
    sin_half_angle = np.sin(half_angle)
    return np.array([
        np.cos(half_angle),
        axis[0]*sin_half_angle,
        axis[1]*sin_half_angle,
        axis[2]*sin_half_angle
    ])
