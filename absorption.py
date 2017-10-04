import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c


# Constants
res_element = 20 * u.km / u.s
damp_const = 6.27E8 / u.s
sigma_v_0 = 1.102E-6 * u.m ** 2 / u.s
lambda_0 = 1215.6702 * u.angstrom


# Broad absorption contribution
def tau_broad(vel_i, vel_j, num_e):
    """

    Args:
        vel_i:
        vel_j:
        num_e:

    Returns:
        tau:
    """
    tau = sigma_v_0 * num_e * damp_const / 4 / np.pi ** 2 * \
          (lambda_0 / (vel_j - vel_i)) ** 2
    return tau


# Narrow absorption contribution
def tau_line(num_e):
    """

    Args:
        num_e:

    Returns:

    """
    tau = sigma_v_0 * num_e * lambda_0 / res_element
    return tau

