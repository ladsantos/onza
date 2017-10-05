import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c


# Lyman-alpha absorption class
class Absorption(object):
    """

    """
    def __init__(self, grid, positions, velocities, supergrid_size=10,
                 stellar_radius=1 * u.solRad, res_element=20 * u.km / u.s):
        self.grid = grid
        self.g_size = len(grid)
        self.pos_starcentric = positions
        self.vel_starcentric = velocities
        self.res_element = res_element
        self.sg_size = supergrid_size
        self.sg_step = self.g_size // supergrid_size
        self.sg_last_step = self.sg_step + self.g_size % supergrid_size

        # Physical constants
        self.damp_const = 6.27E8 / u.s
        self.sigma_v_0 = 1.102E-6 * u.m ** 2 / u.s
        self.lambda_0 = 1215.6702 * u.angstrom

        # Changing positions and velocities from star-centric to numpy
        # array-centric
        self.pos = np.zeros_like(self.pos_starcentric)
        self.pos[0] += self.pos_starcentric[1] + self.grid // 2
        self.pos[1] += self.pos_starcentric[0] + self.grid // 2
        self.pos[2] += self.pos_starcentric[2]
        self.pos = (self.pos / stellar_radius).decompose()
        self.vel = np.zeros_like(self.vel_starcentric)
        self.vel[0] += self.vel_starcentric[1]
        self.vel[1] += self.vel_starcentric[0]
        self.vel[2] += self.vel_starcentric[2]

    # Broad absorption contribution
    def tau_broad(self, vel_i, vel_j, num_e):
        """

        Args:
            vel_i:
            vel_j:
            num_e:

        Returns:
            tau:
        """
        tau = self.sigma_v_0 * num_e * self.damp_const / 4 / np.pi ** 2 * \
            (self.lambda_0 / (vel_j - vel_i)) ** 2
        return tau

    # Narrow absorption contribution
    def tau_line(self, num_e):
        """

        Args:
            num_e:

        Returns:

        """
        tau = self.sigma_v_0 * num_e * self.lambda_0 / self.res_element
        return tau

    # The total absorption
    def tau_h(self):
        pass
