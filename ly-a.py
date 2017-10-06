import numpy as np
import astropy.units as u


# Lyman-alpha absorption class
class Absorption(object):
    """

    Args:
        grid: Image of transit
        positions: Positions of particles
        velocities: Velocities of particles
        cell_size: Size of cell
        stellar_radius: Stellar radius
        res_element: Resolution element of the instrument, in velocity units
        vel_range: Range of velocities of the spectrum (not to be confused with
            the velocities of particles!)
    """
    def __init__(self, grid, positions, velocities, cell_size=10,
                 stellar_radius=1 * u.solRad, res_element=20 * u.km / u.s,
                 vel_range=[-300, 300] * u.km / u.s):
        self.grid = grid
        self.g_size = len(grid)
        self.pos_starcentric = positions
        self.vel_starcentric = velocities
        self.res_element = res_element
        self.c_size = cell_size
        self.c_step = self.g_size // cell_size
        self.c_last_step = self.c_step + self.g_size % cell_size
        self.vel_bins = np.arange(vel_range[0] - res_element / 2,
                                  vel_range[1] + res_element * 3 / 2,
                                  res_element)

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
            vel_i: Reference velocity
            vel_j: Current velocity
            num_e: Number of particles with current velocity

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

    # Compute the number of hydrogen particles inside a cell within a given
    # velocity range in the z-axis
    def num_particles(self, cell_indexes, velocity_range):
        """

        Args:
            cell_indexes: In the form [[i0, j0], [i1, j1]]
            velocity_range: In the form [v0, v1]

        Returns:

        """
        i0, j0, i1, j1 = cell_indexes[0][0], cell_indexes[0][1], \
                         cell_indexes[1][0], cell_indexes[1][1]
        v0, v1 = velocity_range[0], velocity_range[1]
        inds = np.where(i0 < self.pos[:, 0] < i1 and j0 < self.pos[:, 1] < j1
                        and v0 < self.vel[:, 2] < v1)
        n_c_v = len(inds)
        return n_c_v

    # The total optical depth
    def tau(self, cell_indexes, velocity_bin):
        """

        Args:
            cell_indexes: In the form [[i0, j0], [i1, j1]]
            velocity_bin: Index of the bin (left side) of the desired velocity
                range self.vel_bins

        Returns:

        """
        k = velocity_bin
        tau_broad = []
        ref_vel = (self.vel_bins[k] + self.vel_bins[k + 1]) / 2
        ref_vel_range = [self.vel_bins[k], self.vel_bins[k + 1]]
        for i in range(len(self.vel_bins) - 1):
            cur_vel = self.vel_bins[i] + self.res_element / 2
            cur_vel_range = [self.vel_bins[i], self.vel_bins[i + 1]]
            t0 = self.sigma_v_0 * self.num_particles(cell_indexes,
                                                     cur_vel_range)
            t1 = self.damp_const / 4 / np.pi ** 2 * self.lambda_0 ** 2
            t2 = (cur_vel - ref_vel) ** (-2)
            tau_broad.append(t0 * t1 * t2)
        tau_broad = np.array(tau_broad)
        tau_broad = np.delete(tau_broad, k, 0)
        ref_num_e = self.num_particles(cell_indexes, ref_vel_range)
        tau = self.tau_line(ref_num_e) + np.sum(tau_broad)

        return tau
