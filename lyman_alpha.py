import numpy as np
import astropy.units as u
from itertools import product


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
        self.pos_starcentric = (positions / stellar_radius).decompose()
        self.vel_starcentric = velocities   # Star-centric velocities
        self.res_element = res_element
        self.c_size = cell_size
        self.c_step = self.g_size // cell_size  # Cell step size
        self.c_last_step = self.c_step + self.g_size % cell_size
        # Velocity bins are used to compute the spectral absorption in bins
        # of velocity space defined by the spectral resolution element
        self.vel_bins = np.arange(
                (vel_range[0] - res_element / 2).to(u.km / u.s).value,
                (vel_range[1] + res_element * 3 / 2).to(u.km / u.s).value,
                res_element.to(u.km / u.s).value) * u.km / u.s

        # Physical constants
        self.damp_const = 6.27E8 / u.s
        self.sigma_v_0 = 1.102E-6 * u.m ** 2 / u.s
        self.lambda_0 = 1215.6702 * u.angstrom

        # Changing positions and velocities from star-centric to numpy
        # array-centric
        self.pos = np.zeros_like(self.pos_starcentric)
        self.pos[0] += self.pos_starcentric[1].value + self.g_size // 2
        self.pos[1] += self.pos_starcentric[0].value + self.g_size // 2
        self.pos[2] += self.pos_starcentric[2].value
        #self.pos = (self.pos / stellar_radius).decompose()
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
        return tau.decompose()

    # Narrow absorption contribution
    def tau_line(self, num_e):
        """

        Args:
            num_e:

        Returns:

        """
        tau = self.sigma_v_0 * num_e * self.lambda_0 / self.res_element
        return tau.decompose()

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
        # Count where particles are inside the cell and inside the
        # designated velocity range
        n_c_v = 0
        # ``itertools.product`` is used to avoid nested loops
        for x, y, vz in product(self.pos[0, :], self.pos[1, :], self.vel[2, :]):
            if i0 < x < i1 and j0 < y < j1 and v0 < vz < v1:
                n_c_v += 1
        return n_c_v

    # The total optical depth
    def tau(self, cell_indexes, velocity_bin):
        """

        Args:
            cell_indexes: In the form [[i0, j0], [i1, j1]]
            velocity_bin: Index of the bin (left side) of the desired velocity
                range in ``self.vel_bins``

        Returns:

        """
        k = velocity_bin
        tau_broad = []

        # The reference velocity (the part of the spectrum where we want to
        # compute the optical depth, in velocity space)
        ref_vel = (self.vel_bins[k] + self.vel_bins[k + 1]) / 2
        ref_vel_range = [self.vel_bins[k], self.vel_bins[k + 1]]

        # Compute the contribution of broad absorption for each velocity bin
        for i in range(len(self.vel_bins) - 1):
            if i == k:
                pass
            else:
                # Sweep the whole velocity spectrum
                cur_vel = self.vel_bins[i] + self.res_element / 2
                cur_vel_range = [self.vel_bins[i], self.vel_bins[i + 1]]
                t0 = self.sigma_v_0 * self.num_particles(cell_indexes,
                                                         cur_vel_range)
                t1 = self.damp_const / 4 / np.pi ** 2 * self.lambda_0 ** 2
                t2 = (cur_vel - ref_vel) ** (-2)
                tau_broad.append(t0 * t1 * t2)

        # Finally compute the total optical depth
        ref_num_e = self.num_particles(cell_indexes, ref_vel_range)
        tau = self.tau_line(ref_num_e) + sum(tau_broad)

        return tau
