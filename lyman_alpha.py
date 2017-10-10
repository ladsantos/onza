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
        # Computing the cell bins
        self.c_bins = np.arange(0, self.g_size, cell_size)
        if self.c_bins[-1] < self.g_size - 1:
            self.c_bins[-1] = self.g_size - 1
        # Velocity bins are used to compute the spectral absorption in bins
        # of velocity space defined by the spectral resolution element
        self.res_element = res_element
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

        # Initiating useful global variables
        self.hist = None

    # Compute histogram of particles in cells and velocity space
    def compute_hist(self):
        """
        Compute histogram of particles (bins are defined by the cells and
        spectral resolution).

        Returns:

        """
        # First convert velocity arrays to unit-less arrays
        vel_arr = self.vel[2].to(u.km / u.s).value
        vel_bins = self.vel_bins.to(u.km / u.s).value
        sample = np.array([self.pos[0].value, self.pos[1].value, vel_arr]).T
        hist, bins = np.histogramdd(sample, bins=[self.c_bins, self.c_bins,
                                                  vel_bins])
        self.hist = hist

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
    def num_particles(self, cell_indexes, velocity_index):
        """

        Args:
            cell_indexes: In the form [i, j]
            velocity_index: int

        Returns:

        """
        n_c_v = self.hist[cell_indexes[0], cell_indexes[1], velocity_index]
        return n_c_v

    # The total optical depth
    def tau(self, cell_indexes, velocity_bin):
        """

        Args:
            cell_indexes: In the form [i, j]
            velocity_bin: Index of the bin (left side) of the desired velocity
                range in ``self.vel_bins``

        Returns:

        """

        k = velocity_bin
        tau_broad = []

        # The reference velocity (the part of the spectrum where we want to
        # compute the optical depth, in velocity space)
        ref_vel = (self.vel_bins[k] + self.vel_bins[k + 1]) / 2

        # Compute the contribution of broad absorption for each velocity bin
        for i in range(len(self.vel_bins) - 1):
            if i == k:
                pass
            else:
                # Sweep the whole velocity spectrum
                cur_vel = self.vel_bins[i] + self.res_element / 2
                t0 = self.sigma_v_0 * self.num_particles(cell_indexes, i)
                t1 = self.damp_const / 4 / np.pi ** 2 * self.lambda_0 ** 2
                t2 = (cur_vel - ref_vel) ** (-2)
                tau_broad.append(t0 * t1 * t2)

        # Finally compute the total optical depth
        ref_num_e = self.num_particles(cell_indexes, k)
        tau = self.tau_line(ref_num_e) + sum(tau_broad)

        return tau
