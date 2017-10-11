from __future__ import division
import numpy as np
import multiprocessing as mp
from itertools import product


# Lyman-alpha absorption class
class Absorption(object):
    """

    Args:
        grid: Image of transit
        positions: Positions of particles, in stellar radius
        velocities: Velocities of particles, in km / s
        cell_size: Size of cell, in px
        res_element: Resolution element of the instrument, in km / s
        vel_range: Range of velocities of the spectrum (not to be confused with
            the velocities of particles!), in km / s
    """
    def __init__(self, grid, positions, velocities, cell_size=10,
                 res_element=20, vel_range=(-300, 300), atoms_per_part=1E9):
        self.grid = grid
        self.g_size = len(grid)
        self.part_density = atoms_per_part
        self.px_size = 2.0 / self.g_size
        self.pos_starcentric = positions    # Star-centric positions
        self.vel_starcentric = velocities   # Star-centric velocities

        # Computing the cell bins
        self.c_bins = np.arange(0, self.g_size, cell_size)
        if self.c_bins[-1] < self.g_size - 1:
            self.c_bins[-1] = self.g_size - 1

        # Computing the areas of each cell. TODO: Improve this, looks horrible!
        self.c_area = []
        for i in range(len(self.c_bins) - 1):
            areas = []
            for j in range(len(self.c_bins) - 1):
                areas.append((self.c_bins[i + 1] - self.c_bins[i]) ** 2)
            self.c_area.append(areas)
        self.c_area = np.array(self.c_area) * self.px_size ** 2

        # Velocity bins are used to compute the spectral absorption in bins
        # of velocity space defined by the spectral resolution element
        self.res_element = res_element
        self.vel_bins = np.arange(
                vel_range[0] - res_element / 2,
                vel_range[1] + res_element * 3 / 2, res_element)

        # The Doppler shift (reference velocities)
        self.doppler_shift = []
        for i in range(len(self.vel_bins) - 1):
            self.doppler_shift.append((self.vel_bins[i] +
                                       self.vel_bins[i + 1]) / 2)
        self.doppler_shift = np.array(self.doppler_shift)

        # Physical constants
        self.damp_const = 6.27E8        # s ** (-1)
        self.sigma_v_0 = 1.102E-12      # km ** 2 / s
        self.lambda_0 = 1.2156702E-10   # km
        self.c = 299792.458             # km / s

        # Changing positions and velocities from star-centric to numpy
        # array-centric
        self.pos = np.zeros_like(self.pos_starcentric)
        self.pos[0] += self.pos_starcentric[1] + self.g_size // 2
        self.pos[1] += self.pos_starcentric[0] + self.g_size // 2
        self.pos[2] += self.pos_starcentric[2]
        self.vel = np.zeros_like(self.vel_starcentric)
        self.vel[0] += self.vel_starcentric[1]
        self.vel[1] += self.vel_starcentric[0]
        self.vel[2] += self.vel_starcentric[2]

        # Initiating useful global variables
        self.hist = None
        self.wavelength = (self.doppler_shift / self.c * self.lambda_0 +
                           self.lambda_0) * 1E13    # Angstrom
        self.flux = None

    # Compute histogram of particles in cells and velocity space
    def compute_hist(self):
        """
        Compute histogram of particles (bins are defined by the cells and
        spectral resolution).

        Returns:

        """
        # First convert velocity arrays to unit-less arrays
        vel_arr = self.vel[2]
        vel_bins = self.vel_bins
        arr = np.array([self.pos[0], self.pos[1], vel_arr]).T
        hist, bins = np.histogramdd(sample=arr, bins=[self.c_bins, self.c_bins,
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
        t_broad = self.sigma_v_0 * num_e * self.damp_const / 4 / np.pi ** 2 * \
            (self.lambda_0 / (vel_j - vel_i)) ** 2
        return t_broad

    # Narrow absorption contribution
    def tau_line(self, num_e):
        """

        Args:
            num_e:

        Returns:

        """
        t_line = self.sigma_v_0 * num_e * self.lambda_0 / self.res_element
        return t_line

    # Compute the number of hydrogen particles inside a cell within a given
    # velocity range in the z-axis
    def num_particles(self, cell_indexes, velocity_index):
        """

        Args:
            cell_indexes: In the form [i, j]
            velocity_index: int

        Returns:

        """
        n_c_v = self.hist[cell_indexes[0], cell_indexes[1],
                          velocity_index] * self.part_density
        return n_c_v / self.c_area[cell_indexes[0], cell_indexes[1]]

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
        ref_vel = self.doppler_shift[k]

        # Compute the contribution of broad absorption for each velocity bin
        for i in range(len(self.doppler_shift)):
            if i == k:
                pass
            else:
                # Sweep the whole velocity spectrum
                cur_vel = self.doppler_shift[i]
                t0 = self.sigma_v_0 * self.num_particles(cell_indexes, i)
                t1 = self.damp_const / 4 / np.pi ** 2 * self.lambda_0 ** 2
                t2 = (cur_vel - ref_vel) ** (-2)
                tau_broad.append(t0 * t1 * t2)
        tau_broad = np.array(tau_broad)

        # Finally compute the total optical depth
        ref_num_e = self.num_particles(cell_indexes, k)
        tau = self.tau_line(ref_num_e) + np.sum(tau_broad)

        return tau

    # Compute absorption spectrum
    def compute_abs(self):
        """

        Returns:

        """
        self.flux = []

        # For each wavelength, compute the absorption coefficient
        for k in range(len(self.doppler_shift)):
            coeff = 0
            # Sum for each cell
            cells = np.arange(len(self.c_bins) - 1)
            for i, j in product(cells, cells):
                cell_flux = np.sum(
                        self.grid[self.c_bins[i]:self.c_bins[i + 1],
                        self.c_bins[j]:self.c_bins[j + 1]])
                exponent = np.exp(-self.tau([i, j], k))
                coeff += exponent * cell_flux
            #coeff = coeff / len(cells) ** 2
            self.flux.append(coeff)
        self.flux = np.array(self.flux)

    def compute_abs_mp(self, k):

        coeff = 0
        # Sum for each cell
        cells = np.arange(len(self.c_bins) - 1)
        for i, j in product(cells, cells):
            cell_flux = np.sum(
                    self.grid[self.c_bins[i]:self.c_bins[i + 1],
                    self.c_bins[j]:self.c_bins[j + 1]])
            coeff += np.exp(-self.tau([i, j], k)) * cell_flux
        #coeff = coeff / len(cells) ** 2
        return coeff

    # Multiprocessing test
    def test(self):

        # For each wavelength, compute the absorption coefficient
        self.flux = []
        pool = mp.Pool(processes=4)
        k = range(len(self.doppler_shift))
        self.flux = pool.map(self.compute_abs_mp, k)
        self.flux = np.array(self.flux)
