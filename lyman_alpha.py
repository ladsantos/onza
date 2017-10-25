#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Lyman-alpha line computation module.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import scipy.interpolate as si
from itertools import product

__all__ = ["Absorption", "Emission"]


# Lyman-alpha absorption class
class Absorption(object):
    """
    Computes the wavelength-dependent Lyman-alpha absorption coefficient for
    a cloud of neutral hydrogen in front of a star.

    Args:

        grid (`numpy.array`): Two-dimensional image of transit

        positions (`numpy.array`): Positions of particles, in unit of pixels.
            Shape of array must be (3, N), where N is the number of
            pseudo-particles. Positions in lines 0, 1 and 2 must be x, y and z,
            respectively.

        velocities (`numpy.array`): Velocities of particles, in km / s. Shape of
            array must be (3, N), where N is the number of pseudo-particles.
            Velocities in lines 0, 1 and 2 must be x, y and z, respectively.

        densities (`numpy.array` or `None`, optional): Column densities of
            particles. Must have the same shape as `grid`.

        cell_size (`int`): Size of cell, in px. Cells are the regions of the
            transit image where fluxes are computed. Lower cell sizes will
            render a finer computation of fluxes.

        res_element (`float`): Resolution element of the spectrum in km / s.

        vel_range (tuple): Range of velocities of the spectrum (not to be
            confused with the velocities of particles!), in km / s.

        densities (`numpy.array` or `None`, optional): 2-d map of densities of
            particles.

        vel_dist (`numpy.array` or `None`, optional): 1-d distribution of
            velocities inside the interval vel_range. It can have any length, as
            it will be interpolated to the spectral resolution automatically. It
            has to be normalized to 1.0. Future versions should support pixel-
            specific distributions instead of constant.
    """
    def __init__(self, grid, positions=None, velocities=None, cell_size=10,
                 res_element=20, vel_range=(-300, 300), atoms_per_part=1E9,
                 densities=None, vel_dist=None):
        self.grid = grid
        self.g_size = len(grid)
        self.part_density = atoms_per_part
        self.px_size = 2.0 / self.g_size

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

        # The Doppler shift (reference velocities) and wavelengths
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

        # TODO: Also improve this. Probably work on an input object instead.
        # Check if we are working with densities or particles
        if densities is None or vel_dist is None:
            self.pos_starcentric = positions  # Star-centric positions
            self.vel_starcentric = velocities  # Star-centric velocities
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
            # Computing the histogram of particles in cells and velocity space
            self.arr = np.array([self.pos[0], self.pos[1], self.vel[2]]).T
            self.hist, self.hist_bins = np.histogramdd(sample=self.arr,
                                                       bins=[self.c_bins,
                                                             self.c_bins,
                                                             self.vel_bins])
        else:
            # The densities are already the 2-d histogram and we need to append
            # the velocities distribution to produce a 3-d histogram
            # First we verify if need be to interpolate the velocities
            # distribution to the array of Doppler-shift
            if len(vel_dist) != len(self.doppler_shift):
                ref_vel_dist = np.linspace(vel_range[0], vel_range[1],
                                         len(vel_dist))
                f = si.interp1d(ref_vel_dist, vel_dist)
                vel_dist = f(self.doppler_shift)
            else:
                pass
            num_cells = len(self.c_bins) - 1
            self.hist = np.zeros([num_cells, num_cells,
                                  len(self.doppler_shift)])
            cells = np.arange(len(self.c_bins) - 1)
            for i, j in product(cells, cells):

                # The last columns and lines have to be added manually
                if i == cells[-1]:
                    cell_dens = np.sum(
                        densities[self.c_bins[i]:self.c_bins[i + 1] + 1,
                        self.c_bins[j]:self.c_bins[j + 1]])
                elif j == cells[-1]:
                    cell_dens = np.sum(
                        densities[self.c_bins[i]:self.c_bins[i + 1],
                        self.c_bins[j]:self.c_bins[j + 1] + 1])
                else:
                    cell_dens = np.sum(
                        densities[self.c_bins[i]:self.c_bins[i + 1],
                        self.c_bins[j]:self.c_bins[j + 1]])

                self.hist[i, j] = vel_dist * cell_dens * self.res_element

        # Initiating useful global variables
        self.wavelength = (self.doppler_shift / self.c * self.lambda_0 +
                           self.lambda_0) * 1E13  # Angstrom
        self.flux = None

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

                # The last column and lines have to be added manually
                if i == cells[-1]:
                    cell_flux = np.sum(
                            self.grid[self.c_bins[i]:self.c_bins[i + 1] + 1,
                            self.c_bins[j]:self.c_bins[j + 1]])
                elif j == cells[-1]:
                    cell_flux = np.sum(
                            self.grid[self.c_bins[i]:self.c_bins[i + 1],
                            self.c_bins[j]:self.c_bins[j + 1] + 1])
                else:
                    cell_flux = np.sum(
                            self.grid[self.c_bins[i]:self.c_bins[i + 1],
                            self.c_bins[j]:self.c_bins[j + 1]])
                exponent = np.exp(-self.tau([i, j], k))
                coeff += exponent * cell_flux

            self.flux.append(coeff)
        self.flux = np.array(self.flux)


# The Lyman-alpha emission class
class Emission(object):
    def __init__(self):
        raise NotImplementedError()
