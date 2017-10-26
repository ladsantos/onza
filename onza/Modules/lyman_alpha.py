#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Lyman-alpha line computation module.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from itertools import product

__all__ = ["Absorption", "Emission"]


# Lyman-alpha absorption class
class Absorption(object):
    """
    Computes the wavelength-dependent Lyman-alpha absorption coefficient for
    a cloud of neutral hydrogen in front of a star.

    Args:

        transit_grid (`onza.transit.Grid`): Two-dimensional image of transit.

        density_cube (`onza.input` object): Three-dimensional map of densities
            in spatial (x and y) and velocity (z) dimensions.

        res_element (`float`): Resolution element of the spectrum in km / s.

        vel_range (tuple): Range of velocities of the spectrum in km / s.
    """
    def __init__(self, transit_grid, density_cube, res_element=20,
                 vel_range=(-300, 300)):

        self.transit = transit_grid
        self.cube = density_cube

        # Velocity bins are used to compute the spectral absorption in bins
        # of velocity space defined by the spectral resolution element
        self.res_element = res_element
        self.shift_bin = np.arange(
                vel_range[0] - res_element / 2,
                vel_range[1] + res_element * 3 / 2, res_element)

        # The Doppler shift (reference velocities) and wavelengths
        self.doppler_shift = []
        for i in range(len(self.shift_bin) - 1):
            self.doppler_shift.append((self.shift_bin[i] +
                                       self.shift_bin[i + 1]) / 2)
        self.doppler_shift = np.array(self.doppler_shift)

        # Physical constants
        self.damp_const = 6.27E8        # s ** (-1)
        self.sigma_v_0 = 1.102E-12      # km ** 2 / s
        self.lambda_0 = 1.2156702E-10   # km
        self.c = 299792.458             # km / s

        # Initiating useful global variables
        self.wavelength = (self.doppler_shift / self.c * self.lambda_0 +
                           self.lambda_0) * 1E13  # Angstrom
        self.flux = None

    # Narrow absorption contribution
    def tau_line(self, num_e):
        """
        Computes the narrow optical depth contribution from the Lyman-alpha line
        center.

        Args:

            num_e (`float`): Number density of particles per unit area of
            transit pixel.

        Returns:

            t_line (`float`): The optical depth contribution
        """
        t_line = self.sigma_v_0 * num_e * self.lambda_0 / self.res_element
        return t_line

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
                dens = self.cube.density[i, cell_indexes[0], cell_indexes[1]]
                t0 = self.sigma_v_0 * dens
                t1 = self.damp_const / 4 / np.pi ** 2 * self.lambda_0 ** 2
                t2 = (cur_vel - ref_vel) ** (-2)
                tau_broad.append(t0 * t1 * t2)
        tau_broad = np.array(tau_broad)

        # Finally compute the total optical depth
        ref_num_e = self.cube.density[k, cell_indexes[0], cell_indexes[1]]
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
            c_bin = self.transit.cell_bin
            cells = np.arange(len(c_bin) - 1)

            for i, j in product(cells, cells):

                # The last column and lines have to be added manually
                if i == cells[-1]:
                    cell_flux = np.sum(
                        self.transit.grid[c_bin[i]:c_bin[i + 1] + 1,
                                          c_bin[j]:c_bin[j + 1]])
                elif j == cells[-1]:
                    cell_flux = np.sum(
                        self.transit.grid[c_bin[i]:c_bin[i + 1],
                                          c_bin[j]:c_bin[j + 1] + 1])
                else:
                    cell_flux = np.sum(
                        self.transit.grid[c_bin[i]:c_bin[i + 1],
                                          c_bin[j]:c_bin[j + 1]])
                exponent = np.exp(-self.tau([i, j], k))
                coeff += exponent * cell_flux

            self.flux.append(coeff)
        self.flux = np.array(self.flux)


# The Lyman-alpha emission class
class Emission(object):
    def __init__(self):
        raise NotImplementedError()
