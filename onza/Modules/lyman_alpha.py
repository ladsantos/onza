#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Lyman-alpha line computation module.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from schwimmbad import MultiPool, SerialPool
from itertools import product

__all__ = ["Absorption", "Emission"]


# Lyman-alpha absorption class
class Absorption(object):
    """
    Computes the wavelength-dependent Lyman-alpha absorption coefficient for
    a cloud of neutral hydrogen in front of a star.

    Args:

        transit_grid (`transit.Grid` object): Two-dimensional image of transit.

        density_cube (`input` object): Three-dimensional map of densities
            in spatial and velocity dimensions.
    """
    def __init__(self, transit_grid, density_cube, flux=None):

        self.transit = transit_grid
        self.cube = density_cube
        self.flux = flux

        # Velocity bins are used to compute the spectral absorption in bins
        # of velocity space defined by the spectral resolution element
        self.shift_bin = self.cube.vel_bin
        self.res_element = self.cube.res_element

        # The Doppler shift (reference velocities) and wavelengths
        self.doppler_shift = self.cube.doppler_shift

        # Physical constants
        self.damp_const = 6.27E8        # s ** (-1)
        self.sigma_v_0 = 1.102E-12      # km ** 2 / s
        self.lambda_0 = 1.2156702E-10   # km
        self.c = 299792.458             # km / s

        # Compute the flux in each cell. It does not look good, but bear with me
        # for a second. This is necessary to avoid unnecessary loops when
        # computing the absorption profile.
        c_bin = self.transit.cell_bin
        cells = np.arange(len(c_bin) - 1)

        def _expr(inds):
            i = inds[0]
            j = inds[1]
            flux = np.sum(self.transit.grid[c_bin[i]:c_bin[i + 1],
                          c_bin[j]:c_bin[j + 1]])
            return flux

        self.cell_flux = np.array(list(map(_expr, product(cells, cells))))
        self.cell_flux = np.reshape(self.cell_flux, [len(cells), len(cells)])

        # Initiating useful global variables
        self.wavelength = (self.doppler_shift / self.c * self.lambda_0 +
                           self.lambda_0) * 1E13  # Angstrom
        self.abs_profile = None

    # Narrow absorption contribution
    def tau_line(self, num_e):
        """
        Computes the narrow optical depth contribution from the Lyman-alpha line
        center.

        Args:

            num_e (`float`): Number density of transiting particles per km ** 2.

        Returns:

            t_line (`float`): The optical depth narrow contribution.
        """
        t_line = self.sigma_v_0 * num_e * self.lambda_0 / self.res_element
        return t_line

    # The total optical depth
    def tau(self, cell_indexes, velocity_bin):
        """
        Computes the total optical depth taking into account both the narrow and
        broad contributions.

        Args:

            cell_indexes (array-like): In the form [i, j]

            velocity_bin: Index of the bin (left side) of the desired velocity
                range in ``self.vel_bins``

        Returns:

            tau (`float`): Total optical depth in the cell
        """
        k = velocity_bin

        # The reference velocity (the part of the spectrum where we want to
        # compute the optical depth, in velocity space)
        ref_vel = self.doppler_shift[k]

        # We compute tau_broad only for velocities that are not the reference
        other_shift = np.delete(self.doppler_shift, k)
        other_dens = np.delete(
            self.cube.density[:, cell_indexes[0], cell_indexes[1]], k)

        # Some clever `numpy.array` sorcery here to optimize the computation
        delta_v = other_shift - ref_vel
        tau_broad = delta_v ** (-2) * other_dens * self.sigma_v_0 * \
            self.damp_const / 4 / np.pi ** 2 * self.lambda_0 ** 2

        # Finally compute the total optical depth
        ref_num_e = self.cube.density[k, cell_indexes[0], cell_indexes[1]]
        tau = self.tau_line(ref_num_e) + np.sum(tau_broad)
        return tau

    # Compute absorption index for a single Doppler shift bin and a single cell
    def _compute_abs(self, indexes):
        """

        Returns:

        """
        i = indexes[1]
        j = indexes[2]
        k = indexes[0]
        coeff = np.exp(-self.tau([i, j], k)) * self.cell_flux[i, j]
        return coeff

    # Compute absorption using single-process or multiprocessing
    def compute_profile(self, multiprocessing=False):
        """

        Args:
            multiprocessing

        Returns:

        """
        k = list(range(len(self.doppler_shift)))
        cells = list(range(len(self.transit.cell_bin) - 1))
        if multiprocessing is True:
            with MultiPool() as pool:
                self.abs_profile = list(pool.map(self._compute_abs,
                                                 product(k, cells, cells)))
        else:
            with SerialPool() as pool:
                self.abs_profile = list(pool.map(self._compute_abs,
                                                 product(k, cells, cells)))
        self.abs_profile = np.reshape(self.abs_profile,
                                      [len(k), len(cells), len(cells)])
        self.abs_profile = np.sum(self.abs_profile, axis=(1, 2))

        if self.flux is not None:
            self.flux *= self.abs_profile

    # New way to compute the absorption profile
    def fast_profile(self):
        """

        Returns:

        """
        n = len(self.doppler_shift)
        m = len(self.cube.density[0])

        # Some `numpy.array` sorcery is needed to avoid loops and list
        # comprehension, which are too slow.
        # First we need to deal with the computation of tau_broad, which
        # involves removing particular indices of arrays corresponding to each
        # reference velocity in the Doppler shift array

        # First create an array of the indexes to be deleted
        i_del = np.arange(0, n)
        i_del *= (n + 1)
        i_del_l = i_del * m ** 2
        i_del_u = i_del_l + m ** 2
        # Now build the multiple slices to be removed from the densities cube
        slices = ()
        for i in range(n):
            slices += (slice(i_del_l[i], i_del_u[i]), )
        slices = np.r_[tuple(slices)]

        # Now compute the doppler shifts and densities with specific indices
        # removed
        other_shift = np.broadcast_to(self.doppler_shift, [n, n])
        other_shift = np.reshape(np.delete(other_shift, i_del), [n, n-1])
        other_dens = np.broadcast_to(self.cube.density, [n, n, m, m])
        other_dens = np.reshape(np.delete(other_dens, slices),
                                [n, n-1, m, m])

        # Now compute the optical depths and absorption coefficients
        delta_v = (other_shift.T - self.doppler_shift)
        term = delta_v ** (-2) * self.sigma_v_0 * self.damp_const / 4 / \
            np.pi ** 2 * self.lambda_0 ** 2
        tau_broad = (np.sum(other_dens.T * term, axis=-2)).T
        tau_line = self.sigma_v_0 * self.cube.density * self.lambda_0 / \
            self.res_element
        tau = tau_line + tau_broad
        coeff = self.cell_flux * np.exp(-tau)

        # Finally compute absorption profile
        self.abs_profile = np.sum(coeff, axis=(1, 2))


# The Lyman-alpha emission class
class Emission(object):
    def __init__(self):
        raise NotImplementedError()
