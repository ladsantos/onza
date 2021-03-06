#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes input data for the lyman_alpha module.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from itertools import product


__all__ = ["ParticleEnsemble", "DensityMap"]


# The general onza-input parent class
class _OnzaInput(object):
    """
    Generalized `onza` input parent class.

    Args:

        cell_bin (`numpy.array`): The bins of the cell map.

        vel_bin (`numpy.array`): Array containing the bins of Doppler shift
            velocities in which to compute the spectrum.

        cell_area (`numpy.array` or `None`, optional): 2-d array containing the
            area of each cell. If `None`, it will be automatically computed from
            `cell_bin`. It must have dimensions (N-1, N-1), where N is the
            length of `cell_bin`. Default value is `None`.

        px_physical_area (`float`): Physical area of a pixel, in km ** 2.
    """
    def __init__(self, cell_bin, vel_bin, cell_area=None, cell_volume=None,
                 px_physical_area=40680159.61):

        self.cell_bin = cell_bin
        self.vel_bin = vel_bin
        self.cell_volume = cell_volume

        # Check if `cell_area` was provided
        if cell_area is not None:
            self.cell_area = cell_area
        else:
            # Compute the areas if they were not provided
            self.cell_area = []
            for i in range(len(self.cell_bin) - 1):
                area = []
                for j in range(len(self.cell_bin) - 1):
                    area.append((self.cell_bin[i + 1] - self.cell_bin[i]) *
                                (self.cell_bin[j + 1] - self.cell_bin[j]))
                self.cell_area.append(area)
            self.cell_area = np.array(self.cell_area) * px_physical_area

        # Compute the `doppler_shift` array, which consists on the values of the
        # Doppler shift away from the line center computed as the mean of two
        # consecutive values in `vel_bin`
        self.doppler_shift = []
        for i in range(len(self.vel_bin) - 1):
            self.doppler_shift.append((self.vel_bin[i] +
                                       self.vel_bin[i + 1]) / 2)
        self.doppler_shift = np.array(self.doppler_shift)

        # Spectral resolution element, assuming uniform Doppler shift sampling
        self.res_element = self.doppler_shift[1] - self.doppler_shift[0]


# Compute a density cube from an ensemble of pseudo-particles
class ParticleEnsemble(_OnzaInput):
    """
    Compute the density cube (essentially an histogram) of an ensemble of
    pseudo-particles. The third dimension is an histogram of velocities in the
    line-of-sight direction.

    Args:

        positions (`numpy.array`): Positions of particles, in unit of pixels.
            Shape of array must be (2, N), where N is the number of
            pseudo-particles. Positions in lines 0, 1 and 2 must be x and y,
            respectively.

        velocities (`numpy.array`): Velocities of particles in the line-of-sight
            direction, in km / s. Shape of array must be (N, ), where N is the
            number of pseudo-particles.

        cell_bin (array-like): The bins of the cell map.

        vel_bin (array-like): Array containing the bins of Doppler shift
            velocities in which to compute the spectrum.

        cell_area (`numpy.array` or `None`, optional): 2-d array containing the
            area of each cell. If `None`, it will be automatically computed from
            `cell_bin`. It must have dimensions (N-1, N-1), where N is the
            length of `cell_bin`. Default value is `None`.

        atoms_per_particle (`float`, optional): Number of atoms per
            pseudo-particle. Default value is 1E9.
    """
    def __init__(self, positions, velocities, cell_bin, vel_bin, cell_area=None,
                 cell_volume=None, atoms_per_particle=1E9):

        super(ParticleEnsemble, self).__init__(cell_bin, vel_bin, cell_area,
                                               cell_volume)

        self.pos = positions
        self.vel = velocities

        # Computing the histogram of particles in cells and velocity space
        self.arr = np.array([self.vel[2], self.pos[0], self.pos[1]]).T
        self.hist, self.hist_bins = np.histogramdd(sample=self.arr,
                                                   bins=[self.vel_bin,
                                                         self.cell_bin,
                                                         self.cell_bin])
        self.hist *= atoms_per_particle
        self.density = self.hist / self.cell_area

        # The volume density can be useful sometimes
        if self.cell_volume is not None:
            hist_vol, hist_vol_bins = np.histogramdd(sample=self.pos,
                                                     bins=[self.cell_bin,
                                                           self.cell_bin,
                                                           self.cell_bin])
            hist_vol *= atoms_per_particle
            self.volume_density = hist_vol / self.cell_volume


# Compute a density cube from a 2-d density map and a fixed distribution of
# velocities
class DensityMap(_OnzaInput):
    """
    Compute a density cube using as input a 2-d number density of particles map
    and a fixed distribution of velocities of particles in the line of sight.

    Args:

        density_map (`numpy.array`): 2-d array containing the number densities
            of particles inside a pixel.

        vel_bin (`numpy.array`): Array containing the bins of Doppler shift
            velocities in which to compute the spectrum.

        vel_dist (`numpy.array`): Distribution of velocities of particles in
            the direction of the line of sight.

        cell_bin (array-like): The bins of the cell map.

        cell_area (`numpy.array` or `None`, optional): 2-d array containing the
            area of each cell. If `None`, it will be automatically computed from
            `cell_bin`. It must have dimensions (N-1, N-1), where N is the
            length of `cell_bin`. Default value is `None`.
    """
    def __init__(self, density_map, vel_bin, vel_dist, cell_bin,
                 cell_area=None):

        super(DensityMap, self).__init__(cell_bin, vel_bin, cell_area)

        self.map = density_map
        self.vel_dist = vel_dist

        # Computing the coarse map (cells instead of pixels)
        cells = np.arange(len(cell_bin))
        self.map_coarse = np.zeros([len(cells) - 1, len(cells) - 1], float)
        for i, j in product(cells[:-1], cells[:-1]):
            self.map_coarse[i, j] = np.sum(self.map[cell_bin[i]:cell_bin[i + 1],
                                           cell_bin[j]:cell_bin[j + 1]])

        # Computing the density cube
        self.density = np.array([vk * self.map_coarse * self.res_element
                                 for vk in self.vel_dist]) / self.cell_area
