#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes input data for the lyman_alpha module.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np


__all__ = ["ParticleEnsemble"]


# Compute a density cube from an ensemble of pseudo-particles
class ParticleEnsemble(object):
    """
    Compute the density cube (essentially an histogram) of an ensemble of
    pseudo-particles. TThe third dimension is an histogram of velocities in the
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

        vel_bin (array-like): The bins of the distribution of velocities.

        cell_area (`numpy.array` or `None`, optional): 2-d array containing the
            area of each cell. If `None`, it will be automatically computed from
            `cell_bin`. It must have dimensions (N-1, N-1), where N is the
            length of `cell_bin`. Default value is `None`.

        atoms_per_particle (`float`, optional): Number of atoms per
            pseudo-particle. Default value is 1E9.

    """
    def __init__(self, positions, velocities, cell_bin, vel_bin, cell_area=None,
                 atoms_per_particle=1E9):
        """


        """
        self.pos = positions
        self.vel = velocities
        self.cell_bin = cell_bin
        self.vel_bin = vel_bin

        # Computing the histogram of particles in cells and velocity space
        self.arr = np.array([self.pos[0], self.pos[1], self.vel[2]]).T
        self.hist, self.hist_bins = np.histogramdd(sample=self.arr,
                                                   bins=[self.cell_bin,
                                                         self.cell_bin,
                                                         self.vel_bin])
        self.hist *= atoms_per_particle

        # Divide by the area of the cells
        if cell_area is not None:
            self.cell_area = cell_area
            self.density = self.hist[:, :] / self.cell_area
        else:
            # First compute the areas if they were not provided
            self.cell_area = []
            for i in range(len(self.cell_bin) - 1):
                area = []
                for j in range(len(self.cell_bin) - 1):
                    area.append((self.cell_bin[i + 1] - self.cell_bin[i]) *
                                (self.cell_bin[j + 1] - self.cell_bin[j]))
                self.cell_area.append(area)
            self.cell_area = np.array(self.cell_area)
            # And finally compute the densities
            self.density = self.hist[:, :] / self.cell_area
