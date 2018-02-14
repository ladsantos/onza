#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute a grid containing drawing of disks of stars,
planets and pseudo-particle clouds.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import scipy.optimize as sp
from PIL import Image, ImageDraw
from itertools import product

__all__ = ["Grid"]


# The Grid class is used to produce two-dimensional images of a star.
class Grid(object):
    """
    Computes square images of an astronomical event.

    Args:
        size (`int`): Size of the grid in number of pixels
    """
    def __init__(self, size=2001):
        self.grid_size = size
        self.shape = (size, size)

        # The grid itself starts as a zeros numpy-array
        self.grid = np.zeros(self.shape, float)

        # Starting useful global variables
        self.norm = None  # Normalization factor
        self.cloud = None
        self.cell_bin = None
        self.cell_area = None
        self.cell_volume = None

    # Draw a general disk
    def _draw_disk(self, center, radius, value=1.0):
        """
        Computes a general disk (circular ellipse) in the grid.

        Args:
            center (array-like): Coordinates of the center of the disk.
                The origin is the center of the grid.

            radius (`int`): Radius of the disk in units of grid pixels.

            value (`float`): Value to be attributed to each pixel inside the
                disk.

        Returns:
            disk (`numpy.array`): 2-d Array containing the image of the disk in
                a grid of pixels.
        """
        top_left = (center[0] - radius, center[1] - radius)
        bottom_right = (center[0] + radius, center[1] + radius)
        image = Image.new('1', self.shape)
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline=1, fill=1)
        disk = np.reshape(np.array(list(image.getdata())), self.shape) * value
        return disk

    # Draw the disk of a star
    def draw_star(self, center, radius):
        """
        Computes a two-dimensional image of a stellar disk given its position
        and radius.

        Args:

            center (array-like): Coordinates of the center of the star.
                The origin is the center of the grid.

            radius (`int`): Radius of the star in units of grid pixels.
        """
        star = self._draw_disk(center, radius)
        self.norm = np.sum(star)  # Normalization factor is the total flux
        # Adding the star to the grid
        self.grid = star / self.norm

    # Draw the disk of a planet in the grid
    def draw_planet(self, center, radius, imp_param=None, orb_dist=None):
        """
        Computes a two-dimensional image of a planet given its position and
        radius.

        Args:

            center (array-like): Coordinates of the center of the planet.
                The origin is the center of the grid.

            radius (`int`): Radius of the planet in units of grid pixels.
        """
        planet = self._draw_disk(center, radius)
        # Adding the planet to the grid, normalized by the stellar flux
        self.grid -= planet / self.norm
        # The grid must not have negative values (this may happen if the planet
        # disk falls out of the stellar disk)
        self.grid = self.grid.clip(min=0.0)

    # Draw a cloud of pseudo-particles in the grid
    def draw_cloud(self, density_map=None, xy_positions=None):
        """

        Args:
            density_map (`numpy.array`): Must have the same shape as the grid.

        Returns:

        """
        if density_map is not None:
            self.cloud = density_map
        elif xy_positions is not None:
            pixels = np.arange(0, self.grid_size + 1)
            self.cloud, hist_bins = np.histogramdd(sample=xy_positions.T,
                                                   bins=[pixels, pixels])
            self.cloud += 1  # To avoid zero values
            self.cloud = self.cloud.T
        else:
            raise ValueError('Either `density_map` or `xy_positions` has to be '
                             'provided.')

    # Compute a cell grid
    def draw_cells(self, cell_size=10, px_physical_area=40680159.61):
        """

        Args:
            cell_size:

            px_physical_area (`float`): In km ** 2

        Returns:

        """
        # Computing the cell bins
        self.cell_bin = np.arange(0, self.grid_size, cell_size)
        if self.cell_bin[-1] < self.grid_size - 1:
            self.cell_bin = np.append(self.cell_bin, self.grid_size - 1)

        # Computing the areas and volumes of each cell, in pixels
        self.cell_area = []
        self.cell_volume = []
        for i in range(len(self.cell_bin) - 1):
            area = []
            volume = []
            for j in range(len(self.cell_bin) - 1):
                area.append((self.cell_bin[i + 1] - self.cell_bin[i]) *
                            (self.cell_bin[j + 1] - self.cell_bin[j]))
            self.cell_area.append(area)
        self.cell_area = np.array(self.cell_area) * px_physical_area

    # Plot transit just for fun
    def plot_transit(self, plot_cloud=True, plot_cells=True, output_file=None,
                     colorbar=False, densities_factor=1.0, colorbar_label=""):
        """

        Args:
            plot_cloud:
            plot_cells:
            output_file:

        Returns:

        """
        plt.imshow(self.grid)

        # Check if cells have been drawn, and if they have, plot them
        if self.cell_area is not None and plot_cells is True:
            for i, j in product(self.cell_bin, self.cell_bin):
                plt.axvline(x=i, color='k', lw=1)
                plt.axhline(y=j, color='k', lw=1)

        # Check if cloud has been drawn and plot it
        if self.cloud is not None and plot_cloud is True:
            norm = plc.LogNorm()
            plt.imshow(self.grid * self.cloud * self.norm * densities_factor,
                       norm=norm, cmap='viridis_r')

        if colorbar is True:
            plt.colorbar(label=colorbar_label)

        plt.axis('off')
        plt.tight_layout()
        plt.gca().invert_yaxis()
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)
            plt.close()


# The orbit module computes the orbital motion of a planet
class Orbit(object):
    """

    Args:

        ecc: Eccentricity.

        semi_a: Semi-major axis.

        phi0: Phase of periastron passage.

        arg_periapse: Argument of periapse.

        long_asc_node: Longitude of ascending node. Default is pi radians.

        inc: Inclination angle of the orbit in relation to the reference plane.
            Default is pi / 2 radians.
    """
    def __init__(self, ecc, semi_a, phi0, arg_periapse, long_asc_node=np.pi,
                 inc=np.pi/2):
        self.ecc = ecc
        self.semi_a = semi_a
        self.phi0 = phi0
        self.omega = arg_periapse
        self.big_omega = long_asc_node
        self.i = inc

    # Compute Kepler equation
    def kep_eq(self, e_ano, m_ano):
        kep = e_ano - self.ecc * np.sin(e_ano) - m_ano
        return kep

    # Compute position of planet
    def pos(self, phi):
        m_ano = phi - self.phi0  # Mean anomaly

        # Compute eccentric anomaly. First try it as if it were an array-like
        # object. If error is raised, treat as float
        try:
            e_ano = np.array([sp.newton(func=self.kep_eq, x0=mk, args=(mk,))
                              for mk in m_ano])
        except TypeError:
            e_ano = sp.newton(func=self.kep_eq, x0=m_ano, args=(m_ano,))

        # Computing the true anomaly
        f = 2 * np.arctan2(np.sqrt(1. + self.ecc) * np.sin(e_ano / 2),
                           np.sqrt(1. - self.ecc) * np.cos(e_ano / 2))

        # The distance from center of motion
        r = self.semi_a * (1 - self.ecc ** 2) / (1 + self.ecc * np.cos(f))

        # Finally compute the position
        xs = r * (np.cos(self.big_omega) * np.cos(self.omega + f) -
                  np.sin(self.big_omega) * np.sin(self.omega + f) *
                  np.cos(self.i))
        ys = r * (np.sin(self.big_omega) * np.cos(self.omega + f) +
                  np.cos(self.big_omega) * np.sin(self.omega + f) *
                  np.cos(self.i))
        zs = r * np.sin(self.omega + f) * np.sin(self.i)
        position = np.array([xs, ys, zs])

        return position


# Test module
if __name__ == "__main__":
    # a is given in stellar radii
    o = Orbit(0.001, 10, 0, np.pi, inc=np.pi / 4)
    print(o.pos(-np.pi / 2))
