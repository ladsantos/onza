#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute a grid containing drawing of disks of stars,
planets and pseudo-particle clouds.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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
        self.norm = None  # Normalization factor

        # The grid itself starts as a zeros numpy-array
        self.grid = np.zeros(self.shape, float)

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
    def draw_planet(self, center, radius):
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
    def draw_cloud(self):
        raise NotImplementedError('This method is not implemented yet.')


# Test module
if __name__ == "__main__":
    g = Grid(size=221)
    g.draw_star([110, 110], 109)
    g.draw_planet([110, 110], 1)
    _transit = g.grid
    plt.imshow(_transit)
    plt.show()
