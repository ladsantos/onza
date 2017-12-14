#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to perform data manipulation of input files to be fed to
the main data crunching and modelling modules.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from scipy.ndimage import zoom


__all__ = ['nearest_index', 'DensityMap']


# Useful function to find the index in an array corresponding to a specific
# value.
def nearest_index(array, target_value):
    """
    Finds the index of a value in ``array`` that is closest to ``target_value``.

    Args:
        array (``numpy.array``): Target array.
        target_value (``float``): Target value.

    Returns:
        index (``int``): Index of the value in ``array`` that is closest to
            ``target_value``.
    """
    index = array.searchsorted(target_value)
    index = np.clip(index, 1, len(array) - 1)
    left = array[index - 1]
    right = array[index]
    index -= target_value - left < right - target_value
    return index


# Manipulation of a data cube of number densities
class DensityMap(object):
    """
    The idea of this class if to manipulate density arrays to produce finer
    grids, expand and/or extrapolate values beyond the limits and remove
    singularities.

    Args:

        array:
    """
    def __init__(self, array):
        self.array = array
        self.log_array = np.log10(array)

        # Initiating useful variables
        self.result = None
        self.log_result = None

    # Interpolate values to produce a finer grid of densities
    def interpolate(self, factor=10, order=3):
        """

        Args:

            factor:

            order:
        """
        # First we take the log10 of the array because interpolating in
        # log-space gives a better result
        if self.result is None:
            self.log_result = zoom(self.log_array, zoom=factor, order=order)
            self.result = 10 ** self.log_result
        else:
            self.log_result = zoom(self.log_result, zoom=factor, order=order)
            self.result = 10 ** self.log_result

    # Extrapolate array beyond its borders
    def extrapolate(self, add_ij, value=0.0, log_space=True):
        """

        Args:

            add_ij (`int`): Number of cells (lines and columns) to add on each
                side of the array. In the future, this will accept different
                sizes, such as a different number of lines to add on the top
                and bottom, as well as the sides.

            value (`float`, optional): Value towards which the extrapolation
                goes to. Default is 0.0.

            log_space (`bool`, optional): Work in log-space? Default is `True`.
        """
        if self.result is not None:
            ni, nj = np.shape(self.result)
            if log_space is False:
                old_arr = self.result
            else:
                old_arr = self.log_result
        else:
            ni, nj = np.shape(self.array)
            if log_space is False:
                old_arr = self.array
            else:
                old_arr = self.log_array

        arr_i = np.ones(shape=(add_ij, nj)) * value
        arr_j = np.ones(shape=(ni + 2 * add_ij, add_ij)) * value
        # Concatenate arr_i on top and bottom
        new_arr = np.concatenate((arr_i, old_arr, arr_i), axis=0)
        # Concatenate arr_j on the sides
        new_arr = np.concatenate((arr_j, new_arr, arr_j), axis=1)

        if log_space is False:
            self.result = new_arr
            self.log_result = np.log10(self.result)
        else:
            self.log_result = new_arr
            self.result = 10 ** self.log_result


# Test module
import matplotlib.pyplot as plt
import matplotlib.colors as c

if __name__ == "__main__":
    test_array = np.zeros((100, 100))
    for i in range(len(test_array)):
        for j in range(len(test_array[0, :])):
            test_array[i, j] = 10. + 1E10 * np.exp(-((i - 50) ** 2 / 10 ** 2 +
                                                    (j - 50) ** 2 / 10 ** 2))
    dens = DensityMap(test_array)
    dens.interpolate(factor=2)
    dens.extrapolate(10, 1.0, log_space=True)
    test_array = dens.result
    norm = c.LogNorm()
    plt.imshow(test_array, norm=norm)
    plt.show()