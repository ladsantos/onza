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


__all__ = []


# Manipulation of a data cube of number densities
class Density(object):
    """
    The idea of this class if to manipulate density arrays to produce finer
    grids, expand and/or extrapolate values beyond the limits and remove
    singularities.

    """
    def __init__(self, array):
        self.array = array

        # Initiating useful variables
        self.result = None

    # Interpolate values to produce a finer grid of densities
    def interpolate(self, factor=10, order=3):
        """

        Args:
            factor:

        Returns:

        """
        self.result = zoom(self.array, factor, order)