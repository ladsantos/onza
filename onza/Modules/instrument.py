#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes instrumental properties to model theoretical spectral
lines.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from scipy.integrate import simps


# The Hubble STIS instrument class
class HubbleSTIS(object):
    """
    The Hubble STIS spectrograph instrumental configuration class.

    Args:

        resolution (`float`): Target spectral resolution to use in the
            computation of the instrumental response.

        config (`str`, optional): String specifying the configuration to be used
            in the computation of the instrumental response. Available options
            are: `'STIS_HI_HD189_2010'`, `'STIS_HI_HD189_2011'`,
            `'STIS_HI_HD209'`, `'GJ436_HI_0'`, `'GJ436_HI_2'`, `'GJ436_HI_3'`,
            `'Kepler444_HI'`, `'HD97658_HI_23'`, and `'TRAPPIST1_HI'`. If not
            specified, no instrumental response function is set.

        simulation_path (`str`, optional): In case the user wants to use a
            specific configuration instead of the ones available for the
            variable `config`, the user can input a tabulated line spread
            function here. This option is not implemented yet.

        wavelength_range (sequence, optional): Wavelength range in which to
            compute the instrumental response. If not defined, the instrumental
            response is computed according to the specified configuration.
    """
    def __init__(self, resolution, config=None, simulation_path=None,
                 wavelength_range=None):
        # LSF = Line Spread Function
        self.config = config
        self.resolution = resolution
        self.sim_path = simulation_path
        self.wv_range = wavelength_range

        # Instantiating useful global variables
        self.wavelength = None
        self.kernel = None

        # Configuration not set
        if self.config is None:
            self.response_mode = None
            self.kernel_mode = None
            self.coeff_LSF = None
            self.LSF_core = None
            self.LSF_wing = None
            self.hrange_LSF = None
            self.instrument_resolution = None

        # HD189_2010 configuration
        elif self.config == 'STIS_HI_HD189_2010':
            self.response_mode = 'LSF'
            self.kernel_mode = 'double_gaussian'
            self.coeff_LSF = 0.88
            self.LSF_core = 0.0416
            self.LSF_wing = 0.2027
            self.hrange_LSF = 0.626343
            self.instrument_resolution = 0.05335

        # HD189_2011 configuration
        elif self.config == 'STIS_HI_HD189_2011':
            self.response_mode='LSF'
            self.kernel_mode='double_gaussian'
            self.coeff_LSF=0.9456
            self.LSF_core=0.0665
            self.LSF_wing=0.2665
            self.hrange_LSF=0.754044
            self.instrument_resolution=0.05337

        # HD209 configuration
        elif self.config == 'STIS_HI_HD209':
            self.response_mode = 'LSF'
            self.kernel_mode = 'double_gaussian'
            self.coeff_LSF = 0.51785155
            self.LSF_core = 0.094208
            self.LSF_wing = 0.051881
            self.hrange_LSF = 0.33307602
            self.instrument_resolution = 0.05302

        # GJ436 configuration 0
        elif self.config == 'GJ436_HI_0':
            self.response_mode = 'LSF'
            self.kernel_mode = 'single_gaussian'
            self.fwhm_LSF = 0.07652675
            self.hrange_LSF = 0.284463523
            self.instrument_resolution = 0.0530366854

        # GJ436 configuration 2
        elif self.config == 'GJ436_HI_2':
            self.response_mode = 'LSF'
            self.kernel_mode = 'single_gaussian'
            self.fwhm_LSF = 0.05174745
            self.hrange_LSF = 0.192322867
            self.instrument_resolution = 0.0533312197

        # GJ436 configuration 3
        elif self.config == 'GJ436_HI_3':
            self.response_mode = 'LSF'
            self.kernel_mode = 'single_gaussian'
            self.fwhm_LSF = 0.06670499
            self.hrange_LSF = 0.247939299
            self.instrument_resolution = 0.05333425

        # Kepler-444 configuration
        elif self.config == 'Kepler444_HI':
            self.coeff_LSF = 0.93
            self.LSF_core = 0.05172355056
            self.LSF_wing = 0.2479531032
            self.hrange_LSF = 0.84849551915067423

        # HD977658 configuration
        elif self.config == 'HD97658_HI_23':
            self.response_mode = 'LSF'
            self.kernel_mode = 'double_gaussian'
            self.instrument_resolution = 0.0533
            self.coeff_LSF = 0.933
            self.LSF_core = 0.0533
            self.LSF_wing = 0.3069547
            self.hrange_LSF = 1.046715527

        # TRAPPIST1 configuration
        elif self.config == 'TRAPPIST1_HI':
            self.response_mode = 'LSF'
            self.kernel_mode = 'double_gaussian'
            self.instrument_resolution = 0.0533
            self.coeff_LSF = 0.935946
            self.LSF_core = 0.05174577
            self.LSF_wing = 0.245484
            self.hrange_LSF = 0.833664828828

        # Raise an error if the configuration specified by the user is
        # non-existent.
        else:
            raise NotImplementedError("The configuration '%s' is not "
                                      "implemented." % self.config)

    # Compute the discrete kernel function
    def compute_kernel(self):
        """
        Computes the kernel of the instrumental response.
        """
        # Half number of pixels in the kernel table at the resolution of the
        # band spectrum
        if self.wv_range is None:
            half_number = int(self.hrange_LSF / self.resolution) + 1
        else:
            half_number = int(self.wv_range / self.resolution) + 1

        # Centered spectral table with same pixel widths as the band spectrum
        # the kernel is associated to
        n_kernel = 2 * half_number + 1
        self.wavelength = self.resolution * (np.arange(n_kernel) -
                                             (n_kernel - 1) / 2)

        # Single Gaussian
        if self.kernel_mode == 'single_gaussian':
            self.kernel = np.exp(-np.power(self.wavelength /
                                           (np.sqrt(2) * self.fwhm_LSF), 2))

        # Double Gaussian
        elif self.kernel_mode == 'double_gaussian':
            kernel_core = self.coeff_LSF * np.exp(
                -np.power(self.wavelength / (np.sqrt(2) * self.LSF_core), 2))
            kernel_wing = (1 - self.coeff_LSF) * np.exp(
                -np.power(self.wavelength / (np.sqrt(2) * self.LSF_wing), 2))
            self.kernel = kernel_core + kernel_wing

        # Tabulated LSF
        elif self.config is None and self.sim_path is not None:
            raise NotImplementedError('Using a tabulated LSF is not '
                                      'implemented yet.')

        else:
            raise ValueError('Either `config` or `simulation_path` have to be '
                             'provided to compute the discrete kernel '
                             'function.')

        # Normalize kernel
        self.kernel /= simps(self.kernel, self.wavelength)
