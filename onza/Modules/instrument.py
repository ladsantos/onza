#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes instrumental properties to model theoretical spectral
lines.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np


# The Hubble STIS instrument class
class HubbleSTIS(object):
    """

    """
    def __init__(self, config, bandwidth, simulation_path=None):
        self.config = config
        self.bandwidth = bandwidth
        self.sim_path = simulation_path

        # Instantiating useful global variables
        self.wavelength = None
        self.kernel = None

        # No instrumental response
        if self.config is None:
            self.response_mode = None
            self.kernel_mode = None
            self.coeff_LSF = None
            self.LSF_core = None
            self.LSF_wing = None
            self.hrange_LSF = None
            self.dw_inst = None

        # HD189_2010 configuration
        elif self.config == 'STIS_HI_HD189_2010':
            # XXX XXX XXX XXX XXX
            # TODO: I need to know exactly what are these parameters
            # XXX XXX XXX XXX XXX
            self.response_mode = 'LSF'
            self.kernel_mode = 'double_gauss'
            self.coeff_LSF = 0.88
            self.LSF_core = 0.0416
            self.LSF_wing = 0.2027
            self.hrange_LSF = 0.626343
            self.dw_inst = 0.05335

        # HD189_2011 configuration
        elif self.config == 'STIS_HI_HD189_2011':
            self.resp_mode='LSF'
            self.kernel_mode='double_gauss'
            self.coeff_LSF=0.9456
            self.LSF_core=0.0665
            self.LSF_wing=0.2665
            self.hrange_LSF=0.754044
            self.dw_inst=0.05337

        # HD209 configuration
        elif self.config == 'STIS_HI_HD209':
            self.resp_mode = 'LSF'
            self.kernel_mode = 'double_gauss'
            self.coeff_LSF = 0.51785155
            self.LSF_core = 0.094208
            self.LSF_wing = 0.051881
            self.hrange_LSF = 0.33307602
            self.dw_inst = 0.05302

        # GJ436 configuration 0
        elif self.config == 'GJ436_HI_0':
            self.resp_mode = 'LSF'
            self.kernel_mode = 'single_gauss'
            self.dw_LSF = 0.07652675
            self.hrange_LSF = 0.284463523
            self.dw_inst = 0.0530366854

        # GJ436 configuration 2
        elif self.config == 'GJ436_HI_2':
            self.resp_mode = 'LSF'
            self.kernel_mode = 'single_gauss'
            self.dw_LSF = 0.05174745
            self.hrange_LSF = 0.192322867
            self.dw_inst = 0.0533312197

        # GJ436 configuration 3
        elif self.config == 'GJ436_HI_3':
            self.resp_mode = 'LSF'
            self.kernel_mode = 'single_gauss'
            self.dw_LSF = 0.06670499
            self.hrange_LSF = 0.247939299
            self.dw_inst = 0.05333425

        # Kepler-444 configuration
        elif self.config == 'Kepler444_HI':
            self.coeff_LSF = 0.93
            self.LSF_core = 0.05172355056
            self.LSF_wing = 0.2479531032
            self.hrange_LSF = 0.84849551915067423

        # HD977658 configuration
        elif self.config == 'HD97658_HI_23':
            self.resp_mode = 'LSF'
            self.kernel_mode = 'double_gauss'
            self.dw_inst = 0.0533
            self.coeff_LSF = 0.933
            self.LSF_core = 0.0533
            self.LSF_wing = 0.3069547
            self.hrange_LSF = 1.046715527

        # TRAPPIST1 configuration
        elif self.config == 'TRAPPIST1_HI':
            self.resp_mode = 'LSF'
            self.kernel_mode = 'double_gauss'
            self.dw_inst = 0.0533
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

        Returns:

        """
        # Half number of pixels in the kernel table at the resolution of the
        # band spectrum
        half_number = int(self.hrange_LSF / self.bandwidth) + 1

        # Centered spectral table with same pixel widths as the band spectrum
        # the kernel is associated to
        n_kernel = 2 * half_number + 1
        self.wavelength = self.bandwidth * (np.arange(n_kernel) -
                                            (n_kernel - 1) / 2)

        # Single Gaussian
        if self.kernel_mode == 'single_gauss':
            self.kernel = np.exp(-np.power(self.wavelength /
                                           (np.sqrt(2) * self.dw_LSF), 2))

        # Double Gaussian
        if self.kernel_mode == 'double_gaussian':
            kernel_core = self.coeff_LSF * np.exp(
                -np.power(self.wavelength / (np.sqrt(2) * self.LSF_core), 2))
            kernel_wing = (1 - self.coeff_LSF) * np.exp(
                -np.power(self.wavelength / (np.sqrt(2) * self.LSF_wing), 2))
            self.kernel = kernel_core + kernel_wing

        # Normalize kernel
        self.kernel /= np.sum(self.kernel)
