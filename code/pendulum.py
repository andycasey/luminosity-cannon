#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon + Physics = Pendulum """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import numpy as np
import random
import sys
from itertools import chain, combinations
from math import factorial
from warnings import simplefilter

import scipy.optimize as op
from astropy.table import Table

from . import (model, plot)


class PendulumModel(model.BaseModel):

    def __init__(self, labels, wavelengths, fluxes, flux_uncertainties,
        verify=True):
        """
        Initialise a Pendulum model.

        :param labels:
            A table with columns as labels, and stars as rows.

        :type labels:
            :class:`~astropy.table.Table`

        :param wavelengths:
            The wavelengths of the given pixels.

        :type wavelengths:
            :class:`np.array`

        :param fluxes:
            An array of fluxes for each star as shape (num_stars, num_pixels).
            The num_stars should match the rows in `labels`.

        :type fluxes:
            :class:`np.ndarray`

        :param flux_uncertainties:
            An array of 1-sigma flux uncertainties for each star as shape
            (num_stars, num_pixels). The shape of the `flux_uncertainties` array
            should match the `fluxes` array. 

        :type flux_uncertainties:
            :class:`np.ndarray`
        """

        super(self, CannonModel).__init__(self, labels, fluxes,
            flux_uncertainties, verify=verify)
        self._wavelengths = wavelengths
        return None


    def train(self, atomic_wavelengths):
        """
        Train the model.

        # Fit profiles to each transition in each spectrum.
        # Fit EWs as a function of Teff, logg, etc
        # Fit pixel-by-pixel scatter & label vector. 
        """

        raise NotImplementedError
