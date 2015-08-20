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

try:
    from . import (model, plot, utils)
except ValueError:
    import model
    import plot
    import utils


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

        super(self.__class__, self).__init__(labels, fluxes, flux_uncertainties,
            verify=verify, wavelengths=wavelengths)
        return None


    def train(self, atomic_wavelengths, label_vector_description=None, N=None,
        limits=None, pivot=False, **kwargs):
        """
        Train the model.

        # Fit profiles to each transition in each spectrum.
        # Fit EWs as a function of Teff, logg, etc
        # Fit pixel-by-pixel scatter & label vector. 
        """

        if N is not None or limits is not None or pivot:
            raise NotImplementedError

        # For each star, fit the equivalent widths.

        N_stars, N_atomic_lines = self._fluxes.shape[0], atomic_wavelengths.size
        equivalent_widths = np.nan * np.ones((N_stars, N_atomic_lines))

        msg = "Measuring equivalent widths of {0} atomic lines in {1} stars:"\
            .format(N_atomic_lines, N_stars)
        with utils.ProgressBar(msg, kwargs.get("__progressbar", True)) as pb:

            for i in range(N_stars):
                pb.update(i, N_stars)
                equivalent_widths[i, :] = _fit_absorption_profiles(
                    self._wavelengths, self._fluxes[i, :],
                    self._flux_uncertainties[i, :], atomic_wavelengths)


        raise a


        self._label_vector_description = label_vector_description
        
        # Build the label vector array.
        lv = self._parse_label_vector_description(label_vector_description)
        lva, use, offsets = _build_label_vector_array(self._labels, lv,
            N, limits, pivot)

        # Initialise the requisite arrays.
        N_stars, N_pixels = self._fluxes.shape[:2]
        scatter = np.nan * np.ones(N_pixels)
        coefficients = np.nan * np.ones((N_pixels, lva.shape[0]))
        
        # Display a progressbar unless requested otherwise.
        increment = int(N_pixels / 100)
        progressbar = kwargs.pop("__progressbar", True)
        if progressbar:
            sys.stdout.write("\rTraining Cannon model from {0} stars with {1} "\
                "pixels each:\n".format(N_stars, N_pixels))
            sys.stdout.flush()

        for i in xrange(N_pixels):
            if progressbar and (i == 0 or i % increment == 0):
                sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%".format(
                    done="=" * int((i + 1) / increment),
                    not_done=" " * int((N_pixels - i - 1)/ increment),
                    percent=100. * (i + 1)/N_pixels))
                sys.stdout.flush()

            coefficients[i, :], scatter[i] = _fit_pixel(
                self._fluxes[use, i], self._flux_uncertainties[use, i], lva,
                **kwargs)

            if not np.any(np.isfinite(scatter[i] * coefficients[i, :])):
                logger.warn("No finite coefficients at pixel {}!".format(i))

        if progressbar:
            sys.stdout.write("\r\n")
            sys.stdout.flush()

        self._coefficients, self._scatter, self._offsets, self._trained \
            = coefficients, scatter, offsets, True

        return (coefficients, scatter, offsets)
        raise NotImplementedError


    @model.requires_training_wheels
    def save(self, filename):
        """
        Save the (trained) model to disk. 

        :param filename:
            The file path where to save the model to.

        :type filename:
            str

        :param overwrite: [optional]
            Overwrite the existing file path, if it already exists.

        :type overwrite:
            bool

        :returns:
            True

        :raise TypeError:
            If the model has not been trained, since there is nothing to save.
        """

        # Create a hash of the labels, fluxes and flux uncertainties.
        if verify:
            hashes = [hash(str(_)) for _ in \
                (self._labels, self._fluxes, self._flux_uncertainties)]
        else:
            hashes = None

        raise NotImplementedError
        contents = \
            (self._label_vector_description, self._coefficients, self._scatter,
                self._offsets, hashes)
        with open(filename, "w") as fp:
            pickle.dump(contents, fp, -1)

        return True


def _fractional_absorption_depths(wavelengths, fluxes, atomic_wavelengths):
    """
    Return the flux depths at the given rest wavelengths.

    # Assumes normalised continuum
    """

    _ = np.isfinite(fluxes)
    return np.clip(1 - np.interp(atomic_wavelengths, wavelengths[_], fluxes[_],
        left=1, right=1), 0, 1)


def _absorption_line_mask(wavelengths, atomic_wavelengths, window):
    """
    Return pixels that are within +/- window of each of the atomic wavelengths.
    """

    mask = np.zeros(wavelengths.size, dtype=bool)
    for atomic_wavelength in np.atleast_1d(atomic_wavelengths):
        lower, upper = atomic_wavelength - window, atomic_wavelength + window
        mask += (upper > wavelengths) * (wavelengths > lower)
    return mask




def _fit_absorption_profiles(wavelengths, fluxes, flux_uncertainties,
    atomic_wavelengths, DEBUG=False, full_output=False, **kwargs):
    """
    Assumed perfectly normalised, but this is extensible to treat continuum and
    radial velocity.
    """

    # At each of the rest_wavelengths, get the flux depth.


    # Using data +/- 1 Angstrom of the rest_wavelengths, and the initial flux
    # depths, get a good estimate for the FWHM kernel.
    window = kwargs.pop("_initial_kernel_window", 1)
    mask = _absorption_line_mask(wavelengths, atomic_wavelengths, window)
    fds = _fractional_absorption_depths(wavelengths, fluxes, atomic_wavelengths)
    x, y, y_err = wavelengths[mask], fluxes[mask], flux_uncertainties[mask]

    kernel = op.fmin(_nll_single_kernel, 0.1, args=(x, y, y_err, fds,
        atomic_wavelengths), disp=False)

    if DEBUG:

        fig, ax = plt.subplots()
        ax.plot(wavelengths, fluxes, c='k')
        ax.scatter(x, y, facecolor="r")
        model_fluxes = _model_absorption_lines_single_kernel(wavelengths, kernel,
            fds, atomic_wavelengths)
        ax.plot(wavelengths, model_fluxes, c='r')

        # Now actually model the LSF.
        lsf_degree = kwargs.pop("lsf_degree", 4)
        if lsf_degree > 0:

            mask = _absorption_line_mask(wavelengths, atomic_wavelengths, 3*kernel)
            x, y, y_err = wavelengths[mask], fluxes[mask], flux_uncertainties[mask]

            p0 = np.hstack([np.zeros(lsf_degree), kernel])
            kernel_coefficients = op.fmin(_nll_absorption_lines, p0,
                args=(x, y, y_err, fds, atomic_wavelengths), disp=False)

            model_fluxes = _model_absorption_lines(wavelengths, kernel_coefficients,
                fds, atomic_wavelengths)
            ax.plot(wavelengths, model_fluxes, c='b')


    # Fit sigmas to each line?
    # wavelength_tolerance

    # Need to return: EWs for each line, (wavelengths, sigmas, flux depths,)

    #fitted_atomic_wavelengths =

    # Prepare arrays.
    profile_sigmas = kernel * np.ones(atomic_wavelengths.size)
    profile_depths = fds.copy()
    profile_wavelengths = atomic_wavelengths.copy()
    
    mu_tolerance = kwargs.pop("wavelength_tolerance",
        np.diff(wavelengths).mean())

    for i, (atomic_wavelength, depth) in enumerate(zip(atomic_wavelengths, fds)):

        mask = _absorption_line_mask(wavelengths, atomic_wavelength, 5 * kernel)
        if 3 >= mask.sum(): continue

        x, y, y_err = wavelengths[mask], fluxes[mask], flux_uncertainties[mask]

        # Fit a profile to this line.
        p0 = [atomic_wavelength, kernel, depth]

        def _bounded_gaussian(x, mu, sigma, amplitude, mu_bounds=None,
            sigma_bounds=None, amplitude_bounds=None):
            
            if not (mu + mu_tolerance) > mu > (mu - mu_tolerance) \
            or not (1.3 * kernel) > sigma > 0 \
            or not 1 > amplitude > 0:
                return np.inf * np.ones(x.size)
            return gaussian(x, mu, sigma, amplitude)

        p_opt, p_cov = op.curve_fit(_bounded_gaussian, x, y, p0=p0, sigma=y_err,
            absolute_sigma=True)
        
        profile_wavelengths[i], profile_sigmas[i], profile_depths[i] = p_opt

        if DEBUG:

            fig, ax = plt.subplots()
            ax.plot(x, y, c='k')
            ax.plot(x, gaussian(x, *p0), c='r')
            ax.plot(x, gaussian(x, *p_opt), c='b')

            ax.axvline(atomic_wavelength, c='r')
            ax.axvline(p_opt[0], c='b')
            ax.axvline(atomic_wavelength - mu_tolerance, c='g')
            ax.axvline(atomic_wavelength + mu_tolerance, c='g')
            
            ax.set_title(p_opt[1]/kernel)

    # Integrate profiles.
    equivalent_widths = np.sqrt(2*np.pi) * profile_sigmas * profile_depths

    if full_output:
        return (equivalent_widths, profile_wavelengths, profile_sigmas, profile_depths)
    return equivalent_widths

    
    """

    # Using data +/- 3 sigma of each line, fit the line depths and the fwhm
    # simultaneously (the FWHM may be a low-order).
    mask = _absorption_line_mask(wavelengths, atomic_wavelengths, 3*kernel)
    x, y, y_err = wavelengths[mask], fluxes[mask], flux_uncertainties[mask]

    f = lambda x, *p: _model_absorption_lines_single_kernel(x, p[0], p[1:],
        atomic_wavelengths)

    result = op.curve_fit(f, x, y, p0=np.hstack([kernel, fds]), sigma=y_err,
        absolute_sigma=True)
    f = lambda p: _nll_single_kernel(p[0], x, y, y_err, p[1:],
        atomic_wavelengths)

    result = op.fmin_powell(f, np.hstack([kernel, fds]), disp=False)

    model_fluxes2 = _model_absorption_lines_single_kernel(wavelengths, result[0],
        result[1:], atomic_wavelengths)

    ax.scatter(x, y, facecolor="b")
    ax.plot(wavelengths, model_fluxes2, c='b')
    """

def _model_absorption_lines(wavelengths, kernel_coefficients, flux_depths,
    atomic_wavelengths, continuum=1):

    sigmas = np.polyval(kernel_coefficients, atomic_wavelengths)
    continuum *= np.ones(wavelengths.size)
    return continuum * np.product([gaussian(wavelengths, mu, sigma, depth) \
        for mu, sigma, depth, in zip(atomic_wavelengths, sigmas, flux_depths)], axis=0)


def _nll_absorption_lines(kernel_coefficients, wavelengths, fluxes,
    flux_uncertainties, flux_depths, atomic_wavelengths):

    model = _model_absorption_lines(wavelengths, kernel_coefficients,
        flux_depths, atomic_wavelengths)

    chi_sq = (model - fluxes)**2 / flux_uncertainties**2
    return 0.5 * np.nansum(chi_sq)


def _model_absorption_lines_single_kernel(wavelengths, sigma, flux_depths,
    atomic_wavelengths, continuum=1):
    """
    Model absorption lines with a single kernel width.
    """
    continuum = continuum * np.ones(wavelengths.size)
    return continuum * np.product([gaussian(wavelengths, mu, sigma, depth) \
        for mu, depth in zip(atomic_wavelengths, flux_depths)], axis=0)


def _nll_single_kernel(sigma, wavelengths, fluxes, flux_uncertainties,
    flux_depths, atomic_wavelengths):

    if 0 >= sigma:
        return np.inf

    model = _model_absorption_lines_single_kernel(wavelengths, sigma,
        flux_depths, atomic_wavelengths)
    chi_sq = (model - fluxes)**2 / flux_uncertainties**2
    return 0.5 * np.nansum(chi_sq)





def gaussian(x, mu, sigma, amplitude):
    return 1. - amplitude * np.exp(-(x - mu)**2 / (2. * sigma**2))


if __name__ == "__main__":

    from astropy.io import fits
    image = fits.open("../data/APOGEE/Clusters/M71/aspcapStar-r5-v603-2M19533470+1846213.fits")

    wavelengths = 10**(np.arange(image[1].data.size) * image[1].header["CDELT1"] \
        + image[1].header["CRVAL1"])
    fluxes = image[1].data
    fluxes[0 >= fluxes] = np.nan

    atomic_wavelengths = [16099.2, 16106.7]

    plt.plot(wavelengths, fluxes, c='k')

    fd = _fractional_absorption_depths(wavelengths, fluxes, atomic_wavelengths)
    model_fluxes = _model_absorption_lines_single_kernel(wavelengths, 0.1, fd, atomic_wavelengths)


    plt.plot(wavelengths, model_fluxes, c='r')

    uncertainties = image[2].data
    uncertainties[~np.isfinite(fluxes)] = 10e8
    uncertainties[0 >= uncertainties] = 10e8

    


    def vac2air(wave,sdssweb=False):
        """
        NAME:
           vac2air
        PURPOSE:
           Convert from vacuum to air wavelengths (See Allende Prieto technical note: http://hebe.as.utexas.edu/apogee/docs/air_vacuum.pdf)
        INPUT:
           wave - vacuum wavelength in \AA
           sdssweb= (False) if True, use the expression from the SDSS website (http://classic.sdss.org/dr7/products/spectra/vacwavelength.html)
        OUTPUT:
           air wavelength in \AA
        HISTORY:
           2014-12-04 - Written - Bovy (IAS)
           2015-04-27 - Updated to CAP note expression - Bovy (IAS)
        """
        if sdssweb:
            return wave/(1.+2.735182*10.**-4.+131.4182/wave**2.+2.76249*10.**8./wave**4.)
        else:
            return wave/(1.+0.05792105/(238.0185-(10000./wave)**2.)+0.00167917/(57.362-(10000./wave)**2.))

    def air2vac(wave,sdssweb=False):
        """
        NAME:
           air2vac
        PURPOSE:
           Convert from air to vacuum wavelengths (See Allende Prieto technical note: http://hebe.as.utexas.edu/apogee/docs/air_vacuum.pdf)
        INPUT:
           wave - air wavelength in \AA
           sdssweb= (False) if True, use the expression from the SDSS website (http://classic.sdss.org/dr7/products/spectra/vacwavelength.html)
        OUTPUT:
           vacuum wavelength in \AA
        HISTORY:
           2014-12-04 - Written - Bovy (IAS)
           2015-04-27 - Updated to CAP note expression - Bovy (IAS)
        """
        return op.brentq(lambda x: vac2air(x,sdssweb=sdssweb)-wave,
                               wave-20,wave+20.)




    _FEI_lines= [air2vac(l) for l in [15194.492,15207.526,15395.718,15490.339,
                                      15648.510,15964.867,16040.657,16153.247,
                                      16165.032]]
    _FEI_lines.append(16697.635) # one more from Shetrone

    # From Table 5
    _MGI_lines= [air2vac(l) for l in [15740.716,15748.9,15765.8,15879.5,
                                      15886.2,15889.485,15954.477]]
    _ALI_lines= [air2vac(l) for l in [16718.957,16763.359]]
    _SII_lines= [air2vac(l) for l in [15361.161,15376.831,15833.602,15960.063,
                                      16060.009,16094.787,16215.670,16680.770,
                                      16828.159]]
    _KI_lines= [air2vac(l) for l in [15163.067,15168.376]]
    _CAI_lines= [air2vac(l) for l in [16136.823,16150.763,16155.236,16157.364]]
    _TII_lines= [air2vac(l) for l in [15543.756,15602.842,15698.979,15715.573,
                                      16635.161]]
    _VI_lines= [air2vac(15924.)]
    _CRI_lines= [air2vac(l) for l in [15680.063,15860.214]]
    _MNI_lines= [air2vac(l) for l in [15159.,15217.,15262.]]
    _COI_lines= [air2vac(16757.7)]
    _NII_lines= [air2vac(l) for l in [15605.680,15632.654,16584.439,16589.295,
                                      16673.711,16815.471,16818.760]]
    _CUI_lines= [air2vac(16005.7)]
    # From Katia Cunha
    _NAI_lines= [air2vac(16373.86),air2vac(16388.85)]
    # From Matthew Shetrone
    _SI_lines= [15406.540,15426.490,15474.043,15482.712]

    atomic_wavelengths = np.hstack([_FEI_lines, _MGI_lines, _ALI_lines, _SII_lines,
        _KI_lines, _CAI_lines, _TII_lines, _VI_lines, _CRI_lines, _MNI_lines,
        _COI_lines, _NII_lines, _CUI_lines, _NAI_lines, _SI_lines])


    _fit_absorption_profiles(wavelengths, fluxes, uncertainties, atomic_wavelengths)



    # Load model fluxes.
    DATA_PREFIX = "../data/APOGEE-Hipparcos"
    from astropy.table import Table

    stars = Table.read("{}.fits.gz".format(DATA_PREFIX))
    fluxes = np.memmap("{}-flux.memmap".format(DATA_PREFIX), mode="r", dtype=float)
    flux_uncertainties = np.memmap("{}-flux-uncertainties.memmap".format(DATA_PREFIX),
        mode="r", dtype=float)

    # Re-shape
    fluxes = fluxes.reshape((len(stars), -1))
    flux_uncertainties = flux_uncertainties.reshape(fluxes.shape)
    wavelengths = np.memmap("{}-wavelength.memmap".format(DATA_PREFIX), mode="r",
        dtype=float)

    model = PendulumModel(stars, wavelengths, fluxes, flux_uncertainties)
    model.train(atomic_wavelengths)
    raise a



