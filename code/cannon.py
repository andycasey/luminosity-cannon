#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon for stellar distances """

import cPickle as pickle
import logging
import numpy as np
import sys

import scipy.optimize as op
from astropy.table import Table


# Set up logging.
logging.basicConfig(level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sick")


class CannonModel(object):

    def __init__(self, labels, fluxes, flux_uncertainties):
        """
        Initialise a Cannon model.

        :param labels:
            A table with columns as labels, and stars as rows.

        :type labels:
            astropy table.

        :param fluxes:
            An array of fluxes for each star as shape (num_stars, num_pixels).
            The num_stars should match the rows in `labels`.

        :type fluxes:
            np.ndarray

        :param flux_uncertainties:
            An array of 1-sigma flux uncertainties for each star as shape
            (num_stars, num_pixels). The shape of the `flux_uncertainties` array
            should match the `flux` array. 

        :type flux_uncertainties:
            np.ndarray
        """

        fluxes = np.atleast_2d(fluxes)
        flux_uncertainties = np.atleast_2d(flux_uncertainties)

        if len(labels) != fluxes.shape[0]:
            raise ValueError("the fluxes should have shape (n_stars, n_pixels) "
                "where n_stars is the number of rows in the labels array")

        if fluxes.shape != flux_uncertainties.shape:
            raise ValueError("the flux and flux uncertainties array should have"
                " the same shape")

        if len(labels) == 0:
            raise ValueError("no stars given")

        self._labels = labels
        self._fluxes = fluxes
        self._flux_uncertainties = flux_uncertainties
        self._trained = False
        self._label_vector_description = None

        print("TODO: Check for forbidden characters in label names")

        return None




    def train(self, label_vector_description, N=None, limits=None, pivot=False,
        **kwargs):


        # Save and interpret the label vector description.
        self._label_vector_description = label_vector_description
        lv = self._interpret_label_vector_description(label_vector_description)

        # Build the label vector array.
        lva, star_indices, offsets = _build_label_vector_array(
            self._labels, lv, N, limits, pivot)


        N_stars, N_pixels = self._fluxes.shape[:2]
        scatter = np.nan * np.ones(N_pixels)
        #coefficients = np.nan * np.ones((N_pixels, lva.shape[1]))
        coefficients = np.memmap("tmp", dtype=float, mode="w+", shape=(N_pixels, lva.shape[1]))

        increment = int(N_pixels / 100)
        progressbar = kwargs.pop("__progressbar", True)
        if progressbar:
            sys.stdout.write("\rTraining Cannon model from {} points:\n".format(
                star_indices.size))
            sys.stdout.flush()

        for i in xrange(N_pixels):
            if progressbar and (i == 0 or i % increment == 0):
                sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%".format(
                    done="=" * int((i + 1) / increment),
                    not_done=" " * int((N_pixels - i - 1)/ increment),
                    percent=100. * (i + 1)/N_pixels))
                sys.stdout.flush()

            coefficients[i, :], scatter[i] = _fit_pixel(
                self._fluxes[:, i], self._flux_uncertainties[:, i], lva)

        if progressbar:
            sys.stdout.write("\r\n")
            sys.stdout.flush()






        raise NotImplementedError



    def solve_labels(self, fluxes, flux_uncertainties):
        raise NotImplementedError







    def _interpret_label_vector_description(self, label_vector_description):

        if isinstance(label_vector_description, (str, unicode)):
            label_vector_description = label_vector_description.split()

        order = lambda t: int((t.split("^")[1].strip() + " ").split(" ")[0]) \
            if "^" in t else 1
        repr_param = lambda d: (d + "^").split("^")[0].strip()

        theta = []
        for description in label_vector_description:

            # Is it just a parameter?
            try:
                index = self._labels.colnames.index(description.strip())

            except ValueError:
                if "*" in description:
                    # Split by * to evaluate cross-terms.
                    cross_terms = []
                    for cross_term in description.split("*"):
                        try:
                            index = repr_param(cross_term)
                        except ValueError:
                            raise ValueError("couldn't interpret '{0}' in the "\
                                "label '{1}' as a parameter coefficient".format(
                                    *map(str.strip, (cross_term, description))))
                        cross_terms.append((cross_term, order(cross_term)))
                    theta.append(cross_terms)

                elif "^" in description:
                    theta.append([(
                        repr_param(description),
                        order(description)
                    )])

                else:
                    raise ValueError("could not interpret '{0}' as a parameter"\
                        " coefficient description".format(description))
            else:
                theta.append([(description.strip(), order(description))])

        logger.info("Training the Cannon model using the following description "
            "of the label vector: {0}".format(self._repr_label_vector_description(theta)))

        return theta


    def _repr_label_vector_description(self, label_vector_indices):

        string = ["1"]
        for cross_terms in label_vector_indices:
            sub_string = []
            for descr, order in cross_terms:
                if order > 1:
                    sub_string.append("{0}^{1}".format(descr, order))
                else:
                    sub_string.append(descr)
            string.append(" * ".join(sub_string))
        return " + ".join(string)




    def save(self):
        # Save the label vector description, coefficients, etc
        raise NotImplementedError

    def load(self):
        # Load the label vector description, coefficients, etc
        raise NotImplementedError


def _fit_coefficients(intensities, u_intensities, scatter, lv_array,
    full_output=False):

    # For a given scatter, return the best-fit coefficients.    
    variance = u_intensities**2 + scatter**2

    CiA = lv_array * np.tile(1./variance, (lv_array.shape[1], 1)).T
    ATCiAinv = np.linalg.inv(np.dot(lv_array.T, CiA))

    Y = intensities/variance
    ATY = np.dot(lv_array.T, Y)
    coefficients = np.dot(ATCiAinv, ATY)

    if full_output:
        return (coefficients, ATCiAinv)
    return coefficients


def _pixel_scatter_ln_likelihood(ln_scatter, intensities, u_intensities,
    lv_array, debug=False):
    
    scatter = np.exp(ln_scatter)

    try:
        # Calculate the coefficients for this level of scatter.
        coefficients = _fit_coefficients(intensities, u_intensities, scatter,
            lv_array)

    except np.linalg.linalg.LinAlgError:
        if debug: raise
        return -np.inf

    model = np.dot(coefficients, lv_array.T)
    variance = u_intensities**2 + scatter**2

    return -0.5 * np.sum((intensities - model)**2 / variance) \
        - 0.5 * np.sum(np.log(variance))


def _fit_pixel(fluxes, flux_uncertainties, lv_array, debug=False):

    # Get an initial guess of the scatter.
    scatter = np.var(fluxes) - np.median(flux_uncertainties)**2
    scatter = np.sqrt(scatter) if scatter >= 0 else np.std(fluxes)

    ln_scatter = np.log(scatter)

    # Optimise the scatter, and at each scatter value we will calculate the
    # optimal vector coefficients.
    nll = lambda ln_s, *a, **k: -_pixel_scatter_ln_likelihood(ln_s, *a, **k)
    op_scatter = np.exp(op.fmin_powell(nll, ln_scatter,
        args=(fluxes, flux_uncertainties, lv_array), disp=False))

    # Calculate the coefficients at the optimal scatter value.
    # Note that if we can't solve for the coefficients, we should just set them
    # as zero and send back a giant variance.
    try:
        coefficients = _fit_coefficients(fluxes, flux_uncertainties, op_scatter,
            lv_array)

    except np.linalg.linalg.LinAlgError:
        logger.exception("Failed to calculate coefficients")
        if debug: raise

        return (np.zeros(lv_array.shape[1]), 10e8)

    else:
        return (coefficients, op_scatter)


def _build_label_vector_rows(label_vector, labels):
    columns = [np.ones(len(labels))]
    for cross_terms in label_vector:
        column = 1
        for descr, order in cross_terms:
            column *= labels[descr].flatten()**order
        columns.append(column)

    return np.vstack(columns).T


def _build_label_vector_array(labels, label_vector, N=None, limits=None,
    pivot=True):

    logger.debug("Building Cannon label vector array")

    indices = np.ones(len(labels), dtype=bool)
    if limits is not None:
        for parameter, (lower_limit, upper_limit) in limits.items():
            indices *= (upper_limit >= labels[parameter]) * \
                (labels[parameter] >= lower_limit)

    if N is not None:
        _ = np.linspace(0, indices.sum() - 1, N, dtype=int)
        indices = np.where(indices)[0][_]   
    
    else:
        indices = np.where(indices)[0]

    labels = labels[indices]
    if pivot:
        raise NotImplementedError
        offsets = labels.mean(axis=0)
        labels -= offsets
    else:
        offsets = np.zeros(len(labels.colnames))

    return (_build_label_vector_rows(label_vector, labels), indices, offsets)








if __name__ == "__main__":

    with open("hipparcos-spectra.pkl", "rb") as fp:
        stars = pickle.load(fp)

    data = np.memmap("hipparcos-spectra.memmap", mode="r", dtype=float)
    data = data.reshape(len(stars) + 1, -1, 2)

    fluxes = data[1:, :, ::2]
    flux_uncertainties = data[1:, :, 1::2]

    # fuck zeros:
    ok = np.where(np.sum(fluxes > 0, axis=1) > 0)
    fluxes = fluxes[:, ok].reshape(len(stars), -1)
    flux_uncertainties = flux_uncertainties[:, ok].reshape(len(stars), -1)

    labels = Table.read("master_table_hip_harps.dat", format="ascii")
    # Ensure the labels are sorted the same as the stars
    sort_indices = np.array([np.where(labels["Star"] == star)[0] for star in stars])
    labels = labels[sort_indices]

    model = CannonModel(labels, fluxes, flux_uncertainties)
    model.train("plx^2")
    

    raise a
