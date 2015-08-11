#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon for stellar distances """

import cPickle as pickle
import logging
import numpy as np
import sys
from warnings import simplefilter

import scipy.optimize as op
from astropy.table import Table


# Set up logging.
logging.basicConfig(level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sick")

simplefilter("ignore", np.RankWarning)
simplefilter("ignore", RuntimeWarning)



class CannonModel(object):

    def __init__(self, labels, fluxes, flux_uncertainties, verify=True):
        """
        Initialise a Cannon model.

        :param labels:
            A table with columns as labels, and stars as rows.

        :type labels:
            :class:`~astropy.table.Table`

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

        fluxes = np.atleast_2d(fluxes)
        flux_uncertainties = np.atleast_2d(flux_uncertainties)

        if len(labels) != fluxes.shape[0]:
            raise ValueError("the fluxes should have shape (n_stars, n_pixels) "
                "where n_stars is the number of rows in the labels array")

        if fluxes.shape != flux_uncertainties.shape:
            raise ValueError("the flux and flux uncertainties array should have"
                " the same shape")

        if len(labels) == 0:
            raise ValueError("no stars (labels) given")

        self._trained = False
        self._labels = labels
        self._fluxes = fluxes
        self._flux_uncertainties = flux_uncertainties
        self._label_vector_description = None

        if verify:
            self._check_forbidden_label_characters("^*")

        return None



    def train(self, label_vector_description, N=None, limits=None, pivot=False,
        **kwargs):


        # Save and interpret the label vector description.
        self._label_vector_description = label_vector_description
        lv = self._parse_label_vector_description(label_vector_description)

        # Build the label vector array.
        lva, star_indices, offsets = _build_label_vector_array(
            self._labels, lv, N, limits, pivot)


        N_stars, N_pixels = self._fluxes.shape[:2]
        scatter = np.nan * np.ones(N_pixels)
        coefficients = np.nan * np.ones((N_pixels, lva.shape[1]))
        
        increment = int(N_pixels / 100)
        progressbar = kwargs.pop("__progressbar", True)
        if progressbar:
            sys.stdout.write("\rTraining Cannon model from {0} stars with {1} pixels each:\n".format(
                N_stars, N_pixels))
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

        self._coefficients, self._scatter, self._offsets, self._trained \
            = coefficients, scatter, offsets, True

        return (coefficients, scatter, offsets)


    def solve_labels(self, flux, flux_uncertainties=None, **kwargs):

        if flux_uncertainties is None:
            variance = np.zeros_like(flux)
        else:
            variance = flux_uncertainties**2

        # Which parameters are actually in the Cannon model?
        # (These are the ones we have to solve for.)
        label_vector_indices = self._parse_label_vector_description(
            self._label_vector_description, verbose=False)
        indices = np.unique(np.hstack(
            [[term[0] for term in vector_terms if term[1] != 0] \
            for vector_terms in label_vector_indices]))

        finite = np.isfinite(self._coefficients[:, 0] * flux * variance)

        # Get an initial estimate of those parameters from a simple inversion.
        # (This is very much incorrect for non-linear terms).
        Cinv = 1.0 / (self._scatter[finite]**2 + variance[finite])
        A = np.dot(self._coefficients[finite, :].T,
            Cinv[:, None] * self._coefficients[finite, :])
        B = np.dot(self._coefficients[finite, :].T,
            Cinv * flux[finite])
        initial_vector_labels = np.linalg.solve(A, B)
        
        # p0 contains all coefficients, but we need only the linear terms for
        # the initial estimate
        _ = np.array([i for i, vector_terms \
            in enumerate(label_vector_indices) if len(vector_terms) == 1 \
            and vector_terms[0][1] == 1])
        if len(_) == 0:
            raise ValueError("no linear terms in Cannon model")

        p0 = initial_vector_labels[1 + _]


        label_vector_indices2 = self._parse_label_vector_description(
            self._label_vector_description, verbose=False,
            return_indices=True)

        # TODO HACKIEST SHIT EVER
        translate = {}
        vals = []
        for each in label_vector_indices2:
            for cv, o in each:
                vals.append(cv)

        vals = np.unique(np.sort(vals))
        for i, v in enumerate(vals):
            translate[v] = i

        label_vector_indices3 = []
        for each in label_vector_indices2:
            foo = []
            for cv, o in each:
                foo.append((translate[cv], o))
            label_vector_indices3.append(foo)


        # Create the function.
        def f(coefficients, *labels):
            return np.dot(coefficients, _build_label_vector_rows(
                label_vector_indices3, labels).T).flatten()

        # Optimise the curve to solve for the parameters and covariance.
        full_output = kwargs.pop("full_output", False)
        kwds = kwargs.copy()
        kwds.setdefault("maxfev", 10000)
        labels, covariance = op.curve_fit(f, self._coefficients[finite],
            flux[finite], p0=p0, sigma=1.0/np.sqrt(Cinv),
            absolute_sigma=True, **kwds)

        # Since we might not have solved for every parameter, let's return a 
        # dictionary. Don't forget to apply the offsets to the inferred labels.
        labels = dict(zip(np.array(self._labels.colnames)[vals],
            labels))
        print("TODO: IGNORING OFFSETS")

        if full_output:
            return (labels, covariance)

        return labels


    def _check_forbidden_label_characters(self, characters):
        """
        Check the label table for potentially forbidden characters.

        :param characters:
            A string of forbidden characters.

        :type characters:
            str

        :returns:
            True

        :raises ValueError:
            If a forbidden character is in a potential label name.
        """

        for column in self._labels.dtype.names:
            for character in characters:
                if character in column:
                    raise ValueError("forbidden character '{0}' is in potential"
                        " label '{1}' - to ignore this use verify=False".format(
                            character, column))
        return True


    def _parse_label_vector_description(self, label_vector_description,
        return_indices=False, **kwargs):
        """
        Parse a human-readable label vector description into indices (that refer
        back to the labels table) and orders for all of the cross-terms.
        
        :param label_vector_description:
            The human-reable label vector description. These labels are expected
            to be columns in the labels table that was supplied when the class
            was initialised.

        :type label_vector_description:
            str

        :param return_indices: [optional]
            Return the indices corresponding to the label table instead of the
            actual parameter name itself.

        :type return_indices:
            bool

        :returns:
            A list where each item contains indices and order information to
            represent the label vector description.

            *** TODO INCLUDE EXAMPLE ***
        """

        if isinstance(label_vector_description, (str, unicode)):
            label_vector_description = label_vector_description.split()

        # Functions to parse the parameter (or index) and order for each term.
        order = lambda t: int((t.split("^")[1].strip() + " ").split(" ")[0]) \
            if "^" in t else 1

        param = lambda d: (d + "^").split("^")[0].strip()
        if return_indices:
            param = lambda d: self._labels.colnames.index(param(d))

        theta = []
        for description in label_vector_description:

            # Is it just a parameter?
            try:
                index = self._labels.colnames.index(description.strip())

            except ValueError:

                # Either the description is not a column in the labels table, or
                # this is a cross-term.

                if "*" in description:
                    cross_terms = []
                    for ct in description.split("*"):
                        try:
                            index = param(ct)
                        except ValueError:
                            raise ValueError("couldn't interpret '{0}' in the "\
                                "label '{1}' as a parameter coefficient".format(
                                    *map(str.strip, (ct, description))))
                        cross_terms.append((param(ct), order(ct)))
                    theta.append(cross_terms)

                elif "^" in description:
                    theta.append([(
                        param(description),
                        order(description)
                    )])

                else:
                    raise ValueError("could not interpret '{0}' as a parameter"\
                        " coefficient description".format(description))
            else:
                theta.append([(param(description), order(description))])

        if kwargs.pop("verbose", True):
            logger.info("Training Cannon model using label vector description:"\
                " {}".format(self._repr_label_vector_description(theta)))

        return theta


    def _repr_label_vector_description(self, label_vector_indices):
        """
        Represent label vector indices as a readable label vector description.

        :param label_vector_indices:
            A list of label vector indices. Each item in the list is expected to
            be a tuple of cross-terms (each in a list).

        :returns:
            A human-readable string of the label vector description.
        """

        string = ["1"]
        for cross_terms in label_vector_indices:
            sub_string = []
            for descr, order in cross_terms:
                if order > 1:
                    sub_string.append("{0}^{1}".format(descr, order))
                else:
                    sub_string.append(descr)
            string.append(" * ".join(sub_string))
        raise FORDOCCO
        return " + ".join(string)


    def save(self, filename, overwrite=False):
        """
        Save the (trained) model to disk. This will save the label vector
        description, the optimised coefficients and scatter, and pivot offsets.

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

        if not self._trained:
            raise TypeError("the model has not been trained; there is nothing "
                "to save")

        contents = \
            (self._label_vector_description, self._coefficients, self._scatter,
                self._offsets)
        with open(filename, "w") as fp:
            pickle.dump(contents, fp, -1)

        return True


    def load(self, filename):
        """
        Load a trained model from disk.

        :param filename:
            The file path where to load the model from.

        :type filename:
            str

        :returns:
            True

        :raises IOError:
            If the model could not be loaded.
        """

        with open(filename, "r") as fp:
            contents = pickle.load(fp)

        self._label_vector_description, self._coefficients, self._scatter, \
            self._offsets = contents
        self._trained = True

        return True




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

    try:
        return np.vstack(columns).T
    except ValueError:
        columns[0] = np.ones(1)
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

