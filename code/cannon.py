#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon for absolute stellar luminosities. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import hashlib
import logging
import numpy as np
import random
import sys
from itertools import chain, combinations
from math import factorial
from warnings import simplefilter

import scipy.optimize as op
from astropy.table import Table


# Set up logger.
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cannon")

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

        self._check_data(labels, fluxes, flux_uncertainties)

        self._trained = False
        self._labels = labels
        self._label_vector_description = None
        self._fluxes, self._flux_uncertainties = fluxes, flux_uncertainties

        if verify:
            self._check_forbidden_label_characters("^*")
        return None


    @property
    def _trained_hash(self):
        """
        Return a hash of the trained state.
        """

        if not self._trained:
            return None
        args = (self._coefficients, self._scatter, self._offsets,
            self._label_vector_description)
        return "".join([str(hash(str(each)))[:10] for each in args])


    def _check_data(self, labels, fluxes, flux_uncertainties):
        """
        Check that the labels, flux and flux uncertainty data is OK.

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

        return None


    def train(self, label_vector_description, N=None, limits=None, pivot=False,
        **kwargs):
        """
        Train a Cannon model based on the label vector description provided.

        :params label_vector_description:
            The human-readable form of the label vector description.

        :type label_vector_description:
            str

        :param N: [optional]
            Limit the number of stars used in the training set. If left to None,
            all stars will be used.

        :type N:
            None or int

        :param limits: [optional]
            A dictionary containing labels (keys) and upper/lower limits (as a
            two-length tuple).

        :type limits:
            dict

        :param pivot: [optional]
            Pivot the data about the labels.

        :type pivot:
            bool

        :returns:
            A three-length tuple containing the model coefficients, the scatter
            in each pixel, and the label offsets.
        """

        self._label_vector_description = label_vector_description
        
        # Build the label vector array.
        lv = self._parse_label_vector_description(label_vector_description)
        lva, star_indices, offsets = _build_label_vector_array(self._labels, lv,
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
                self._fluxes[:, i], self._flux_uncertainties[:, i], lva)

        if progressbar:
            sys.stdout.write("\r\n")
            sys.stdout.flush()

        self._coefficients, self._scatter, self._offsets, self._trained \
            = coefficients, scatter, offsets, True

        return (coefficients, scatter, offsets)


    def _get_linear_indices(self, label_vector_description, full_output=False):
        """
        Return indices of linear labels in the given label vector description.

        :param label_vector_description:
            The human-readable label vector description.

        :type label_vector_description:
            str

        :param full_output: [optional]
            Return the indices and the names of the corresponding linear labels.

        :type full_output:
            bool

        :returns:
            The indices of the linear terms in the label vector description. If
            `full_output` is True, then the corresponding label names are also
            provided. The indices are index-zeroed (i.e., there is no adjustment
            for the zeroth term of a label vector array typically being '1').
        """

        lvi = self._parse_label_vector_description(label_vector_description,
            verbose=False)

        names = []
        indices = []
        for i, term in enumerate(lvi):
            if len(term) == 1 and term[0][1] == 1:
                names.append(term[0][0])
                indices.append(i)
        names, indices = tuple(names), np.array(indices)

        if full_output:
            return (indices, names)
        return indices


    def solve_labels(self, flux, flux_uncertainties, **kwargs):
        """
        Solve the labels for given fluxes (and uncertainties) using the trained
        model.

        :param fluxes:
            The normalised fluxes. These should be on the same wavelength scale
            as the trained data.

        :type fluxes:
            :class:`~np.array`

        :param flux_uncertainties:
            The 1-sigma uncertainties in the fluxes. This should have the same
            shape as `fluxes`.

        :type flux_uncertainties:
            :class:`~np.array`

        :returns:
            The labels for the given fluxes as a dictionary.

        :raises TypeError:
            If the model is not trained.
        """

        if not self._trained:
            raise TypeError("the model has not been trained!")

        # Get an initial estimate of those parameters from a simple inversion.
        # (This is very much incorrect for non-linear terms).
        finite = np.isfinite(self._coefficients[:, 0]*flux *flux_uncertainties)
        Cinv = 1.0 / (self._scatter[finite]**2 + flux_uncertainties[finite]**2)
        A = np.dot(self._coefficients[finite, :].T,
            Cinv[:, None] * self._coefficients[finite, :])
        B = np.dot(self._coefficients[finite, :].T,
            Cinv * flux[finite])
        initial_vector_p0 = np.linalg.solve(A, B)

        # p0 contains all coefficients, but we only want the linear terms to
        # make an initial estimate.
        indices, names = self._get_linear_indices(self._label_vector_description,
            full_output=True)
        if len(indices) == 0:
            raise NotImplementedError("no linear terms in Cannon model -- TODO")

        # Get the initial guess of just the linear parameters.
        # (Here we make a + 1 adjustment for the first '1' term)
        p0 = initial_vector_p0[indices + 1]

        logger.debug("Initial guess: {0}".format(dict(zip(names, p0))))

        # Now we need to build up label vector rows by indexing relative to the
        # labels that we will actually be solving for (in this case it's the
        # variable 'names'), and not the labels as they are currently referenced
        # in self._labels
        label_vector_indices = self._parse_label_vector_description(
            self._label_vector_description, return_indices=True,
            __columns=names)

        # Create the function.
        def f(coefficients, *labels):
            return np.dot(coefficients, _build_label_vector_rows(
                label_vector_indices, labels).T).flatten()

        # Optimise the curve to solve for the parameters and covariance.
        full_output = kwargs.pop("full_output", False)
        kwds = kwargs.copy()
        kwds.setdefault("maxfev", 10000)
        labels, covariance = op.curve_fit(f, self._coefficients[finite],
            flux[finite], p0=p0, sigma=1.0/np.sqrt(Cinv), absolute_sigma=True,
            **kwds)

        # We might have solved for any number of parameters, so we return a
        # dictionary.
        logger.warn("TODO: apply offsets as required")
        labels = dict(zip(names, labels))

        logger.debug("Final solution: {0}".format(labels))

        if full_output:
            return (labels, covariance)
        return labels


    @property
    def residuals(self):
        """
        Calculate the residuals of the inferred labels using the trained model.

        :returns:
            The corresponding label names, the expected labels for all stars in
            the training set, and the inferred labels.

        :raises TypeError:
            If the model is not trained.
        """

        if not self._trained:
            raise TypeError("model must be trained first")

        try:
            if self._residuals_hash == self._trained_hash \
            and self._trained_hash is not None:
                return self._residuals_cache

        except AttributeError:
            None

        label_indices, label_names = self._get_linear_indices(
            self._label_vector_description, full_output=True)
        expected_labels = np.zeros((self._fluxes.shape[0], len(label_names)))
        inferred_labels = np.zeros((self._fluxes.shape[0], len(label_names)))

        for i, (flux, uncertainty) \
        in enumerate(zip(self._fluxes, self._flux_uncertainties)):
            inferred = self.solve_labels(flux, uncertainty)
            for j, label_name in enumerate(label_names):
                expected_labels[i, j] = self._labels[label_name][i]
                inferred_labels[i, j] = inferred[label_name]

        # Cache for future, unless the training state changes.
        self._residuals_hash = self._trained_hash
        self._residuals_cache = (label_names, expected_labels, inferred_labels)
        return self._residuals_cache


    def cross_validate(self, label_vector_description, N=1, max_combinations=1000,
        **kwargs):
        """
        Perform cross-validation on the trained model.

        :params label_vector_description:
            The human-readable form of the label vector description.

        :type label_vector_description:
            str

        :param N: [optional]
            The number of stars (rows) to use in the testing dataset.

        :type N:
            int

        :param max_combinations: [optional]
            The maximum number of cross-validation combinations to run. If None
            or -1 is given then all possible combinations based on `N` will be
            run. Note that for `N > 1`, this can result in a very large number
            of combinations.

        :type max_combinations:
            int
        """

        R = len(self._labels)
        M = range(R)
        if max_combinations in (None, -1):
            max_combinations = factorial(N) / factorial(R) / factorial(N - R)

        # Shake it all about.
        random.shuffle(M)

        # Initialise arrays.
        label_indices, label_names = self._get_linear_indices(
            label_vector_description, full_output=True)

        inferred_test_labels = np.empty((max_combinations, N, len(label_names)))
        expected_test_labels = np.empty((max_combinations, N, len(label_names)))
        testing_set_indices = np.empty((max_combinations, N))

        # Go through each combination.
        for i, indices in enumerate(combinations(M, N)):
            if i >= max_combinations:
                break

            indices = np.array(indices)
            train = np.ones(len(self._labels), dtype=bool)
            train[indices] = False

            # Create a model to use so we don't overwrite self.
            model = self.__class__(self._labels[train], self._fluxes[train, :],
                self._flux_uncertainties[train, :])
            model.train(label_vector_description, **kwargs)

            # Now solve for the testing set.
            results = []
            for j, (all_labels, fluxes, flux_uncertainties) \
            in enumerate(zip(self._labels[~train], self._fluxes[~train, :],
                self._flux_uncertainties[~train, :])):

                inferred_labels = model.solve_labels(fluxes, flux_uncertainties)

                # Save the results.
                for k, name in enumerate(label_names):
                    expected_test_labels[i, j, k] = all_labels[name]
                    inferred_test_labels[i, j, k] = inferred_labels[name]

            # Save the indices.
            testing_set_indices[i, :] = indices

        # [TODO] Need to shuffle the combinations some how.
        # [TODO] Thread everything.
        # [TODO] Make it fail-safe against falling over on a single permutation.
        # [TODO] Safeguard against dumb input.
        raise a


        # Create a model that we will use so we don't overwrite self.


        # For each combination:
        # - train the model without the test set.
        # - Solve for labels on the test set
        # - Keep the (training set indices, expected results, solved results)

        # Return everything separately:



        if not self._trained:
            raise TypeError("model must be trained before cross-validation")

        raise NotImplementedError("soon...")


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
        """

        if isinstance(label_vector_description, (str, unicode)):
            label_vector_description = label_vector_description.split()

        # This allows us to parse arbitrary columns instead of referencing
        # based on the self._labels. Use with caution! Note: if using this,
        # always turn off verbosity.
        columns = kwargs.pop("__columns", self._labels.colnames)

        # Functions to parse the parameter (or index) and order for each term.
        order = lambda t: int((t.split("^")[1].strip() + " ").split(" ")[0]) \
            if "^" in t else 1

        if return_indices:
            param = lambda d: columns.index((d + "^").split("^")[0].strip())
        else:
            param = lambda d: (d + "^").split("^")[0].strip()

        theta = []
        for description in label_vector_description:

            # Is it just a parameter?
            try:
                index = columns.index(description.strip())

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

        if kwargs.pop("verbose", True) and not return_indices:
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

            if len(sub_string) > 1:
                string.append("({})".format(" * ".join(sub_string)))
            else:
                string.append(sub_string[0])

        return " + ".join(string)


    def save(self, filename, overwrite=False, verify=True):
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

        # Create a hash of the labels, fluxes and flux uncertainties.
        if verify:
            hashes = [hash(str(_)) for _ in \
                (self._labels, self._fluxes, self._flux_uncertainties)]
        else:
            hashes = None

        contents = \
            (self._label_vector_description, self._coefficients, self._scatter,
                self._offsets, hashes)
        with open(filename, "w") as fp:
            pickle.dump(contents, fp, -1)

        return True


    def load(self, filename, verify=True):
        """
        Load a trained model from disk.

        :param filename:
            The file path where to load the model from.

        :type filename:
            str

        :param verify: [optional]
            Verify whether the hashes in the stored filename match what is
            expected from the label, flux and flux uncertainty arrays.

        :type verify:
            bool

        :returns:
            True

        :raises IOError:
            If the model could not be loaded.

        :raises ValueError:
            If the current hash of the labels, fluxes, or flux uncertainties is
            different than what was stored in the filename. Disable this option
            (at your own risk) by setting `verify` to False.
        """

        with open(filename, "r") as fp:
            contents = pickle.load(fp)

        hashes = contents[-1]
        if verify and hashes is not None:
            exp_hash = [hash(str(_)) for _ in \
                (self._labels, self._fluxes, self._flux_uncertainties)]
            descriptions = ("labels", "fluxes", "flux_uncertainties")
            for e_hash, r_hash, descr in zip(exp_hash, hashes, descriptions):
                if e_hash != r_hash:
                    raise ValueError("expected hash for {0} ({1}) is different "
                        "to that stored in {2} ({3})".format(descr, e_hash,
                            filename, r_hash)) 

        self._label_vector_description, self._coefficients, self._scatter, \
            self._offsets, hashes = contents
        self._trained = True

        return True


def _fit_coefficients(fluxes, flux_uncertainties, scatter, lv_array,
    full_output=False):
    """
    Fit model coefficients and scatter to a given set of normalised fluxes for a
    single pixel.

    :param fluxes:
        The normalised fluxes for a single pixel (in many stars).

    :type fluxes:
        :class:`~np.array`

    :param flux_uncertainties:
        The 1-sigma uncertainties in normalised fluxes. This should have the
        same shape as `fluxes`.

    :type flux_uncertainties:
        :class:`~np.array`

    :param lv_array:
        The label vector array for each pixel.

    :type lv_array:
        :class:`~np.ndarray`

    :param full_output: [optional]
        Return the coefficients and the covariance matrix.

    :type full_output:
        bool

    :returns:
        The label vector coefficients for the pixel, the inverse variance matrix
        and the total pixel variance.
    """

    variance = flux_uncertainties**2 + scatter**2
    CiA = lv_array.T * np.tile(1./variance, (lv_array.shape[0], 1)).T
    ATCiAinv = np.linalg.inv(np.dot(lv_array, CiA))

    ATY = np.dot(lv_array, fluxes/variance)
    coefficients = np.dot(ATCiAinv, ATY)

    return (coefficients, ATCiAinv, variance)
    

def _pixel_scatter_nll(scatter, fluxes, flux_uncertainties, lv_array,
    debug=False):
    """
    Return the negative log-likelihood for the scatter in a single pixel.

    :param scatter:
        The model scatter in the pixel.

    :type scatter:
        float

    :param fluxes:
        The fluxes for a given pixel (in many stars).

    :type fluxes:
        :class:`~np.array`

    :param flux_uncertainties:
        The 1-sigma uncertainties in the fluxes for a given pixel. This should
        have the same shape as `fluxes`.

    :type flux_uncertainties:
        :class:`~np.array`

    :param lv_array:
        The label vector array for each star, for the given pixel.

    :type lv_array:
        :class:`~np.ndarray`

    :param debug: [optional]
        Re-raise captured exceptions.

    :type debug:
        bool

    :returns:
        The log-likelihood of the log scatter, given the fluxes and the label
        vector array.

    :raises np.linalg.linalg.LinAlgError:
        If there was an error in inverting a matrix, and `debug` is set to True.
    """
    
    if 0 > scatter:
        return -np.inf

    try:
        # Calculate the coefficients for the given level of scatter.
        coefficients, ATCiAinv, variance = _fit_coefficients(fluxes,
            flux_uncertainties, scatter, lv_array)

    except np.linalg.linalg.LinAlgError:
        if debug: raise
        return -np.inf

    model = np.dot(coefficients, lv_array)

    return 0.5 * np.sum((fluxes - model)**2 / variance) \
        + 0.5 * np.sum(np.log(variance))


def _fit_pixel(fluxes, flux_uncertainties, lv_array, debug=False):
    """
    Return the optimal label vector coefficients and scatter for a pixel, given
    the fluxes, uncertainties, and the label vector array.

    :param fluxes:
        The fluxes for the given pixel, from all stars.

    :type fluxes:
        :class:`~np.array`

    :param flux_uncertainties:
        The 1-sigma flux uncertainties for the given pixel, from all stars.

    :type flux_uncertainties:
        :class:`~np.array`

    :param lv_array:
        The label vector array. This should have shape `(N_stars, N_terms + 1)`.

    :type lv_array:
        :class:`~np.ndarray`

    :param debug: [optional]
        Re-raise exceptions that would otherwise be suppressed (and a large
        scatter provided).

    :type debug:
        bool

    :returns:
        The optimised label vector coefficients and scatter for this pixel.
    """

    # Get an initial guess of the scatter.
    scatter = np.var(fluxes) - np.median(flux_uncertainties)**2
    scatter = np.sqrt(scatter) if scatter >= 0 else np.std(fluxes)
    
    # Optimise the scatter, and at each scatter value we will calculate the
    # optimal vector coefficients.
    op_scatter = op.fmin_powell(_pixel_scatter_nll, scatter,
        args=(fluxes, flux_uncertainties, lv_array), disp=False) 

    # Calculate the coefficients at the optimal scatter value.
    # Note that if we can't solve for the coefficients, we should just set them
    # as zero and send back a giant variance.
    try:
        coefficients, _, __ = _fit_coefficients(fluxes, flux_uncertainties,
            op_scatter, lv_array)

    except np.linalg.linalg.LinAlgError:
        logger.exception("Failed to calculate coefficients")
        if debug: raise

        return (np.zeros(lv_array.shape[0]), 10e8)

    else:
        return (coefficients, op_scatter)


def _build_label_vector_rows(label_vector, labels):
    """
    Build a label vector row from a description of the label vector (as indices
    and orders to the power of) and the label values themselves.

    For example: if the first item of `labels` is `A`, and the label vector
    description is `A^3` then the first item of `label_vector` would be:

    `[[(0, 3)], ...`

    This indicates the first label item (index `0`) to the power `3`.

    :param label_vector:
        An `(index, order)` description of the label vector. 

    :type label_vector:
        list

    :param labels:
        The values of the corresponding labels.

    :type labels:
        list

    :returns:
        The corresponding label vector row.
    """
    
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
    """
    Build the label vector array.

    :param labels:
        The labels for each star as a table. This should have `N_star` rows.

    :type labels:
        :class:`~astropy.table.Table`

    :param label_vector:
        The label vector description in `(index, order)` form.

    :type label_vector`
        list

    :param N: [optional]
        The number of stars to use for the label vector array. If None is given,
        then all stars (rows) will be used.

    :type N:
        int

    :param limits: [optional]
        Place limits on the labels to train with. This should be provided as a
        dictionary where the labels are keys and two-length tuples of
        `(lower, upper)` values are given as values.

    :type limits:
        dict

    :param pivot: [optional]
        Pivot about some mean label values.

    :type pivot:
        bool

    :returns:
        A three-length tuple containing the label vector array, the indices of
        stars (rows) from `labels` that were used (in case `N` and/or `limits`
        were provided) and the coefficient offsets (for when `pivot` is True).
    """

    logger.debug("Building Cannon label vector array")

    indices = np.ones(len(labels), dtype=bool)
    if limits is not None:
        for parameter, (lower_limit, upper_limit) in limits.items():
            indices *= (upper_limit >= labels[parameter]) * \
                (labels[parameter] >= lower_limit)

    indices = np.where(indices)[0]
    if N is not None:
        indices = indices[np.linspace(0, indices.sum() - 1, N, dtype=int)]
    
    labels = labels[indices]
    if pivot:
        raise NotImplementedError
        offsets = labels.mean(axis=0)
        labels -= offsets
    else:
        offsets = np.zeros(len(labels.colnames))

    return (_build_label_vector_rows(label_vector, labels).T, indices, offsets)

