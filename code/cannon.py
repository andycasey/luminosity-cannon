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
        coefficients = np.nan * np.ones((N_pixels, lva.shape[1]))
        #coefficients = np.memmap("tmp", dtype=float, mode="w+", shape=(N_pixels, lva.shape[1]))

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
        label_vector_indices = self._interpret_label_vector_description(
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


        label_vector_indices2 = self._interpret_label_vector_description(
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




    def _interpret_label_vector_description(self, label_vector_description,
        verbose=True, return_indices=False):

        if isinstance(label_vector_description, (str, unicode)):
            label_vector_description = label_vector_description.split()

        order = lambda t: int((t.split("^")[1].strip() + " ").split(" ")[0]) \
            if "^" in t else 1

        if return_indices:
            repr_param = lambda d: \
                self._labels.colnames.index((d + "^").split("^")[0].strip())
        else:
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
                        cross_terms.append((repr_param(cross_term), order(cross_term)))
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
                theta.append([(repr_param(description), order(description))])

        if verbose:
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






    def save(self, filename):

        assert self._trained
        # Save the label vector description, coefficients, etc
        with open(filename, "w") as fp:
            pickle.dump((self._label_vector_description, self._coefficients,
                self._scatter, self._offsets), fp, -1)

        return True


    def load(self, filename):
        # Load the label vector description, coefficients, etc
        with open(filename, "r") as fp:
            self._label_vector_description, self._coefficients, self._scatter, \
                self._offsets = pickle.load(fp)

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








if __name__ == "__main__":

    with open("hipparcos-spectra.pkl", "rb") as fp:
        stars = pickle.load(fp)

    data = np.memmap("hipparcos-spectra.memmap", mode="r", dtype=float)
    data = data.reshape(len(stars) + 1, -1, 2)


    N_px, sample_rate = None, 10

    wavelengths = data[0, ::sample_rate, 0]
    fluxes = data[1:, ::sample_rate, ::2].reshape(len(stars), -1)
    flux_uncertainties = data[1:, ::sample_rate, 1::2].reshape(len(stars), -1)

    """
    import matplotlib.pyplot as plt
    for i, star in enumerate(stars):
        print("plotting {}".format(star))

        fig, ax = plt.subplots(1)
        ax.plot(wavelengths, fluxes[i], c='k')
        ax.fill_between(wavelengths,
            fluxes[i] - flux_uncertainties[i],
            fluxes[i] + flux_uncertainties[i],
            facecolor="#666666", edgecolor="#666666", zorder=-1)

	ax.set_ylim(0, 1.2)
        ax.set_title(star)
        fig.tight_layout()

        #fig.savefig("figures/{}.png".format(star))
        plt.close("all")
    """

    #fluxes = fluxes[:, ok].reshape(len(stars), -1)[:, :N_px]
    #flux_uncertainties = flux_uncertainties[:, ok].reshape(len(stars), -1)[:, :N_px]

    labels = Table.read("master_table_hip_harps.dat", format="ascii")
    labels = Table.read("table.dat", format="ascii")
    # Ensure the labels are sorted the same as the stars
    sort_indices = np.array([np.where(labels["Star"] == star)[0] for star in stars])
    labels = labels[sort_indices]

    quality = ((labels["qual2mass"] == "AAA") * (labels["n(in1arcmin)"] == 1)).flatten()
    #quality = ((labels["sourceVmag"] == "G") * (labels["n(in1arcmin)"] == 1)).flatten()
    #quality = np.ones(len(labels), dtype=bool)


    model = CannonModel(labels[quality], fluxes[quality, :],
        flux_uncertainties[quality, :])


    #model.train("""K2mass_absolute^3 K2mass_absolute^2 K2mass_absolute 
    #    JmK^3 JmK^2 JmK JmK^2*K2mass_absolute JmK*K2mass_absolute^2 
    #    JmK*K2mass_absolute""".split())
    #model.save("jk-qual")

    model.load("jk-qual")

    #model.load("tmp")

    #model.train("""V_absolute^3 V_absolute^2 V_absolute 
    #    b-v^3 b-v^2 b-v b-v^2*V_absolute b-v*V_absolute^2 
    #    b-v*V_absolute""".split())
    #model.save("tmp-no-qual")



    inferred_labels = []
    for i, (label, flux, flux_uncertainty) \
    in enumerate(zip(labels[quality], fluxes[quality, :], flux_uncertainties[quality, :])):


        inferred = model.solve_labels(flux, flux_uncertainty)

        row_data = {
            "expected_JmK": label["JmK"],
            "expected_distance": 1./label["plx"],
            "expected_distance_err": 1./label["e_plx"],
            "expected_K2mass": label["K2mass"],
            "expected_K2mass_absolute":  label["K2mass_absolute"],
            "expected_bmv": label["b-v"],
            "expected_V_absolute": label["V_absolute"],
        }
        for k, v in inferred.items():
            row_data["inferred_{}".format(k)] = v

        # mu = apparent - absolute = 5 * log_10(d) - 5
        # mu/5 = log_10(d) - 1
        # d = 10**(mu/5 + 1)
        mu = label["K2mass"] - inferred["K2mass_absolute"]
        row_data["inferred_distance"] = 10**(mu/5. + 1)
        #mu = label["vmag"] - inferred["V_absolute"]
        #row_data["inferred_distance_V"] = 10**(mu/5. + 1)

        inferred_labels.append(row_data)

        print("Done {}".format(i))

    foo = Table(rows=inferred_labels)


    #dist = "inferred_distance_V"
    dist = "inferred_distance"


    fig, ax = plt.subplots()

    #ax.errorbar(1000*foo["expected_distance"].flatten(), 1000*foo["inferred_distance"].flatten(),
    #    xerr=foo["expected_distance_err"].flatten() * 1000, fmt=None, ecolor="k")

    ax.scatter(1000*foo["expected_distance"].flatten(), 1000*foo[dist].flatten(),
        facecolor="k")

    #ax.set_xlim(0, 1500)
    #ax.set_ylim(0, 1500)
    lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot([0, lim], [0, lim], ":", c="#666666", zorder=-1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_xlabel("Hipparcos")
    ax.set_ylabel("Inferred")

    #fig.savefig("distances-qual.png")


    a = "expected_JmK"
    b_exp = "expected_K2mass"
    b_exp_abs = "expected_K2mass_absolute"


    #a = "expected_bmv"
    #b_exp = "expected_vmag"
    #b_exp_abs = "expected_V_absolute"


    difference = 1000 * (foo[dist] - foo["expected_distance"])
    fig, ax = plt.subplots(1, 3)

    ax[0].scatter(foo[a], difference, facecolor="k")
    ax[0].set_xlabel(a)
    ax[0].set_ylabel("Distance difference")

    ax[1].scatter(foo[b_exp], difference, facecolor="k")
    ax[1].set_xlabel(b_exp)

    ax[2].scatter(foo[b_exp_abs], difference, facecolor="k")
    ax[2].set_xlabel(b_exp_abs)

    #fig.savefig("differences-qual.png")


    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(foo["inferred_JmK"].flatten() - foo[a].flatten(), difference,
        facecolor="k")

    ax[0].set_xlabel("difference in JmK")
    ax[0].set_ylabel("difference in distance")

    ax[1].scatter(foo["inferred_JmK"].flatten() - foo[a].flatten(),
        difference/foo["expected_distance"],
        facecolor="k")

    ax[1].set_xlabel("difference in JmK")
    ax[1].set_ylabel("difference in distance [%]")

    #fig.savefig("differences-jmk-qual.png")

    raise a
