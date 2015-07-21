#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon for stellar distances """

import numpy as np

from astropy.table import Table


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




    def train(self, label_vector_description):
        raise NotImplementedError



    def solve_labels(self, fluxes, flux_uncertainties):
        raise NotImplementedError




    def _interpret_label_vector_description(self, label_vector_description):

        if isinstance(label_vector_description, (str, unicode)):
            label_vector_description = label_vector_description.split()

        order = lambda t: int((t.split("^")[1].strip() + " ").split(" ")[0]) \
            if "^" in t else 1
        parameter_index = lambda d: \
            self._labels.colnames.index((d + "^").split("^")[0].strip())

        theta = []
        for description in human_readable_label_vector:

            # Is it just a parameter?
            try:
                index = self._labels.colnames.index(description.strip())

            except ValueError:
                if "*" in description:
                    # Split by * to evaluate cross-terms.
                    cross_terms = []
                    for cross_term in description.split("*"):
                        try:
                            index = parameter_index(cross_term)
                        except ValueError:
                            raise ValueError("couldn't interpret '{0}' in the "\
                                "label '{1}' as a parameter coefficient".format(
                                    *map(str.strip, (cross_term, description))))
                        cross_terms.append((index, order(cross_term)))
                    theta.append(cross_terms)

                elif "^" in description:
                    theta.append([(
                        parameter_index(description),
                        order(description)
                    )])

                else:
                    raise ValueError("could not interpret '{0}' as a parameter"\
                        " coefficient description".format(description))
            else:
                theta.append([(index, order(description))])

        logger.info("Training the Cannon model using the following description "
            "of the label vector: {0}".format(self._repr_label_vector(theta)))

        return theta


    def _repr_label_vector_description(self, label_vector_indices):

        string = ["1"]
        for cross_terms in label_vector_indices:
            sub_string = []
            for index, order in cross_terms:
                _ = self.grid_points.dtype.names[index]
                if order > 1:
                    sub_string.append("{0}^{1}".format(_, order))
                else:
                    sub_string.append(_)
            string.append(" * ".join(sub_string))
        return " + ".join(string)




    def save(self):
        # Save the label vector description, coefficients, etc
        raise NotImplementedError

    def load(self):
        # Load the label vector description, coefficients, etc
        raise NotImplementedError







if __name__ == "__main__":

    with open("hipparcos-spectra.pkl", "rb") as fp:
        stars = pickle.load(fp)

    data = np.memmap("hipparcos-spectra.memmap", mode="r", dtype=float)
    data = data.reshape(len(stars), -1, 2)

    fluxes, flux_uncertainties = data[1:, :, ::2], data[1:, :, 1::2]

    labels = Table.read("master_table_hip_harps.dat", format="ascii")
    # Ensure the labels are sorted the same as the stars
    sort_indices = np.array([labels["Star"].index(star) for star in stars])
    labels = labels[sort_indices]

    a = CannonModel(labels, fluxes, flux_uncertainties)

    raise a