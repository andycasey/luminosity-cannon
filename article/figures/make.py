#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Make figures for the article. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import matplotlib
from matplotlib.ticker import MaxNLocator
from astropy.io import fits

# Plotting parameters.
matplotlib.rcParams["text.usetex"] = True

# Common keywords for plt.scatter
SCATTER_KWDS = {
    "s": 30,
    "facecolor": "k"
}

def figure_1(data_table="../../data/APOGEE-Hipparcos.fits.gz"):
    """
    HR diagram (TEFF, logg) for all of the Hipparcos stars.

    :param data_table: [optional]
        The path of the data table containing information for the Hipparcos
        sample.

    :type data_table:
        str
    """

    stars = fits.open(data_table)[1].data

    fig, ax = plt.subplots()
    kwds = {}
    kwds.update(SCATTER_KWDS)
    ax.scatter(stars["TEFF"], stars["LOGG"], **kwds)

    # Axes and labels.
    ax.set_xlabel(r"$T_{\rm eff}$ $[{\rm K}]$")
    ax.set_ylabel(r"$\log{g}$")

    ax.set_xlim(6000, 3500)
    ax.set_ylim(4.5, 0)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()
    return fig


if __name__ == "__main__":

    fig_fns = (figure_1, )

    for i, fig_fn in enumerate(fig_fns):
        fig = fig_fn()
        fig.savefig("figure-{0}.png".format(i), dpi=300)
        fig.savefig("figure-{0}.pdf".format(i), dpi=300)
        print("Created figure-{0}.pdf and figure-{0}.png".format(i))


