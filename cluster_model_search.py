#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Test models for the APOGEE Cluster sample. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.table import Column, Table
from code.cannon import CannonModel


SHOW_AS_PERCENT = True
    
DATA_PREFIX = "data/APOGEE-Clusters"
LABEL_VECTOR_DESCRIPTION = "TEFF LOGG LOGG^2 TEFF*LOGG PARAM_M_H PARAM_M_H*TEFF PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^2 JmK_ABS JmK_ABS*K_ABS"
LABEL_VECTOR_DESCRIPTION = "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^3 K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS" 

if __name__ == "__main__":


    stars = Table.read("{}.fits.gz".format(DATA_PREFIX))
    fluxes = np.memmap("{}-flux.memmap".format(DATA_PREFIX), mode="r", dtype=float)
    flux_uncertainties = np.memmap("{}-flux-uncertainties.memmap".format(DATA_PREFIX),
        mode="r", dtype=float)

    # Re-shape
    fluxes = fluxes.reshape((len(stars), -1))
    flux_uncertainties = flux_uncertainties.reshape(fluxes.shape)

    # Calculate absolute magnitudes.
    magnitudes = ("J", "H", "K")
    for magnitude in magnitudes:
        stars.add_column(Column(name="{}_ABS".format(magnitude),
            data=stars[magnitude] - stars["mu"]))

    # Add colours.
    stars.add_column(Column(name="JmK_ABS", data=stars["J_ABS"] - stars["K_ABS"]))
    stars.add_column(Column(name="HmK_ABS", data=stars["H_ABS"] - stars["K_ABS"]))
    stars.add_column(Column(name="JmH_ABS", data=stars["J_ABS"] - stars["H_ABS"]))

    """
    #ok = np.array((stars["e_Hpmag"] < 0.01) * np.isfinite(stars["J"] * stars["H"] * stars["K"]) \
    #    * (stars["Plx"] > 0) * (stars["e_Plx"]/stars["Plx"] < 0.1) * (stars["K_ERR"] < 0.10))

    stars = stars[ok]
    fluxes = fluxes[ok]
    flux_uncertainties = flux_uncertainties[ok]
    """
    model = CannonModel(stars, fluxes, flux_uncertainties)
    #print("LOADING")
    #model.load("tmp")
    model.train(LABEL_VECTOR_DESCRIPTION)
    
    # Calculate residuals.
    labels, expected, inferred = model.label_residuals

    # Plot residuals.
    fig, axes = plt.subplots(2, int(np.ceil(len(labels)/2.)))
    if len(labels) == 1: axes = np.array([axes])
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.scatter(expected[:, i], inferred[:, i], facecolor="k")
        ax.set_xlabel(label)
        ax.set_ylabel("Inferred {}".format(label))
        ax.set_title("mean / median / |sigma| = {0:.2f} / {1:.2f} / {2:.2f}".format(
            np.nanmean(expected[:, i] - inferred[:, i]),
            np.nanmedian(expected[:, i] - inferred[:, i]),
            np.nanstd(np.abs(expected[:, i] - inferred[:, i]))),
        fontsize=6)

        min_limit = min([ax.get_xlim()[0], ax.get_ylim()[0]])
        max_limit = max([ax.get_xlim()[1], ax.get_ylim()[1]])
        ax.plot([min_limit, max_limit], [min_limit, max_limit], c="#666666",
            zorder=-100)
        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)


    # Do distance errors.
    distance_labels = []
    for label in labels:
        if label.rstrip("_ABS") in magnitudes:
            distance_labels.append(label[:-4])

    if len(distance_labels) > 0:
        for label in distance_labels:

            gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
            ax = map(plt.subplot, gs)

            expected_distance = 10**((5 + stars["mu"])/5.) / 1000. # [kpc]
            mu = stars[label] - inferred[:, list(labels).index("{}_ABS".format(label))]
            inferred_distance = 10**((5 + mu)/5.) / 1000. # [kpc]

            difference_absolute = inferred_distance - expected_distance
            difference_percent = 100 * (inferred_distance - expected_distance)/expected_distance
            ax[0].axhline(0, c="#666666", zorder=-1)

            if SHOW_AS_PERCENT:
                difference, units = difference_percent, r"\%"
            else:
                difference, units = difference_absolute, r"kpc"

            ax[0].scatter(expected_distance, difference, facecolor="k")
            ax[0].set_ylabel(r"$\Delta{}D$ $[" + units + "]$")
            
            limit = max(np.abs(ax[0].get_ylim()))

            sigma = np.std(np.abs(difference))
            ax[0].axhspan(-sigma, +sigma, 0, limit, facecolor="#CCCCCC",
                zorder=-1000)

            ax[0].set_ylim(-limit, +limit)

            ax[1].scatter(expected_distance, inferred_distance, facecolor="k")
            ax[1].set_xlabel("Expected distance [kpc]")
            ax[1].set_ylabel("Inferred distance [kpc]")

            limit = max([ax[1].get_xlim()[1], ax[1].get_ylim()[1]])
            ax[1].plot([0, limit], [0, limit], c="#666666", zorder=-100)
            
            ax[0].set_xticklabels([])
            ax[0].set_xlim(0, limit)

            ax[1].set_xlim(0, limit)

            ax[1].set_ylim(0, limit)
            #ax[0].set_yticks([-limit/2, 0, +limit/2])

            ax[1].xaxis.set_major_locator(MaxNLocator(5))
            ax[1].yaxis.set_major_locator(MaxNLocator(5))
            
            ax[0].set_title("mean / median / |sigma| / |sigma| \% = {0:.2f} / {1:.2f} / {2:.2f} / {3:.2f}".format(
                np.nanmean(expected_distance - inferred_distance),
                np.nanmedian(expected_distance - inferred_distance),
                np.nanstd(np.abs(difference_absolute)),
                np.nanstd(np.abs(difference_percent))),
                fontsize=6)

            #fig.tight_layout()
            #ax[0].figure.tight_layout()
            ax[0].figure.subplots_adjust(hspace=0)

            #mu = app - abs = 5 * np.log10(1000./(stars["Plx"])) - 5



    raise a
