#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Model search. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Column, Table
from code.cannon import CannonModel


# Utilities.
def random_word():
    with open("/usr/share/dict/words", "r") as fp:
        line = next(fp)
        for num, aline in enumerate(fp):
            if random.randrange(num + 2): continue
            line = aline
    return line.rstrip()


# Setup 
OUTPUT_DIR = "model_search"
FILENAME_PREFIX = "test-Cannon"

stars = Table.read("{}.fits.gz".format(FILENAME_PREFIX))
fluxes = np.memmap("{}-flux.memmap".format(FILENAME_PREFIX), mode="r",
    dtype=float).reshape((len(stars), -1))
flux_uncertainties = np.memmap("{}-flux-uncertainties.memmap".format(
    FILENAME_PREFIX), mode="r", dtype=float).reshape(fluxes.shape)

# Calculate distance modulus
mu = 5 * np.log10(1000./(stars["Plx"])) - 5

# Apparent magnitudes:
# JHK all present.
magnitudes = ("J", "H", "K")
for magnitude in magnitudes:
    stars.add_column(Column(name="{}_ABS".format(magnitude),
        data=stars[magnitude] - mu))

#stars.write("temp.fits.gz")
#stars = Table.read("temp.fits.gz")

# Specify different label vector descriptions.
label_vector_descriptions = [
    "TEFF^4 TEFF^3 TEFF^2 TEFF",
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 LOGG^3 LOGG^4 LOGG^5",
#    "J_ABS J_ABS^2 J_ABS^3",
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 LOGG^3 LOGG^4 LOGG^5 J_ABS J_ABS^2 J_ABS^3",
    "J_ABS H_ABS K_ABS J_ABS*H_ABS J_ABS*K_ABS H_ABS*K_ABS",
    "J_ABS H_ABS K_ABS J_ABS*H_ABS J_ABS*K_ABS H_ABS*K_ABS J_ABS^2 H_ABS^2 K_ABS^2",
    "J_ABS H_ABS K_ABS J_ABS*H_ABS J_ABS*K_ABS H_ABS*K_ABS J_ABS^2 H_ABS^2 K_ABS^2 J_ABS^3 H_ABS^3 K_ABS^3",
]

for i, label_vector_description in enumerate(label_vector_descriptions):

    # Has this already been tried?
    label_vector_hash = str(hash(label_vector_description))[:10]
    output_prefix = "/".join([OUTPUT_DIR, label_vector_hash])

    if os.path.exists("{}.pkl".format(output_prefix)):
        print("Skipping {0} because it has been done already ({1}.pkl)".format(
            label_vector_description, output_prefix))
        continue

    # Train the model.
    model = CannonModel(stars, fluxes, flux_uncertainties)
    model.train(label_vector_description)

    # Calculate residuals.
    labels, expected, inferred = model.label_residuals

    # Plot residuals.
    fig, axes = plt.subplots(1, len(labels))
    if len(labels) == 1: axes = [axes]
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
        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)

    a = random_word()
    b = random_word()
    fig.savefig("{0}/{1}-{2}.{3}.png".format(OUTPUT_DIR, a, b,
        label_vector_hash))

    fig, ax = plt.subplots()
    ax.plot(model._scatter, c='k')
    fig.savefig("{0}/{1}-{2}-scatter.{3}.png".format(OUTPUT_DIR, a, b,
        label_vector_hash))

    #plt.close("all")
    
    # Save the model.
    model.save("{0}.pkl".format(output_prefix))

    