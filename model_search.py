#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Model search. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
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

# Specify different label vector descriptions.
label_vector_descriptions = [
    "TEFF",
    "TEFF^2 TEFF",
    "TEFF^3 TEFF^2 TEFF",
    "TEFF^4 TEFF^3 TEFF^2 TEFF",
    "TEFF^5 TEFF^4 TEFF^3 TEFF^2 TEFF"
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
    labels, expected, inferred = model.residuals

    # Plot residuals.
    fig, axes = plt.subplots(1, len(labels))
    if len(labels) == 1: axes = [axes]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.scatter(expected[:, i], inferred[:, i], facecolor="k")
        ax.set_xlabel(label)
        ax.set_ylabel("Inferred {}".format(label))
        ax.set_title("mean / median / |sigma| = {0:.2f} / {1:.2f} / {2:.2f}".format(
            np.mean(expected[:, i] - inferred[:, i]),
            np.median(expected[:, i] - inferred[:, i]),
            np.std(np.abs(expected[:, i] - inferred[:, i]))))

        min_limit = min([ax.get_xlim()[0], ax.get_ylim()[0]])
        max_limit = max([ax.get_xlim()[1], ax.get_ylim()[1]])
        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)

    fig.savefig("{0}/{1}-{2}.{3}.png".format(OUTPUT_DIR,
        random_word(), random_word(), label_vector_hash))
    #plt.close("all")
    
    # Save the model.
    model.save("{0}.pkl".format(output_prefix))

    