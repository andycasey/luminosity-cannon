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
RELOAD = False
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

stars.add_column(Column(name="JmK_ABS", data=stars["J_ABS"] - stars["K_ABS"]))
stars.add_column(Column(name="HmK_ABS", data=stars["H_ABS"] - stars["K_ABS"]))
stars.add_column(Column(name="JmH_ABS", data=stars["J_ABS"] - stars["H_ABS"]))

ok = (stars["e_Hpmag"] < 0.01) * np.isfinite(stars["J"] * stars["H"] * stars["K"]) \
    * (stars["Plx"] > 0) * (stars["e_Plx"]/stars["Plx"] < 0.1) * (stars["K_ERR"] < 0.10) # 10% error in parallax
print("OK: {}".format(ok.sum()))
stars = stars[ok]
fluxes = fluxes[ok, :]
flux_uncertainties = flux_uncertainties[ok, :]

#stars.write("temp.fits.gz")
#stars = Table.read("temp.fits.gz")

# Specify different label vector descriptions.
"""
    "K_ABS^3 K_ABS^2 K_ABS JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS", # from old.harps-hipparcos.py
    "K_ABS^5 K_ABS^4 K_ABS^3 K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", # JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*J_ABS JmK_ABS*J_ABS^2 
    "TEFF^4 TEFF^3 TEFF^2 TEFF K_ABS^3 K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", # JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*J_ABS JmK_ABS*J_ABS^2 
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 K_ABS^3 K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", # JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*J_ABS JmK_ABS*J_ABS^2 

    "K_ABS^3 K_ABS^2 K_ABS  JmK_ABS^3 JmK_ABS^2 JmK_ABS  JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS  JmK_ABS^2*K_ABS^2 JmK_ABS^3*K_ABS^3",
#    "TEFF^4 TEFF^3 TEFF^2 TEFF",
#    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 LOGG^3 LOGG^4 LOGG^5",
#    "J_ABS J_ABS^2 J_ABS^3",
 #   "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 LOGG^3 LOGG^4 LOGG^5 J_ABS J_ABS^2 J_ABS^3",
  #  "J_ABS H_ABS K_ABS J_ABS*H_ABS J_ABS*K_ABS H_ABS*K_ABS",
 #   "J_ABS H_ABS K_ABS J_ABS*H_ABS J_ABS*K_ABS H_ABS*K_ABS J_ABS^2 H_ABS^2 K_ABS^2",
#    "J_ABS H_ABS K_ABS J_ABS*H_ABS J_ABS*K_ABS H_ABS*K_ABS J_ABS^2 H_ABS^2 K_ABS^2 J_ABS^3 H_ABS^3 K_ABS^3",
#    "JmK_ABS HmK_ABS JmH_ABS J_ABS H_ABS K_ABS J_ABS*H_ABS J_ABS*K_ABS H_ABS*K_ABS J_ABS^2 H_ABS^2 K_ABS^2 J_ABS^3 H_ABS^3 K_ABS^3",
#    "JmK_ABS J_ABS H_ABS K_ABS J_ABS*H_ABS J_ABS*K_ABS H_ABS*K_ABS J_ABS^2 H_ABS^2 K_ABS^2 J_ABS^3 H_ABS^3 K_ABS^3",    
    # JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*J_ABS JmK_ABS*J_ABS^2 

#    "ITEFF^4 ITEFF^3 ITEFF^2 ITEFF ILOGG ILOGG^2  ITEFF*ILOGG ITEFF^2*ILOGG ITEFF*ILOGG^2 IPARAM_M_H IPARAM_M_H*ITEFF IPARAM_M_H*ITEFF^2 IPARAM_ALPHA_M IPARAM_M_H*IPARAM_ALPHA_M K_ABS^3 K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", 

#    "TEFF*N_H TEFF*O_H TEFF*C_H TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^3 K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", 

#    "C_H N_H O_H TEFF*N_H TEFF*O_H TEFF*C_H TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^3 K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", 
 #   "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 LOGG^3 LOGG^4 LOGG^5 K_ABS^3 K_ABS^2 K_ABS JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS",
    
    """

# TEST WITH RANK INDICES 
stars["ITEFF"] = np.array(np.argsort(stars["TEFF"]), dtype=float)
stars["ILOGG"] = np.array(np.argsort(stars["LOGG"]), dtype=float)
stars["IPARAM_M_H"] = np.array(np.argsort(stars["PARAM_M_H"]), dtype=float)
stars["IPARAM_ALPHA_M"] = np.array(np.argsort(stars["PARAM_ALPHA_M"]), dtype=float)

label_vector_descriptions = [
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^3 K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", 
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^5 JmK_ABS^4 JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", 
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^3 K_ABS^2 K_ABS JmK_ABS^3 JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", 
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^2 JmK_ABS JmK_ABS^2*K_ABS JmK_ABS*K_ABS^2 JmK_ABS*K_ABS ", 
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^2 JmK_ABS JmK_ABS*K_ABS ", 
       
    "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG PARAM_M_H PARAM_M_H*TEFF PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^2 JmK_ABS JmK_ABS*K_ABS", 
    "TEFF^3 TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG PARAM_M_H PARAM_M_H*TEFF PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^2 JmK_ABS JmK_ABS*K_ABS", 
    "TEFF^2 TEFF LOGG LOGG^2  TEFF*LOGG PARAM_M_H PARAM_M_H*TEFF PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^2 JmK_ABS JmK_ABS*K_ABS", 
    "TEFF LOGG LOGG^2  TEFF*LOGG PARAM_M_H PARAM_M_H*TEFF PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^2 JmK_ABS JmK_ABS*K_ABS", 
    "TEFF LOGG TEFF*LOGG PARAM_M_H PARAM_M_H*TEFF PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS^2 K_ABS JmK_ABS^2 JmK_ABS JmK_ABS*K_ABS", 

    "TEFF LOGG TEFF*LOGG PARAM_M_H PARAM_M_H*TEFF PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS K_ABS JmK_ABS^2 JmK_ABS JmK_ABS*K_ABS", 
    #"TEFF LOGG TEFF*LOGG PARAM_M_H PARAM_M_H*TEFF PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M K_ABS K_ABS JmK_ABS JmK_ABS JmK_ABS*K_ABS", 

]

for i, label_vector_description in enumerate(label_vector_descriptions):

    # Has this already been tried?
    label_vector_hash = str(hash(label_vector_description))[:10]
    output_prefix = "/".join([OUTPUT_DIR, label_vector_hash])

    model = CannonModel(stars, fluxes, flux_uncertainties)

    if os.path.exists("{}.pkl".format(output_prefix)):
        if RELOAD:
            model.load("{}.pkl".format(output_prefix))

        else:
            print("Skipping {0} because it has been done already ({1}.pkl)".format(
                label_vector_description, output_prefix))
            continue

    # Train the model.
    else:
        model.train(label_vector_description)

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

    a = random_word()
    b = random_word()
    #fig.tight_layout()
    fig.savefig("{0}/{1}-{2}.{3}.png".format(OUTPUT_DIR, a, b,
        label_vector_hash))

    # Do distance errors.
    distance_labels = []
    for label in labels:
        if label.rstrip("_ABS") in magnitudes:
            distance_labels.append(label[:-4])

    if len(distance_labels) > 0:
        fig, axes = plt.subplots(2, int(np.ceil(len(distance_labels)/2.)))
        if len(distance_labels) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for ax, label in zip(axes, distance_labels):

            # Calculate expected distance.
            expected_distance = 1000./stars["Plx"] # Parsecs
            mu = stars[label] - inferred[:, list(labels).index("{}_ABS".format(label))]
            inferred_distance = 10**((5 + mu)/5.) # Parsecs

            e = 1000.0/(stars["Plx"] + stars["e_Plx"]) - 1000./stars["Plx"]
            #e = np.abs(1000./stars["Plx"] - 1000./(stars["Plx"] + stars["e_Plx"]))
            ax.errorbar(expected_distance, inferred_distance,
                xerr=e, fmt=None, elinecolor='k', zorder=-100)
            ax.scatter(expected_distance, inferred_distance, facecolor="k")
            ax.set_xlabel("Hipparcos distance [pc]")
            ax.set_ylabel("Inferred distance [pc]")

            limit = max([ax.get_xlim()[1], ax.get_ylim()[1]])
            ax.plot([0, limit], [0, limit], c="#666666", zorder=-100)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)

            inferred_distance[0 >= inferred_distance] = np.nan
            ax.set_title("mean / median / |sigma| = {0:.0f} / {1:.0f} / {2:.0f}".format(
                np.nanmean(expected_distance - inferred_distance),
                np.nanmedian(expected_distance - inferred_distance),
                np.nanstd(np.abs(expected_distance - inferred_distance))))

            #mu = app - abs = 5 * np.log10(1000./(stars["Plx"])) - 5

        #fig.tight_layout()
        fig.savefig("{0}/{1}-{2}-distance.{3}.png".format(OUTPUT_DIR, a, b,
            label_vector_hash))


    if len(distance_labels) > 0:
        fig, axes = plt.subplots(2, int(np.ceil(len(distance_labels)/2.)))
        if len(distance_labels) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for ax, label in zip(axes, distance_labels):

            # Calculate expected distance.
            expected_plx = stars["Plx"] # Parsecs
            mu = stars[label] - inferred[:, list(labels).index("{}_ABS".format(label))]
            inferred_plx = 1000.0/(10**((5 + mu)/5.)) # Parsecs

            #e = np.abs(1000./stars["Plx"] - 1000./(stars["Plx"] + stars["e_Plx"]))
            e = stars["e_Plx"]
            ax.errorbar(expected_plx, inferred_plx,
                xerr=e, fmt=None, elinecolor='k', zorder=-100)
            ax.scatter(expected_plx, inferred_plx, facecolor="k")
            ax.set_xlabel("Hipparcos parallax")
            ax.set_ylabel("Inferred parallax")

            limit = max([ax.get_xlim()[1], ax.get_ylim()[1]])
            ax.plot([0, limit], [0, limit], c="#666666", zorder=-100)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_title(label)

            inferred_plx[0 >= inferred_plx] = np.nan
            ax.set_title("mean / median / |sigma| / mean_data_sigma = {0:.2f} / {1:.2f} / {2:.2f} / {3:.2f}".format(
                np.nanmean(expected_plx - inferred_plx),
                np.nanmedian(expected_plx - inferred_plx),
                np.nanstd(np.abs(expected_plx - inferred_plx)),
                np.nanmean(stars["e_Plx"])), fontsize=6)

            #mu = app - abs = 5 * np.log10(1000./(stars["Plx"])) - 5

        #fig.tight_layout()
        fig.savefig("{0}/{1}-{2}-parallax.{3}.png".format(OUTPUT_DIR, a, b,
            label_vector_hash))

    fig, ax = plt.subplots()
    ax.plot(model._scatter, c='k')
    #fig.tight_layout()
    fig.savefig("{0}/{1}-{2}-scatter.{3}.png".format(OUTPUT_DIR, a, b,
        label_vector_hash))

    #plt.close("all")
    
    # Save the model.
    model.save("{0}.pkl".format(output_prefix))



"""
WTF is that band??


start, end = 2500, 3000

# Predict spectra for all stars
# Calculate total residuals between that range

indices, names = model._get_linear_indices(model._label_vector_description, True)

# See what the total residual correlates with.
chi_sqs = np.zeros(model._fluxes.shape[0])
for i in range(model._fluxes.shape[0]):

    data = model._fluxes[i, :]
    m = model.predict([model._labels[name][i] for name in names])

    chi = (data - m)**2/(model._flux_uncertainties[i, :]**2 + model._scatter**2)
    chi_sqs[i] = np.nansum(chi[start:end])
    print(i)

"""



    