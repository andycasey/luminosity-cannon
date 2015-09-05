#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create and run a model with some data. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from astropy.io import fits
from astropy.table import Table
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from code import cannon, plot


OUTPUT_DIR = ""
CONFIG_FILENAME = "simple_photometry2.yaml"
APOGEE_FILENAME = "APOGEE_xm_Hipparcos.fits.gz"

THREADS = 1
LOCO_CV = True
LOO_CV = True


try:
    APOGEE
except NameError:
    APOGEE = fits.open(APOGEE_FILENAME)[1].data


CLUSTER_DISTANCES = {
    # Distance in parsecs, core radius in arcmin, half-light radius in arcmin, distance reference.
    # R_core from Davenport et al. 2010
    "M67": (   908, 8.40, "Kharchenko et al. (2005)"),
    "M2":  (11.5e3, 0.32, "Harris"),
    "M3":  (10.2e3, 0.37, "Harris"),
    "M5":  ( 7.5e3, 0.44, "Harris"),
    "M13": ( 7.1e3, 0.62, "Harris"),
    "M15": (10.4e3, 0.14, "Harris"), # c
    "M53": (17.9e3, 0.35, "Harris"),
    "M71": ( 4.0e3, 0.63, "Harris"),
    "M92": ( 8.3e3, 0.26, "Harris"),
    "M107": (6.4e3, 0.56, "Harris"),
    "N188": (2301.442, np.nan, "2010MNRAS.403.1491P"),
    "N2158": (8472.274, np.nan, "2010MNRAS.403.1491P"),
    # NGC 2420 distance modulus in V 11.88 ± 0.27
    # mu = 5 * log10(D) - 5
    "N2420": (2376.8, np.nan, "http://www.aanda.org/articles/aa/pdf/2010/03/aa12965-09.pdf"),
    "N4147": (19.3e3, np.nan, "Harris"),
    "N5466": (16.0e3, np.nan, "Harris"),
    "N6791": (4000.0, np.nan, "http://www.aanda.org/articles/aa/pdf/2008/46/aa10749-08.pdf"),
    "N6819": (2910.0, np.nan, "https://iopscience.iop.org/article/10.1088/0004-637X/786/1/10/pdf"), # Mv = 12.32 ± 0.03
    "N7789": (2337.0, np.nan, "2005A&A...438.1163K")
}

R = {
    "J": 0.717,
    "H": 0.464,
    "K": 0.306,
}



def data_paths(config):
    paths = []
    for wildmask in config.get("data", []):
        paths.extend(glob.glob(wildmask))
    return paths


def load_spectra(filenames):

    # Return common wavelengths, QC-fixed fluxes and flux uncertainties

    N_stars = len(filenames)
    with fits.open(filenames[0]) as image:
        N_pixels = image[1].data.size
        wavelengths = 10**(image[1].header["CRVAL1"] + \
            np.arange(image[1].data.size) * image[1].header["CDELT1"])
 
    fluxes = np.zeros((N_stars, N_pixels))
    flux_uncertainties = np.zeros((N_stars, N_pixels))

    for i, filename in enumerate(filenames):
        print("{0}/{1}: {2}".format(i + 1, N_stars, filename))
        with fits.open(filename) as image:
            fluxes[i, :] = np.atleast_2d(image[1].data)[0, :].copy()
            flux_uncertainties[i, :] = np.atleast_2d(image[2].data)[0, :].copy()

    # QC for pixels.
    unphysical = ((0 >= flux_uncertainties) + ~np.isfinite(flux_uncertainties))\
        + ((0 >= fluxes) + ~np.isfinite(fluxes))
    fluxes[unphysical] = 1.0
    flux_uncertainties[unphysical] = 1e8

    return (wavelengths, fluxes, flux_uncertainties)


def match_spectra(filenames, photometric_filters=("J", "H", "K")):
    """
    Match the data paths to the APOGEE stars.
    """

    # Distance modulii
    N = len(filenames)
    mu = np.nan * np.ones(N, dtype=float)

    indices = np.nan * np.ones(N, dtype=int)
    descr = []
    for i, filename in enumerate(filenames):
        basename = os.path.basename(filename)
        is_hipparcos = basename.startswith("HIP")

        if is_hipparcos:
            idx = np.where(APOGEE["HIP"] == int(basename[3:].split(".")[0]))[0]
            indices[i] = idx[0]
            if len(idx) != 1:
                print("Multiple ({0}) matches for {1}".format(len(idx),
                    basename))

            assert APOGEE["HIP"][idx[0]] > 0

            mu[i] = 5 * np.log10(1000./APOGEE["Plx"][idx[0]]) - 5
            descr.append("HIP")

        else:
            _ = basename.replace("aspcapStar-r5-v603-", "apStar-r5-")
            idx = np.where(APOGEE["FILE"] == _)[0]
            #assert len(idx) == 1
            indices[i] = idx[0]

            # TODO: Hacky
            cluster_name = filename.split("/")[-2]
            distance = CLUSTER_DISTANCES[cluster_name][0]
            mu[i] = 5 * np.log10(distance) - 5
            descr.append(cluster_name)

    stars = Table(APOGEE[indices.astype(int)])
    stars["mu"] = mu
    stars["DESCR"] = descr

    # Calculate absolute magnitudes using the distance modulus and a correction
    # for dust.
    stars["SFD_EBV"][stars["SFD_EBV"] < 0] = 0
    print("Setting E(B-V) = 0 if not known")

    for pf in photometric_filters:
        stars["{}_ABS".format(pf)] = \
            stars[pf] - mu - stars["SFD_EBV"] * R.get(pf, 0)
        if pf not in R:
            print("No Rx (extinction) value found for filter {}".format(pf))

    return stars


def quality_mask(stars, rules=None):

    N_stars = len(stars)
    ok = np.ones(N_stars, dtype=bool)
    rules = [] if rules is None else rules
    for rule in rules:
        print("Applying {}".format(rule))
        _ok = eval(rule, {"stars": stars})
        num_affected = N_stars - sum(_ok)
        print("{0} excluded by rule {1}".format(num_affected, rule))
        ok *= _ok
    return ok


if __name__ == "__main__":

    # Load the configuration filename.
    with open(CONFIG_FILENAME, "r") as fp:
            config = yaml.load(fp)

    output_prefix, _ = os.path.splitext(CONFIG_FILENAME)
    model_filename = CONFIG_FILENAME.replace(".yaml", ".pkl")
    if os.path.exists(model_filename):
        model = cannon.CannonModel.from_filename(model_filename)

        # TODO: Check the model has not changed.

    else:
        # Get the data filenames and collate the tabular data.
        filenames = data_paths(config)
        stars = match_spectra(filenames)

        # Apply quality cuts so we don't load unnecessary files.
        QC = quality_mask(stars, config.get("qc", None))
        stars = stars[QC]
        filenames = np.array(filenames)[QC]

        # Some bespoke columns.
        stars["JmK"] = stars["J"] - stars["K"]
        stars["JmH"] = stars["J"] - stars["H"]
        stars["HmK"] = stars["H"] - stars["K"]
            
        # Load the spectra
        wavelengths, fluxes, flux_uncertainties = load_spectra(filenames)

        # Apply quality cuts before initiating the model.
        model = cannon.CannonModel(stars, wavelengths, fluxes, flux_uncertainties)
        model.train(config["label_vector_description"], threads=THREADS)
        model.save(model_filename, with_data=True)


    # Plot the label residuals.
    _ = os.path.join(OUTPUT_DIR, "{}-label-residuals.png".format(output_prefix))
    fig = model.plot_label_residuals(aux=config.get("plot", {}).get("aux", None))
    fig.savefig(_)
    print("Created {}".format(_))

    # Plot the scatter.
    _ = os.path.join(OUTPUT_DIR, "{}-scatter.png".format(output_prefix))
    fig = model.plot_model_scatter()
    fig.savefig(_)
    print("Created {}".format(_))


    # Plot a random spectrum.
    _ = os.path.join(OUTPUT_DIR, "{}-random-spectrum.png".format(output_prefix))
    fig = model._plot_random_spectrum()
    fig.savefig(_)
    print("Created {}".format(_))

    # Plot the expected distance vs inferred distance.
    percentile = True
    labels, expected, inferred = model.label_residuals
    filters = [_[:-4] for _ in labels if _.endswith("_ABS")]
    N_filters = len(filters)

    fig = plt.figure(figsize=(N_filters * 4, 4))
    gs = GridSpec(2, N_filters, height_ratios=[1, 4])
    axes = [fig.add_subplot(_) for _ in gs]
    for i, pf in enumerate(filters):

        expected_distance = 10**((5 + model._labels["mu"])/5.) / 1000. # [kpc]
        mu = model._labels[pf] \
            - inferred[:, list(labels).index("{}_ABS".format(pf))] \
            - model._labels["SFD_EBV"] * R.get(pf, 0)

        inferred_distance = 10**((5 + mu)/5.) / 1000. # [kpc]
        residual_distance = inferred_distance - expected_distance
        difference_absolute = residual_distance
        difference_percent = 100 * residual_distance / expected_distance

        ax_relation, ax_diff = axes[N_filters + i], axes[i]

        ax_relation.scatter(expected_distance, inferred_distance, facecolor="k")

        limit = max([ax_relation.get_xlim()[1], ax_relation.get_ylim()[1]])
        ax_relation.plot([0, limit], [0, limit], c="#cccccc", zorder=-100)
        ax_relation.set_xlim(0, limit)
        ax_relation.set_ylim(0, limit)
        ax_relation.set_title(pf)

        ax_diff.axhline(0, c="#cccccc", zorder=-100)
        _ = difference_percent if percentile else difference_absolute
        ax_diff.scatter(expected_distance, _, facecolor="k")
        ax_diff.set_xlim(0, limit)

        ylimit = np.max(np.abs(ax_diff.get_ylim()))
        ax_diff.set_ylim(-ylimit, +ylimit)

        ax_relation.set_xlabel("Expected distance [kpc]")
        ax_relation.set_ylabel("Inferred distance [kpc]")
        ax_relation.xaxis.set_major_locator(MaxNLocator(5))
        ax_relation.yaxis.set_major_locator(MaxNLocator(5))

        _ = r"$\Delta{D}\,\,[\%]$" if percentile else r"$\Delta{D}\,\,[{\rm kpc}]$"
        ax_diff.set_ylabel(_)
        ax_diff.set_xticklabels([])
        ax_diff.yaxis.set_major_locator(MaxNLocator(5))
        ax_diff.set_title("mean / median / sigma / |sigma| [%] = {0:.1f} / {1:.1f}"
            " / {2:.1f} / {3:.1f}".format(
            np.nanmean(residual_distance),
            np.nanmedian(residual_distance),
            np.nanstd(difference_absolute),
            np.nanstd(np.abs(difference_percent))),
            fontsize=8)


    fig.tight_layout()
    _ = os.path.join(OUTPUT_DIR, "{}-distances.png".format(output_prefix))
    fig.savefig(_)
    print("Created {}".format(_))


    # Plot the expected parallax vs inferred parallax.
    if np.any(model._labels["Plx"] > 0):
        fig, axes = plt.subplots(len(filters))
        axes = np.array([axes]) if N_filters == 1 else axes.flatten()
        
        for ax, pf in zip(axes, filters):

            expected_plx = model._labels["Plx"]
            inferred_mu = model._labels[pf] \
                - inferred[:, list(labels).index("{}_ABS".format(pf))] \
                - model._labels["SFD_EBV"] * R.get(pf, 0)

            inferred_plx = 1000./(10**((5 + inferred_mu)/5.))

            ax.errorbar(expected_plx, inferred_plx, model._labels["e_Plx"],
                fmt=None, ecolor="k", zorder=-1)
            ax.scatter(expected_plx, inferred_plx, facecolor="k")
            ax.set_xlabel("Hipparcos parallax [mas/yr]")
            ax.set_ylabel("Inferred parallax [mas/yr]")

            limit = max([ax.get_xlim()[1], ax.get_ylim()[1]])
            ax.plot([0, limit], [0, limit], c="#666666", zorder=-100)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)

            residual_plx = inferred_plx - expected_plx
            ax.set_title("mean / median / sigma = {0:.1f} / {1:.1f} / {2:.1f}"\
                .format(np.nanmean(residual_plx), np.nanmedian(residual_plx),
                    np.nanstd(residual_plx)))

        _ = os.path.join(OUTPUT_DIR, "{}-plx.png".format(output_prefix))
        fig.savefig(_)
        print("Created {}".format(_))

    else:
        print("No Hipparcos stars in sample")


    # Do cross-validation for each cluster.
    if LOCO_CV:

        _ = os.path.join(OUTPUT_DIR, "{}-loco-cv.pkl".format(output_prefix))
        if os.path.exists(_):
            print("Loading LOCO-CV results from {}".format(_))
            with open(_, "rb") as fp:
                labels, expected, inferred = pickle.load(fp)
        else:            
            labels, expected, inferred = model.cross_validate_by_label("DESCR",
                threads=THREADS)
            
            with open(_, "wb") as fp:
                pickle.dump((labels, expected, inferred), fp, -1)
            print("Saved LOCO-CV results to {}".format(_))
        
        aux = config.get("plot", {}).get("aux", None)
        if aux is not None:
            aux = model._labels[aux]
        _ = os.path.join(OUTPUT_DIR, "{}-label-residuals-loco-cv.png".format(
            output_prefix))
        fig = plot.label_residuals(labels, expected, inferred, aux)
        fig.savefig(_)
        print("Created {}".format(_))

        # Using the cross-validated values, do expected distance vs inferred.
        filters = [_[:-4] for _ in labels if _.endswith("_ABS")]
        N_filters = len(filters)

        fig = plt.figure(figsize=(N_filters * 4, 4))
        gs = GridSpec(2, N_filters, height_ratios=[1, 4])
        axes = [fig.add_subplot(_) for _ in gs]
        for i, pf in enumerate(filters):

            expected_distance = 10**((5 + model._labels["mu"])/5.) / 1000. # [kpc]
            mu = model._labels[pf] \
                - inferred[:, list(labels).index("{}_ABS".format(pf))] \
                - model._labels["SFD_EBV"] * R.get(pf, 0)

            inferred_distance = 10**((5 + mu)/5.) / 1000. # [kpc]
            residual_distance = inferred_distance - expected_distance
            difference_absolute = residual_distance
            difference_percent = 100 * residual_distance / expected_distance

            ax_relation, ax_diff = axes[N_filters + i], axes[i]

            ax_relation.scatter(expected_distance, inferred_distance, facecolor="k")

            limit = max([ax_relation.get_xlim()[1], ax_relation.get_ylim()[1]])
            ax_relation.plot([0, limit], [0, limit], c="#cccccc", zorder=-100)
            ax_relation.set_xlim(0, limit)
            ax_relation.set_ylim(0, limit)

            ax_diff.axhline(0, c="#cccccc", zorder=-100)
            _ = difference_percent if percentile else difference_absolute
            ax_diff.scatter(expected_distance, _, facecolor="k")
            ax_diff.set_xlim(0, limit)

            ylimit = np.max(np.abs(ax_diff.get_ylim()))
            ax_diff.set_ylim(-ylimit, +ylimit)

            ax_relation.set_xlabel("Expected distance [kpc]")
            ax_relation.set_ylabel("Inferred distance [kpc]")
            ax_relation.xaxis.set_major_locator(MaxNLocator(5))
            ax_relation.yaxis.set_major_locator(MaxNLocator(5))

            _ = r"$\Delta{D}\,\,[\%]$" if percentile else r"$\Delta{D}\,\,[{\rm kpc}]$"
            ax_diff.set_ylabel(_)
            ax_diff.set_xticklabels([])
            ax_diff.yaxis.set_major_locator(MaxNLocator(5))
            ax_diff.set_title("mean / median / sigma / |sigma| [%] = {0:.1f} / {1:.1f}"
                " / {2:.1f} / {3:.1f}".format(
                np.nanmean(residual_distance),
                np.nanmedian(residual_distance),
                np.nanstd(difference_absolute),
                np.nanstd(np.abs(difference_percent))),
                fontsize=8)

        fig.tight_layout()
        _ = os.path.join(OUTPUT_DIR, "{}-distances-loco-cv.png".format(output_prefix))
        fig.savefig(_)
        print("Created {}".format(_))

        if np.any(model._labels["Plx"] > 0):
            fig, axes = plt.subplots(len(filters))
            axes = np.array([axes]) if N_filters == 1 else axes.flatten()
            
            for ax, pf in zip(axes, filters):

                expected_plx = model._labels["Plx"]
                inferred_mu = model._labels[pf] \
                    - inferred[:, list(labels).index("{}_ABS".format(pf))] \
                    - stars["SFD_EBV"] * R.get(pf, 0)

                inferred_plx = 1000./(10**((5 + inferred_mu)/5.))

                ax.errorbar(expected_plx, inferred_plx, model._labels["e_Plx"],
                    fmt=None, ecolor="k", zorder=-1)
                ax.scatter(expected_plx, inferred_plx, facecolor="k")
                ax.set_xlabel("Hipparcos parallax [mas/yr]")
                ax.set_ylabel("Inferred parallax [mas/yr]")

                limit = max([ax.get_xlim()[1], ax.get_ylim()[1]])
                ax.plot([0, limit], [0, limit], c="#666666", zorder=-100)
                ax.set_xlim(0, limit)
                ax.set_ylim(0, limit)

                residual_plx = inferred_plx - expected_plx
                ax.set_title("mean / median / sigma = {0:.1f} / {1:.1f} / {2:.1f}"\
                    .format(np.nanmean(residual_plx), np.nanmedian(residual_plx),
                        np.nanstd(residual_plx)))

            _ = os.path.join(OUTPUT_DIR, "{}-plx-loco-cv.png".format(output_prefix))
            fig.savefig(_)
            print("Created {}".format(_))


    # Do cross-validation for individual stars.
    if LOO_CV:

        _ = os.path.join(OUTPUT_DIR, "{}-loo-cv.pkl".format(output_prefix))
        if os.path.exists(_):
            print("Loading LOO-CV results from {}".format(_))
            with open(_, "rb") as fp:
                labels, expected, inferred = pickle.load(fp)
        else:            
            labels, expected, inferred = model.cross_validate(threads=THREADS)
            with open(_, "wb") as fp:
                pickle.dump((labels, expected, inferred), fp, -1)
            print("Saved LOO-CV results to {}".format(_))
        
        aux = config.get("plot", {}).get("aux", None)
        if aux is not None:
            aux = model._labels[aux]
        _ = os.path.join(OUTPUT_DIR, "{}-label-residuals-loo-cv.png".format(
            output_prefix))
        fig = plot.label_residuals(labels, expected, inferred, aux)
        fig.savefig(_)
        print("Created {}".format(_))

        # Using the cross-validated values, do expected distance vs inferred.
        filters = [_[:-4] for _ in labels if _.endswith("_ABS")]
        N_filters = len(filters)


        fig = plt.figure(figsize=(N_filters * 4, 4))
        gs = GridSpec(2, N_filters, height_ratios=[1, 4])
        axes = [fig.add_subplot(_) for _ in gs]
        for i, pf in enumerate(filters):

            expected_distance = 10**((5 + model._labels["mu"])/5.) / 1000. # [kpc]
            mu = model._labels[pf] \
                - inferred[:, list(labels).index("{}_ABS".format(pf))] \
                - model._labels["SFD_EBV"] * R.get(pf, 0)
            inferred_distance = 10**((5 + mu)/5.) / 1000. # [kpc]
            residual_distance = inferred_distance - expected_distance
            difference_absolute = residual_distance
            difference_percent = 100 * residual_distance / expected_distance

            ax_relation, ax_diff = axes[N_filters + i], axes[i]
            ax_relation.scatter(expected_distance, inferred_distance, facecolor="k")

            limit = max([ax_relation.get_xlim()[1], ax_relation.get_ylim()[1]])
            ax_relation.plot([0, limit], [0, limit], c="#cccccc", zorder=-100)
            ax_relation.set_xlim(0, limit)
            ax_relation.set_ylim(0, limit)

            ax_diff.axhline(0, c="#cccccc", zorder=-100)
            _ = difference_percent if percentile else difference_absolute
            ax_diff.scatter(expected_distance, _, facecolor="k")
            ax_diff.set_xlim(0, limit)

            ylimit = np.max(np.abs(ax_diff.get_ylim()))
            ax_diff.set_ylim(-ylimit, +ylimit)

            ax_relation.set_xlabel("Expected distance [kpc]")
            ax_relation.set_ylabel("Inferred distance [kpc]")
            ax_relation.xaxis.set_major_locator(MaxNLocator(5))
            ax_relation.yaxis.set_major_locator(MaxNLocator(5))

            _ = r"$\Delta{D}\,\,[\%]$" if percentile else r"$\Delta{D}\,\,[{\rm kpc}]$"
            ax_diff.set_ylabel(_)
            ax_diff.set_xticklabels([])
            ax_diff.yaxis.set_major_locator(MaxNLocator(5))
            ax_diff.set_title("mean / median / sigma / |sigma| [%] = {0:.1f} / {1:.1f}"
                " / {2:.1f} / {3:.1f}".format(
                np.nanmean(residual_distance),
                np.nanmedian(residual_distance),
                np.nanstd(difference_absolute),
                np.nanstd(np.abs(difference_percent))),
                fontsize=8)

        fig.tight_layout()
        _ = os.path.join(OUTPUT_DIR, "{}-distances-loo-cv.png".format(output_prefix))
        fig.savefig(_)
        print("Created {}".format(_))

        if np.any(model._labels["Plx"] > 0):
            fig, axes = plt.subplots(len(filters))
            axes = np.array([axes]) if N_filters == 1 else axes.flatten()
            
            for ax, pf in zip(axes, filters):

                expected_plx = model._labels["Plx"]
                inferred_mu = model._labels[pf] \
                    - inferred[:, list(labels).index("{}_ABS".format(pf))] \
                    - model._labels["SFD_EBV"] * R.get(pf, 0)
                inferred_plx = 1000./(10**((5 + inferred_mu)/5.))

                ax.errorbar(expected_plx, inferred_plx, model._labels["e_Plx"],
                    fmt=None, ecolor="k", zorder=-1)
                ax.scatter(expected_plx, inferred_plx, facecolor="k")
                ax.set_xlabel("Hipparcos parallax [mas/yr]")
                ax.set_ylabel("Inferred parallax [mas/yr]")

                limit = max([ax.get_xlim()[1], ax.get_ylim()[1]])
                ax.plot([0, limit], [0, limit], c="#666666", zorder=-100)
                ax.set_xlim(0, limit)
                ax.set_ylim(0, limit)

                residual_plx = inferred_plx - expected_plx
                ax.set_title("mean / median / sigma = {0:.1f} / {1:.1f} / {2:.1f}"\
                    .format(np.nanmean(residual_plx), np.nanmedian(residual_plx),
                        np.nanstd(residual_plx)))

            _ = os.path.join(OUTPUT_DIR, "{}-plx-loo-cv.png".format(output_prefix))
            fig.savefig(_)
            print("Created {}".format(_))




    raise a



# Supply:
# run model.pkl config.yaml --plot 

# Configuration includes:
# - bring data from where [data]
# - quality cuts [qc]
# - label vector description [label_vector_description]

# [ARG] Load in from somewhere else?
    # Include data from where?

    # Load in all of the fluxes, uncertainties, etc

    # Is it a Hipparcos or cluster star?

    # Get mu from Hipparcos parallax or assumed cluster distance

    # Exclude stars based on some QCs

    # Train model based on a label vector description

# Save the model with data.

# [ARG] Plot expected versus residuals from in/out

# [ARG] Plot expected versus residuals for any Hipparcos stars

# Save the in/out residuals.

# Do cross-validation?

# Save the expected/inferred.

# Plot results

# Do one-cluster-out cross-validation?

# Save the expected/inferred

# Plot results

