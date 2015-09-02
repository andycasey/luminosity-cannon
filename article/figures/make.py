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
LINE_KWDS = {}
SCATTER_KWDS = {
    "s": 30,
    "facecolor": "k"
}

ERRORBAR_KWDS = {
    "fmt": None,
    "ecolor": "k",
    "zorder": -1
}

def hipparcos_hrd(data_table="../../data/APOGEE-Hipparcos.fits.gz"):
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


def cluster_hrds(data_table="../../data/APOGEE-Clusters.fits.gz"):
    """
    HR diagram for clusters.
    """

    stars = fits.open(data_table)[1].data

    # Cluster name is stored in 'FIELD'.
    clusters = sorted(set(stars["FIELD"]))
    N = len(clusters)

    # (N, M) figure size.
    cols = np.ceil(N**0.5)
    rows = np.ceil(N / cols)
    cols, rows = map(int, (cols, rows))

    kwds = {}
    kwds.update(SCATTER_KWDS)

    e_kwds = {}
    e_kwds.update(ERRORBAR_KWDS)

    fig, axes = plt.subplots(cols, rows)
    if cols * rows == 1: axes = np.array([axes])
    for i, (ax, cluster) in enumerate(zip(axes.flatten(), clusters)):

        # Plot this stuff.
        members = stars["FIELD"] == cluster
        ax.errorbar(stars["TEFF"][members], stars["LOGG"][members],
            xerr=stars["TEFF_ERR"][members], yerr=stars["LOGG_ERR"][members],
            **e_kwds)
        ax.scatter(stars["TEFF"][members], stars["LOGG"][members], **kwds)

        # Show the name of the cluster.
        ax.text(5500, 0.5, r'${\rm ' + cluster + '}$',
            horizontalalignment="left", verticalalignment="bottom")

        # [TODO] Show an isochrone.

    # Set everything at the same axes.
    x_lims, y_lims = map(list, (ax.get_xlim(), ax.get_ylim()))
    for ax in axes.flatten()[:N]:
        # Get limits.
        if ax.get_xlim()[0] < x_lims[0]:
            x_lims[0] = ax.get_xlim()[0]
        if ax.get_xlim()[1] > x_lims[1]:
            x_lims[1] = ax.get_xlim()[1]

        if ax.get_ylim()[0] < y_lims[0]:
            y_lims[0] = ax.get_ylim()[0]
        if ax.get_ylim()[1] > y_lims[1]:
            y_lims[1] = ax.get_ylim()[1]

    # Set limits and ticks.
    for ax in axes.flatten()[:N]:
        ax.set_xlim(x_lims[::-1])
        ax.set_ylim(y_lims[::-1])

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        # Set labels.
        if ax.is_last_row() or ax in axes.flatten()[N - cols + 1:N]:
            ax.set_xlabel(r"$T_{\rm eff}$ $[{\rm K}]$")
        else:
            ax.set_xticklabels([])

        if ax.is_first_col():
            ax.set_ylabel(r"$\log{g}$")
        else:
            ax.set_yticklabels([])

    # Hide extra axes.
    fig.tight_layout()
    for ax in axes.flatten()[N:]:
        ax.set_visible(False)

    # Last visible in a row?
    # [TODO]
    return fig



def cluster_members(data_table="../../data/APOGEE-All_Clusters.fits.gz"):
    """
    HR diagram for clusters.
    """

    stars = fits.open(data_table)[1].data

    # Cluster name is stored in 'FIELD'.
    clusters = sorted(set(stars["FIELD"]))
    N = len(clusters)

    # (N, M) figure size.
    cols = np.ceil(N**0.5)
    rows = np.ceil(N / cols)
    cols, rows = map(int, (cols, rows))

    kwds = {}
    kwds.update(SCATTER_KWDS)

    e_kwds = {}
    e_kwds.update(ERRORBAR_KWDS)

    fig, axes = plt.subplots(cols, rows)
    if cols * rows == 1: axes = np.array([axes])
    for i, (ax, cluster) in enumerate(zip(axes.flatten(), clusters)):

        # Plot this stuff.
        members = (stars["FIELD"] == cluster) * (stars["TEFF"] > 0)
        #ax.errorbar(stars["TEFF"][members], stars["LOGG"][members],
        #    xerr=stars["TEFF_ERR"][members], yerr=stars["LOGG_ERR"][members],
        #    **e_kwds)
        ax.scatter(stars["VHELIO_AVG"][members], stars["PARAM_M_H"][members],
            **kwds)

        # Show the name of the cluster.
        ax.set_title(cluster)

        #ax.text(5500, 0.5, r'${\rm ' + cluster + '}$',
        #    horizontalalignment="left", verticalalignment="bottom")

        # [TODO] Show an isochrone.

    # Set everything at the same axis range.
    x_ptp, y_ptp = map(np.ptp, (ax.get_xlim(), ax.get_ylim()))
    for ax in axes.flatten()[:N]:

        xi_ptp, yi_ptp = map(np.ptp, (ax.get_xlim(), ax.get_ylim()))
        if xi_ptp > x_ptp:
            x_ptp = xi_ptp
        if yi_ptp > y_ptp:
            y_ptp = yi_ptp

    # Set limits and ticks.
    for ax in axes.flatten()[:N]:
        x_mean, y_mean = map(np.mean, (ax.get_xlim(), ax.get_ylim()))

        ax.set_xlim(x_mean - 0.5 * x_ptp, x_mean + 0.5 * x_ptp)
        ax.set_ylim(y_mean - 0.5 * y_ptp, y_mean + 0.5 * y_ptp)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        # Set labels.
        if ax.is_last_row() or ax in axes.flatten()[N - cols + 1:N]:
            ax.set_xlabel(r"$v$ $[{\rm km s}^{-1}]$")
        else:
            ax.set_xticklabels([])

        if ax.is_first_col():
            ax.set_ylabel(r"$[{\rm M/H}]$")
        else:
            ax.set_yticklabels([])

    # Hide extra axes.
    for ax in axes.flatten()[N:]:
        ax.set_visible(False)

    return fig


def inferred_distance(filename="../../data/APOGEE-All_Clusters-results.fits.gz",
    photometric_filter=None, percentile=False, col="K_ERR"):


    data = fits.open(filename)[1].data
    data = data[(data["K_ERR"] > 0) * (data["K_ERR"] < 0.05)]


    dist_labels = [_ for _ in data.dtype.names if _.startswith("D_inferred_")]

    if len(dist_labels) == 0:
        raise ValueError("no distance labels found (none starting with "
            "D_inferred_*")

    figures = []
    for label in dist_labels:

        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax_diff, ax_relation = map(plt.subplot, gs)

        scatter_kwds = {}
        scatter_kwds.update(SCATTER_KWDS)

        line_kwds = {}
        line_kwds.update(LINE_KWDS)
        line_kwds.update({
            "c": "#666666",
            "zorder": -100
        })

        ax_relation.scatter(data["D_expected_mu"], data[label], **scatter_kwds)

        # Plot limits.
        limits = max([ax_relation.get_xlim()[1], ax_relation.get_ylim()[1]])
        ax_relation.plot([0, limits], [0, limits], **line_kwds)
        ax_relation.set_xlim(0, limits)
        ax_relation.set_ylim(0, limits)

        ax_relation.set_xlabel(r"${\rm Expected}$ ${\rm distance}$ $[{\rm kpc}]$")
        ax_relation.set_ylabel(r"${\rm Inferred}$ ${\rm distance}$ $[{\rm kpc}]$")

        ax_relation.xaxis.set_major_locator(MaxNLocator(5))
        ax_relation.yaxis.set_major_locator(MaxNLocator(5))

        # Show the difference.
        difference = data[label] - data["D_expected_mu"]
        if percentile:
            difference *= 100./data["D_expected_mu"]

        ax_diff.scatter(data["D_expected_mu"], difference,
            **scatter_kwds)
        
        ax_diff.plot([0, limits], [0, 0], **line_kwds)

        ylabel = r"$\Delta{D}$ $[{\rm \%}]$" if percentile \
            else r"$\Delta{D}$ $[{\rm kpc}]$"
        ax_diff.set_ylabel(ylabel)

        ax_diff.set_xlim(ax_relation.get_xlim())
        ax_diff.set_xticks(ax_relation.get_xticks())
        ax_diff.set_xticklabels([])

        # Mark the positions of the clusters (in expected distance).
        clusters = set(data["FIELD"])
        y_text = -10 if not percentile else -15
        cluster_distances = []
        for cluster in clusters:
            cluster_distances.append(
                np.median(data["D_expected_mu"][data["FIELD"] == cluster]))

        idx = np.argsort(cluster_distances)
        clusters = np.array(list(clusters))[idx]
        cluster_distances = np.array(cluster_distances)[idx]

        print(label)
        for j, (cluster, distance) in enumerate(zip(clusters, cluster_distances)):
            #print(cluster, distance)
            nearby = np.abs(distance - np.array(cluster_distances)[:j])
            ax_relation.text(distance, distance + 5 + np.sum(nearby < 3),
                cluster, fontsize=10, horizontalalignment="right")


            errors = distance - data[label][data["FIELD"] == cluster]
            errors_percent = errors * 100./data[label][data["FIELD"] == cluster]
            print("{0} mean {1:.1f} std. dev {2:.1f} kpc {3:.1f} \%".format(
                cluster, np.mean(errors), np.std(errors), np.std(errors_percent)))

        ylims = max(np.abs(ax_diff.get_ylim()))
        ax_diff.set_ylim(-ylims, +ylims)
        ax_diff.yaxis.set_major_locator(MaxNLocator(5))

        if not percentile:
            # Show percentile lines.
            ax_relation.plot([0, limits], [0, 0.9 * limits], **line_kwds)
            ax_relation.plot([0, limits], [0, 1.1 * limits], **line_kwds)

            ax_relation.fill_between([0, limits],
                [0, 0.9 * limits], [0, 1.1 * limits],
                facecolor="#cccccc", zorder=-100)


        fig = ax_relation.figure
        fig.tight_layout()

        if not percentile:
            difference *= 100./data["D_expected_mu"]

        print("Error mean {0:.1f} kpc std. dev. {1:.1f} kpc".format(
            difference.mean(), np.std(difference)))

        figures.append(fig)

        """
        fig, ax = plt.subplots()
        ax.scatter(data[col], difference, **scatter_kwds)

        ax.set_xlabel("COL")
        ax.set_ylabel("ERROR (percent)")

        """

    return figures




def inferred_parallax(filename="../../data/APOGEE-Hipparcos-results.fits.gz",
    photometric_filter=None, percentile=False):


    data = fits.open(filename)[1].data

    dist_labels = [_ for _ in data.dtype.names if _.startswith("Plx_inferred_")]

    if len(dist_labels) == 0:
        raise ValueError("no parallax labels found (none starting with "
            "Plx_inferred_*")

    figures = []
    for label in dist_labels:

        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax_diff, ax_relation = map(plt.subplot, gs)

        scatter_kwds = {}
        scatter_kwds.update(SCATTER_KWDS)

        line_kwds = {}
        line_kwds.update(LINE_KWDS)
        line_kwds.update({
            "c": "#666666",
            "zorder": -100
        })

        ax_relation.errorbar(data["Plx"], data[label],
            xerr=data["e_Plx"], **ERRORBAR_KWDS)
        ax_relation.scatter(data["Plx"], data[label], **scatter_kwds)

        # Plot limits.
        limits = max([ax_relation.get_xlim()[1], ax_relation.get_ylim()[1]])
        ax_relation.plot([0, limits], [0, limits], **line_kwds)
        ax_relation.set_xlim(0, limits)
        ax_relation.set_ylim(0, limits)

        ax_relation.set_xlabel(r"$Hipparcos$ ${\rm parallax}$ $[{\rm mas\,\, yr^{-1}}]$")
        ax_relation.set_ylabel(r"${\rm Inferred}$ ${\rm parallax}$ $[{\rm mas\,\, yr^{-1}}]$")

        ax_relation.xaxis.set_major_locator(MaxNLocator(5))
        ax_relation.yaxis.set_major_locator(MaxNLocator(5))

        # Show the difference.
        difference = data[label] - data["Plx"]
        if percentile:
            difference *= 100./data["Plx"]

        ax_diff.errorbar(data["Plx"], difference, xerr=data["e_Plx"],
            **ERRORBAR_KWDS)
        ax_diff.scatter(data["Plx"], difference, **scatter_kwds)
        
        ax_diff.plot([0, limits], [0, 0], **line_kwds)

        ylabel = r"$\Delta{\pi}$ $[{\rm \%}]$" if percentile \
            else r"$\Delta{\pi}$ $[{\rm mas\,\,yr^{-1}}]$"
        ax_diff.set_ylabel(ylabel)

        ax_diff.set_xlim(ax_relation.get_xlim())
        ax_diff.set_xticks(ax_relation.get_xticks())
        ax_diff.set_xticklabels([])

        ylims = max(np.abs(ax_diff.get_ylim()))
        ax_diff.set_ylim(-ylims, +ylims)
        ax_diff.yaxis.set_major_locator(MaxNLocator(5))

        fig = ax_relation.figure
        fig.tight_layout()

        figures.append(fig)

    return figures


if __name__ == "__main__":

    from collections import OrderedDict
    
    FIGURES = {
        "figure-1": hipparcos_hrd,
        "figure-2": inferred_parallax,
        "figure-3": cluster_hrds,
        "tmp-cluster-members": cluster_members,
        "figure-4": inferred_distance
    }

    for filename_prefix, function in FIGURES.items():
        fig = function()
        if isinstance(fig, list) and len(fig) > 1:
            for i, f in enumerate(fig):
                f.savefig("{}.png".format(filename_prefix), dpi=300)
                f.savefig("{}.pdf".format(filename_prefix), dpi=300)
    
        elif isinstance(fig, list):
            fig = fig[0]
            fig.savefig("{}.png".format(filename_prefix), dpi=300)
            fig.savefig("{}.pdf".format(filename_prefix), dpi=300)
    
        else:
            fig.savefig("{}.png".format(filename_prefix), dpi=300)
            fig.savefig("{}.pdf".format(filename_prefix), dpi=300)

        print("Created {0}.png and {0}.pdf".format(filename_prefix))


