#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Download cluster spectra from APOGEE. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
import matplotlib.pyplot as plt
from astropy.io import fits

# Clusters that I will use which are also in Ness et al. (2015)
cluster_names = ("M2", "M3", "M13", "M15", "M53", "M67", "M71", "M92")

# Other clusters that are used by Ness et al. (2015) but not used here yet:
#   "M107": "NGC 6171", Difficult to unambiguously separate from field.
#   "N4147": "NGC 4147", # No clear membership criterion could be drawn. 
#   "N5466": "NGC 5466", # Harris says [Fe/H] = -2. No clear members.
#   "M35N2158": "M 3", # M3 
#   "M5PAL5": "M 5", 
#   N188,   # 
#   N2420,  # 
#   N6791,  # 
#   N6819,  # 
#   N7789   # 
#   Pleades. # 

# Other clusters that are not used in Ness et al. (2015):
#   N6229 (NGC 6229): No clear memebers.
#   M54SGRC1 (M 54).
#   N1333
#   N5634SGR2
#   N2243

command = ("wget -O APOGEE/Clusters/{cluster_name}/{file}"
    " http://data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/"
    "stars/{telescope}/{location_id}/{file}")

apogee = fits.open("APOGEE-allStar-v603.fits")[1].data
ok = (apogee["SNR"] > 80.) * (np.abs(apogee["VHELIO_AVG"]) < 500) \
    * (apogee["ASPCAPFLAG"] == 0) * (apogee["STARFLAG"] == 0) \
    * (apogee["TEFF"] > 0)

membership_criteria = {
    # Harris says [Fe/H] = -0.78 and V_helio = -22.8
    "M71": (-15 > apogee["VHELIO_AVG"]) * (apogee["VHELIO_AVG"] > -30) \
        * (-0.6 > apogee["PARAM_M_H"]) * (apogee["PARAM_M_H"] > -1),
    # Harris says NGC 7078 has [Fe/H] = -2.37 and V_helio = -107.0
    "M15": (-2 > apogee["PARAM_M_H"]),
    # Harris says NGC 6205 has [Fe/H] = -1.53 and V_helio = -244.2
    "M13": (-1.5 > apogee["PARAM_M_H"]) * (-230 > apogee["VHELIO_AVG"]) \
        * (apogee["VHELIO_AVG"] > -250),
    # Harris says M92 has [Fe/H] = -2.31 and V_helio = -120.0
    "M92": (-100 > apogee["VHELIO_AVG"]) * (apogee["VHELIO_AVG"] > -130) \
        * (apogee["PARAM_M_H"] < -2.0),
    # Clump at V_helio ~ 0 and [Fe/H] < -1.3
    "M2": (20 > apogee["VHELIO_AVG"]) * (apogee["VHELIO_AVG"] > -20) \
        * (apogee["PARAM_M_H"] < -1.3),
    # Harris says M3 has [Fe/H] = -1.50 and V_helio = -147.6
    "M3": (apogee["PARAM_M_H"] < -1.2) * (apogee["VHELIO_AVG"] < -136),
    "M67": (35 > apogee["VHELIO_AVG"]) * (apogee["VHELIO_AVG"] > 33) \
        * (0.17 > apogee["PARAM_M_H"]) * (apogee["PARAM_M_H"] > -0.10),
    "M53": (-50 > apogee["VHELIO_AVG"]) * (apogee["VHELIO_AVG"] > -70) \
        * (-1.9 > apogee["PARAM_M_H"]) * (apogee["PARAM_M_H"] > -2.3)
}

if not os.path.exists("APOGEE/Clusters/plots/"):
    os.mkdirs("APOGEE/Clusters/plots/")

for cluster_name in cluster_names:

    candidates = (apogee["FIELD"] == cluster_name) * ok

    if cluster_name in membership_criteria:
        members = membership_criteria[cluster_name] * candidates
    else:
        members = candidates

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(apogee["RA"][candidates], apogee["DEC"][candidates],
        facecolor="k")
    ax[0].scatter(apogee["RA"][members], apogee["DEC"][members], facecolor="r")
    ax[0].set_xlabel("RA")
    ax[0].set_ylabel("DEC")

    ax[1].scatter(apogee["VHELIO_AVG"][candidates],
        apogee["PARAM_M_H"][candidates], facecolor="k")
    ax[1].scatter(apogee["VHELIO_AVG"][members],
        apogee["PARAM_M_H"][members], facecolor="r")
    ax[1].set_title(cluster_name)

    ax[1].set_ylabel("[M/H]")
    ax[1].set_xlabel("V_helio")
    
    ax[2].scatter(apogee["TEFF"][candidates], apogee["LOGG"][candidates],
        facecolor="k")
    ax[2].scatter(apogee["TEFF"][members], apogee["LOGG"][members],
        facecolor="r")
    ax[2].set_xlabel("Teff")
    ax[2].set_ylabel("logg")
    ax[2].set_xlim(ax[2].get_xlim()[::-1])
    ax[2].set_ylim(ax[2].get_ylim()[::-1])

    fig.tight_layout()
    fig.savefig("APOGEE/Clusters/plots/{0}.png".format(cluster_name), dpi=300)

    # Download the cluster spectra.
    if not os.path.exists("APOGEE/Clusters/{}".format(cluster_name)):
        os.mkdir("APOGEE/Clusters/{}".format(cluster_name))

    for member in apogee[members]:
        os.system(command.format(cluster_name=cluster_name,
            telescope=member["TELESCOPE"], location_id=member["LOCATION_ID"],
            file=member["FILE"]))
