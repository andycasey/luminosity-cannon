#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Download APOGEE spectra for Hipparcos stars. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
from glob import glob
from astropy.io import fits

# Which spectra should be downloaded, and to where?
APSTAR, ASPCAP = (True, True)
APOGEE_DATA_FOLDER = "APOGEE/"

data = fits.open("APOGEE-allStar-v603-Hipparcos.fits.gz")[1].data

if APSTAR:

    command = ("wget -O {APOGEE_DATA_FOLDER}/apStar/HIP{hip}.fits"
        " http://data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/"
        "stars/{telescope}/{field}/{file}")

    if not os.path.exists("{}/apStar/".format(APOGEE_DATA_FOLDER)):
        os.mkdir("{}/apStar/".format(APOGEE_DATA_FOLDER))

    for row in data:
        os.system(command.format(
            APOGEE_DATA_FOLDER=APOGEE_DATA_FOLDER,
            telescope=row["TELESCOPE"],
            file=row["FILE"],
            field=row["LOCATION_ID"] if row["LOCATION_ID"] > 1 else row["FIELD"],
            hip=row["HIP"]))
    
if ASPCAP:

    # Get the aspCap stuff.
    command = ("wget -O {APOGEE_DATA_FOLDER}/aspCap/HIP{hip}.fits"
        " http://data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/"
        "stars/l25_6d/v603/{field}/{file}")

    if not os.path.exists("{}/aspCap/".format(APOGEE_DATA_FOLDER)):
        os.mkdir("{}/aspCap/".format(APOGEE_DATA_FOLDER))

    for row in data:
        if 0 > row["TEFF"]: continue

        # Download the aspCap stuff.
        os.system(command.format(
            APOGEE_DATA_FOLDER=APOGEE_DATA_FOLDER,
            file=row["FILE"].replace("apStar-r5-", "aspcapStar-r5-v603-"),
            field=row["LOCATION_ID"] if row["LOCATION_ID"] > 1 else row["FIELD"],
            hip=row["HIP"]))


    