#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Download APOGEE spectra for Hipparcos stars. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
from glob import glob
from astropy.io import fits

APOGEE_DATA_FOLDER = "APOGEE/"

data = fits.open("APOGEE-allStar-v603-Hipparcos.fits.gz")[1].data
command = ("wget -O {APOGEE_DATA_FOLDER}/HIP{hip}.fits http://data.sdss3.org/"
    "sas/dr12/apogee/spectro/redux/r5/stars/{telescope}/{field}/{file}")

if not os.path.exists(APOGEE_DATA_FOLDER):
    os.mkdir(APOGEE_DATA_FOLDER)

before = len(glob("{}/*.fits".format(APOGEE_DATA_FOLDER)))
for row in data:
    os.system(command.format(
        APOGEE_DATA_FOLDER=APOGEE_DATA_FOLDER,
        telescope=row["TELESCOPE"],
        file=row["FILE"],
        field=row["LOCATION_ID"] if row["LOCATION_ID"] > 1 else "hip",
        hip=row["HIP"]))
after = len(glob("{}/*.fits".format(APOGEE_DATA_FOLDER)))

print("New files in {0}/: {1} (from {2} rows)".format(APOGEE_DATA_FOLDER,
    after - before, len(data)))
