#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Download APOGEE cluster spectra used in Ness et al. (2015). """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

ness_path = np.loadtxt("Ness-Clusters.txt", usecols=(0, ), dtype=str)

command = ("wget -O APOGEE/Ness_Clusters/{cluster_name}/{basename}"
        " http://data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/"
        "stars/l25_6d/v603/{field}/{basename}")

expected_downloads = len(ness_path)
for path in ness_path:

    _ = path.split('/')[-2]
    field, cluster_name = _.split('_')

    if not os.path.exists("APOGEE/Ness_Clusters/{}".format(cluster_name)):
        os.mkdir("APOGEE/Ness_Clusters/{}".format(cluster_name))

    os.system(command.format(field=field,
        cluster_name=cluster_name,
        basename=os.path.basename(path).replace("aspcapStar-v304-",
            "aspcapStar-r5-v603-")))

