#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Put the APOGEE Hipparcos and Cluster data into a convenient format. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.table import Table, vstack

np.random.seed(123)

__cluster_distances = {
    # Distance in parsecs, core radius in arcmin, half-light radius in arcmin, distance reference.
    # R_core from Davenport et al. 2010
    "M67": (   908, 8.40, "Kharchenko et al. (2005)"), 
    "M2":  (11.5e3, 0.32, "Harris"),
    "M3":  (10.2e3, 0.37, "Harris"),
    "M13": ( 7.1e3, 0.62, "Harris"),
    "M15": (10.4e3, 0.14, "Harris"), # c
    "M53": (17.9e3, 0.35, "Harris"),
    "M71": ( 4.0e3, 0.63, "Harris"),
    "M92": ( 8.3e3, 0.26, "Harris"),
}


def get_apogee_cluster_sample(perturb_distances=True):
    """
    Prepare APOGEE Cluster spectra and data table in a convenient format.
    """

    spectra_filenames = glob("APOGEE/Clusters/*/*.fits")

    N_stars = len(spectra_filenames)
    with fits.open(spectra_filenames[0]) as image:
        N_pixels = image[1].data.size

    fluxes = np.zeros((N_stars, N_pixels))
    flux_uncertainties = np.zeros((N_stars, N_pixels))

    for i, filename in enumerate(spectra_filenames):
        print("{0}/{1}: {2}".format(i + 1, N_stars, filename))

        with fits.open(filename) as image:
            fluxes[i, :] = np.atleast_2d(image[1].data)[0, :].copy()
            flux_uncertainties[i, :] = np.atleast_2d(image[2].data)[0, :].copy()

    # Quality control.
    zero_pixels_everytime = np.all(0 >= fluxes, axis=0)
    fluxes = fluxes[:, ~zero_pixels_everytime]
    flux_uncertainties = flux_uncertainties[:, ~zero_pixels_everytime]

    unphysical_uncertainties = (0 >= flux_uncertainties) \
        + ~np.isfinite(flux_uncertainties)
    flux_uncertainties[unphysical_uncertainties] = 1e8

    unphysical_fluxes = (0 >= fluxes) + ~np.isfinite(fluxes)
    fluxes[unphysical_fluxes] = 1.0
    flux_uncertainties[unphysical_fluxes] = 1e8

    # Match each filename to the APOGEE stars.
    apogee = fits.open("APOGEE-allStar-v603.fits")[1].data
    indices = np.ones(N_stars, dtype=int)
    mu = np.zeros(N_stars)
    for i, filename in enumerate(spectra_filenames):
        _ = os.path.basename(filename).replace(
            "aspcapStar-r5-v603-", "apStar-r5-")

        index = np.where(apogee["FILE"] == _)[0]
        assert len(index) == 1
        indices[i] = index[0]

        cluster_name = filename.split("/")[-2]
        #mu = 5 * log_10(d [pc]) - 5
        # 10**((mu + 5)/5.)
        distance, core_radius_arcmin = __cluster_distances[cluster_name][:2]

        if perturb_distances:            
            # Calculate core_radius in parsecs.
            core_radius = distance * np.tan(np.pi/180 * core_radius_arcmin/60.)
            distance_realisation = np.random.normal(distance, core_radius)

        else:
            distance_realisation = distance

        mu[i] = 5. * np.log10(distance_realisation) - 5

    cluster_stars = Table(apogee[indices])
    cluster_stars["mu"] = mu

    return (cluster_stars, fluxes, flux_uncertainties)


def get_apogee_hipparcos_sample():
    """
    Prepare APOGEE Hipparcos spectra and data table in a convenient format.
    """

    stars = fits.open("APOGEE-allStar-v603-Hipparcos.fits.gz")[1].data

    QUALITY_MASK = (stars["TEFF"] > 0) * (5500 > stars["TEFF"]) \
        * (stars["ASPCAPFLAG"] == 0)

    good_stars = Table(stars[QUALITY_MASK])
    
    # Load the first image to get pixel information.
    with fits.open("APOGEE/aspCap/HIP{}.fits".format(good_stars["HIP"][0])) \
    as image:
        N_pixels = image[1].data.size

    N_stars = len(good_stars)

    # Should really do this as a memory map first, but since it's small and we 
    # need to check for bad pixels, let's just use arrays.
    fluxes = np.zeros((N_stars, N_pixels))
    flux_uncertainties = np.zeros((N_stars, N_pixels))

    # Fill with default (bad) values.
    for i, star in enumerate(good_stars):
        print("{0}/{1}: HIP{2}".format(i + 1, N_stars, star["HIP"]))

        with fits.open("APOGEE/aspCap/HIP{}.fits".format(star["HIP"])) as im:
            fluxes[i, :] = np.atleast_2d(im[1].data)[0, :].copy()
            flux_uncertainties[i, :] = np.atleast_2d(im[2].data)[0, :].copy()

    # Do QC on the arrays:
    # - zeros in all fluxes --> delete pixel
    # - zero/negative flux uncertainties --> set to large values.
    # - non-finite or zero fluxes --> set to 1 and set uncertainty as large

    zero_pixels_everytime = np.all(0 >= fluxes, axis=0)
    fluxes = fluxes[:, ~zero_pixels_everytime]
    flux_uncertainties = flux_uncertainties[:, ~zero_pixels_everytime]

    unphysical_uncertainties = (0 >= flux_uncertainties) \
        + ~np.isfinite(flux_uncertainties)
    flux_uncertainties[unphysical_uncertainties] = 1e8

    unphysical_fluxes = (0 >= fluxes) + ~np.isfinite(fluxes)
    fluxes[unphysical_fluxes] = 1.0
    flux_uncertainties[unphysical_fluxes] = 1e8

    # Calculate distance moduli.
    good_stars["mu"] = 5 * np.log10(1000./(good_stars["Plx"])) - 5

    return (good_stars, fluxes, flux_uncertainties)


def save_sample(stars, fluxes, flux_uncertainties, prefix, clobber=True):
    """
    Save a table and data arrays to disk.
    """

    stars.write("{}.fits.gz".format(prefix), overwrite=clobber)

    if (os.path.exists("{}-flux.memmap".format(prefix)) \
    or os.path.exists("{}-flux-uncertainties.memmap".format(prefix))) \
    and not clobber:
        raise IOError("file exists")

    fluxes_memmap = np.memmap("{}-flux.memmap".format(prefix), mode="w+",
        dtype=float, shape=fluxes.shape)
    flux_uncertainties_memmap = np.memmap("{}-flux-uncertainties.memmap".format(
        prefix), mode="w+", dtype=float, shape=flux_uncertainties.shape)
    fluxes_memmap[:] = fluxes[:]
    flux_uncertainties_memmap[:] = flux_uncertainties[:]
    del fluxes_memmap, flux_uncertainties_memmap

    return True


if __name__ == "__main__":

    # Hipparcos (is saaaaah hipstar)
    hipstars, hipfluxes, hipflux_uncertainties = get_apogee_hipparcos_sample()
    save_sample(hipstars, hipfluxes, hipflux_uncertainties, "APOGEE-Hipparcos")

    # Clusters
    clstars, clfluxes, clflux_uncertainties = get_apogee_cluster_sample()
    save_sample(clstars, clfluxes, clflux_uncertainties, "APOGEE-Clusters")

    # Both.
    clstars["SAMPLE"] = "CL"
    hipstars["SAMPLE"] = "HIP"

    del hipstars["PARAM_COV"], hipstars["FPARAM_COV"]

    common = set(hipstars.dtype.names).intersection(clstars.dtype.names)
    for column, (dtype, _) in hipstars.dtype.fields.items():
        if column not in common:
            _ = hipstars[column].copy()[:len(clstars)]
            _.data[:] = np.nan
            clstars[column] = _
        
    both_samples = vstack([hipstars, clstars])
    fluxes = np.vstack([hipfluxes, clfluxes])
    flux_uncertainties = np.vstack([hipflux_uncertainties, clflux_uncertainties])
    save_sample(both_samples, fluxes, flux_uncertainties, "APOGEE-Clusters+Hipparcos")

    
