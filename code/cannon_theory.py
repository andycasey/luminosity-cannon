#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate theoretical EWs to see how flexible our SP-EW model needs to be.
"""

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"


import cPickle as pickle
from astropy.table import Table

import scipy.optimize as op
import oracle

DATA_PREFIX = "../data/APOGEE-Hipparcos"

stars = Table.read("{}.fits.gz".format(DATA_PREFIX))

atomic_lines = Table(np.array([
    [15194.492, 26.0, 2.223, -4.779],
    [15207.526, 26.0, 5.385, +0.080],
    [15395.718, 26.0, 5.620, -0.341],
    [15490.339, 26.0, 2.198, -4.807],
    [15648.510, 26.0, 5.426, -0.701],
    [15964.867, 26.0, 5.921, -0.128],
    [16040.657, 26.0, 5.874, +0.066],
    [16153.247, 26.0, 5.351, -0.743],
    [16165.032, 26.0, 6.319, +0.723]
    ]), names=["wavelength", "species", "excitation_potential", "loggf"])



equivalent_widths = np.arange(10, 300, 10)

abundances = np.zeros((len(stars), len(atomic_lines), equivalent_widths.size))

import matplotlib.pyplot as plt



for i, star in enumerate(stars[:N]):
    print("Star {0}/{1}".format(i, len(stars)))

    for j, equivalent_width in enumerate(equivalent_widths):

        atomic_lines["equivalent_width"] = equivalent_width
        abundances[i, :, j] = oracle.synthesis.moog.atomic_abundances(
            atomic_lines,
            [star["TEFF"], star["LOGG"], star["PARAM_M_H"]],
            1.0, photosphere_kwargs={"kind": "Castelli/Kurucz"})

    x = np.log(np.repeat(equivalent_widths, len(atomic_lines))/np.tile(atomic_lines["wavelength"], len(equivalent_widths)))
    y = abundances[0, :, :].T.flatten()
    fig, ax = plt.subplots()
    color = np.tile(np.log(np.exp(atomic_lines["loggf"]) * atomic_lines["wavelength"]), len(equivalent_widths)).flatten() \
        - (5040. * np.tile(atomic_lines["excitation_potential"], len(equivalent_widths)).flatten())/star["TEFF"]
    scat = ax.scatter(x, y, c=color)
    plt.colorbar(scat)

    raise a



def c_atomic(name):
    c = np.zeros(abundances.shape)

    for i in range(N):
        for j in range(len(atomic_lines)):
            for k in range(len(equivalent_widths)):
                c[i, j, k] = atomic_lines[name][j]
    return c

def c_star(name):
    c = np.zeros(abundances.shape)
    for i in range(N):
        for j in range(len(atomic_lines)):
            for k in range(len(equivalent_widths)):
                c[i, j, k] = stars[name][i]
    return c


# Fit them all.
def f(xdata, a, b,c,d):# c, d, e, f):
    return np.dot([a, b, c, d], xdata)

#xdata = np.array([self._labels[n] for n in ("TEFF", "LOGG", "CA_H")])
"""
xdata = np.array([
    np.ones(N * len(equivalent_widths) * len(atomic_lines)),
    np.repeat(np.repeat(stars["TEFF"][:N], len(equivalent_widths)), len(atomic_lines)),
    np.repeat(np.repeat(stars["LOGG"][:N], len(equivalent_widths)), len(atomic_lines)),
    np.repeat(np.repeat(stars["FE_H"][:N], len(equivalent_widths)), len(atomic_lines))
    ])
"""
M = 4
xdata = np.ones((M, N, eqw.shape[1], eqw.shape[2]))

ydata = eqw[:N].flatten()

for i, star in enumerate(stars[:N]):
    for j, atomic_line in enumerate(atomic_lines):
        for k, equivalent_width in enumerate(equivalent_widths):
#            xdata[1, i, j, k] = star["TEFF"]**3
#            xdata[2, i, j, k] = star["TEFF"]**2
            xdata[2, i, j, k] = star["LOGG"]
            xdata[3, i, j, k] = star["TEFF"]
            xdata[1, i, j, k] = abundances[i, j, k]

xdata = xdata.reshape(M, -1)

p_opt, p_cov = op.curve_fit(f, xdata, ydata,
    p0=np.hstack([np.zeros(xdata.shape[0] - 1), 1]))

fig, ax = plt.subplots()
model = f(xdata, *p_opt)
scat = ax.scatter(ydata - model, c_star("TEFF")[:N].flatten())
#plt.colorbar(scat)
#ax.set_xlabel("Expected EW")
#ax.set_ylabel("Inferred EW")



