#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate theoretical EWs to see how flexible our SP-EW model needs to be.
"""

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"


import cPickle as pickle
from astropy.table import Table

import matplotlib
matplotlib.rcParams["text.usetex"] = True
import scipy.optimize as op
import oracle

DATA_PREFIX = "../data/APOGEE-Hipparcos"

stars = Table.read("{}.fits.gz".format(DATA_PREFIX))

atomic_lines = Table(np.array([
    [15194.492, 26.0, 2.223, -4.779],
    [15207.526, 26.0, 5.385, +0.080],
    [15395.718, 26.0, 5.620, -0.341],
    [15490.339, 26.0, 2.198, -4.807],
# Missing offset is entirely attributable to loggf differences
    [15648.510, 26.0, 5.426, -0.701],
    [15964.867, 26.0, 5.921, -0.128],
    [16040.657, 26.0, 5.874, +0.066],
    [16153.247, 26.0, 5.351, -0.743],
    [16165.032, 26.0, 6.319, +0.723]
    ]), names=["wavelength", "species", "excitation_potential", "loggf"])




equivalent_widths = np.arange(10, 300, 10)

abundances = np.zeros((len(stars), len(atomic_lines), equivalent_widths.size))

import matplotlib.pyplot as plt

eqws = np.zeros(abundances.shape)
offset = np.zeros(abundances.shape)

# These are from Irwin (1980) and probably way too old..
# (they better be too old)
fe_coefficients = np.array([
    -1.15609527e+3,
    +7.46597652e+2,
    -1.92865672e+2,
    +2.49658410e+1,
    -1.61934455e+0,
    +4.21182087e-2
])

if False:
    N = 1
    for i, star in enumerate(stars[:N]):
        print("Star {0}/{1}".format(i, len(stars)))

        for j, equivalent_width in enumerate(equivalent_widths):

            atomic_lines["equivalent_width"] = equivalent_width

            eqws[i, :, j] = equivalent_width
            abundances[i, :, j] = oracle.synthesis.moog.atomic_abundances(
                atomic_lines,
                [star["TEFF"], star["LOGG"], star["PARAM_M_H"]],
                1.0, photosphere_kwargs={"kind": "Castelli/Kurucz"})

            # The rest will vary with temperature (via partition functions) and
            # surface gravity (through ionisation states).

            ln_q = -np.sum(np.polyval(fe_coefficients[::-1], np.log(star["TEFF"])))
            offset[i, :, j] = (-5040. * atomic_lines["excitation_potential"])/star["TEFF"] \
                + ln_q \
                + np.log(atomic_lines["wavelength"] * np.exp(atomic_lines["loggf"]))
#                + np.log(atomic_lines["wavelength"] * np.exp(atomic_lines["loggf"])) \

    #with open("t.pkl", "wb") as fp:
    #    pickle.dump((eqws, abundances, offset), fp, -1)
else:
    with open("t.pkl", "rb") as fp:
        eqws, abundances, offset = pickle.load(fp)



# Fit the differences in a single line across all SPs.

p = np.zeros(abundances.shape)
c = np.zeros(abundances.shape)
for i, star in enumerate(stars):
    for j, equivalent_width in enumerate(equivalent_widths):
        p[i, :, j] = np.arange(len(atomic_lines))
        c[i, :, j] = star["LOGG"]



for idx in range(9):

    #plt.scatter(eqws[:, idx, :].flatten(), abundances[:, idx, :].flatten(), c=c[:, idx, :].flatten())


    def f(xdata, a, b, c, d, e,f,g, h, i, j,k,l,m,n,o):# c, d, e, f):
        return np.dot([a, b, c,d, e, f,g, h, i, j, k,l,m,n,o], xdata)

    # Some mapping from
    # EW + SP --> Abundance
    # Abundance + SP --> EW

    xdata = np.zeros((15, len(stars), len(equivalent_widths)))
    xdata[0, :, :] = 1.
    xdata[1, :, :] = np.log(eqws[:, idx, :]/atomic_lines["wavelength"][idx])
    xdata[2, :, :] = np.log(eqws[:, idx, :]/atomic_lines["wavelength"][idx])**2
    xdata[3, :, :] = np.log(eqws[:, idx, :]/atomic_lines["wavelength"][idx])**3
    #xdata[11, :, :] = np.log(eqws[:, idx, :]/atomic_lines["wavelength"][idx])**4

    #xdata[2, :, :] = eqws[:, idx, :]**2
    #xdata[3, :, :] = eqws[:, idx, :]**3
    #xdata[4, :, :] = eqws[:, idx, :]**4

    # Notes: It's not just a function of EW (perhaps obviously)
    # O(EW) = 1 does as good as O(EW) = 4
    # REW BRINGS DOWN TO 0.04
    # AND METALLICITY COMES DOWN TO 0.039
    # MUCH BETTER TO USE REW OVER EW

    ydata = abundances[:, idx, :].flatten()

    for i, star in enumerate(stars):
        for j, equivalent_width in enumerate(equivalent_widths):
            xdata[4, i, j] = np.log(star["TEFF"]) # 0.081
            xdata[5, i, j] = np.log(star["TEFF"])**2 #equivalent_width # down to 0.079
            xdata[6, i, j] = np.log(star["TEFF"])**3 # down to 0.058
            xdata[7, i, j] = np.log(star["TEFF"]) * star["LOGG"]
            xdata[8, i, j] = np.log(star["TEFF"]) * star["PARAM_M_H"]
            xdata[9, i, j] = star["LOGG"] * star["PARAM_M_H"]
            xdata[10, i, j] = star["PARAM_M_H"]
            xdata[11, i, j] = star["LOGG"]

    #        xdata[10,i, j] = star["PARAM_M_H"]**2
    #        xdata[8, i, j] = star["TEFF"] * star["PARAM_M_H"]
            #xdata[4, i, j] = 0#star["PARAM_M_H"]
            #xdata[5, i, j] = 0#star["TEFF"]**4
            #xdata[6, i, j] = 0#star["TEFF"]*star["LOGG"]

    xdata_orig = xdata.copy()
    xdata = xdata.reshape(xdata.shape[0], -1)

    ok = xdata_orig[1, :, :] < -4.5

    xdata = xdata[:, ok.flatten()]
    ydata = ydata[ok.flatten()]
    xdata_orig = xdata_orig[:, ok]

    p_opt, p_cov = op.curve_fit(f, xdata, ydata,
        p0=np.hstack([1, np.zeros(xdata.shape[0] - 1)]))

    fig, ax = plt.subplots(1)
    model = f(xdata, *p_opt)
    scat = ax.scatter(ydata, model, c=xdata_orig[1].flatten(),
        cmap="winter")
    cbar = plt.colorbar(scat)
    limits = [
        min([ax.get_xlim()[0], ax.get_ylim()[0]]),
        max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ]
    ax.plot(limits, limits, c="#666666", zorder=-100)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_xlabel(r"$\log_\epsilon({\rm X})$ ${\rm [MOOG]}$")
    ax.set_ylabel(r"$\log_\epsilon({\rm X})$ ${\rm [Model]}$")
    cbar.set_label(r"$\log(\frac{\rm EW}{\lambda})$")
    ax.set_title(r"$\sigma = {0:.3f}$".format(np.abs(model - ydata).std()) + r" ${\rm dex}$")

    fig.tight_layout()
    fig.savefig("APOGEE-atomic-line-{0}.png".format(idx))

# HERE 
raise a

stuff = np.zeros((5, len(stars), len(equivalent_widths)))
for i, star in enumerate(stars):
    for j, equivalent_width in enumerate(equivalent_widths):
        stuff[0, i, j] = star["TEFF"]
        stuff[1, i, j] = star["LOGG"]
        stuff[2, i, j] = star["PARAM_M_H"]
        stuff[3, i, j] = equivalent_width


fig, ax = plt.subplots(4, 2)
ax = ax.flatten()
ax[0].scatter(xdata_orig[1].flatten(), ydata)
ax[1].scatter(xdata_orig[1].flatten() * stuff[0].flatten(), ydata,
    c=stuff[2].flatten())
ax[2].scatter(xdata_orig[1].flatten() * stuff[1].flatten(), ydata,
    c=stuff[0].flatten())
ax[3].scatter(xdata_orig[1].flatten() * (stuff[2].flatten() + 7.5), ydata,
    c=stuff[2].flatten())
ax[4].scatter(stuff[3].flatten() - ydata, stuff[2].flatten())
ax[5].scatter(xdata_orig[1].flatten() * stuff[0].flatten(), ydata)
scat = ax[6].scatter(xdata_orig[1].flatten(), ydata, c=offset[:, idx, :].flatten())
plt.colorbar(scat)


"""
p = np.zeros(abundances.shape)
for i, star in enumerate(stars):
    for j, equivalent_width in enumerate(equivalent_widths):
        p[i, :, j] = np.arange(len(atomic_lines))

x = np.log(np.repeat(equivalent_widths, len(atomic_lines))/np.tile(atomic_lines["wavelength"], len(equivalent_widths)))
y = abundances[0, :, :].T.flatten()
fig, ax = plt.subplots(3)
color = np.tile(np.log(np.exp(atomic_lines["loggf"]) * atomic_lines["wavelength"]), len(equivalent_widths)).flatten() \
    - (5040. * np.tile(atomic_lines["excitation_potential"], len(equivalent_widths)).flatten())/star["TEFF"]
scat = ax[0].scatter(x, y, c=offset[0, :, :].T.flatten())

z = y + offset[0, :, :].T.flatten()
scat = ax[1].scatter(x, z,
    c=np.tile(atomic_lines["loggf"], len(equivalent_widths)).flatten())
plt.colorbar(scat)

z = z.reshape(len(equivalent_widths), -1)
diffs = z[:, 1:] - z[:, 0].reshape(len(equivalent_widths), 1)
z[:, 1:] -= np.median(diffs, axis=0)

s = ax[2].scatter(x, z.flatten(), c=np.repeat(equivalent_widths, len(atomic_lines)).flatten())
plt.colorbar(s)

raise a


def c_star(name):

    p = np.zeros(abundances.shape)
    for i, star in enumerate(stars[:N]):
        for j, equivalent_width in enumerate(equivalent_widths):
            p[i, :, j] = star[name]
    return p


def c_atomic(name):

    p = np.zeros(abundances.shape)
    for i, star in enumerate(stars[:N]):
        for j, equivalent_width in enumerate(equivalent_widths):
            p[i, :, j] = atomic_lines[name]
    return p

# Fit them all.
def f(xdata, a, b,c,d):# c, d, e, f):
    return np.dot([a, b, c,d], xdata)

#xdata = np.array([self._labels[n] for n in ("TEFF", "LOGG", "CA_H")])


M = 3

# Some mapping from
# EW + SP --> Abundance
# Abundance + SP --> EW
#N = len(stars[:N])
N = 1
xdata = np.ones((M+1, N, eqws.shape[1], eqws.shape[2]))
xdata[0, :] = abundances[:N, :, :]

ydata = eqws[:N].flatten()

for i, star in enumerate(stars[:N]):
    for j, equivalent_width in enumerate(equivalent_widths):

        xdata[1, i, :, j] = np.log(star["TEFF"])
        xdata[2, i, :, j] = star["LOGG"]
        xdata[3, i, :, j] = atomic_lines["excitation_potential"]


xdata = xdata.reshape(M+1, -1)

p_opt, p_cov = op.curve_fit(f, xdata, ydata,
    p0=np.hstack([np.zeros(xdata.shape[0] - 1), 1]))

fig, ax = plt.subplots()
model = f(xdata, *p_opt)
ax.scatter(ydata, model, c=c_star("TEFF")[:N].flatten())
ax.set_title(np.abs(model - ydata).std())


plt.close("all")
fig, axes = plt.subplots(5)
rew = np.log(model/c_atomic("wavelength")[:N].flatten()) - np.log(ydata/c_atomic("wavelength")[:N].flatten())

axes[0].scatter(rew, c_atomic("excitation_potential")[:N].flatten())
axes[1].scatter(rew, c_atomic("loggf")[:N].flatten())
axes[2].scatter(rew, c_atomic("wavelength")[:N].flatten())

chi = c_atomic("excitation_potential")[:N].flatten()
loggf = c_atomic("loggf")[:N].flatten()
axes[3].scatter(rew, loggf*chi**2)

axes[4].scatter(rew, np.exp(loggf)*chi)

#plt.scatter(model - ydata, (c_atomic("excitation_potential")[:N].flatten()/c_atomic("loggf")[:N].flatten()))

#scat = ax.scatter(ydata - model, c_star("TEFF")[:N].flatten())
#plt.colorbar(scat)
#ax.set_xlabel("Expected EW")
#ax.set_ylabel("Inferred EW")
"""


