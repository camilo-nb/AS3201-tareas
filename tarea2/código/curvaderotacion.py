#!/usr/bin/python3

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt
from astropy import units as u

R0 = 8.5*u.kpc
v0 = 220*u.km/u.s

def get_values_from(header: fits.header.Header, axis: int) -> np.ndarray:
    """Get real coordinate values from `header` at `axis` instead of pixels."""
    naxis = header[f"NAXIS{axis}"]  # number of pixels on `axis`
    crpix = header[f"CRPIX{axis}"]  # reference pixel for `axis`
    crval = header[f"CRVAL{axis}"]  # coordinate value at `crpix`
    cdelt = header[f"CDELT{axis}"]  # pixel spacing for `axis`
    return crval + (1 - crpix + np.arange(naxis)) * cdelt

cube = fits.open("../southgal_fixbadc.fits")
data = cube[0].data  # indexing: [galactic latitude, galactic longitude, velocity]
header = cube[0].header

v = get_values_from(header, 1)  # VELO-LSR
l = get_values_from(header, 2)  # GLON-FLT
b = get_values_from(header, 3)  # GLAT-FLT

v_maximorum = np.zeros_like(l)
b_v_maximorum = np.zeros_like(l)

sigclip = SigmaClip(sigma_lower=5, sigma_upper=5, maxiters=None, cenfunc=np.mean, stdfunc=np.std)
T_noise = sigclip(data, axis=2)
rms = np.sqrt(np.mean(T_noise**2, axis=2))
T_rms = np.repeat(rms[..., np.newaxis], len(v), axis=2)
iv_terminal = (data > 5*T_rms).argmax(axis=2)
for il in range(len(l)):
    v_terminal = np.zeros_like(b)
    for ib in range(len(b)):
        i = iv_terminal[ib][il]
        v_terminal[ib] = v[i] if i else np.nan
    i = np.nanargmin(v_terminal)
    v_maximorum[il] = v_terminal[i]
    b_v_maximorum[il] = b[i]

vrot = v0*np.sin(l*np.pi/180.)+v_maximorum*u.km/u.s
R = R0*np.sin(l*np.pi/180.)

if __name__ == "__main__":
    
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.plot(v, data[14][115], c='k', lw=0.5)
    ax.axhline(5*rms[14][115], c='k', lw=0.5, ls="--")
    ax.axvline(v[iv_terminal[14][115]], c='k', lw=0.5, ls="--")
    ax.set_xlabel(r"$v$ [km/s]")
    ax.set_ylabel(r"$T$ [K]")
    ax.text(0.1, 0.9, f"$l=${l[115]:.2f}", ha='left', va='center', transform=ax.transAxes)
    ax.text(0.1, 0.85, f"$b=${b[14]:.2f}", ha='left', va='center', transform=ax.transAxes)
    ax.text(0.1, 0.8, f"$v=${v[iv_terminal[14][115]]:.2f}", ha='left', va='center', transform=ax.transAxes)
    ax.yaxis.set_tick_params(rotation=90)
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig("../informe/rsc/vterminal.pdf")
    plt.show()
    
    w0 = v0/R0.to(u.km)

    w = w0+vrot/R.to(u.km)

    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.plot(-R, vrot, c='k', lw=0.5)
    ax.set_xlabel(r"$-R_{\odot}\sin\,l$ [kpc]")
    ax.set_ylabel(r"$v_\mathrm{rot}$ [km/s]")
    ax.yaxis.set_tick_params(rotation=90)
    ax.tick_params(direction="in", top=False, right=True)
    axl = ax.secondary_xaxis("top", functions=(lambda R: (180./np.pi*np.arcsin(-R/R0.value)-360)%360, lambda l: -R0.value*np.sin(l*np.pi/180.)))
    axl.set_xlabel(r"$l$ [°]")
    axl.tick_params(direction="in", top=True)
    fig.savefig("../informe/rsc/vrot.pdf")
    plt.show()

    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.plot(-R, w, c='k', lw=0.5)
    ax.set_xlabel(r"$-R_{\odot}\sin\,l$ [kpc]")
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.yaxis.set_tick_params(rotation=90)
    ax.tick_params(direction="in", top=False, right=True)
    axl = ax.secondary_xaxis("top", functions=(lambda R: (180./np.pi*np.arcsin(-R/R0.value)-360)%360, lambda l: -R0.value*np.sin(l*np.pi/180.)))
    axl.set_xlabel(r"$l$ [°]")
    axl.tick_params(direction="in", top=True)
    fig.savefig("../informe/rsc/w.pdf")
    plt.show()
