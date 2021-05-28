import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt
from astropy import units as u

R0 = 8.5*u.kpc
v0 = 220*u.km/u.s
w0 = v0/R0.to(u.km)

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

sigclip = SigmaClip(sigma_upper=5, sigma_lower=np.inf, maxiters=1, cenfunc=np.mean, stdfunc=np.std)
filtered_data = sigclip(data, axis=2)
iv = filtered_data.mask.argmax(axis=2)
for il in range(len(l)):
    v_tan = np.zeros_like(b)
    for ib in range(len(b)):
        i = iv[ib][il]
        v_tan[ib] = v[i] if i else np.nan
    i = np.nanargmin(v_tan)
    v_maximorum[il] = v_tan[i]
    b_v_maximorum[il] = b[i]

vtan = v0*abs(np.sin(l*np.pi/180.))+v_maximorum*np.sign(np.sin(l*np.pi/180.))*u.km/u.s
R = abs(R0*np.sin(l*np.pi/180.))
