import numpy as np
from astropy.io import fits


def get_values_from(header: fits.header.Header, axis: int) -> np.ndarray:
    """Get real coordinate values from `header` at `axis` instead of pixels."""
    naxis = header[f"NAXIS{axis}"]  # number of pixels on `axis`
    crpix = header[f"CRPIX{axis}"]  # reference pixel for `axis`
    crval = header[f"CRVAL{axis}"]  # coordinate value at `crpix`
    cdelt = header[f"CDELT{axis}"]  # pixel spacing for `axis`
    return crval + (1 - crpix + np.arange(naxis)) * cdelt


cube = fits.open("../southgal_fixbadc.fits")
data = cube[0].data
header = cube[0].header

vel = get_values_from(header, 1)  # VELO-LSR
lon = get_values_from(header, 2)  # GLON-FLT
lat = get_values_from(header, 3)  # GLAT-FLT
