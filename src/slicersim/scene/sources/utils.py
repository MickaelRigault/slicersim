""" Sources: tools shared by the source models. """

import numpy as np
from astropy.cosmology import Planck18 as cosmology


def obsmag_to_redshift(mag, magabs,
                       redshift_scan="0.001:3:0.01",
                       cosmo=cosmology):
    """Linearly interpolate apparent magnitude into redshift given an absolute magnitude.

    The distance modulus is tabulated on a redshift ramp and the requested
    apparent magnitude(s) are linearly interpolated back into redshift.

    Parameters
    ----------
    mag : float or array_like
        Apparent magnitude(s).
    magabs : float
        Absolute magnitude.
    redshift_scan : str, optional
        Redshift ramp (argument to `numpy.r_`). Default is "0.001:3:0.01".
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology model. Default is `astropy.cosmology.Planck18`.

    Returns
    -------
    float or numpy.ndarray
        Linearly interpolated redshift(s). Same shape as ``mag``.

    Notes
    -----
    Magnitudes outside the range spanned by ``redshift_scan`` are clipped to
    the ramp boundaries by `numpy.interp`, they are not extrapolated.

    Examples
    --------
    >>> obsmag_to_redshift(22., -19.3)  # doctest: +SKIP
    0.19...
    """
    zz = eval(f"np.r_[{redshift_scan}]")
    obsmag = cosmo.distmod(zz).value + magabs

    # Linear interpolation
    return np.interp(mag, obsmag, zz)
