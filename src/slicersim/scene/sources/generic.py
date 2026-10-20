"""This module handles generic analytic sources (e.g. power law)."""

import warnings
import numpy as np
from astropy.cosmology import Planck18 as cosmology

from sncosmo import Spectrum
# internal function
#
def generic_get_flux(lbda, lbda_ref, flux_ref,
                    magobs=None, magabs=None,
                    band="sdssr",
                    redshift=0.1, cosmology=cosmology):
    """Internal model function: spectrum modulated by the Sersic profile.

    Parameters
    ----------
    lbda : array_like
        Wavelength array in Angstrom.
    mag : float or None
        Target magnitude. If None, the reference spectrum is used
        as-is (no rescaling).
    band : str
        Bandpass name.
    r_eff : float, optional
        Effective (half-light) radius, in arcsec. Default is 1.
    n : float, optional
        Sersic index. Default is 1.0.
    ellip : float, optional
        Ellipticity, 1 - b/a. Default is 0.0.
    theta : float, optional
        Position angle, in degrees. Default is 0.0.
    pixel_scale : float, optional
        Arcsec per pixel of the output grid. Default is 0.1.
    shape : tuple of int, optional
        Output cube spatial shape as ``(nx, ny)``. Default is ``(10, 10)``.
    position : tuple of float, optional
        Centroid offset ``(dx, dy)`` in pixels. Default is None.
    oversample : int, optional
        Oversampling factor per axis. Default is 4.

    Returns
    -------
    numpy.ndarray
        3D datacube of shape ``(len(lbda), ny, nx)``: the flux at each
        wavelength scaled by the (flux-normalized) Sersic profile.
    """
    if magabs is None and magobs is None:
        flux_ratio = 1
    else:
        if magobs is None: # but magabs given
            magobs = magabs + cosmology.distmod(redshift).value
        elif magabs is not None:
            warnings.warn("both magobs and magabs are given. magobs is used.")

        in_mag = Spectrum(lbda_ref*(1+redshift), flux_ref).bandmag(band, "ab")
        flux_ratio = 10 ** (-0.4 * (magobs - in_mag))

    flux_ = np.interp(lbda, lbda_ref*(1+redshift), flux_ref,
                      left=np.nan, right=np.nan)
    flux_ *= flux_ratio

    return flux_

def get_powerlaw_flux(lbda, alpha, mag, lambda_ref=5500.,
                      band="sdssr", magsys="ab"):
    r"""Get the flux of a power-law source.

    The flux density is :math:`F_\lambda \propto (\lambda/\lambda_{ref})^\alpha`,
    rescaled so that the resulting spectrum has the requested magnitude in
    the given band and magnitude system.

    Parameters
    ----------
    lbda : array_like
        Wavelength array in Angstrom.
    alpha : float
        Power-law index of :math:`F_\lambda` (e.g. -1 falls toward the red).
    mag : float
        Target magnitude in the given band.
    lambda_ref : float, optional
        Reference wavelength in Angstrom. Default is 5500.
    band : str, optional
        Name of the bandpass (must be known by ``sncosmo``).
        Default is ``"sdssr"``.
    magsys : str, optional
        Name of the magnitude system (see ``sncosmo``). Default is ``"ab"``.

    Returns
    -------
    numpy.ndarray
        The power-law flux in erg/s/cm^2/A, matching the shape of ``lbda``.

    Notes
    -----
    Since ``lambda_ref`` only sets the normalization of the unscaled
    power law, it has no effect on the output once rescaled to ``mag``.

    Examples
    --------
    >>> import numpy as np
    >>> lbda = np.linspace(4000, 9000, 100)
    >>> flux = get_powerlaw_flux(lbda, alpha=-2, mag=20)
    """
    from sncosmo import Spectrum

    lbda = np.asarray(lbda, dtype=float)  # assumed Angstrom
    # flux with whatever magnitude.
    flux = (lbda / lambda_ref) ** alpha

    # let's get it to the target mag using sncosmo
    native_mag = Spectrum(wave=lbda, flux=flux).bandmag(band, magsys)
    fluxcoef_for_target_mag = 10 ** (-0.4 * (mag - native_mag))

    return flux * fluxcoef_for_target_mag  # numpy array
