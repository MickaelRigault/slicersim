"""This module handles generic analytic sources (e.g. power law)."""

import warnings
import numpy as np
from astropy.cosmology import Planck18 as cosmology

from sncosmo import Spectrum
# internal function
#
def generic_get_flux(lbda, lbda_ref, flux_ref,
                    mag=None, magabs=None,
                    band="sdssr",
                    redshift=0.1, cosmology=cosmology):
    """Get the flux of a reference spectrum, redshifted and normalized to a magnitude.

    Parameters
    ----------
    lbda : array_like
        Observer-frame wavelength array in Angstrom.
    lbda_ref : array_like
        Rest-frame wavelength array of the reference spectrum, in Angstrom.
    flux_ref : array_like
        Flux of the reference spectrum, sampled at ``lbda_ref``, in
        erg/s/cm^2/A.
    mag : float or None, optional
        Target (observed) AB magnitude in ``band``. Default is None.
    magabs : float or None, optional
        Target absolute AB magnitude in the rest-frame ``band``, used if
        ``mag`` is None: the reference (rest-frame) spectrum is scaled to
        ``magabs`` at 10 pc, then dimmed by ``distmod(redshift)`` and by
        ``(1 + redshift)`` (flux density per unit wavelength), i.e. it is
        K-corrected, as ``sncosmo.Model.set_source_peakabsmag``.
        If both ``mag`` and ``magabs`` are None, the reference spectrum is
        used as is (no rescaling). Default is None.
    band : str, optional
        Bandpass name (must be known by ``sncosmo``). Default is ``"sdssr"``.
    redshift : float, optional
        Redshift applied to the reference wavelengths. With ``mag``, it is a
        wavelength stretch only (the dimming is in ``mag``). Default is 0.1.
    cosmology : astropy.cosmology.Cosmology, optional
        Cosmology used to convert ``magabs``. Default is Planck18.

    Returns
    -------
    numpy.ndarray
        The flux in erg/s/cm^2/A at ``lbda``, NaN outside of the
        (redshifted) reference wavelength range.

    Warns
    -----
    UserWarning
        If both ``mag`` and ``magabs`` are given (``magabs`` is then ignored).
    """
    if magabs is None and mag is None:
        flux_ratio = 1

    elif mag is None: # but magabs given
        # rest-frame spectrum at 10pc has magabs in band,
        # then it is dimmed by the distance modulus and by (1+z) [per Angstrom].
        rest_mag = Spectrum(lbda_ref, flux_ref).bandmag(band, "ab")
        flux_ratio = 10 ** (-0.4 * (magabs + cosmology.distmod(redshift).value - rest_mag)
                            ) / (1 + redshift)
    else:
        if magabs is not None:
            warnings.warn("mag and magabs have been specified. magabs ignored.")

        in_mag = Spectrum(lbda_ref*(1+redshift), flux_ref).bandmag(band, "ab")
        flux_ratio = 10 ** (-0.4 * (mag - in_mag))

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
