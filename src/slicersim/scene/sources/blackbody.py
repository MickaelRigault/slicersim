"""This module handles blackbody sources."""

import warnings

from astropy import units
from astropy.cosmology import Planck18 as cosmology
from astropy.modeling.models import BlackBody

from sncosmo import Spectrum, get_bandpass

_FLAMBDA_units = units.erg / (units.cm ** 2 * units.s * units.AA)

def get_blackbody_flux(lbda, temperature, mag=None,
                        magabs=None, redshift=0,
                        cosmology=cosmology,
                        band="sdssr", magsys="ab"):
    """Get the flux of a blackbody source.

    The Planck function is evaluated at the rest-frame wavelengths
    (``lbda / (1 + redshift)``) and rescaled either to the observed
    magnitude ``mag`` (observer-frame spectrum in ``band``) or to the
    absolute magnitude ``magabs`` (rest-frame spectrum in ``band``, i.e.
    K-corrected, as ``sncosmo.Model.set_source_peakabsmag``).
    The normalization does not depend on ``lbda``.

    Parameters
    ----------
    lbda : array_like or astropy.units.Quantity
        Observer-frame wavelength array. Assumed to be in Angstrom if it
        carries no unit.
    temperature : float
        Temperature of the blackbody in K.
    mag : float or None, optional
        Target (observed) magnitude in the given band. Default is None.
    magabs : float or None, optional
        Target absolute magnitude in the rest-frame ``band``, used if ``mag``
        is None. The observed flux is then the rest-frame spectrum at 10 pc,
        dimmed by ``distmod(redshift)`` and by ``(1 + redshift)`` (flux
        density per unit wavelength). Default is None.
    redshift : float, optional
        Redshift of the source. With ``mag``, only the wavelengths are
        stretched (the dimming is in ``mag``). With ``magabs``, it must be
        strictly positive. Default is 0.
    cosmology : astropy.cosmology.Cosmology, optional
        Cosmology used to convert ``magabs``. Default is Planck18.
    band : str, optional
        Name of the bandpass (must be known by ``sncosmo``).
        Default is ``"sdssr"``.
    magsys : str, optional
        Name of the magnitude system (see ``sncosmo``). Default is ``"ab"``.

    Returns
    -------
    numpy.ndarray
        The blackbody flux in erg/s/cm^2/A, matching the shape of ``lbda``.

    Notes
    -----
    The normalisation is purely photometric: the blackbody shape only depends
    on ``temperature`` while the amplitude is set by ``mag``. The solid angle
    is therefore not a free parameter.

    Examples
    --------
    >>> import numpy as np
    >>> lbda = np.linspace(4000, 9000, 100)
    >>> flux = get_blackbody_flux(lbda, temperature=6000, mag=20)
    """
    if not hasattr(lbda, 'unit'):  # assumed Angstrom
        lbda = units.Quantity(lbda, units.AA)

    blackbody = BlackBody(temperature=temperature * units.K)

    def _get_flux(lbda_):
        """ blackbody flux [erg/s/cm2/A] (whatever mag) at the rest-frame lbda_ [AA] """
        lbda_ = units.Quantity(lbda_, units.AA)
        flux_nu = blackbody(lbda_) * units.sr  # rm sr in unit
        return flux_nu.to(_FLAMBDA_units, units.spectral_density(lbda_)).value

    # observed flux with whatever magnitude.
    flux = _get_flux(lbda.to_value(units.AA) / (1 + redshift))
    band_wave = get_bandpass(band).wave

    if mag is None:
        if magabs is None:
            raise ValueError("either mag or magabs must be given.")
        # absolute magnitude: rest-frame spectrum at 10pc has magabs in band,
        # then it is dimmed by the distance modulus and by (1+z) [per Angstrom].
        rest_mag = Spectrum(wave=band_wave, flux=_get_flux(band_wave)
                            ).bandmag(band, magsys)
        fluxcoef = 10 ** (-0.4 * (magabs + cosmology.distmod(redshift).value - rest_mag)
                          ) / (1 + redshift)
    else:
        if magabs is not None:
            warnings.warn("mag and magabs have been specified. magabs ignored.")
        # the "whatever mag", computed on the band's own wavelength grid
        native_mag = Spectrum(wave=band_wave, flux=_get_flux(band_wave / (1 + redshift))
                              ).bandmag(band, magsys)
        fluxcoef = 10 ** (-0.4 * (mag - native_mag))

    return flux * fluxcoef  # numpy array
