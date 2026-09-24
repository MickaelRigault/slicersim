""" This module handles Blackbody sources """

from astropy import units
from astropy.modeling.models import BlackBody
from sncosmo import Spectrum

_FLAMBDA_units = units.erg / (units.cm ** 2 * units.s * units.AA)

def  get_blackbody_pointsource(temperature=6000, mag=20,
                                band="sdssr", magsys="ab",
                                position=[1, 0.5],
                                **kwargs):
    """Initialize the LazuliBlackBody.

    Parameters
    ----------
    temperature : float, optional
        Temperature of the blackbody in Kelvin. Default is 6000.
    mag : float, optional
        Target magnitude in the given band. Default is 20.
    band : str, optional
        Name of the bandpass (from sncosmo). Default is "sdssr".
    magsys : str, optional
        Name of the magnitude system (see sncosmo). Default is "ab".
    position : list, optional
        Position in the MLA in spaxels. Default is [1, 0.5].
    background : str or dict, optional
        Background to use. Default is "zodi".
    **kwargs
        Goes to `simulation.Simulation.from_config()`.
    """
    # build the scene config | background (str or dict) is merged in by get_config
    return {"name": "blackbody",
            "source": "blackbody",
            "temperature": temperature,
            "mag": mag,
            "band": band,
            "magsys": magsys,
            "position": position} | kwargs

def get_blackbody_flux(lbda, temperature, mag,
                       band="sdssr", magsys="ab"):
    """Get the flux of a blackbody source.

    The Planck function is evaluated at the requested wavelengths and rescaled
    so that the resulting spectrum has the requested magnitude in the given
    band and magnitude system.

    Parameters
    ----------
    lbda : array_like or astropy.units.Quantity
        Wavelength array. Assumed to be in Angstrom if it carries no unit.
    temperature : float
        Temperature of the blackbody in Kelvin.
    mag : float
        Target magnitude in the given band.
    band : str, optional
        Name of the bandpass (must be known by `sncosmo`). Default is "sdssr".
    magsys : str, optional
        Name of the magnitude system (see `sncosmo`). Default is "ab".

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
    flux_nu = blackbody(lbda) * units.sr  # rm sr in unit

    # flux with whatever magnitude.
    flux = flux_nu.to(_FLAMBDA_units, units.spectral_density(lbda))

    # let's get it to the target mag using sncosmo
    spec_in = Spectrum(wave=lbda.value, flux=flux.value)
    # the "whatever mag"
    native_mag = spec_in.bandmag(band, magsys)
    # conver the flux to the good amplitude
    fluxcoef_for_target_mag = 10 ** (-0.4 * (mag - native_mag))

    return flux.value * fluxcoef_for_target_mag  # numpy array
