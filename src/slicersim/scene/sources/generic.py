import numpy as np

def get_powerlaw_flux(lbda, alpha, mag, lambda_ref=5500.,
                      band="sdssr", magsys="ab"):
    """Get the flux of a power-law source: F_lambda proportional to (lambda/lambda_ref)**alpha.

    Parameters
    ----------
    lbda : array_like
        Wavelength array in Angstrom.
    alpha : float
        Power-law index of F_lambda (e.g. -1 falls toward the red).
    mag : float
        Target magnitude in the given band.
    lambda_ref : float, optional
        Reference wavelength in Angstrom. Default is 5500.
    band : str, optional
        Name of the bandpass (from sncosmo). Default is "sdssr".
    magsys : str, optional
        Name of the magnitude system (see sncosmo). Default is "ab".

    Returns
    -------
    array_like
        The power-law flux in erg/s/cm^2/A.
    """
    from sncosmo import Spectrum

    lbda = np.asarray(lbda, dtype=float)  # assumed Angstrom
    # flux with whatever magnitude.
    flux = (lbda / lambda_ref) ** alpha

    # let's get it to the target mag using sncosmo
    native_mag = Spectrum(wave=lbda, flux=flux).bandmag(band, magsys)
    fluxcoef_for_target_mag = 10 ** (-0.4 * (mag - native_mag))

    return flux * fluxcoef_for_target_mag  # numpy array
