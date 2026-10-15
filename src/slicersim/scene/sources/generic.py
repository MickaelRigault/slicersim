"""This module handles generic analytic sources (e.g. power law)."""

import numpy as np

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
