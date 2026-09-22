""" This module handles Supernovae sources """

import warnings

import numpy as np
from astropy.cosmology import Planck18 as cosmology

from .utils import obsmag_to_redshift

try:
    from twins_embedding import TwinsEmbeddingModel
    twins_embedding_model = TwinsEmbeddingModel()
except ImportError:
    twins_embedding_model = None


def get_snia_pointsource(model="salt", **kwargs):
    """Get a generic configuration for a SN Ia point source.

    Parameters
    ----------
    model : str, optional
        Name of the SN Ia model to use.
        - "salt" (or any sncosmo salt source name).
        - "twin"
        Default is "salt".
    **kwargs
        Additional parameters to update the configuration.

    Returns
    -------
    dict
        Configuration dictionary for a SN Ia point source, ready to be passed
        to `~slicersim.scene.pointsource.PointSource.from_config`.

    Raises
    ------
    NotImplementedError
        If ``model`` matches neither the "salt" nor the "twin" family.

    See Also
    --------
    slicersim.scene.get_sn_scene : Build a full scene around a SN Ia.

    Examples
    --------
    >>> get_snia_pointsource(model="salt", redshift=0.5)  # doctest: +SKIP
    {'name': 'SN Ia', 'redshift': 0.5, ...}
    """
    generic = {'name': 'SN Ia',
               'redshift': 1.5,
               'phase': 0, 'position': [1, 0.5]}

    if "salt" in model.lower():
        if model == "salt":
            model = "salt2-extended"

        pointsource = {'source': model, 'MBmax': -19.3, 'c': 0., 'x1': 0.}

    elif "twin" in model.lower():
        pointsource = {'magnitude': 0., 'color': 0., 'coordinates': (0., 0., 0.)}

    else:
        raise NotImplementedError(f"no SN Ia configuration defined for model: {model}")

    return generic | pointsource | kwargs


def get_saltmodel(redshift=0.1,
                  MBmax=-19.3, source="salt2-extended", cosmo=cosmology,
                  x1=0, c=0, alpha=-0.14, beta=3.15):
    """Get a sncosmo SALT2 SN Ia model.

    The returned model is monkey-patched with an additional method `get_flux`
    which returns null fluxes outside the valid spectral domain of the model.

    Parameters
    ----------
    redshift : float, optional
        Redshift of the SN Ia. Default is 0.1.
    MBmax : float, optional
        Peak absolute Bessell-B AB-magnitude. Default is -19.3.
    source : str, optional
        Model source name (see `sncosmo`). Default is "salt2-extended".
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology model. Default is `astropy.cosmology.Planck18`.
    x1 : float, optional
        SALT2 stretch parameter. Default is 0.
    c : float, optional
        SALT2 color parameter. Default is 0.
    alpha : float, optional
        Stretch standardization factor. Default is -0.14.
    beta : float, optional
        Color standardization factor. Default is 3.15.

    Returns
    -------
    sncosmo.Model
        A monkey-patched `sncosmo.Model` instance, carrying an extra
        ``get_flux(wave, time)`` method.

    Notes
    -----
    The peak absolute magnitude is standardised following the Tripp relation,
    ``eff_mbmax = MBmax + x1 * alpha + c * beta``, without environmental bias.
    """
    import sncosmo

    model = sncosmo.Model(source=source)
    model.set(z=redshift, c=c, x1=x1)  # SALT2 parameters
    eff_mbmax = MBmax + (x1 * alpha + c * beta)  # Tripp relation (no env bias)

    # set effective peak magnitude
    model.set_source_peakabsmag(eff_mbmax,
                                "bessellb", "AB", cosmo=cosmo)

    def get_flux(wave, time):
        """Return the model flux, filling zeros outside the model wavelength range.

        Parameters
        ----------
        wave : array_like
            Wavelengths in Angstroms at which to evaluate the flux.
        time : float
            Rest-frame phase in days relative to peak brightness.

        Returns
        -------
        numpy.ndarray
            Flux array in erg/s/cm²/Å, with zeros where ``wave`` falls
            outside the model's valid wavelength range.
        """
        wmin, wmax = model.minwave(), model.maxwave()
        wave = np.atleast_1d(wave)
        sel = (wave > wmin) & (wave < wmax)
        flux = np.zeros_like(wave)
        flux[sel] = model.flux(time, wave[sel])

        return flux

    model.get_flux = get_flux  # Monkey patching
    return model


# explicit here the parameters to enable mutable_parameters parsing
def get_saltmodel_flux(lbda, phase,
                       abmag=None,  # extra
                       redshift=0.1,
                       MBmax=-19.3, source="salt2-extended", cosmo=cosmology,
                       x1=0, c=0, alpha=-0.14, beta=3.15):
    """Get the flux of a SALT model.

    Parameters
    ----------
    lbda : array_like
        Wavelength array in Angstrom.
    phase : float
        Phase of the SN Ia.
    abmag : float, optional
        Apparent magnitude. If given, `redshift` is ignored and derived from `abmag`.
        Default is None.
    redshift : float, optional
        Redshift of the SN Ia. Default is 0.1.
    MBmax : float, optional
        Peak absolute Bessell-B AB-magnitude. Default is -19.3.
    source : str, optional
        Model source name (see `sncosmo`). Default is "salt2-extended".
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology model. Default is `astropy.cosmology.Planck18`.
    x1 : float, optional
        SALT2 stretch parameter. Default is 0.
    c : float, optional
        SALT2 color parameter. Default is 0.
    alpha : float, optional
        Stretch standardization factor. Default is -0.14.
    beta : float, optional
        Color standardization factor. Default is 3.15.

    Returns
    -------
    numpy.ndarray
        The SALT model flux in erg/s/cm^2/A, zero outside the wavelength
        range of the model.

    Warns
    -----
    UserWarning
        If both ``abmag`` and ``redshift`` are given, since ``redshift`` is
        then ignored and re-derived from ``abmag``.

    See Also
    --------
    get_saltmodel : The underlying `sncosmo.Model`.
    """
    if abmag is not None:
        if redshift is not None:
            warnings.warn("abmag and redshift are set, redshift is ignored and derived from abmag.")

        redshift = obsmag_to_redshift(abmag, MBmax, cosmo=cosmo)

    model = get_saltmodel(redshift=redshift, MBmax=MBmax,
                          source=source, cosmo=cosmo,
                          x1=x1, c=c, alpha=alpha, beta=beta)

    return model.get_flux(lbda, phase)


def get_twins_embedding_flux(lbda, phase, redshift=0.05,
                             magnitude=0., color=0., coordinates=(0., 0., 0.),
                             cosmo=cosmology, ref_redshift=0.05,
                             norm=1e-15):
    """Get a SN Ia spectrum assuming the Twin Embedding model.

    Parameters
    ----------
    lbda : array_like
        Wavelength array in Angstrom.
    phase : float
        Rest-frame phase in days.
    redshift : float, optional
        Redshift of the simulated flux. Default is 0.05.
    magnitude : float, optional
        Magnitude offset for the twin model (dmag). Default is 0.
    color : float, optional
        Color term of the twin model (Av). Default is 0.
    coordinates : tuple, optional
        Embedding parameters for the twin model (xi). Default is (0., 0., 0.).
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology to be used to redshift the simulated target.
        Default is `astropy.cosmology.Planck18`.
    ref_redshift : float, optional
        Reference redshift. Default is 0.05.
    norm : float, optional
        Normalization of the output flux. Default is 1e-15.

    Returns
    -------
    numpy.ndarray
        The Twin Embedding model flux in erg/s/cm^2/A, interpolated onto
        ``lbda``. Values outside the model wavelength coverage are `numpy.nan`.

    Raises
    ------
    ImportError
        If the optional ``twins_embedding`` package is not installed.

    Notes
    -----
    ``twins_embedding`` is an optional dependency, it is imported at module
    load time and silently ignored if missing.
    """
    if twins_embedding_model is None:
        raise ImportError("The 'twins_embedding' module is not available.")

    flux, *_ = twins_embedding_model.evaluate(phase, magnitude, color, list(coordinates))
    wl_obs = twins_embedding_model.wave * (1. + redshift)
    dist_ratio = cosmo.luminosity_distance(ref_redshift).value / cosmo.luminosity_distance(redshift).value
    cosmo_k_corr = (1. + ref_redshift) / (1. + redshift)
    flux_obs = flux * norm * dist_ratio ** 2. * cosmo_k_corr

    return np.interp(lbda, wl_obs, flux_obs, left=np.nan, right=np.nan)
