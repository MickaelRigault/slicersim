""" Scene: pointsource module """

import warnings

import numpy as np
from astropy import units
from astropy.cosmology import Planck18 as cosmology

from .base import SceneElement

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
        Configuration dictionary for a SN Ia point source.
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

    return generic | pointsource | kwargs


# ==================== #
#  Top level shortcut  #
# ==================== #
def source_to_modelfunc(source):
    """Get the model function from a source name.

    Parameters
    ----------
    source : str
        Name of the source model.
        - "salt"
        - "twins-embedding"
        - "blackbody"

    Returns
    -------
    callable
        The model function for the given source.
    """
    if "salt" in source:
        model_func = get_saltmodel_flux

    elif source == 'twins-embedding':
        model_func = get_twins_embedding_flux

    elif source == "blackbody":
        model_func = get_blackbody_flux

    else:
        raise NotImplementedError(f"no model_func defined for source: {source}")

    return model_func


def obsmag_to_redshift(mag, magabs,
                       redshift_scan="0.001:3:0.01",
                       cosmo=cosmology):
    """Linearly interpolate apparent magnitude into redshift given an absolute magnitude.

    Parameters
    ----------
    mag : float or array_like
        Apparent magnitude(s).
    magabs : float
        Absolute magnitude.
    redshift_scan : str, optional
        Redshift ramp (argument to `np.r_`). Default is "0.001:3:0.01".
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology model. Default is `astropy.cosmology.Planck18`.

    Returns
    -------
    float or array_like
        Linearly interpolated redshift(s).
    """

    zz = eval(f"np.r_[{redshift_scan}]")
    obsmag = cosmo.distmod(zz).value + magabs

    # Linear interpolation
    return np.interp(mag, obsmag, zz)


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
        A monkey-patched `sncosmo.Model` instance.
    """
    import sncosmo

    model = sncosmo.Model(source=source)
    model.set(z=redshift, c=c, x1=x1)  # SALT2 parameters
    eff_mbmax = MBmax + (x1 * alpha + c * beta)  # Tripp relation (no env bias)

    # set effective peak magnitude
    model.set_source_peakabsmag(eff_mbmax,
                                "bessellb", "AB", cosmo=cosmo)

    def get_flux(wave, time):  # make sure get_flux exists.

        wmin, wmax = model.minwave(), model.maxwave()
        wave = np.atleast_1d(wave)
        sel = (wave > wmin) & (wave < wmax)
        flux = np.zeros_like(wave)
        flux[sel] = model.flux(time, wave[sel])

        return flux

    model.get_flux = get_flux  # Monkey patching
    return model


# ============== #
#                #
#   Models       #
#                #
# ============== #
_FLAMBDA_units = units.erg / (units.cm ** 2 * units.s * units.AA)


def get_blackbody_flux(lbda, temperature, mag,
                       band="sdssr", magsys="ab"):
    """Get the flux of a blackbody source.

    Parameters
    ----------
    lbda : array_like
        Wavelength array in Angstrom.
    temperature : float
        Temperature of the blackbody in Kelvin.
    mag : float
        Target magnitude in the given band.
    band : str, optional
        Name of the bandpass (from sncosmo). Default is "sdssr".
    magsys : str, optional
        Name of the magnitude system (see sncosmo). Default is "ab".

    Returns
    -------
    array_like
        The blackbody flux in erg/s/cm^2/A.
    """
    from sncosmo import Spectrum
    from astropy.modeling.models import BlackBody

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
    array_like
        The SALT model flux in erg/s/cm^2/A.
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
    array_like
        The Twin Embedding model flux.
    """
    if twins_embedding_model is None:
        raise ImportError("The 'twins_embedding' module is not available.")

    flux, flux_error = twins_embedding_model.evaluate(phase, magnitude, color, list(coordinates))
    wl_obs = twins_embedding_model.wave * (1. + redshift)
    dist_ratio = cosmo.luminosity_distance(ref_redshift).value / cosmo.luminosity_distance(redshift).value
    cosmo_k_corr = (1. + ref_redshift) / (1. + redshift)
    flux_obs = flux * norm * dist_ratio ** 2. * cosmo_k_corr

    return np.interp(lbda, wl_obs, flux_obs, left=np.nan, right=np.nan)


# ============= #
#               #
#  PointSource  #
#               #
# ============= #
class PointSource(SceneElement):
    """A `SceneElement` with a position."""

    def __init__(self, model_func, position, lbda=None, meta={}):
        """Initialize the PointSource.

        Parameters
        ----------
        model_func : callable
            Function that returns the spectrum of the element.
            It must take `lbda` as first argument.
        position : tuple
            Position of the point source (x, y).
        lbda : array_like, optional
            Wavelength array in Angstrom. Default is None.
        meta : dict, optional
            Dictionary of parameters for the `model_func`.
            Default is {}.
        """
        meta = meta.copy()
        meta["position"] = position
        super().__init__(model_func=model_func, lbda=lbda, meta=meta)

    @classmethod
    def from_config(cls, config):
        """Generate a `PointSource` from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
            It must contain:
            - position: tuple, optional
            - model_func: callable, optional
            - source: str, optional

        Returns
        -------
        PointSource
            An instance of the `PointSource` class.
        """
        position = config.get("position", (0, 0))
        model_func = config.get("model_func", None)
        # look for one
        if model_func is None:
            if "source" in config:
                source_ = config["source"]
                if type(source_) in [str, np.str_]:
                    model_func = source_to_modelfunc(config["source"])
                else:  # assume it's a spectrum
                    mag = config.get("mag", None)
                    band = config.get("band", "bessellb")
                    lbda_, flux_ = source_
                    return cls.from_spectrum(lbda_, flux_,
                                             mag=mag, band=band,
                                             position=position, meta=config.copy())
            else:
                raise ValueError("neither 'model_func' nor 'source' in the config. One is needed.")

        return cls(model_func=model_func, position=position,
                   meta=config.copy())

    @classmethod
    def from_spectrum(cls, lbda_, flux_, mag=20, band="bessellb",
                      position=(0, 0), lbda=None, meta={}):
        """Generate a `PointSource` from a spectrum.

        Parameters
        ----------
        lbda_ : array_like
            Wavelength array in Angstrom of the reference spectrum.
        flux_ : array_like
            Flux of the reference spectrum.
        mag : float, optional
            Default magnitude of the target. Default is 20.
        band : str, optional
            Name of the bandpass (must be known by sncosmo). Default is "bessellb".
        position : tuple, optional
            Position of the point source (x, y). Default is (0, 0).
        lbda : array_like, optional
            Default wavelength at which the spectrum may be generated.
            Default is None.
        meta : dict, optional
            Meta information carried by the object. Default is {}.

        Returns
        -------
        PointSource
            An instance of the `PointSource` class.
        """
        from sncosmo import Spectrum
        meta["mag"] = mag
        meta["band"] = band
        meta["flux_ref"] = flux_
        meta["lbda_ref"] = lbda_
        this = cls(None, position=position, lbda=lbda, meta=meta)

        # internal function
        def _internal_get_flux(lbda, mag, band):
            """Internal function to get the flux of the spectrum.

            Parameters
            ----------
            lbda : array_like
                Wavelength array in Angstrom.
            mag : float
                Target magnitude.
            band : str
                Bandpass name.

            Returns
            -------
            array_like
                The flux of the spectrum.
            """
            if mag is None:
                flux_ratio = 1
            else:
                in_mag = Spectrum(meta["lbda_ref"], meta["flux_ref"]
                                  ).bandmag(band, "ab")
                flux_ratio = 10 ** (-0.4 * (mag - in_mag))

            flux_ = np.interp(lbda, meta["lbda_ref"], meta["flux_ref"],
                              left=np.nan, right=np.nan)
            return flux_ * flux_ratio

        this._model_func = _internal_get_flux
        return this

    # ========== #
    #  Getter    #
    # ========== #
    def get_spectrum(self, lbda=None, phase=None, restframe=False):
        """Get the spectrum at a given phase.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, `self.lbda` is used.
            Default is None.
        phase : float, optional
            Phase with respect to maximum light. If None, `self.phase` is used.
            Default is None.
        restframe : bool, optional
            Is the input phase in the rest frame or observer frame?
            Default is False.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        flux : array_like
            Flux in erg/s/cm^2/A.
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        if phase is None:
            phase = self.phase

        elif restframe and phase != 0:
            if (z := self.redshift) is None:
                raise ValueError("no known redshift for the target. Cannot use restrame")

            phase = phase / (1 + z)

        # Actual flux
        model_kwargs = self._parse_model_kwargs_()
        if phase is not None:
            model_kwargs |= {"phase": phase}  # allow

        flux = self.model_func(lbda, **model_kwargs)  # compute spectrum
        return lbda, flux

    # ================ #
    #   Properties     #
    # ================ #
    @property
    def redshift(self):
        """Redshift of the point source."""
        return self.meta.get("redshift", None)

    @property
    def position(self):
        """Position of the point source."""
        return self.meta.get("position", (0,0))

    @property
    def phase(self):
        """Phase of the point source."""
        return self.meta.get("phase", None)

    @property
    def mutable_parameters(self):
        """List of mutable parameters."""
        return self._model_mutables + ["position"]
