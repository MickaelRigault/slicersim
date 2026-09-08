""" Scene: extendedsource module

Structured background (host galaxy) scene element.

The extended source is described by:
- a spectrum: the *total* (spatially integrated) flux of the galaxy
  in erg/s/cm²/Å, normalized to a total AB magnitude in a band ;
- a spatial profile: a Sersic profile of arbitrary index, effective radius,
  ellipticity and position angle, normalized such that it integrates to 1
  over the (infinite) sky. The profile hence distributes the total spectrum
  spatially ; flux falling outside the field of view is naturally lost.
"""

import numpy as np
from astropy import units

from .base import SceneElement

_FLAMBDA_units = units.erg / (units.cm ** 2 * units.s * units.AA)

#: Parameters describing the spatial (Sersic) profile of the host.
PROFILE_PARAMETERS = ["index", "r_eff", "ellip", "theta"]


def get_host_extendedsource(spectrum="elliptical", profile="sersic", **kwargs):
    """Get a configuration for an extended-source host with
    a specified profile and spectrum.

    Parameters
    ----------
    spectrum : str, optional
        Name of the galaxy model spectrum to use:
        - "elliptical": typical red/dead galaxy spectrum (default).
        - "flat": flat (in AB magnitude) spectrum.
    profile : str, optional
        Name of the spatial profile of the extended source.
        Only "sersic" is implemented. Default is "sersic".
    **kwargs
        Additional parameters to update the configuration
        (e.g., index=1, r_eff=0.5, mag=21, position=[2, 1]).

    Returns
    -------
    dict
        Configuration dictionary for a sersic extended source.
    """
    generic = {'name': 'host',
               'profile': 'sersic',
               'index': 4.,       # Sersic index
               'r_eff': 1.,       # effective radius [arcsec]
               'ellip': 0.,       # ellipticity (1 - b/a)
               'theta': 0.,       # position angle [rad]
               'mag': 20.,        # total AB magnitude
               'band': 'sdssr',   # band of the AB magnitude
               'redshift': 0.,
               'position': [0, 0]}  # position in the MLA [spx]

    if "sersic" not in profile.lower():
        raise NotImplementedError("Only the Sersic profile has been implemented.")

    extendedsource = {'source': spectrum}

    return generic | extendedsource | kwargs


# ==================== #
#  Top level shortcut  #
# ==================== #
def source_to_modelfunc(source):
    """Get the model function from a source name.

    Parameters
    ----------
    source : str
        Name of the source model.
        - "elliptical" - typical red/dead galaxy spectrum
        - "flat" - flat (in AB mag) spectrum of specified magnitude

    Returns
    -------
    callable
        The model function for the given source.
    """
    if "elliptical" in source.lower():
        model_func = get_elliptical_flux

    elif "flat" in source.lower():
        model_func = get_flat_flux

    else:
        raise NotImplementedError(f"no model_func defined for source: {source}")

    return model_func


# ============== #
#                #
#   Models       #
#                #
# ============== #
def elliptical_template_flux(lbda, temperature=4600,
                             break_lbda=4000, break_dmag=1.2,
                             break_width=300):
    """Rest-frame elliptical (early-type) galaxy template spectrum.

    The template approximates the spectral energy distribution of an old
    stellar population by a cool blackbody continuum combined with a smooth
    flux suppression bluewards of the 4000 Å break. It reproduces the red
    continuum and strong 4000 Å break characteristic of elliptical galaxy
    templates (e.g., Kinney+96) over the optical/near-IR range.

    The output is *not* normalized (see `get_elliptical_flux`).

    Parameters
    ----------
    lbda : array_like
        Rest-frame wavelength array in Angstrom.
    temperature : float, optional
        Blackbody temperature of the continuum in Kelvin. Default is 4600.
    break_lbda : float, optional
        Wavelength of the break in Angstrom. Default is 4000.
    break_dmag : float, optional
        Amplitude of the break in magnitudes (flux suppression far bluewards
        of the break). Default is 1.2.
    break_width : float, optional
        Smoothing width of the break in Angstrom. Default is 300.

    Returns
    -------
    array_like
        Template flux density (f_lambda, arbitrary normalization).
    """
    from astropy.modeling.models import BlackBody

    if not hasattr(lbda, 'unit'):  # assumed Angstrom
        lbda = units.Quantity(lbda, units.AA)

    blackbody = BlackBody(temperature=temperature * units.K)
    flux_nu = blackbody(lbda) * units.sr  # rm sr in unit
    flux = flux_nu.to(_FLAMBDA_units, units.spectral_density(lbda)).value

    # smooth 4000 Å break: full suppression (break_dmag) far bluewards,
    # none redwards.
    blue_fraction = 0.5 * (1 - np.tanh((lbda.value - break_lbda) / break_width))
    flux *= 10 ** (-0.4 * break_dmag * blue_fraction)

    return flux


# explicit here the parameters to enable mutable_parameters parsing
def get_elliptical_flux(lbda, mag=20, band="sdssr", magsys="ab",
                        redshift=0.):
    """Get the total flux spectrum of an elliptical galaxy.

    The spectral shape comes from `elliptical_template_flux` (redshifted),
    normalized such that the *total* (spatially integrated) spectrum has
    the requested observed magnitude.

    Parameters
    ----------
    lbda : array_like
        Observer-frame wavelength array in Angstrom.
    mag : float
        Total (spatially integrated) target magnitude in the given band.
        Default is 20.
    band : str, optional
        Name of the bandpass (from sncosmo). Default is "sdssr".
    magsys : str, optional
        Name of the magnitude system (see sncosmo). Default is "ab".
    redshift : float, optional
        Redshift of the galaxy (shifts the template). Default is 0.

    Returns
    -------
    array_like
        The galaxy total flux in erg/s/cm^2/A.
    """
    from sncosmo import Spectrum

    lbda = np.atleast_1d(lbda)
    # template shape, observed frame
    flux = elliptical_template_flux(lbda / (1 + redshift))

    # normalize to the requested observed magnitude using sncosmo
    spec_in = Spectrum(wave=lbda, flux=flux)
    native_mag = spec_in.bandmag(band, magsys)
    fluxcoef_for_target_mag = 10 ** (-0.4 * (mag - native_mag))

    return flux * fluxcoef_for_target_mag  # numpy array


def get_flat_flux(lbda, mag=20):
    """Get a flat (in AB magnitude) total flux spectrum.

    Parameters
    ----------
    lbda : array_like
        Wavelength array in Angstrom.
    mag : float
        Total AB magnitude (per definition, in any band). Default is 20.

    Returns
    -------
    array_like
        The flux in erg/s/cm^2/A.
    """
    lbda = np.atleast_1d(lbda)
    fnu = 10 ** (-0.4 * (mag + 48.60))  # erg/s/cm²/Hz
    return fnu * 2.99792458e18 / lbda ** 2  # erg/s/cm²/Å


# ================ #
#                  #
#  ExtendedSource  #
#                  #
# ================ #
class ExtendedSource(SceneElement):
    """A `SceneElement` with a position and a spatial (Sersic) profile.

    The element's spectrum (`get_spectrum`) is the *total* spatially
    integrated flux of the source in erg/s/cm²/Å. The spatial distribution
    is given by `get_profile`, a unit-integral Sersic profile, such that
    spectrum x profile is a surface brightness in erg/s/cm²/Å/arcsec².
    """

    def __init__(self, model_func, position=(0, 0), lbda=None, meta={}):
        """Initialize the ExtendedSource.

        Parameters
        ----------
        model_func : callable
            Function that returns the total spectrum of the element.
            It must take `lbda` as first argument.
        position : tuple
            Position of the profile center (x, y) in spaxel units.
        lbda : array_like, optional
            Wavelength array in Angstrom. Default is None.
        meta : dict, optional
            Dictionary of parameters for the `model_func` and the
            spatial profile (index, r_eff, ellip, theta). Default is {}.
        """
        meta = meta.copy()
        meta["position"] = position
        super().__init__(model_func=model_func, lbda=lbda, meta=meta)

    @classmethod
    def from_config(cls, config):
        """Generate an `ExtendedSource` from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
            It must contain:
            - model_func: callable, optional
            - source: str or (lbda, flux) spectrum, optional
            one of the two is mandatory. And may contain:
            - position: tuple, optional
            - profile parameters: index, r_eff, ellip, theta, optional
            - mag, band: normalization of the total spectrum, optional

        Returns
        -------
        ExtendedSource
            An instance of the `ExtendedSource` class.
        """
        position = config.get("position", (0, 0))
        model_func = config.get("model_func", None)
        # look for one
        if model_func is None:
            if "source" in config:
                source_ = config["source"]
                if type(source_) in [str, np.str_]:
                    model_func = source_to_modelfunc(source_)
                else:  # assume it's a spectrum
                    mag = config.get("mag", None)
                    band = config.get("band", "sdssr")
                    lbda_, flux_ = source_
                    return cls.from_spectrum(lbda_, flux_,
                                             mag=mag, band=band,
                                             position=position, meta=config.copy())
            else:
                raise ValueError("neither 'model_func' nor 'source' in the config. One is needed.")

        return cls(model_func=model_func, position=position,
                   meta=config.copy())

    @classmethod
    def from_spectrum(cls, lbda_, flux_, mag=20, band="sdssr",
                      position=(0, 0), lbda=None, meta={}):
        """Generate an `ExtendedSource` from a total spectrum.

        Parameters
        ----------
        lbda_ : array_like
            Wavelength array in Angstrom of the reference spectrum.
        flux_ : array_like
            Total flux of the reference spectrum.
        mag : float, optional
            Total magnitude of the source. Default is 20.
        band : str, optional
            Name of the bandpass (must be known by sncosmo). Default is "sdssr".
        position : tuple, optional
            Position of the profile center (x, y) in spaxels. Default is (0, 0).
        lbda : array_like, optional
            Default wavelength at which the spectrum may be generated.
            Default is None.
        meta : dict, optional
            Meta information carried by the object (including profile
            parameters). Default is {}.

        Returns
        -------
        ExtendedSource
            An instance of the `ExtendedSource` class.
        """
        from sncosmo import Spectrum
        meta = meta.copy()
        meta["mag"] = mag
        meta["band"] = band
        meta["flux_ref"] = flux_
        meta["lbda_ref"] = lbda_
        this = cls(None, position=position, lbda=lbda, meta=meta)

        # internal function
        def _internal_get_flux(lbda, mag, band):
            """Get the flux of the reference spectrum scaled to mag."""
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
    def get_profile(self, xx, yy, position=(0, 0)):
        """Get the normalized spatial profile of the source.

        The profile integrates to 1 over the (infinite) sky, such that
        `get_spectrum()[1][:, None, None] * get_profile(xx, yy)` is a
        surface brightness in erg/s/cm²/Å/arcsec².

        Parameters
        ----------
        xx, yy : array_like
            (Broadcastable) coordinates where to evaluate the profile,
            in arcsec, with respect to the field-of-view center.
        position : tuple, optional
            Center of the profile (x, y) in arcsec. Default is (0, 0).
            (`self.position` is in spaxels; the spaxel-to-arcsec conversion
            is the spectrograph's job, see
            `Spectrograph.generate_structured_background`.)

        Returns
        -------
        array_like
            The spatial profile in 1/arcsec².
        """
        from ..profiles import get_profilemodel

        params = self.profile_parameters
        profile_func = get_profilemodel("sersic",
                                        position=position,
                                        normalized=True,
                                        n=params["index"],
                                        r_eff=params["r_eff"],
                                        ellip=params["ellip"],
                                        theta=params["theta"])
        return profile_func(xx, yy)

    # ================ #
    #   Properties     #
    # ================ #
    @property
    def profile(self):
        """Name of the spatial profile."""
        return self.meta.get("profile", "sersic")

    @property
    def profile_parameters(self):
        """Spatial (Sersic) profile parameters as a dict."""
        defaults = {"index": 4., "r_eff": 1., "ellip": 0., "theta": 0.}
        return {k: self.meta.get(k, default_) for k, default_ in defaults.items()}

    @property
    def sersic_index(self):
        """Sersic index of the spatial profile."""
        return self.profile_parameters["index"]

    @property
    def r_eff(self):
        """Effective (half-light) radius of the profile in arcsec."""
        return self.profile_parameters["r_eff"]

    @property
    def redshift(self):
        """Redshift of the extended source."""
        return self.meta.get("redshift", None)

    @property
    def position(self):
        """Position of the profile center (x, y) in spaxels."""
        return self.meta.get("position", (0, 0))

    @property
    def mutable_parameters(self):
        """List of mutable parameters."""
        return self._model_mutables + ["position"] + PROFILE_PARAMETERS
