""" Scene: pointsource module """

import numpy as np
from copy import deepcopy
from .base import SceneElement
from .sources import source_to_modelfunc


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
            Configuration dictionary. It must contain either
            ``model_func`` or ``source``:

            - position: tuple, optional (default (0, 0))
            - model_func: callable, optional. Used as is if given.
            - source: str or (lbda, flux), optional. If str, the model
              function is obtained from
              `~slicersim.scene.sources.source_to_modelfunc` (e.g.
              "salt2-extended", "blackbody", "bulla23"). Otherwise, it is
              assumed to be a spectrum (see `from_spectrum`).

            Any other entry is stored in the meta and used as parameter of
            the model function (e.g. redshift, phase).
            The input dict is not modified.

        Returns
        -------
        PointSource
            An instance of the `PointSource` class.

        Raises
        ------
        ValueError
            If neither ``model_func`` nor ``source`` is in `config`.

        Examples
        --------
        >>> ps = PointSource.from_config({"source": "bulla23", "redshift": 0.1,
        ...                               "phase": 1.4, "position": [1, 0.5]})
        """
        # do not affect the input config
        config = deepcopy(config)

        position = config.get("position", (0, 0))
        model_func = config.get("model_func", None)
        # look for one
        if model_func is None:
            if "source" not in config:
                raise ValueError("neither 'model_func' nor 'source' in the config. One is needed.")

            source_ = config["source"]
            if type(source_) in [str, np.str_]:
                model_func = source_to_modelfunc(source_)
            else:
                # assume it's a spectrum
                mag = config.get("mag", None)
                band = config.get("band", "bessellb")
                lbda_, flux_ = source_
                return cls.from_spectrum(lbda_, flux_,
                                            mag=mag, band=band,
                                            position=position,
                                            meta=config)

        return cls(model_func=model_func, position=position,
                   meta=config)

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
