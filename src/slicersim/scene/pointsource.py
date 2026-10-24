"""Scene: point source module.

`PointSource` is the `~slicersim.scene.base.SceneElement` describing an
unresolved source (e.g. a supernova, a kilonova or a standard star).
"""

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
    """An unresolved scene element with a position.

    A `~slicersim.scene.base.SceneElement` whose meta also carries the
    ``position`` of the source. Use `from_config` or `from_spectrum` to
    build it.

    Parameters
    ----------
    model_func : callable
        Function that returns the spectrum of the element, in
        erg/s/cm^2/A. It must take the wavelength (``lbda``) as first
        argument.
    position : tuple of float
        Position of the point source ``(x, y)``, in spaxels.
    lbda : array_like, optional
        Default wavelength array in Angstrom. Default is None.
    meta : dict, optional
        Parameters of ``model_func`` (and any other metadata).
        Default is an empty dict.
    """

    def __init__(self, model_func, position, lbda=None, meta={}):
        """Initialize the PointSource."""
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

            - ``position``: tuple of float, optional (default ``(0, 0)``), in
              spaxels.
            - ``model_func``: callable, optional. Used as is if given.
            - ``source``: str or ``(lbda, flux)``, optional. If str, the
              model function is obtained from
              `~slicersim.scene.sources.source_to_modelfunc` (e.g.
              ``"salt2-extended"``, ``"blackbody"``, ``"bulla23"``).
              Otherwise, it is assumed to be a tabulated spectrum (see
              `from_spectrum`), normalized by the optional ``mag`` (default
              None, i.e. no rescaling) and ``band`` (default
              ``"bessellb"``) entries.

            Any other entry is stored in the meta and used as parameter of
            the model function (e.g. ``redshift``, ``phase``).
            The input dict is not modified.

        Returns
        -------
        PointSource
            An instance of the `PointSource` class.

        Raises
        ------
        ValueError
            If neither ``model_func`` nor ``source`` is in ``config``.

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
        """Generate a `PointSource` from a tabulated spectrum.

        The spectrum is linearly interpolated at the requested wavelengths
        (NaN outside of ``lbda_``) and rescaled so that its AB magnitude in
        ``band`` equals ``mag``.

        Parameters
        ----------
        lbda_ : array_like
            Wavelength array of the reference spectrum, in Angstrom.
        flux_ : array_like
            Flux of the reference spectrum, in erg/s/cm^2/A.
        mag : float or None, optional
            AB magnitude of the target in ``band``. If None, the reference
            spectrum is used as is. Default is 20.
        band : str, optional
            Name of the bandpass (must be known by ``sncosmo``).
            Default is ``"bessellb"``.
        position : tuple of float, optional
            Position of the point source ``(x, y)``, in spaxels. Default is ``(0, 0)``.
        lbda : array_like, optional
            Default wavelength array, in Angstrom, at which the spectrum is
            generated. Default is None.
        meta : dict, optional
            Meta information carried by the object. It is updated in place
            with ``mag``, ``band``, ``lbda_ref`` and ``flux_ref``.
            Default is an empty dict.

        Returns
        -------
        PointSource
            An instance of the `PointSource` class, whose mutable
            parameters are ``lbda``, ``mag``, ``band`` and ``position``.
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
            Wavelength array in Angstrom. If None, the default wavelength
            (``lbda`` attribute) is used. Default is None.
        phase : float, optional
            Phase in days, with respect to the model reference time (maximum
            light for SNe Ia, merger for kilonovae). If None, `phase` (the
            property) is used. Default is None.
        restframe : bool, optional
            If True, the given ``phase`` is divided by ``1 + redshift``
            before being passed to the model. Ignored if ``phase`` is None.
            Default is False.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        flux : numpy.ndarray
            Flux in erg/s/cm^2/A.

        Raises
        ------
        ValueError
            If ``lbda`` is None and no default wavelength is set, or if
            ``restframe`` is True and the redshift is unknown.
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
        """Redshift of the point source (None if not set)."""
        return self.meta.get("redshift", None)

    @property
    def position(self):
        """Position of the point source ``(x, y)``, in spaxels."""
        return self.meta.get("position", (0,0))

    @property
    def phase(self):
        """Phase of the point source in days (None if not set)."""
        return self.meta.get("phase", None)

    @property
    def mutable_parameters(self):
        """List of mutable parameters (the arguments of ``model_func`` and ``position``)."""
        return self._model_mutables + ["position"]
