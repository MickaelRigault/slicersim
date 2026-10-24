"""Scene: host module.

`Host` is the `~slicersim.scene.base.SceneElement` describing the host galaxy of the observed
target: a spatially extended source, modeled as a spectrum (spectral model)
multiplied by a 2D surface-brightness profile (spatial model).

The model is named ``"spatial:spectral"``, e.g. ``"sersic:bb-5000"``:

- spatial models: ``"sersic"`` (analytic profile, see
  `~slicersim.scene.sources.galaxy.get_sersic_profile`) or an NGC galaxy
  name, e.g. ``"ngc0628"`` (its r-band image, see
  `~slicersim.scene.sources.galaxy.GalaxySpatialModel`);
- spectral models: ``"bb"`` or ``"blackbody"``, optionally followed by the
  temperature in K (``"bb-5000"``, see
  `~slicersim.scene.sources.blackbody.get_blackbody_flux`), or an NGC
  galaxy name, e.g. ``"ngc0628"`` (its spectral template, see
  `~slicersim.scene.sources.generic.generic_get_flux`).

A bare NGC name, e.g. ``"ngc0628"``, is understood as ``"ngc0628:ngc0628"``
(image and spectrum of the same galaxy). NGC names are case insensitive and
NGC data require the ``lazuli_astrosim_data`` package.

Use `get_host_config` to get a default configuration for a model name, and
`Host.from_config` (or `Host.from_modelname`) to build the host.
"""
from copy import deepcopy
import numpy as np

from .base import SceneElement
from .sources import galaxy
from .sources.galaxy import get_sersic_profile
from ..utils import inspect_func


def _is_ngc(name):
    """Whether ``name`` is an NGC galaxy name (case insensitive)."""
    return name.lower().startswith("ngc")


def _is_blackbody(name):
    """Whether ``name`` is a blackbody spectral model (``bb[-T]`` or ``blackbody[-T]``)."""
    return name.split("-")[0].lower() in ["bb", "blackbody"]


def parse_modelname(modelname):
    """Split a host model name into its spatial and spectral models.

    Parameters
    ----------
    modelname : str
        Model name, ``"spatial:spectral"`` (e.g. ``"sersic:bb-5000"``,
        ``"ngc0628:bb-6000"``) or a bare NGC name (e.g. ``"ngc0628"``, read
        as ``"ngc0628:ngc0628"``).

    Returns
    -------
    spatialmodel, spectralmodel : str
        NGC names are lower-cased.

    Raises
    ------
    NotImplementedError
        If ``modelname`` is not in the ``"spatial:spectral"`` format, or if
        the spatial or spectral model is not known.
    """
    if ":" not in modelname:
        if not _is_ngc(modelname):
            raise NotImplementedError(f"modelname '{modelname}' is not recognized. Must be 'spatial:spectral' or an NGC name.")
        # simply modelname="ngc0628" => "ngc0628:ngc0628"
        modelname = f"{modelname}:{modelname}"

    spatialmodel, spectralmodel = modelname.split(":", 1)
    if _is_ngc(spatialmodel):
        spatialmodel = spatialmodel.lower()
    elif spatialmodel != "sersic":
        raise NotImplementedError(f"spatial model '{spatialmodel}' is not recognized. Must be 'sersic' or an NGC galaxy name.")

    if _is_ngc(spectralmodel):
        spectralmodel = spectralmodel.lower()
    elif not _is_blackbody(spectralmodel):
        raise NotImplementedError(f"spectral model '{spectralmodel}' is not recognized. Must be 'bb[-<temperature>]' or an NGC galaxy name.")

    return spatialmodel, spectralmodel


def get_host_config(modelname, redshift=0.1, **kwargs):
    """Get a default host configuration for a given model name.

    Parameters
    ----------
    modelname : str
        Model name, in the format ``"spatial:spectral"``, e.g.
        ``"sersic:bb"``, ``"sersic:bb-6000"``, ``"sersic:ngc4536"`` or simply
        ``"ngc0628"`` (see `parse_modelname` and the module documentation).
    redshift : float, optional
        Redshift of the host. It sets both the angular size of the spatial
        model (via ``r_kpc`` for ``"sersic"``, via the galaxy's distance for
        an NGC image) and the observed spectrum. Default is 0.1.
    **kwargs
        Entries overwriting or completing the default configuration.

    Returns
    -------
    dict
        Configuration to be given to `Host.from_config`. It contains
        ``refmodel`` (= ``modelname``), ``redshift`` and the defaults of the
        chosen models:

        - ``"sersic"``: ``r_kpc=5``, ``n=2``, ``ellip=0.3``, ``theta=-30``;
        - ``"ngcXXXX"`` (spatial): nothing (``position``, ``rotation``);
        - ``"bb"``/``"blackbody"``: ``magabs=-21``, ``band="sdssr"``, and
          ``temperature=5000`` if not given in the model name;
        - ``"ngcXXXX"`` (spectral): ``magabs=-21``, ``band="sdssr"``.

        ``mag`` is the observed magnitude and ``magabs`` the absolute one
        (rest-frame ``band``); if ``mag`` is given (and not ``magabs``), the
        default ``magabs`` is dropped.

    Raises
    ------
    NotImplementedError
        If ``modelname`` is not in the ``"spatial:spectral"`` format, or if
        the spatial or spectral model is not known.

    Examples
    --------
    >>> config = get_host_config("sersic:bb-6000", redshift=1, position=(15, -15))
    >>> host = Host.from_config(config)  # doctest: +SKIP
    """
    spatialmodel, spectralmodel = parse_modelname(modelname)
    config = {"refmodel": f"{spatialmodel}:{spectralmodel}", "redshift": redshift}

    # Spatial is Sersic (NGC images have no default parameter)
    if spatialmodel == "sersic":
        config |= {"r_kpc": 5, "n": 2, "ellip": 0.3, "theta": -30}

    # Spectral: blackbody or NGC galaxy template
    config |= {"magabs": -21, "band": "sdssr"}
    if _is_blackbody(spectralmodel) and "-" not in spectralmodel:
        config |= {"temperature": 5000}

    # an observed magnitude replaces the default absolute one.
    if "mag" in kwargs and "magabs" not in kwargs:
        config.pop("magabs", None)

    return config | kwargs



class Host(SceneElement):
    """Structured background representing the host galaxy of the observed target.

    Models a spatially extended source (e.g. the galaxy hosting a supernova)
    as the product of two functions:

    - a spectral function ``spectral_func(lbda, ...)`` returning the total
      host spectrum (erg/s/cm^2/A), e.g. a blackbody normalized to a given
      magnitude;
    - a spatial function ``spatial_func(...)`` returning a flux-normalized
      ``(ny, nx)`` image, e.g. a Sersic profile (see `get_sersic_profile`).

    The host datacube is ``spectrum[:, None, None] * image[None]``, of shape
    ``(nlbda, ny, nx)``, in erg/s/cm^2/A per pixel.
    The mutable parameters are the arguments of both functions (see
    `mutable_parameters`); arguments shared by both (e.g. ``redshift``) apply
    to both. The image is cached and recomputed only when a spatial parameter
    is updated.

    In a `~slicersim.simulation.Simulation`, the host is rendered on the
    spaxel grid (``shape`` and ``pixel_scale`` given to `get_cube`) and
    convolved by the spatial PSF.

    Use `from_config` or `from_modelname` to build it.

    .. note::
        Host galaxy support is currently in development; the interface is
        subject to change.

    Examples
    --------
    >>> host = Host.from_modelname("sersic:bb-5000", redshift=1, position=(15, -15))  # doctest: +SKIP
    >>> lbda, cube = host.get_cube(lbda, shape=(58, 58), pixel_scale=0.04)  # doctest: +SKIP
    """
    @classmethod
    def from_config(cls, config):
        """Generate a `Host` from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary, typically from `get_host_config`:

            - ``refmodel``: str, optional (default ``"sersic:bb-5000"``).
              Model name ``"spatial:spectral"`` (see `parse_modelname`).
              For ``"bb-<temperature>"``, the temperature (in K) overwrites
              any ``temperature`` entry. For an NGC spectral model, the
              template is stored as ``lbda_ref`` and ``flux_ref``.

            All other entries are the parameters of the spatial and spectral
            functions (stored in the instance meta), e.g. for
            ``"sersic:bb"``:

            - spatial (see `get_sersic_profile`): ``r_kpc`` and ``redshift``
              or ``r_eff``, ``n``, ``ellip``, ``theta``, ``position``,
              ``pixel_scale``, ``shape``, ``oversample``; for an NGC image
              (see `~slicersim.scene.sources.galaxy.GalaxySpatialModel.project_to`):
              ``redshift``, ``position``, ``rotation``, ``shape``,
              ``pixel_scale``;
            - spectral (see
              `~slicersim.scene.sources.blackbody.get_blackbody_flux`):
              ``temperature``, ``mag`` (observed) or ``magabs`` (absolute,
              converted with ``redshift``), ``band``, ``redshift``.

            Parameters not given take the function defaults.
            The input dict is not modified.

        Returns
        -------
        Host
            An instance of the `Host` class.

        Raises
        ------
        NotImplementedError
            If ``refmodel`` is not in the ``"spatial:spectral"`` format, or if
            the spatial or spectral model is not known.

        See Also
        --------
        get_host_config : Default configuration for a model name.
        from_modelname : `get_host_config` and `from_config` in one call.

        Examples
        --------
        >>> host = Host.from_config({"refmodel": "sersic:bb-5000", "mag": 21,
        ...                          "r_eff": 0.5, "n": 1, "ellip": 0.3,
        ...                          "shape": (20, 20)})  # doctest: +SKIP
        """
        # do not affect the input config
        config = deepcopy(config)

        # Parse the model 'spatial:spectral', e.g. 'sersic:bb-5000'
        spatialmodel, spectralmodel = parse_modelname(config.pop("refmodel", "sersic:bb-5000"))

        # --------------------- #
        # => Spatial structure  #
        # --------------------- #
        if spatialmodel == "sersic":
            spatial_func = get_sersic_profile
        else: # NGC galaxy image
            galmodel = galaxy.GalaxySpatialModel.from_name(spatialmodel)
            spatial_func = galmodel.project_to

        # --------------------- #
        # => Spectral structure #
        # --------------------- #
        if _is_blackbody(spectralmodel):
            from .sources.blackbody import get_blackbody_flux
            if "-" in spectralmodel:
                config["temperature"] = float(spectralmodel.split("-")[1])

            spectral_func = get_blackbody_flux

        else: # NGC galaxy spectral template
            from astropy.io import fits
            from ..iotools import get_galaxy_filepath
            from .sources.generic import generic_get_flux
            galdata = fits.getdata(get_galaxy_filepath(spectralmodel, which="spectrum"))
            config["lbda_ref"], config["flux_ref"] = galdata["wavelength"], galdata["flux"]
            spectral_func = generic_get_flux

        # Spatial source.
        return cls._from_spatial_and_spectral_func(spatial_func, spectral_func, **config)


    @classmethod
    def _from_spatial_and_spectral_func(cls, spatial_func, spectral_func, **kwargs):
        """Build a `Host` from a spatial and a spectral function.

        Parameters
        ----------
        spatial_func : callable
            Function returning the flux-normalized ``(ny, nx)`` image. All
            its arguments must have defaults.
        spectral_func : callable
            Function returning the spectrum; its first argument is the
            wavelength (``lbda``).
        **kwargs
            Parameters of the two functions (and any other metadata), stored
            in the instance meta.

        Returns
        -------
        Host
        """
        # takes default values as initial setting
        default_meta_spatial = inspect_func(spatial_func)[1]
        default_meta_spectral = inspect_func(spectral_func)[1]
        # overwrite them by any equivalent entry in kwargs.
        # => Note if 2 entries have the same name in spatial and spectral, spectral wins.
        meta = default_meta_spatial | default_meta_spectral | kwargs

        this = cls("None", meta = meta )
        this._model_func_spectral = spectral_func
        this._model_func_spatial = spatial_func
        return this


    #
    # convinience from_ functions.
    #
    @classmethod
    def from_modelname(cls, modelname, redshift=0.1, **kwargs):
        """Build a `Host` from a model name, with the default configuration.

        Shortcut for ``Host.from_config(get_host_config(modelname, redshift, **kwargs))``.

        Parameters
        ----------
        modelname : str
            Model name ``"spatial:spectral"``, e.g. ``"sersic:bb-5000"``.
        redshift : float, optional
            Redshift of the host. Default is 0.1.
        **kwargs
            Entries overwriting the default configuration, see `get_host_config`.

        Returns
        -------
        Host
        """
        config = get_host_config(modelname, redshift=redshift, **kwargs)
        return cls.from_config(config)


    @classmethod
    def from_sersic_and_blackbody(cls, temperature, mag=None, band="sdssr", redshift=0.1,
                                 r_eff=0.1, n=1.0, ellip=0.0, theta=0.0,
                                 pixel_scale=0.1, shape=(10, 10), position=None,
                                 oversample=4,
                                 **kwargs):
        """Build a `Host` from a Sersic surface-brightness profile and a blackbody spectrum.

        Parameters
        ----------
        temperature : float
            Blackbody temperature, in K.
        mag : float or None, optional
            Total observer-frame AB magnitude of the host in ``band``. If
            None, ``magabs`` must be given in ``kwargs``. Default is None.
        band : str, optional
            Bandpass name (must be known by ``sncosmo``) in which ``mag`` is
            defined. Default is ``"sdssr"``.
        redshift : float, optional
            Redshift of the spectrum (see
            `~slicersim.scene.sources.blackbody.get_blackbody_flux`).
            Default is 0.1.
        r_eff, n, ellip, theta, pixel_scale, shape, position, oversample : optional
            Sersic profile parameters, see `get_sersic_profile`. ``r_eff``
            is in arcsec (default 0.1).
        **kwargs
            Other parameters of the two functions (e.g. ``magabs``) or
            metadata, stored in the instance meta.

        Returns
        -------
        Host
        """
        from .sources.blackbody import get_blackbody_flux
        meta = {key: values for key, values in locals().items()
                       if key not in ["cls" ,"kwargs", "get_blackbody_flux"]
                } | kwargs

        return cls._from_spatial_and_spectral_func(spatial_func=get_sersic_profile,
                                                   spectral_func=get_blackbody_flux,
                                                   **meta)


    @classmethod
    def from_sersic_and_spectrum(cls, lbda, flux, mag=None, band="sdssr", redshift=0.1,
                                 r_eff=0.1, n=1.0, ellip=0.0, theta=0.0,
                                 pixel_scale=0.1, shape=(10, 10), position=None,
                                 oversample=4,
                                 **kwargs):
        """Build a `Host` from a Sersic surface-brightness profile and a reference spectrum.

        The host's spectral shape is taken from a reference spectrum
        (``lbda``, ``flux``), redshifted by ``redshift`` and rescaled to
        match ``mag`` in ``band`` (see
        `~slicersim.scene.sources.generic.generic_get_flux`), and its spatial
        shape is a Sersic profile rendered onto a pixel grid (see
        `get_sersic_profile`).

        Parameters
        ----------
        lbda : array_like
            Wavelength array of the reference spectrum, in Angstrom
            (rest-frame, i.e. before applying ``redshift``). Stored as
            ``lbda_ref``.
        flux : array_like
            Flux of the reference spectrum, sampled at ``lbda``, in
            erg/s/cm^2/A. Stored as ``flux_ref``.
        mag : float or None, optional
            Total observer-frame AB magnitude of the host in ``band``.
            If None and no ``magabs`` is given, the
            reference spectrum is used as is (no rescaling).
            Default is None.
        band : str, optional
            Bandpass name (must be known by ``sncosmo``) in which ``mag`` is
            defined. Default is ``"sdssr"``.
        redshift : float, optional
            Redshift applied to the reference spectrum wavelength axis (no
            flux dimming). Default is 0.1.
        r_eff, n, ellip, theta, pixel_scale, shape, position, oversample : optional
            Sersic profile parameters, see `get_sersic_profile`. ``r_eff``
            is in arcsec (default 0.1).
        **kwargs
            Other parameters of the two functions (e.g. ``magabs``) or
            metadata, stored in the instance meta.

        Returns
        -------
        Host
        """
        from .sources.generic import generic_get_flux
        meta = {key: values for key, values in locals().items()
                       if key not in ["cls", "lbda", "flux", "kwargs", "generic_get_flux"]
                } | {"lbda_ref": lbda, "flux_ref": flux} | kwargs


        return cls._from_spatial_and_spectral_func(spatial_func=get_sersic_profile,
                                                   spectral_func=generic_get_flux,
                                                   **meta)


    # ========== #
    #  Methods   #
    # ========== #
    def update(self, reset_others=False, **kwargs):
        """Update mutable parameters.

        The cached image is recomputed if a spatial parameter changed.

        Parameters
        ----------
        reset_others : bool, optional
            If True, parameters not given in ``kwargs`` are reset to their
            initial values. Default is False.
        **kwargs
            Parameters to update, see `mutable_parameters`.

        Returns
        -------
        list of str
            Names of the updated model parameters.
        """
        updated_parameters = super().update(reset_others=reset_others, **kwargs)
        if reset_others or np.any([key in self._mutable_parameters_spatial for key in updated_parameters]):
            self._update_spatial_model()

        return updated_parameters

    def _model_func(self, lbda, **kwargs):
        """Host datacube: spectrum times image.

        Parameters
        ----------
        lbda : array_like
            Wavelength array in Angstrom.
        **kwargs
            Parameters overwriting the meta for this call only. If any
            spatial parameter is given, the image is recomputed for this call
            (the cache is not changed).

        Returns
        -------
        numpy.ndarray
            ``(nlbda, ny, nx)`` datacube.
        """
        spatial_kwargs = {k: v for k, v in kwargs.items() if k in self._mutable_parameters_spatial}
        spectral_kwargs = {k: v for k, v in kwargs.items() if k in self._mutable_parameters_spectral}
        if unknown := set(kwargs) - set(spatial_kwargs) - set(spectral_kwargs):
            raise ValueError(f"unknown host parameters: {sorted(unknown)}")

        flux = self._model_func_spectral(lbda, **(self._meta_spectral | spectral_kwargs) )
        if spatial_kwargs:
            spatial = self._model_func_spatial(**(self._meta_spatial | spatial_kwargs))
        else:
            spatial = self._current_spatial_model

        return flux[:, None, None] * spatial[None, :, :]


    def _update_spatial_model(self):
        """Recompute the cached image from the current meta."""
        self._spatial_model = self._model_func_spatial(**self._meta_spatial)


    # ========== #
    #  Getter    #
    # ========== #
    def get_cube(self, lbda=None, **kwargs):
        """Get the host datacube.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, the default wavelength
            (``lbda`` attribute) is used. Default is None.
        **kwargs
            Overwrite the current model parameters for this call only,
            typically the grid the host is rendered on (``shape`` and
            ``pixel_scale``), as done by the
            `~slicersim.simulation.Simulation`.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        cube : numpy.ndarray
            Datacube of shape ``(nlbda, ny, nx)``, flux per pixel in
            erg/s/cm^2/A.

        Raises
        ------
        ValueError
            If ``lbda`` is None and no default wavelength is set, or if a
            parameter in ``kwargs`` is not a host parameter.

        Examples
        --------
        Render the host on a 58x58 grid of 0.04 arcsec pixels:

        >>> lbda, cube = host.get_cube(lbda, shape=(58, 58), pixel_scale=0.04)  # doctest: +SKIP
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        return lbda, self.model_func(lbda, **kwargs)


    def get_spectrum(self, lbda=None, from_cube=False):
        """Get the host spectrum.

        By default, this is the total host spectrum (the spectral function).
        With ``from_cube=True``, it is `get_cube` summed over the spatial
        axes, i.e. the flux falling within the host grid (``shape`` x
        ``pixel_scale`` in meta), which can be less than the total spectrum.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, the default wavelength
            (``lbda`` attribute) is used. Default is None.

        from_cube : bool, optional
            If True, the spectrum is computed from the datacube (see
            `get_cube`), otherwise it is computed directly from the spectral
            function. Default is False.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        flux : numpy.ndarray
            Flux in erg/s/cm^2/A.
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        if from_cube:
            lbda, cube = self.get_cube(lbda)
            spec = np.sum(cube, axis=(-2, -1))
        else:
            spec = self._model_func_spectral(lbda, **self._meta_spectral)

        return lbda, spec

    def show_image(self, ax=None, **kwargs):
        """Show the current (cached) host image, in log scale.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If None, a new figure is created.
        **kwargs
            Goes to `matplotlib.axes.Axes.imshow`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        image = self._current_spatial_model
        prop = dict(origin="lower", cmap="cividis", norm=LogNorm(image.max() * 1e-4))
        ax.imshow(image, **( prop | kwargs))
        return fig


    def show_spectrum(self, lbda=None, ax=None, **kwargs):
        """Plot the total host spectrum (see `get_spectrum`).

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, the reference template
            wavelength (``lbda_ref`` in meta, NGC spectral models only) is
            used. Default is None.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If None, a new figure is created.
        **kwargs
            Goes to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if lbda is None:
            lbda = self.meta.get("lbda_ref", None)
            if lbda is None:
                raise ValueError("No lbda given, none available")

        _, spectrum = self.get_spectrum(lbda)
        ax.plot(lbda, spectrum, **kwargs)
        return fig

    # ============ #
    # properties   #
    # ============ #
    @property
    def _current_spatial_model(self):
        """Cached ``(ny, nx)`` image, computed from the meta if not yet done."""
        if getattr(self, "_spatial_model", None) is None:
            self._update_spatial_model()

        return self._spatial_model

    @property
    def _mutable_parameters_spatial(self):
        """Arguments of the spatial function."""
        return inspect_func(self._model_func_spatial )[0]

    @property
    def _mutable_parameters_spectral(self):
        """Arguments of the spectral function, but the wavelength."""
        all_spectral = inspect_func(self._model_func_spectral )[0]
        return [key for key in all_spectral if key not in  ["lbda"] ]

    @property
    def mutable_parameters(self):
        """List of mutable parameters: the arguments of the spatial and spectral functions (without duplicates)."""
        return list(dict.fromkeys(list(self._mutable_parameters_spatial)
                                  + list(self._mutable_parameters_spectral)))

    @property
    def _meta_spatial(self):
        """Current values of the spatial parameters set in meta."""
        return {k: self.meta[k] for k in self._mutable_parameters_spatial if k in self.meta}

    @property
    def _meta_spectral(self):
        """Current values of the spectral parameters set in meta."""
        return {k: self.meta[k] for k in self._mutable_parameters_spectral if k in self.meta}
