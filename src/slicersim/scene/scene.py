"""Scene: scene module.

`Scene` gathers the scene elements (point source, background, host) that
are observed together, and dispatches parameter updates to them.
"""
import warnings

import numpy as np


# ================= #
#                   #
#     Class         #
#                   #
# ================= #
class Scene:
    """Scene description.

    A scene can contain three elements:

    - a point source (e.g. a supernova or a standard star), see
      `~slicersim.scene.pointsource.PointSource`.
    - a uniform background (e.g. the zodiacal light), see
      `~slicersim.scene.background.Background`.
    - a structured background (e.g. the host galaxy), see
      `~slicersim.scene.host.Host`.

    .. warning::
        The scene does not include the thermal background from the telescope.

    Parameters
    ----------
    pointsource : slicersim.scene.pointsource.PointSource, optional
        Point source. Default is None.
    background : slicersim.scene.background.Background, optional
        Uniform background. Default is None.
    host : slicersim.scene.host.Host, optional
        Structured background. Default is None.
    lbda : array_like, optional
        Default wavelength array in Angstrom. Default is None.
    meta : dict, optional
        Dictionary of metadata for the scene. Default is an empty dict.

    See Also
    --------
    from_config : Load the scene from a configuration dictionary.
    """

    def __init__(
        self, pointsource=None, background=None, host=None, lbda=None, meta={}
    ):
        """Initialize the Scene."""

        self._pointsource = pointsource
        self._background = background
        self._lbda = lbda

        self.set_host(host) # special as it may adapt to the pointsource if any.
        self._scene_meta = meta.copy()

    @classmethod
    def from_config(cls, config, lbda=None):
        """Initialize scene from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with the optional keys
            ``"pointsource"``, ``"background"`` and ``"host"``, each being
            the configuration dict of the corresponding element (see the
            ``from_config`` method of
            `~slicersim.scene.pointsource.PointSource`,
            `~slicersim.scene.background.Background` and
            `~slicersim.scene.host.Host`). A missing or empty element has no
            contribution to the scene. Any other entry is stored in the scene
            meta. The host can also be a model name (e.g. ``"ngc0628"`` or
            ``"sersic:bb-5000"``); in both cases, the host is built with the
            defaults of `~slicersim.scene.host.get_host_config`, overwritten
            by the given entries (the model name is the ``"refmodel"`` entry).
            If the scene has a point source with a redshift, the host takes
            its redshift.
        lbda : array_like, optional
            Default wavelength array in Angstrom. Default is None.

        Returns
        -------
        Scene
            An instance of the `Scene` class.

        See Also
        --------
        slicersim.scene.get_scene : Build a scene configuration.
        """
        from . import background, pointsource
        from .host import Host

        config = config.copy()
        pointsource_config = config.pop("pointsource", {})  # rename pointsource ?
        background_config = config.pop("background", {})
        host_config = config.pop("host", {})

        if pointsource_config:  # initialize point-source spectrum from config
            pointsource = pointsource.PointSource.from_config(pointsource_config)
        else:
            pointsource = None

        if background_config:  # initialize background spectrum from config
            background = background.Background.from_config(background_config)
        else:
            background = None

        if host_config:  # initialize host from config
            if type(host_config) is str:
                host_config = {"refmodel": host_config}
            # host defaults (get_host_config) overwritten by the input entries.
            host_config = dict(host_config)
            host = Host.from_modelname(host_config.pop("refmodel", "sersic:bb-5000"),
                                       **host_config)
        else:
            host = None

        return cls(
            pointsource=pointsource,
            background=background,
            host=host,
            lbda=lbda,
            meta=config,
        )  # config don't contains elements anymore.

    # =========== #
    #  Method     #
    # =========== #
    def set_host(self, host):
        """ Set the host element of the scene. """
        # no setting test as you are already pretty deep inside the code.
        # It's likely you know what you are doing.
        self._host = host
        if host is not None and self.has_element("pointsource") and \
            ((redshift := self.pointsource.redshift) is not None):
            self.update(host__redshift = redshift)

    def has_element(self, element):
        """Check if an element is present in the scene.

        Parameters
        ----------
        element : str
            Name of the element ("pointsource", "background", "host").

        Returns
        -------
        bool
            True if the element is present, False otherwise.
        """

        return getattr(self, element, None) is not None


    def update(self, reset_others=False, **kwargs):
        """Update mutable parameters.

        For convenience, the update method respects the django ``__`` format,
        such that, e.g. ``pointsource__phase`` is understood as
        ``pointsource.phase``. For backward compatibility, ``target__`` can
        be used in place of ``pointsource__``.

        Parameters
        ----------
        reset_others : bool, optional
            If True, parameters not given in ``kwargs`` are reset to their
            initial values. Default is False.
        **kwargs
            Parameters to update, see `mutable_parameters`. ``lbda`` updates
            the scene default wavelength. None values are skipped.

        Warns
        -----
        UserWarning
            If a given parameter is not mutable (it is then ignored).

        Examples
        --------
        >>> scene.update(pointsource__phase=-1)  # doctest: +SKIP
        """
        updates_pointsource = {}
        updates_host = {}
        updates_background = {}

        for k, v in kwargs.items():
            # special case
            if k == "lbda":
                self._lbda = v
                continue

            k = k.replace("__", ".")  # django like
            k = k.replace(
                "target.", "pointsource."
            )  # convenience and backward compatiblity

            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            if v is None:  # Skip
                continue

            if k.startswith("pointsource."):
                updates_pointsource[k.replace("pointsource.", "")] = v

            elif k.startswith("background."):
                updates_background[k.replace("background.", "")] = v

            elif k.startswith("host."):
                updates_host[k.replace("host.", "")] = v

            else:
                raise ValueError(f"Unknown scene parameter {k!r}.")

        if self.pointsource is not None:
            self.pointsource.update(reset_others=reset_others, **updates_pointsource)

        if self.background is not None:
            self.background.update(reset_others=reset_others, **updates_background)

        if self.host is not None:
            self.host.update(reset_others=reset_others, **updates_host)


    def get_element_spectrum(self, which, lbda=None, fillna=0, **kwargs):
        """Get the spectrum of a given element.

        Parameters
        ----------
        which : str
            Name of the element ("pointsource", "background", "host").
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, the scene default
            wavelength is used. Default is None.
        fillna : float, optional
            Value to fill the spectrum with if the element is not present.
            Default is 0.
        **kwargs
            Additional arguments passed to the element's ``get_spectrum``
            method (e.g. ``phase`` for the point source).

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        spec : numpy.ndarray
            Spectrum of the element, in erg/s/cm^2/A
            (erg/s/cm^2/A/arcsec^2 for the background).

        Raises
        ------
        ValueError
            If ``lbda`` is None and no default wavelength is set.
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        element = getattr(self, which)
        if element is None:
            spec = np.full_like(lbda, fillna)
        else:
            _, spec = element.get_spectrum(lbda, **kwargs)

        return lbda, spec


    def get_stacked_spectra(self, lbda=None, fillna=0):
        """Get the stack of all scene element spectra.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, the scene default
            wavelength is used. Default is None.
        fillna : float, optional
            Value to fill the spectra with if an element is not present.
            Default is 0.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        spectra : numpy.ndarray
            Array of shape ``(3, nlbda)`` with, in that order, the point
            source, host and background spectra (see
            `get_element_spectrum` for units).

        Raises
        ------
        ValueError
            If ``lbda`` is None and no default wavelength is set.
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        spectra = [
            s_.get_spectrum(lbda)[1] if s_ is not None else np.full_like(lbda, fillna)
            for s_ in [self.pointsource, self.host, self.background]
        ]

        return lbda, np.stack(spectra)


    # -------- #
    #  PLOTTER #
    # -------- #
    def show_background(self, ax=None, in_log=True, **kwargs):
        """Plot the background spectrum.

        The flux (erg/s/cm^2/A/arcsec^2) is plotted as a function of wavelength in micron.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        in_log : bool, optional
            If True, plot the log10 of the flux. Default is True.
        **kwargs
            Additional arguments passed to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        flux = self.background  # background spectrum [erg/s/cm²/Å/arcsec²]
        if in_log:
            flux = np.log10(flux)
            ylabel = "log(Flux [fλ/arcsec²])"
        else:
            ylabel = "Flux [fλ/arcsec²]"

        default = dict()  # ds='steps-mid')
        model_name = self.meta.get("background", {}).get("model", "unknown")
        ax.plot(self.lbda / 1e4, flux, **(default | kwargs) )
        ax.set( xlabel="Wavelength [µm]",
                ylabel=ylabel,
                title=f"Background spectrum ({model_name})",
                )

        return fig

    def show_host(self, ax=None, **kwargs):
        """Plot the total host spectrum (see `~slicersim.scene.host.Host.show_spectrum`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If None, a new figure is created.
        **kwargs
            Goes to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self.has_element("host"):
            raise ValueError("No host element in the scene.")

        return self.host.show_spectrum(lbda=self.lbda, ax=ax, **kwargs)

    def show_pointsource(self, ax=None, in_log=True, **kwargs):
        """Plot the point source spectrum.

        The flux (erg/s/cm^2/A) is plotted as a function of wavelength in micron.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        in_log : bool, optional
            If True, plot the log10 of the flux. Default is True.
        **kwargs
            Additional arguments passed to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        flux = self.pointsource  # point-source spectrum [erg/s/cm²/Å]
        if in_log:
            flux = np.log10(flux)
            ylabel = "log(Flux [fλ])"
        else:
            ylabel = "Flux [fλ]"

        if self.meta.get("pointsource", None) is None:
            title = None
        else:
            title = (
                f"{self.meta['pointsource']['name']} "
                f"({self.meta['pointsource']['source']})"
            )

        default = dict()  # ds='steps-mid')
        ax.plot(self.lbda / 1e4, flux, **{**default, **kwargs})
        ax.set(xlabel="Wavelength [µm]", ylabel=ylabel, title=title)

        return fig

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def pointsource(self):
        """The scene's point source (None if not set)."""
        return self._pointsource

    @property
    def background(self):
        """The scene's spatially flat background (None if not set)."""
        return self._background

    @property
    def host(self):
        """The scene's structured background, i.e. the host (None if not set)."""
        return self._host

    @property
    def mutable_parameters(self):
        """List of mutable parameters, as ``"lbda"`` and ``"{element}.{parameter}"``."""
        # lbda is special. it is a shared parameters
        if self.pointsource is not None:
            pointsource_ = [
                f"pointsource.{pname_}"
                for pname_ in self.pointsource.mutable_parameters
                if pname_ != "lbda"
            ]
        else:
            pointsource_ = []

        if self.background is not None:
            background_ = [
                f"background.{pname_}"
                for pname_ in self.background.mutable_parameters
                if pname_ != "lbda"
            ]
        else:
            background_ = []

        if self.host is not None:
            host_ = [
                f"host.{pname_}"
                for pname_ in self.host.mutable_parameters
                if pname_ != "lbda"
            ]
        else:
            host_ = []

        return ["lbda"] + pointsource_ + background_ + host_

    @property
    def meta(self):
        """Meta parameters of the scene, including the meta of each element."""
        return {
            "pointsource": self.pointsource.meta
            if self.pointsource is not None
            else None,
            "background": self.background.meta if self.background is not None else None,
            "host": self.host.meta if self.host is not None else None,
        } | self._scene_meta

    @property
    def pointsource_position(self):
        """Position of the point source ``(x, y)``, in spaxels."""
        return self.pointsource.position
