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
    - a pointsource point source (e.g., a supernova or a standard star).
    - a uniform background (e.g., the zodiacal background).
    - a structured background (e.g., the host galaxy).

    .. warning::
        The structured `host` is not implemented, and the scene does not
        include the thermal background from the telescope.
    """

    def __init__(self, pointsource=None, background=None, host=None,
                 lbda=None, meta={}):
        """Initialize the Scene.

        Parameters
        ----------
        pointsource : PointSource, optional
            Pointsource point source. Default is None.
        background : Background, optional
            Uniform background. Default is None.
        host : SceneElement, optional
            Structured background (not implemented). Default is None.
        lbda : array_like, optional
            Wavelength array in Angstrom. Default is None.
        meta : dict, optional
            Dictionary of metadata for the scene. Default is {}.

        See Also
        --------
        from_config : Load the scene from a configuration file.
        """

        self._pointsource = pointsource
        self._background = background
        self._host = host
        self._lbda = lbda
        self._scene_meta = meta.copy()

    @classmethod
    def from_config(cls, config, lbda=None):
        """Initialize scene from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing (or not)
            `{"pointsource": {}, "background": {}, "host":{}}`.
            A missing element has no contribution to the scene.
        lbda : array_like, optional
            Wavelength array in Angstrom. Default is None.

        Returns
        -------
        Scene
            An instance of the Scene class.
        """
        from . import pointsource, background

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
            # raise NotImplementedError("Host element not implemented.")
            warnings.warn("Host element not implemented, ignored.")
            host = None
        else:
            host = None

        return cls(pointsource=pointsource,
                   background=background,
                   host=host,
                   lbda=lbda,
                   meta=config)  # config don't contains elements anymore.

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

        For convenience, the update method respects the django '__' format,
        such that, e.g. 'pointsource__phase' is understood as 'pointsource.phase'.
        This way, one can do:

        >>> scene.update(pointsource__phase=-1)  # doctest: +SKIP

        For convenience and backward compatiblity, you can use 
        "target__" in place of "pointsource__".

        Parameters
        ----------
        reset_others : bool, optional
            If True, parameters not given in `kwargs` are reset to their
            initial values. Default is False.
        **kwargs
            Parameters to update.
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
            k = k.replace("target.", "pointsource.") # convenience and backward compatiblity
            
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
            Wavelength array in Angstrom. If None, `self.lbda` is used.
            Default is None.
        fillna : float, optional
            Value to fill the spectrum with if the element is not present.
            Default is 0.
        **kwargs
            Additional arguments passed to the element's `get_spectrum` method.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        spec : array_like
            Spectrum of the element.
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

        The stack contains the spectra of the pointsource, host, and background.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, `self.lbda` is used.
            Default is None.
        fillna : float, optional
            Value to fill the spectra with if an element is not present.
            Default is 0.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        spectra : array_like
            Stacked spectra of the scene elements.
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        spectra = [s_.get_spectrum(lbda)[1] if s_ is not None else np.full_like(lbda, fillna)
                   for s_ in [self.pointsource, self.host, self.background]]

        return lbda, np.stack(spectra)

    def plot_background(self, ax=None, in_log=True, **kwargs):
        """Plot the background spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        in_log : bool, optional
            If True, plot the spectrum in log-scale. Default is True.
        **kwargs
            Additional arguments passed to `ax.plot()`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

        flux = self.background  # background spectrum [erg/s/cm²/Å/arcsec²]
        if in_log:
            flux = np.log10(flux)
            ylabel = "log(Flux [fλ/arcsec²])"
        else:
            ylabel = "Flux [fλ/arcsec²]"

        default = dict()  # ds='steps-mid')
        model_name = self.meta.get('background').get('model')
        ax.plot(self.lbda / 1e4, flux, **{**default, **kwargs})
        ax.set(xlabel="Wavelength [µm]",
               ylabel=ylabel,
               title=f"Background spectrum ({model_name})")

        return ax

    def plot_pointsource(self, ax=None, in_log=True, **kwargs):
        """Plot the pointsource spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        in_log : bool, optional
            If True, plot the spectrum in log-scale. Default is True.
        **kwargs
            Additional arguments passed to `ax.plot()`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

        flux = self.pointsource  # point-source spectrum [erg/s/cm²/Å]
        if in_log:
            flux = np.log10(flux)
            ylabel = "log(Flux [fλ])"
        else:
            ylabel = "Flux [fλ]"

        if self.meta.get("pointsource", None) is None:
            title = None
        else:
            title = (f"{self.meta['pointsource']['name']} "
                     f"({self.meta['pointsource']['source']})")

        default = dict()  # ds='steps-mid')
        ax.plot(self.lbda / 1e4, flux, **{**default, **kwargs})
        ax.set(xlabel="Wavelength [µm]",
               ylabel=ylabel,
               title=title)

        return ax

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def pointsource(self):
        """The scene's point source."""
        return self._pointsource

    @property
    def background(self):
        """The scene's spatially flat background."""
        return self._background

    @property
    def host(self):
        """The scene's structured background."""
        return self._host

    @property
    def mutable_parameters(self):
        """List of mutable parameters."""
        # lbda is special. it is a shared parameters
        if self.pointsource is not None:
            pointsource_ = [f"pointsource.{pname_}" for pname_ in self.pointsource.mutable_parameters if pname_ != "lbda"]
        else:
            pointsource_ = []

        if self.background is not None:
            background_ = [f"background.{pname_}" for pname_ in self.background.mutable_parameters if pname_ != "lbda"]
        else:
            background_ = []

        if self.host is not None:
            host_ = [f"host.{pname_}" for pname_ in self.host.mutable_parameters if pname_ != "lbda"]
        else:
            host_ = []

        return ["lbda"] + pointsource_ + background_ + host_

    @property
    def meta(self):
        """Metaparameters of the scene."""
        return {"pointsource": self.pointsource.meta if self.pointsource is not None else None,
                "background": self.background.meta if self.background is not None else None,
                "host": self.host.meta if self.host is not None else None
                } | self._scene_meta

    @property
    def pointsource_position(self):
        """Position of the pointsource."""
        return self.pointsource.position
