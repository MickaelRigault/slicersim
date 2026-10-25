import warnings

warnings.simplefilter('always', UserWarning)

import numpy as np
import pandas

from .detector import Detector
from .scene import Scene
from .spectrograph import SlicerSpectrograph
from .telescope import Telescope

__all__ = ["Simulation"]

COLORS = {# detector
          "dark": "#00283D",
          "roic_glow": "#005566",
          "thermal_dark": "#399378",
          "ron": "#9FC680",
          # scene
          "pointsource": "#E7D49D",
          "background": "#EE9B00",
          "host":"#CA6602",
          # BB3E03
          # AE2012
          # thermal
          "thermal":"#9B2226"
          }

class Simulation:
    """Main class to simulate an observation.

    This class is the main entry point to the simulation. It combines a
    scene (point source, background, host), a telescope, a spectrograph and
    a detector, together with extraction parameters (e.g. ``nramps``), to
    simulate the observed cube and the extracted spectrum with its variance.

    Instances are usually created with one of the class methods
    `Simulation.from_scene`, `Simulation.from_source` or
    `Simulation.from_config` rather than directly.

    Parameters
    ----------
    scene : `~slicersim.scene.scene.Scene`, optional
        The scene to observe. Default is None.
    spectrograph : `~slicersim.spectrograph.Spectrograph`, optional
        The spectrograph used for the observation. Default is None.
    detector : `~slicersim.detector.Detector`, optional
        The detector used for the observation. Default is None.
    telescope : `~slicersim.telescope.Telescope`, optional
        The telescope used for the observation. Default is None.
    extraction : dict, optional
        Extraction parameters (e.g. ``nramps``, ``aperture_radius``).
        ``aperture`` (default ``"1-arcsec"``) and ``aperture.position``
        (default ``"host"``, i.e. the host position) define the spectrum
        extraction of a scene without point source, see
        `Simulation.get_aperture`. Default is ``{}``.
    meta : dict, optional
        Input configuration, used by `Simulation.reset` to restore the
        initial state of each element. Default is ``{}``.

    Attributes
    ----------
    scene : `~slicersim.scene.scene.Scene`
        The scene to observe.
    spectrograph : `~slicersim.spectrograph.Spectrograph`
        The spectrograph used for the observation.
    detector : `~slicersim.detector.Detector`
        The detector used for the observation.
    telescope : `~slicersim.telescope.Telescope`
        The telescope used for the observation.
    extraction : dict
        Extraction parameters.

    """
    VARIANCE_SOURCES = ["dark", "roic_glow", "thermal_dark", "ron",
                        "pointsource", "background", "host", "thermal",
                        ]

    def __init__(self,
                 scene=None,
                 spectrograph=None,
                 detector=None,
                 telescope=None,
                 extraction={},
                 meta={}):
        """Initialize the simulation (see `Simulation` for parameters)."""
        self.scene = scene                #: Scene instance
        self.spectrograph = spectrograph  #: Spectrograph instance
        self.detector = detector          #: Detector instance
        self.extraction = extraction      #: Extraction parameters
        # host spectrum extraction, used if no pointsource (see get_aperture)
        self.extraction.setdefault("aperture", "1-arcsec")
        self.extraction.setdefault("aperture.position", "host") # follows the host position
        self.telescope = telescope        #: telescope
        self._in_meta = meta              #: Meta-parameters

    @classmethod
    def from_source(cls, lbda, flux, mag=None,
                        band="bessellb", position=(0, 0),
                        instrument="lazuli_cbe.toml",
                        background="zodi",
                        host=None,
                        snr=None, lbda_range=[4000, 6800], frame='obs',
                        **kwargs
                     ):
        """Load the simulation observing a known source (like a star).

        Parameters
        ----------
        lbda : array_like
            (High resolution) wavelength of the source [Angstrom].
        flux : array_like
            Flux of the input source [erg/s/cm^2/A].
        mag : float, optional
            Requested magnitude of the source (rescales the ``flux``
            amplitude). If None, ``flux`` is used as is. Default is None.
        band : str, optional
            Name of the band used to compute the magnitude (should be known by
            ``sncosmo``). Default is ``"bessellb"``.
        position : tuple of float, optional
            Location of the point source within the IFU, ``(x, y)`` in
            spaxels.
            Default is ``(0, 0)``.
        instrument : str, dict or list, optional
            Configuration of the instrument (see
            `~slicersim.iotools.get_config`). Default is ``"lazuli_cbe.toml"``.
        background : str, dict, list or None, optional
            Configuration of the (spatially flat) scene background.
            If None, no background is added. Default is ``"zodi"``.
        host : str, dict, list or None, optional
            Configuration of the (spatially structured) host component of the
            scene. If None, no host is added. Default is None.
        snr : float or None, optional
            If not None, the simulation is set up (see
            `Simulation.fetch_snr`) to reach this average SNR per wavelength
            bin over ``lbda_range``. Default is None.
        lbda_range : list of float, optional
            (wmin, wmax) wavelength range used to average the SNR
            [Angstrom]. Ignored if ``snr`` is None. Default is
            ``[4000, 6800]``.
        frame : str, optional
            Frame of ``lbda_range`` (``"obs"`` or ``"rest"``). Ignored if
            ``snr`` is None. Default is ``"obs"``.
        **kwargs
            Goes to `Simulation.update`.

        Returns
        -------
        Simulation
            The loaded simulation.

        See Also
        --------
        Simulation.from_scene : load a simulation from a scene configuration.

        """
        # build the scene config
        scene_config = [{"scene":
                            {"pointsource":
                                 {"source": [lbda, flux],
                                  "mag": mag,
                                  "band": band,
                                  "position":position},
                            }
                        }]

        if background is not None:
            scene_config += [background]

        if host is not None:
            scene_config += [host]

        # now let's use it to load the scene.
        return cls.from_scene(scene=scene_config,
                              instrument=instrument,
                              snr=snr, lbda_range=lbda_range, frame=frame,
                            **kwargs)

    @classmethod
    def from_scene(cls, redshift=None, snr=20,
                       lbda_range=[4000, 6800], frame='rest',
                       scene='supernova.toml',
                       instrument='lazuli_cbe.toml',
                       slicer=True,
                       **kwargs):
        """Load the simulation from scene and instrument configurations.

        The exposure configuration (``nmd`` and ``nramps``) is then set to
        reach the requested SNR on the pointsource, unless ``snr`` is None.

        Parameters
        ----------
        redshift : float or None, optional
            Redshift of the pointsource. If None, the one from the scene
            configuration is used. Default is None.
        snr : float or None, optional
            Targeted average SNR per wavelength bin (see ``lbda_range`` and
            ``frame``), reached using `Simulation.fetch_snr`. If None, the
            exposure configuration is not changed. Default is 20.
        lbda_range : list of float, optional
            (wmin, wmax) wavelength range where the SNR is estimated
            [Angstrom]. Default is ``[4000, 6800]``.
        frame : str, optional
            Wavelength frame of ``lbda_range`` (``"obs"`` or ``"rest"``).
            Default is ``"rest"``.
        scene : str, dict or list, optional
            Scene configuration (see `~slicersim.iotools.get_config`).
            Default is ``"supernova.toml"``.
        instrument : str, dict or list, optional
            Instrument configuration (see `~slicersim.iotools.get_config`).
            Default is ``"lazuli_cbe.toml"``.
        slicer : bool, optional
            If True, use a `~slicersim.spectrograph.SlicerSpectrograph`,
            otherwise a `~slicersim.spectrograph.MLASpectrograph`.
            Default is True.
        **kwargs
            Goes to `Simulation.update` to change any default configuration
            (applied before the SNR optimisation).

        Returns
        -------
        Simulation
            The loaded simulation.

        """

        from .iotools import get_config

        # create the simulator
        config = get_config(instrument=instrument, scene=scene)
        this = cls.from_config(config, slicer=slicer)

        # update to expecting target
        if redshift is not None: # pointsource and host (if any)
            this.update(redshift=redshift)

        if kwargs:
            this.update(**kwargs)

        if snr is not None:
            # fetch the config respecting the target SNR
            new_config, snr, _ = this.fetch_snr(snr, lbda_range=lbda_range, frame=frame)
            this.update(**new_config)

        return this

    @classmethod
    def from_scene_and_instconfig(cls, scene, config, slicer=True):
        """Load the simulation from a scene and an instrument configuration.

        Parameters
        ----------
        scene : `~slicersim.scene.scene.Scene` or dict
            The scene to observe, either as a scene object or as a scene
            configuration dictionary.
        config : dict or str
            Instrument configuration containing the ``"telescope"``,
            ``"spectrograph"``, ``"detector"`` and ``"extraction"`` entries.
            If a str, it is loaded with `~slicersim.iotools.get_config`.
        slicer : bool, optional
            If True, use a `~slicersim.spectrograph.SlicerSpectrograph`,
            otherwise a `~slicersim.spectrograph.MLASpectrograph`.
            Default is True.

        Returns
        -------
        Simulation
            A simulation with the given scene and configuration.

        """
        if type(config) is str:
            from .iotools import get_config
            config = get_config(config)

        this = cls.from_config(config, slicer)

        # scene can either be a Scene object or the config of
        if type(scene) is dict:
            scene = Scene.from_config(scene, lbda=this.spectrograph.lbda)

        this.scene = scene
        return this

    @classmethod
    def from_config(cls, config, slicer=True):
        """Initialize the simulation from a nested configuration dictionary.

        Parameters
        ----------
        config : dict
            Top level configuration (e.g. as returned by
            `~slicersim.iotools.get_config`). It must contain the
            ``"telescope"``, ``"spectrograph"``, ``"detector"`` and
            ``"extraction"`` entries, and may contain a ``"scene"`` entry
            (otherwise the scene is None). It is stored as the reference
            configuration used by `Simulation.reset`.
        slicer : bool, optional
            If True, use a `~slicersim.spectrograph.SlicerSpectrograph`,
            otherwise a `~slicersim.spectrograph.MLASpectrograph`.
            Default is True.

        Returns
        -------
        Simulation
            The loaded simulation.

        """

        # First initialize spectrograph to set wavelengths, then other elements
        # using spectrograph wavelengths.

        # Initialize the telescope (mirror)
        telescope = Telescope.from_config(config["telescope"])

        # Initialize the spectrograph from config
        if slicer:
            spectrograph = SlicerSpectrograph.from_config(config["spectrograph"], telescope=telescope)
        else:
            from .spectrograph import MLASpectrograph
            spectrograph = MLASpectrograph.from_config(config["spectrograph"], telescope=telescope)

        # Initialize the detector from config
        detector = Detector.from_config(config["detector"],
                                        lbda=spectrograph.lbda,
                                        thermaloptics=spectrograph.optics)

        # Initialize extraction parameters from config
        extraction = config["extraction"]

        # Initialize the scene from config (wavelength from spectrograph)
        if "scene" in config:
            scene = Scene.from_config(config["scene"], lbda=spectrograph.lbda)
        else:
            scene = None

        return cls(scene=scene,
                   telescope=telescope,
                   spectrograph=spectrograph,
                   detector=detector,
                   extraction=extraction,
                   meta=config)  # Store config dict in meta

    # =============== #
    #   Methods       #
    # =============== #
    def _fetch_mutable_parameters(self, key):
        """Fetch mutable parameters matching a key.

        Parameters
        ----------
        key : str
            The key to search for in the mutable parameters. Any mutable parameter
            that ends with this key will be returned.
            e.g., ron => detector.ron
            Extraction parameters are only returned if no other parameter
            matches, e.g. position => pointsource.position, not
            aperture.position.

        Returns
        -------
        list or None
            A list of matching mutable parameters, or None if no match is found.

        """
        mutable_ = [param_ for param_ in self.mutable_parameters if param_.endswith(key)]
        # extraction parameters only if nothing else matches
        # e.g., position => pointsource.position (or host.position), not aperture.position
        if len(mutable_) > 1:
            mutable_ = [param_ for param_ in mutable_ if param_ not in self.extraction] or mutable_

        if len(mutable_)==0:
            mutable_ = None

        return mutable_, key

    def change_pointsource(self, pointsource):
        """Override the considered pointsource.

        Parameters
        ----------
        pointsource : `~slicersim.scene.pointsource.PointSource`
            New pointsource to be used.

        Returns
        -------
        None

        """
        self.scene._pointsource = pointsource

    def change_spectrograph_resolution(self, rmin, spotsize=2):
        """Change the spectrograph's spectral resolution.

        This updates the spectrograph dispersion configuration
        (``dispersion_scale``, ``dispersion_resolution`` and ``spotsize``)
        so that the resulting minimal resolving power scales with ``rmin``.
        The reference is a dispersion resolution of 2 pixels for
        ``rmin=100``, such that
        ``dispersion_scale = 1.3 * (rmin / 100) * (spotsize / 2)``.

        Parameters
        ----------
        rmin : float
            Requested minimal resolving power.
        spotsize : float, optional
            Spot size (in pixels) used to compute the dispersion scale.
            Default is 2.

        Returns
        -------
        None
            Updates the simulation in place (see `Simulation.update`).

        """

        reference_rmin = 100
        reference_resolution = 2
        dispersion_scale = 1.3 * rmin/reference_rmin * spotsize/reference_resolution
        config = dict(dispersion_scale=dispersion_scale,
                          dispersion_resolution=reference_resolution,
                          spotsize=spotsize)
        return self.update(**config)

    def update_from_config(self, config):
        """Update the simulation from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary of configurations, keyed by element name
            (``"scene"``, ``"telescope"``, ``"spectrograph"``,
            ``"detector"``, ``"extraction"``). Each value is passed as
            keyword arguments to the corresponding element's ``update``
            method (or merged into ``extraction``).

        Returns
        -------
        None

        Warns
        -----
        UserWarning
            If a key of ``config`` is not an element name (it is ignored).

        """

        for key, thisconfig in config.items():
            if key == "scene":
                self.scene.update(**thisconfig)

            elif key == "telescope":
                self.telescope.update(**thisconfig)

            elif key == "spectrograph":
                self.spectrograph.update(**thisconfig)

            elif key == "detector":
                self.detector.update(**thisconfig)

            elif key == "extraction":
                self.extraction |= thisconfig
            else:
                warnings.warn(f"the {key=}:{thisconfig} is not parsed  and thus ignored.")


    def update(self, reset_others=False, **kwargs):
        """Update any mutable parameter of the simulation.

        For convenience, the update method respects the django ``__`` format,
        such that, e.g. ``pointsource__phase`` is understood as
        ``pointsource.phase``. This way, one can do:

        >>> sim.update(pointsource__phase=-1)  # doctest: +SKIP

        You can also specify a short name, like ``phase``. If it matches a
        single entry of `Simulation.mutable_parameters`, it is accepted.

        >>> sim.update(phase=-1)  # doctest: +SKIP

        For convenience and backward compatibility, you can use ``target__``
        in place of ``pointsource__``.

        Parameters
        ----------
        reset_others : bool, optional
            If True, the scene, spectrograph and detector parameters that are
            not given are reset to their default values. Default is False.
        **kwargs
            Parameters to update, as ``name=value``. Parameters set to None
            are ignored. Parameters not belonging to the scene, the
            spectrograph or the detector update ``extraction``.

        Warns
        -----
        UserWarning
            If a parameter is not mutable or if a short name is ambiguous
            (matches several mutable parameters); it is then ignored.

        """
        updates_scene = {"reset_others": reset_others}
        updates_detector = {"reset_others": reset_others}
        updates_spectrograph = {"reset_others": reset_others}
        updates_extraction = {}

        for k, v in kwargs.items():
            k = k.replace("__", ".") # django like

            # explicit mentions, skip test for within these objects
            # Spectrograph, special traitement of throughput__noptics to allow throughput__noptics__bla
            if k.startswith(("spectrograph.", "throughput.noptics")):
                updates_spectrograph[k.replace("spectrograph.", "")] = v
                continue

            # Detector
            if k.startswith("detector."):
                updates_detector[k.replace("detector.", "")] = v
                continue

            # Scene
            if k.startswith("scene."):
                updates_scene[k.replace("scene.", "")] = v
                continue

            # look for mutable parameters
            if k not in self.mutable_parameters:
                k, k_input = self._fetch_mutable_parameters(k)
                if k is None:
                    warnings.warn(f"Parameter {k_input!r} is not mutable.")
                    continue

                if len(k) == 1:
                    k = k[0]
                # Special case for which you want to actually change them all.
                # This is limited to the scene.
                elif k_input in ["redshift"]:
                    for subkey in k:
                        updates_scene[subkey] = v

                    continue

                else:
                    warnings.warn(f"Parameter {k_input} is not well defined. several correspondance {k}.")
                    continue

            if v is None:       # Nothing to do
                continue

            if k in self.scene.mutable_parameters or k.startswith("scene."):
                updates_scene[k.replace("scene.", "")] = v

            elif k in self.spectrograph.mutable_parameters or k.startswith("spectrograph."):
                updates_spectrograph[k.replace("spectrograph.", "")] = v

            elif k in self.detector.mutable_parameters or k.startswith("detector."):
                updates_detector[k.replace("detector.", "")] = v

            else:
                updates_extraction[k] = v

        # Update spectrograph 1st because it sets the wavelengths
        spectro_change = self.spectrograph.update(**updates_spectrograph)

        # let's updaet scene and detector for the new wavelength.
        if "lbda" in spectro_change:
            updates_scene["lbda"] = self.spectrograph.lbda
            updates_detector["lbda"] = self.spectrograph.lbda

        self.scene.update(**updates_scene)
        self.detector.update(**updates_detector)

        # Convert extraction parameters in relative units to absolute units
        self.extraction.update(**updates_extraction)
        # Do NOT update meta, so reset still works as expected.

    def reset(self, which="*"):
        """Reset simulation elements to their initial configuration.

        The initial configuration is the one given as ``meta`` when loading
        the simulation (see `Simulation.from_config`). Careful, this creates
        a new element and erases the current one.

        Parameters
        ----------
        which : str, optional
            Element to reset: ``"*"`` or ``"all"`` for every element, or one
            of ``"scene"``, ``"telescope"``, ``"spectrograph"``,
            ``"detector"`` or ``"extraction"``. Default is ``"*"``.

        Raises
        ------
        ValueError
            If ``which`` is not a known element.

        """

        if which not in ['all', '*'] + list(self._elements):
            raise ValueError(f"cannot parse the '{which}' reset.")

        if which in ['all', '*']:
            which = self._elements  # Reset all elements

        if "telescope" in which:
            telescope = Telescope.from_config( self._in_meta.get("telescope") )

        if "spectrograph" in which:
            telescope = self.telescope
            self.spectrograph = self.spectrograph.__class__.from_config(self._in_meta.get("spectrograph"),
                                                             telescope=telescope)

        if "scene" in which:
            self.scene = Scene.from_config(self._in_meta.get("scene"), lbda=self.spectrograph.lbda)

        if "detector" in which:
            self.detector = Detector.from_config(
                self._in_meta.get("detector"),
                lbda = self.spectrograph.lbda,
                thermaloptics = self.spectrograph.optics)

        if "extraction" in which:
            self.extraction = self._in_meta.get("extraction")

    # -------- #
    #  GETTER  #
    # -------- #
    def get_lbda(self):
        """Get the wavelength array.

        This is a shortcut to ``self.spectrograph.lbda``.

        Returns
        -------
        numpy.ndarray
            Wavelength of the spectral bins [Angstrom].

        """
        return self.spectrograph.lbda

    def get_input_spectrum(self, which="pointsource"):
        """Get the input spectrum of a scene element.

        This is a shortcut to ``self.scene.get_element_spectrum``.

        Parameters
        ----------
        which : str, optional
            Which input spectrum to return:

            - individual elements: ``"pointsource"``, ``"background"``,
              ``"host"``;
            - all elements summed: ``"stacked"``.

            Default is ``"pointsource"``.

        Returns
        -------
        lbda : numpy.ndarray
            Wavelength [Angstrom].
        flux : numpy.ndarray
            Input flux [erg/s/cm^2/A].

        """
        if which == "stacked":
            lbda, specs = self.scene.get_stacked_spectra()
            flux = np.sum(specs, axis=0)
        else:
            lbda, flux = self.scene.get_element_spectrum(which)

        return lbda, flux

    def get_effective_transmission(self):
        """Get the effective total transmission of the instrument.

        Product of the spectroscopic throughput
        (``spectrograph.flambda2photon``) and the detector efficiency
        (``detector.photonflux_to_adu``), for a single ramp.

        Returns
        -------
        lbda : numpy.ndarray
            Wavelength [Angstrom].
        throughput : numpy.ndarray
            Effective throughput, i.e. conversion from flux to signal
            [ADU / (erg/s/cm^2/A)].

        """
        lbda = self.spectrograph.lbda # make sure it is up to date.
        return lbda, self.spectrograph.flambda2photon * self.detector.photonflux_to_adu(lbda)

    def get_resolving_power(self):
        r"""Get the spectral resolving power :math:`R`.

        This is a shortcut to
        `~slicersim.spectrograph.Spectrograph.get_resolving_power`:

        .. math::

            R = \frac{\lambda}{\Delta\lambda \, \sigma}

        with :math:`\Delta\lambda` the spectral step [Angstrom] and
        :math:`\sigma` the dispersion resolution [pixel].

        Returns
        -------
        lbda : numpy.ndarray
            Wavelength [Angstrom].
        resolving_power : numpy.ndarray
            Resolving power :math:`R` (dimensionless).

        """
        return self.spectrograph.lbda, self.spectrograph.get_resolving_power()

    def get_pixel_variance(self, flux=0, **kwargs):
        """Get the variance associated with a single detector pixel.

        The variance includes all ``nramps`` ramps.

        Parameters
        ----------
        flux : float or array_like, optional
            Incident flux on the pixel [ph/s]. Default is 0.
        **kwargs
            Goes to `~slicersim.detector.Detector.estimate_pixel_signal`.

        Returns
        -------
        float or numpy.ndarray
            Variance of the pixel [ADU^2].

        """
        lbda = self.spectrograph.lbda
        _, pixel_var = self.detector.estimate_pixel_signal(flux, lbda=lbda, **kwargs)
        pixel_var *= self.extraction["nramps"] # incl. multi ramp approach.
        return pixel_var

    def get_nea(self, nea_spatial=None, nea_pixels=None): # pragma: no cover
        """Get the Noise Equivalent Area (NEA).

        The NEA is the product of the spatial NEA and the pixel NEA.

        Parameters
        ----------
        nea_spatial : float or array_like, optional
            The spatial NEA [spaxel^2]. If None, it is computed.
            Default is None.
        nea_pixels : float or array_like, optional
            The pixel NEA [pixel]. If None, it is computed.
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            The total NEA [pixel].

        """
        return self.spectrograph.get_nea(position = self.scene.pointsource.position,
                                          nea_spatial=nea_spatial, nea_pixels=nea_pixels)

    def get_nea_variance(self, spectrum=None, nea_spatial=None, nea_pixels=None): # pragma: no cover
        """Get the variance estimated from the Noise Equivalent Area.

        Parameters
        ----------
        spectrum : array_like or None, optional
            The input spectrum [erg/s/cm^2/A]. If given, its Poisson
            variance is added (approximation). Default is None.
        nea_spatial : float or array_like, optional
            The spatial NEA [spaxel^2]. If None, it is computed.
            Default is None.
        nea_pixels : float or array_like, optional
            The pixel NEA [pixel]. If None, it is computed.
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            The estimated variance [ADU^2].

        Warns
        -----
        UserWarning
            If ``spectrum`` is given and the configuration differs from the
            validated one (``spatial_scale=0.04``,
            ``psf_sigma_spectral=0.03``).

        """
        nea = self.get_nea(nea_spatial=nea_spatial, nea_pixels=nea_pixels)

        # "background" variance of a single pixels
        pixel_var = self.get_pixel_variance(0)
        variance_pixels = nea * pixel_var

        # adding spectrum. But very inclear if correct...
        if spectrum is not None:
            # test default tests case:
            right_config = self.get_parameter(["spatial_scale", "psf_sigma_spectral"])
            if right_config != {'spatial_scale': 0.04, 'psf_sigma_spectral': 0.03}:
                warnings.warn("inclusion of spectrum variance has been validated for {'spatial_scale': 0.04, 'psf_sigma_spectral': 0.03} only. Likely wrong if too far.")

            spec_adu = self.convert_units(flux_in=spectrum, units_in="flambda", units_out="adu")
            # This is an approximation that seem to work, not clear why... (so the above warning)
            poisson_adu_to_variance = np.sqrt(self.spectrograph.get_nea_spatial()  / self.spectrograph.get_nea_pixels() )
            var_source = spec_adu * poisson_adu_to_variance
        else:
            var_source = 0

        return variance_pixels + var_source

    def get_parameter(self, which=None, default=None, as_dict=True):
        """Get simulation parameter(s).

        The parameter should be an extraction parameter or an attribute (or
        ``meta`` entry) of one of the elements of the simulation, e.g.
        ``gain`` for ``Simulation.detector.gain``. The ``element__name``
        format is also accepted: ``element`` is first looked for as an
        attribute of the simulation (e.g. ``detector__gain``), then as a
        sub-element of the scene, spectrograph or detector (e.g.
        ``optics__emissivity`` for ``Simulation.spectrograph.optics``).

        Parameters
        ----------
        which : str, list of str or None, optional
            Parameter name (or list of). If None, all parameters are
            returned. Default is None.
        default : object, optional
            Value returned if the parameter is not found. Default is None.
        as_dict : bool, optional
            If the parameter is not found, return ``{which: default}``
            rather than ``default``. Default is True.

        Returns
        -------
        object or dict
            Parameter value. If ``which`` is a list, a dict
            ``{name: value}`` is returned.

        """
        if which is None:       # Get all of them
            which = [item
                      for sublist in self._elements
                      for item in sublist ]

        if np.ndim(which):  # If a list, loop over elements
            return {param: self.get_parameter(param, as_dict=False)
                     for param in which }

        # allowing to provide origin___{bla}

        # Extraction parameter (e.g. aperture__position => aperture.position)
        if which.startswith("extraction__") or which.replace("__", ".") in self.extraction:
            return self.extraction[ which.replace("extraction__", "").replace("__", ".") ]

        # Short cuts
        if which in ["detector__ngroup", "ngroup"]:
            return self.detector.nmd[0]

        # Short cuts
        if which in ["detector__nframe", "nframe"]:
            return self.detector.nmd[0]

        # Generic source__value term
        if "__" in which:
            # you provide a self.source.value as source__value
            if len(source_and_value := which.split("__")) == 2:
                source, value = source_and_value
                if hasattr(self, source):
                    instance = getattr(self, source)
                    if hasattr(instance, value):
                        return getattr(instance, value)

        # Otherwise, look at individual elements
        elements = ["scene", "spectrograph", "detector"]
        for element in elements:
            instance = getattr(self, element)
            which = which.replace("__", ".") # original format.
            if '.' in which:    # Composed key: key1.key2
                k1, k2 = which.split('.')
                if hasattr(instance, k1):
                    instance = getattr(instance, k1)
                    which = k2

            if hasattr(instance, which):            # 'self'.which
                return getattr(instance, which)

            if which in instance.meta:  # self.meta['which']
                return instance.meta[which]

            if element == "scene": # for scene: self.meta['pointsource']['which']
                meta_in_scene = {}
                # try if in pointsource
                for sceneelement in ["pointsource", "host", "background"]:
                    meta_scenelement = instance.meta.get(sceneelement, None)
                    if meta_scenelement is not None and which in meta_scenelement:
                        meta_in_scene[sceneelement] = meta_scenelement[which]

                values = list(meta_in_scene.values())
                # unique, or shared by all elements (e.g. redshift of pointsource and host)
                if len(values) == 1 or (len(values) > 1 and
                                        all(np.array_equal(v, values[0]) for v in values[1:])):
                    return values[0]

                elif len(meta_in_scene) >1 :
                    warnings.warn(f"several {which=} in scene ({meta_in_scene.keys()}). specify: scenelementname__which")

        return {which: default} if as_dict else default

    def get_background_spectrum(self, unit="adu", skyarea=None, per_ramp=False, apply_lsf=True):
        """Get the (flat) background flux within a given sky area.

        Parameters
        ----------
        unit : str, optional
            Unit of the output flux: ``"adu"`` (total signal),
            ``"flambda"`` [erg/s/cm^2/A] or ``"ph"``, ``"photon"``,
            ``"photons"`` [ph/s]. Default is ``"adu"``.
        skyarea : float or None, optional
            Sky area [arcsec^2]. If None, the area of one spaxel is used.
            Default is None.
        per_ramp : bool, optional
            If True, the flux is not multiplied by ``nramps`` (for single
            ramp). Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.

        Returns
        -------
        numpy.ndarray
            Background flux per wavelength bin, shape (nlbda,).

        Raises
        ------
        ValueError
            If ``unit`` is unknown.

        """
        if skyarea is None:
            spx_spatial = np.broadcast_to(self.spectrograph.spx_spatial_scale, (2,))
            skyarea = np.prod(spx_spatial)

        # erg/s/cm²/Å / erg/ph * cm² * Å = ph/s
        background_flux = self.scene.background.get_spectrum(self.spectrograph.lbda)[1]
        flux = background_flux * skyarea
        if unit in "flambda":
            coef = 1
        elif unit in ["ph", "photon", "photons"]:
            coef = self.spectrograph.flambda2photon
        elif unit in ["adu"]:
            coef = self.spectrograph.flambda2photon * self.detector.photonflux_to_adu(self.spectrograph.lbda)
        else:
            raise ValueError(f"unknown {unit=}")

        if not per_ramp:
            coef *= self.get_parameter("nramps")

        flux = flux*coef

        if apply_lsf:
            flux = self.spectrograph.apply_line_spread_function(flux)

        return flux

    def get_scene_cubes(self, unit="adu", psf_profile="default",
                            as_oversampled=False, oversampling=None,
                            as_dict=True, per_ramp=False, apply_lsf=True,
                            **kwargs):
        """Get the cubes of each scene component (noise free).

        Parameters
        ----------
        unit : str, optional
            Unit of the output flux: ``"adu"`` (or ``"default"``),
            ``"flambda"`` [erg/s/cm^2/A] or ``"ph"``, ``"photon"``,
            ``"photons"`` [ph/s]. Default is ``"adu"``.
        psf_profile : str, optional
            PSF profile to use. Default is ``"default"``.
        as_oversampled : bool, optional
            If True, return the oversampled pointsource cube. Default is
            False.
        oversampling : int or None, optional
            Oversampling factor. Default is None.
        as_dict : bool, optional
            If True, return a dictionary of cubes. Default is True.
        per_ramp : bool, optional
            If True, the cubes are not multiplied by ``nramps`` (for single
            ramp). Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to the spectrograph ``generate_pointsource`` and
            ``generate_structured_background`` methods.

        Returns
        -------
        dict or numpy.ndarray
            If ``as_dict``, a dictionary with keys ``"pointsource"``,
            ``"background"``, ``"host_cube"`` and ``"thermal_cube"``, each a
            (nlbda, ny, nx) cube. Otherwise, these four cubes stacked in a
            (4, nlbda, ny, nx) array.

        Raises
        ------
        ValueError
            If ``unit`` is unknown.
        NotImplementedError
            If the scene has a host and ``as_oversampled`` is True.

        """
        # by default spectrograph. give flux in ph/s.

        lbda = self.spectrograph.lbda

        # pointsource
        pointsource_spec = self.scene.pointsource.get_spectrum(lbda)[1]
        pointsource_cube = self.spectrograph.generate_pointsource(pointsource_spec,
                                                            position=self.scene.pointsource_position,
                                                            psf_profile=psf_profile,
                                                            oversampling=oversampling,
                                                            as_oversampled=as_oversampled,
                                                            apply_lsf=apply_lsf,
                                                            **kwargs)
        # background
        background_spec = self.scene.background.get_spectrum(lbda)[1]
        background_cube = self.spectrograph.generate_background(background_spec,
                                                            oversampling=oversampling,
                                                            apply_lsf=apply_lsf)

        # host | empty
        if self.scene.has_element("host"):
            if as_oversampled:
                raise NotImplementedError("host not available for oversampled cubes.")
            host_cube = self.spectrograph.generate_structured_background(self._get_host_cube_(lbda),
                                                            psf_profile=psf_profile,
                                                            oversampling=oversampling,
                                                            apply_lsf=apply_lsf,
                                                            **kwargs)
        else:
            host_cube = np.zeros( (self.spectrograph.nlbda, *self.spectrograph.get_spectrograph_shape(oversampling=oversampling)) )  # (nlbda, ny, nx)

        # thermal
        thermal_cube = self.spectrograph.generate_thermal_signal(as_cube=True,
                                                            oversampling=oversampling,
                                                            apply_lsf=apply_lsf) # [ph/s]

        # changing the unit.
        if unit.lower() in ["ph", "photons", "photon"]:
            coef = 1
        elif unit.lower() in "flambda":
            coef = 1/self.spectrograph.flambda2photon[:, np.newaxis, np.newaxis]

        elif unit.lower() in ["adu", "default"]:
            coef = self.detector.photonflux_to_adu(self.spectrograph.lbda)[:, np.newaxis, np.newaxis]
        else:
            raise ValueError(f"cannot parse requested unit {unit=}, flambda, photons, adu available.")

        if not per_ramp:
            coef *= self.get_parameter("nramps")

        pointsource_cube *= coef
        background_cube *= coef
        host_cube *= coef
        thermal_cube *= coef

        if as_dict:
            return {"pointsource": pointsource_cube,
                    "background": background_cube,
                    "host_cube": host_cube,
                    "thermal_cube": thermal_cube}

        return np.stack([pointsource_cube, background_cube, host_cube, thermal_cube])

    def get_projected_scene(self, in_photons=True, switch_off=[],
                                psf_profile="default",
                                as_oversampled=False, oversampling=None,
                                apply_lsf=True, cached=False,
                                **kwargs):
        """Project the scene through the spectrograph and get the flux cube.

        The cube is the noise-free incident flux per spaxel, including the
        pointsource, host, background and thermal contributions (unless
        switched off).

        Parameters
        ----------
        in_photons : bool, optional
            If True, return the cube in [ph/s], otherwise in
            [erg/s/cm^2/A]. Default is True.
        switch_off : list of str, optional
            Components to discard: any of ``"pointsource"``, ``"host"``,
            ``"background"`` and ``"thermal"``. Default is ``[]``.
        psf_profile : str, optional
            PSF profile to use. Default is ``"default"``.
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        oversampling : int or None, optional
            Oversampling factor. If None, 3 is used. Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        cached : bool, optional
            If True, return the cube stored by a previous call made with
            ``cached=True`` (if any), otherwise compute and store it. The
            other arguments are then ignored. Default is False.
        **kwargs
            Goes to the spectrograph ``generate_pointsource`` and
            ``generate_structured_background`` methods.

        Returns
        -------
        numpy.ndarray
            (nlbda, ny, nx) cube in [ph/s] or [erg/s/cm^2/A].

        Raises
        ------
        NotImplementedError
            If the host is included and ``as_oversampled`` is True.

        """
        # should this be recomputed ?
        if cached and hasattr(self, "_cube_cached") and self._cube_cached is not None:
            # cached exists and use to the cached cube requested.
            return self._cube_cached

        # ok cube to be computed.
        if oversampling is None:
            oversampling = 3

        #if not as_oversampled:
        #    oversampling = None
        cube = self.spectrograph.get_empty_cube(filled=0, oversampling=None if not as_oversampled else oversampling)

        # spectra (3, nlbda):
        # * point source spectrum [erg/s/cm²/Å]
        # * host cube [erg/s/cm²/Å/spx]
        # * background spectrum [erg/s/cm²/Å/arcsec²]
        lbda = self.spectrograph.lbda # make sure this is up to date

        # Fill the cube with scene elements in photons/s/spx
        if "pointsource" not in switch_off and self.scene.has_element("pointsource"):
            pointsource_spec = self.scene.pointsource.get_spectrum(lbda)[1]
            cube += self.spectrograph.generate_pointsource(pointsource_spec,
                                                            position=self.scene.pointsource_position,
                                                            psf_profile=psf_profile,
                                                            oversampling=oversampling,
                                                            as_oversampled=as_oversampled,
                                                            apply_lsf=False, # applied once, at the end
                                                            **kwargs)

        if "host" not in switch_off and self.scene.has_element("host"):
            if as_oversampled:
                raise NotImplementedError("host not available for oversampled cubes.")
            cube += self.spectrograph.generate_structured_background(self._get_host_cube_(lbda),
                                                            psf_profile=psf_profile,
                                                            oversampling=oversampling,
                                                            apply_lsf=False, # applied once, at the end
                                                            **kwargs)

        if "background" not in switch_off and self.scene.has_element("background"):
            background_spec = self.scene.background.get_spectrum(lbda)[1]
            cube += self.spectrograph.generate_background(background_spec,
                                                          oversampling=None if not as_oversampled else oversampling,
                                                          apply_lsf=False)

        if "thermal" not in switch_off:
            cube += self.spectrograph.generate_thermal_signal(as_cube=True, oversampling=None if not as_oversampled else oversampling,
                                                                apply_lsf=False) # [ph/s]

        if not in_photons:      # Convert back to flambda
            cube /= self.spectrograph.flambda2photon[:, np.newaxis, np.newaxis]

        # apply_lsf once
        if apply_lsf:
            cube = self.spectrograph.apply_line_spread_function(cube)

        # cached requested and non existed (see top). Hence let's store it.
        if cached:
            self._cube_cached = cube

        return cube  # (nlbda, ny, nx) | [ph/s] or [erg/cm²/Å/s]

    def _get_host_cube_(self, lbda):
        """Render the host on the spaxel grid, before the PSF.

        Parameters
        ----------
        lbda : array_like
            Wavelength array in Angstrom.

        Returns
        -------
        numpy.ndarray
            (nlbda, ny, nx) host flux per spaxel [erg/s/cm²/Å].
        """
        # like the PSF, the host is rendered on spx_shape using
        # spx_spatial_scale along both axes.
        if np.ndim(self.spectrograph.spx_spatial_scale) != 0:
            raise NotImplementedError("host requires a scalar spatial_scale.")

        ny, nx = self.spectrograph.spx_shape
        return self.scene.host.get_cube(lbda, shape=(nx, ny), # host shape is (nx, ny)
                                        pixel_scale=self.spectrograph.spx_spatial_scale)[1]

    def get_cube(self, switch_off=[], psf_profile="default",
                     as_oversampled=False, oversampling=None,
                     per_ramp=False, apply_lsf=True, cached=False,
                     **kwargs):
        """Get the signal and variance cubes as extracted from the exposure.

        Parameters
        ----------
        switch_off : str or list of str, optional
            Name of the variance sources to switch off, any of
            `Simulation.variance_sources` (``"dark"``, ``"roic_glow"``,
            ``"thermal_dark"``, ``"ron"``, ``"pointsource"``,
            ``"background"``, ``"host"``, ``"thermal"``) or
            ``"effective_dark"`` (``"dark"`` and ``"thermal_dark"``).
            Detector effects are only switched off during the computation.
            Default is ``[]``.
        psf_profile : str, optional
            PSF profile to use. Default is ``"default"``.
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        oversampling : int or None, optional
            Oversampling factor. Default is None.
        per_ramp : bool, optional
            If True, the cubes are not multiplied by ``nramps`` (for single
            ramp). Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        cached : bool, optional
            If True, use (or store) the cached projected scene and thermal
            signal (see `Simulation.get_projected_scene`). Default is False.
        **kwargs
            Goes to `Simulation.get_projected_scene`.

        Returns
        -------
        cube_signal : numpy.ndarray
            (nlbda, ny, nx) signal cube [ADU].
        cube_variance : numpy.ndarray
            (nlbda, ny, nx) variance cube [ADU^2].

        Raises
        ------
        ValueError
            If the spectrograph type is neither MLA nor slicer.

        Warns
        -----
        UserWarning
            If an entry of ``switch_off`` is unknown.

        """

        # change accepted.
        #
        # component within the cube
        #
        AVAILABLE_SWITCHOFF = ["effective_dark"] + self.variance_sources
        switch_off = np.atleast_1d(switch_off).tolist() # as list

        if np.any(effect_bool := [effect not in AVAILABLE_SWITCHOFF for effect in switch_off]):
            unknown_effect = np.asarray(switch_off)[effect_bool]
            warnings.warn(f"unknown switch off effect(s) {unknown_effect}.")

        #
        # switch off detector effect
        #
        if "effective_dark" in switch_off:
            switch_off.append("dark")
            switch_off.append("thermal_dark")
            switch_off = np.unique(switch_off).tolist()

        # detector dark
        if "dark" in switch_off:
            current_dark = self.get_parameter("detector__dark")
            self.update(detector__dark = 0)            # Switch off dark
        else:
            current_dark = None

        # thermal induced dark-like signal.
        if "thermal_dark" in switch_off:
            # switch off optics emissivity
            current_optics_emissivity = self.get_parameter("optics__emissivity")
            self.update(optics__emissivity = 0)    # Switch off dark
        else:
            current_optics_emissivity = None

        if "ron" in switch_off:
            current_ron = self.get_parameter("detector__ron")
            self.update(detector__ron=0)              # Switch off ron
        else:
            current_ron = None

        # detector dark
        if "roic_glow" in switch_off:
            current_roic_glow = self.get_parameter("detector__roic_glow")
            self.update(detector__roic_glow = 0)   # Switch off roic glow
        else:
            current_roic_glow = None

        #
        # Get cubes
        #
        cube_prop = dict(psf_profile=psf_profile,
                          switch_off=switch_off,
                          as_oversampled=as_oversampled, oversampling=oversampling,
                          apply_lsf=apply_lsf,
                          cached=cached,
                        ) | kwargs

        # mla
        if self.spectrograph.type in ["mla", "spaxel", "spx"]:
            sig_cube, var_cube = self._get_mla_cube_(**cube_prop)

        # slicer
        elif self.spectrograph.type in ["slicer", "slice"]:
            sig_cube, var_cube = self._get_slicer_cube_(**cube_prop)

        # unknown not accepted.
        else:
            raise ValueError(f"unknown spectrograph type {self.spectrograph.type}: mla or slicer accepted.")

        #
        # switch back in detector effects (if any)
        #
        if current_dark is not None:
            self.update(detector__dark=current_dark)

        if current_optics_emissivity is not None:
            self.update(optics__emissivity=current_optics_emissivity)

        if current_ron is not None:
            self.update(detector__ron=current_ron)

        if current_roic_glow is not None:
            self.update(detector__roic_glow=current_roic_glow)

        #
        # include ramps.
        #
        if not per_ramp:
            nramps = self.get_parameter("nramps")
            sig_cube *= nramps
            var_cube *= nramps

        return sig_cube, var_cube

    def _get_slicer_cube_(self, psf_profile="default", switch_off=[],
                           oversampling=None, as_oversampled=False,
                           apply_lsf=True, cached=False,
                           **kwargs):
        """Get slicer cube.

        Parameters
        ----------
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        switch_off : list, optional
            List of discarded scene elements (pointsource, host, background) +
            thermal. Default is [].
        oversampling : int, optional
            Oversampling factor. Default is None.
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        cached : bool, optional
            If True, use cached projected scene and thermal signal.
            Default is False.
        **kwargs
            Goes to `Simulation.get_projected_scene`.

        Returns
        -------
        sig_cube, var_cube : numpy.ndarray
            (nlbda, ny, nx) signal [ADU] and variance [ADU^2], per ramp.

        """
        # scene split per slices
        cube = self.get_projected_scene(in_photons=True,
                                        switch_off=switch_off,
                                        psf_profile=psf_profile,
                                        as_oversampled=as_oversampled,
                                        oversampling=oversampling,
                                        apply_lsf=apply_lsf,
                                        cached=cached,
                                        **kwargs)  # (nlbda, nslices_with_anamorphose, npixels)

        # slicers projected onto the detector: 1 lbda for 1 slice corresponds to 1 pixel
        lbda = self.spectrograph.lbda # make sure it is up to date
        sig_cube, var_cube = self.detector.estimate_pixel_signal(cube, lbda=lbda,
                                                                 npix_per_spaxel=self.spectrograph._ANAMORPHOSE,
                                                                 cached_thermal=cached) # expects input in [ph/...]

        return sig_cube, var_cube

    def _get_mla_cube_(self, psf_profile="default", switch_off=[],
                           as_oversampled=False, oversampling=None,
                           apply_lsf=True, cached=False,
                           **kwargs): # pragma: no cover
        """Get MLA cube.

        Parameters
        ----------
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        switch_off : list, optional
            List of discarded scene elements (pointsource, host, background) +
            thermal. Default is [].
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        oversampling : int, optional
            Oversampling factor. Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        cached : bool, optional
            If True, use the cached projected scene. Default is False.
        **kwargs
            Goes to `Simulation.get_projected_scene`.

        Returns
        -------
        sig_cube, var_cube : numpy.ndarray
            (nlbda, ny, nx) signal [ADU] and variance [ADU^2], per ramp.

        """

        # Scene effect
        # takes 65.1 ms
        cube = self.get_projected_scene(in_photons=True,
                                        switch_off=switch_off,
                                        psf_profile=psf_profile,
                                        as_oversampled=as_oversampled,
                                        oversampling=oversampling,
                                        apply_lsf=apply_lsf,
                                        cached=cached,
                                        **kwargs)  # (nlbda, ny, nx)

        xdisp_sigmas = self.spectrograph.get_xdisp_sigma_spectral()   # (nlbda,)

        try:
            width = self.extraction["xdisp_width"]
        except KeyError:
            # See Spectrograph.rescale_parameters
            width = round(self.extraction["xdisp_width_insigma"] * np.median(xdisp_sigmas) )

        # This assumes optimal extraction, only the variance depends
        # on detector parameters.
        # takes ~300ms
        sig_cube, var_cube = self.detector.estimate_spx_spectrum(
            cube,               # (nlbda, ny, nx) [ph/s]
            sigma=xdisp_sigmas,       # (nlbda,) [px]
            width=width)


        return sig_cube, var_cube  # (nlbda, ny, nx) [ADU, ADU²]

    def get_slice(self, lbda_range, frame="obs", switch_off=[], incl_error=False,
                  squeeze=False, psf_profile="default",
                  as_oversampled=False, oversampling=None,
                  apply_lsf=True, cached=False,
                  **kwargs):
        """Get images of the cube summed over wavelength ranges.

        Parameters
        ----------
        lbda_range : tuple of float or list of tuple
            (lbda_min, lbda_max) for a single slice, or
            ((lbda_min, lbda_max), (lbda_min, lbda_max), ...) for several
            slices [Angstrom].
        frame : str, optional
            Wavelength frame of ``lbda_range`` (``"obs"`` or ``"rest"``).
            Default is ``"obs"``.
        switch_off : list of str, optional
            Variance sources to switch off (see `Simulation.get_cube`).
            Default is ``[]``.
        incl_error : bool, optional
            If True, the signal is randomly scattered following its
            (Gaussian) error. Default is False.
        squeeze : bool, optional
            If True, squeeze the output arrays. Default is False.
        psf_profile : str, optional
            PSF profile to use. Default is ``"default"``.
        as_oversampled : bool, optional
            If True, use the oversampled cube. Default is False.
        oversampling : int or None, optional
            Oversampling factor. Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        cached : bool, optional
            If True, use cached intermediate products (see
            `Simulation.get_cube`). Default is False.
        **kwargs
            Goes to `Simulation.get_cube`.

        Returns
        -------
        signal : numpy.ndarray
            (nslice, ny, nx) signal [ADU].
        variance : numpy.ndarray
            (nslice, ny, nx) variance [ADU^2].

        Raises
        ------
        ValueError
            If ``frame`` is neither ``"obs"`` nor ``"rest"``.

        """
        lbda_range = np.atleast_2d(lbda_range) # [[wmin, wmax]]

        if frame not in ["obs", "rest"]:
            raise ValueError(f"Unknown wavelength frame {frame!r}.")

        if frame == "rest" and (redshift := self.get_parameter("redshift", as_dict=False)) is not None:
            lbda_range = lbda_range * (1 + redshift)


        cube_signal, cube_variance = self.get_cube(switch_off=switch_off,
                                                   psf_profile=psf_profile,
                                                   as_oversampled=as_oversampled,
                                                   oversampling=oversampling,
                                                   apply_lsf=apply_lsf, cached=cached,
                                                   **kwargs)

        band_sigs = self.spectrograph.cube_to_slice(cube_signal, lbda_range, func=np.nansum, squeeze=squeeze)
        band_vars = self.spectrograph.cube_to_slice(cube_variance, lbda_range, func=np.nansum, squeeze=squeeze)

        if incl_error:
            band_sigs = np.random.normal(loc=band_sigs, scale=np.sqrt(band_vars)) # normal noise

        return band_sigs, band_vars

    def get_spectrum(self, switch_off=[], incl_error=False,
                         of_host=False,
                         psf_profile="default",
                         apply_lsf=True, cached=False):
        """Get the extracted spectrum and its variance.

        If the scene has a point source, its signal is assumed to be
        perfectly extracted, while its variance is computed from the variance
        cube within the extraction aperture.

        Otherwise (e.g. a galaxy alone), the host spectrum is extracted
        within the aperture defined by the ``aperture`` and
        ``aperture.position`` extraction parameters (see
        `Simulation.get_aperture`): the signal is measured from the
        noise-free host cube and the variance from the variance cube, with
        the same weights.

        Both include all ``nramps`` ramps.

        Parameters
        ----------
        switch_off : list of str, optional
            Variance sources to switch off (see `Simulation.get_cube`).
            Default is ``[]``.
        incl_error : bool, optional
            If True, the signal is randomly scattered following its
            (Gaussian) error. Default is False.
        of_host: bool, optional
            If True, extract the host spectrum (even if a point source is
            present). Default is False.
        psf_profile : str, optional
            PSF profile to use. Default is ``"default"``.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        cached : bool, optional
            If True, use cached intermediate products (see
            `Simulation.get_cube`). Default is False.

        Returns
        -------
        lbda : numpy.ndarray
            (nlbda,) wavelength [Angstrom].
        signal : numpy.ndarray
            (nlbda,) signal [ADU].
        variance : numpy.ndarray
            (nlbda,) variance [ADU^2].

        """
        # (lbda, nx, ny) [ADU]
        _, var_cube = self.get_cube(switch_off=switch_off,
                                               per_ramp=True, # see later.
                                               psf_profile=psf_profile,
                                               apply_lsf=apply_lsf, cached=cached)

        # no pointsource, e.g. a galaxy alone: extract the host from the cubes
        if not self.scene.has_element("pointsource") or of_host:
            spec_signal, spec_variance = self._get_host_spectrum_(var_cube, switch_off=switch_off,
                                                                  psf_profile=psf_profile,
                                                                  apply_lsf=apply_lsf)

        else:
            # radius is used to limit where PSF and variance are considered, see pointsource_variance()
            spec_variance = self.spectrograph.pointsource_variance(var_cube,
                                                    position=self.scene.pointsource_position,
                                                    radius=self._get_pointsource_radius_(),
                                                    psf_profile=psf_profile)

            # Assume the spectrum is perfectly extracted
            if "pointsource" not in switch_off:
                # the input spectrum *is not* derived from the cube.
                lbda = self.spectrograph.lbda
                _, pointsource_flambda = self.scene.get_element_spectrum('pointsource', lbda=lbda)
                if apply_lsf:
                    # apply LSF on *true* spectrum.
                    pointsource_flambda = self.spectrograph.apply_line_spread_function(pointsource_flambda)

                pointsource_phflux = pointsource_flambda*self.spectrograph.flambda2photon
                spec_signal = pointsource_phflux * self.detector.photonflux_to_adu(lbda)
            else:
                spec_signal = np.zeros_like(self.spectrograph.lbda)

        if (nexp := self.extraction["nramps"]) > 1:  # Nb of exposures
            spec_signal *= nexp
            spec_variance *= nexp

        if incl_error:
            spec_signal += np.random.normal(loc=0, scale=np.sqrt(spec_variance))

        return self.spectrograph.lbda, spec_signal, spec_variance  # (nlbda,) [ADU, ADU²]

    def get_aperture(self, aperture=None, position=None):
        """Get the aperture used to extract the host spectrum.

        It is used by `Simulation.get_spectrum` when the scene has no point
        source (e.g. a galaxy alone).

        Parameters
        ----------
        aperture : str or None, optional
            Aperture definition:

            - ``"core"``: the core of the galaxy, extracted as a point source
              would be, i.e. optimal (PSF-weighted) extraction within the
              point-source aperture radius (``aperture_radius`` or
              ``aperture_radius_insigma``);
            - ``"<n>-arcsec"``: aperture photometry (plain summation) within
              a radius of ``n`` arcsec, e.g. ``"1-arcsec"``;
            - ``"<n>-kpc"``: aperture photometry within a radius of ``n`` kpc
              (proper) at the scene redshift, e.g. ``"2-kpc"``.

            If None, the ``aperture`` extraction parameter is used (default
            ``"1-arcsec"``). Default is None.
        position : tuple of float, str or None, optional
            ``(x, y)`` aperture center, in spaxels, or ``"host"`` for the
            host position. If None, the ``aperture.position`` extraction
            parameter is used (default ``"host"``). Default is None.

        Returns
        -------
        position : tuple of float
            ``(x, y)`` aperture center, in spaxels.
        radius : float
            Aperture radius, in spaxels.
        optimal : bool
            True for an optimal (PSF-weighted) extraction (``"core"``), False
            for a plain summation.

        Raises
        ------
        ValueError
            If ``aperture`` cannot be parsed, or if a kpc aperture is
            requested without redshift.
        NotImplementedError
            If the spaxels are not square (non scalar ``spx_spatial_scale``).
        """
        if aperture is None:
            aperture = self.extraction.get("aperture", "1-arcsec")

        # position
        if position is None:
            position = self.extraction.get("aperture.position", "host")

        if isinstance(position, str) and position == "host":
            position = self.scene.host.meta.get("position", None) if self.scene.has_element("host") else None

        if position is None: # grid center
            position = (0, 0)

        # radius
        if aperture == "core": # as a pointsource
            return tuple(position), self._get_pointsource_radius_(), True

        try:
            size, unit = aperture.rsplit("-", 1)
            size = float(size)
        except (AttributeError, ValueError):
            raise ValueError(f"cannot parse {aperture=}. Must be 'core', '<n>-arcsec' or '<n>-kpc'.")

        if unit == "kpc":
            from astropy.cosmology import Planck18
            if (redshift := self.get_parameter("redshift", as_dict=False)) is None:
                raise ValueError(f"{aperture=} requires a redshift.")

            cosmology = self.scene.host.meta.get("cosmology", Planck18) if self.scene.has_element("host") else Planck18
            size = size * cosmology.arcsec_per_kpc_proper( float(redshift) ).value # [arcsec]

        elif unit != "arcsec":
            raise ValueError(f"cannot parse {aperture=}. Must be 'core', '<n>-arcsec' or '<n>-kpc'.")

        if np.ndim(self.spectrograph.spx_spatial_scale) != 0:
            raise NotImplementedError("aperture requires a scalar spatial_scale.")

        return tuple(position), size / self.spectrograph.spx_spatial_scale, False

    def _get_host_spectrum_(self, var_cube, switch_off=[], psf_profile="default", apply_lsf=True):
        """Host spectrum extracted within the aperture (see `Simulation.get_aperture`).

        Parameters
        ----------
        var_cube : numpy.ndarray
            (nlbda, ny, nx) variance cube [ADU^2], per ramp.
        switch_off : list of str, optional
            If ``"host"`` is in it, the signal is zero. Default is ``[]``.
        psf_profile : str, optional
            PSF profile to use. Default is ``"default"``.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.

        Returns
        -------
        signal, variance : numpy.ndarray
            (nlbda,) signal [ADU] and variance [ADU^2], per ramp.
        """
        position, radius, optimal = self.get_aperture()

        if "host" not in switch_off and self.scene.has_element("host"):
            # noise-free host cube | [ph/s] -> [ADU]
            host_cube = self.get_projected_scene(in_photons=True,
                                                 switch_off=["pointsource", "background", "thermal"],
                                                 psf_profile=psf_profile,
                                                 apply_lsf=apply_lsf)
            host_cube *= np.reshape(self.detector.photonflux_to_adu(self.spectrograph.lbda), (-1, 1, 1))
        else:
            host_cube = np.zeros_like(var_cube)

        return self.spectrograph.aperture_extraction(host_cube, var_cube,
                                                     position=position, radius=radius,
                                                     psf_profile=psf_profile,
                                                     optimal=optimal)

    def _get_pointsource_radius_(self):
        """Point source extraction aperture radius [spx/slicers]."""
        try:
            radius = self.extraction["aperture_radius"]  # [spx/slicers]
        except KeyError:
            # See Spectrograph.rescale_parameters
            radius = (self.extraction["aperture_radius_insigma"] *
                      self.spectrograph.psf_sigma_spectral /  # [arcsec]
                      self.spectrograph.spx_spatial_scale)    # [arcsec/spx]

        return radius

    def get_snr(self, switch_off=[]):
        """Get the signal-to-noise ratio spectrum.

        See `Simulation.get_spectrum`.

        Parameters
        ----------
        switch_off : list of str, optional
            Variance sources to switch off (see `Simulation.get_cube`).
            Default is ``[]``.

        Returns
        -------
        numpy.ndarray
            (nlbda,) signal-to-noise ratio per wavelength bin.

        """
        _, signal, variance = self.get_spectrum(switch_off=switch_off)
        return signal / variance**0.5

    def get_band_flux(self, lbda_range, frame="obs",
                        statistic=np.nanmean,
                        squeeze=True, **kwargs):
        """Estimate the mean signal and variance over a spectral domain.

        The photometry is computed assuming a top-hat filter and simply
        considering wavelength bins within the spectral domain (no
        interpolation nor weighting).

        Parameters
        ----------
        lbda_range : tuple of float or list of tuple
            (wmin, wmax) test wavelength range (or list of) [Angstrom].
        frame : str, optional
            Wavelength frame of ``lbda_range`` (``"obs"`` or ``"rest"``).
            Default is ``"obs"``.
        statistic : callable, optional
            Function applied on the signal and variance of the wavelength
            bins within the test domain (called with ``axis=0``).
            Default is `numpy.nanmean`.
        squeeze : bool, optional
            If True and a single range is given, return floats rather than
            arrays. Default is True.
        **kwargs
            Goes to `Simulation.get_spectrum` (e.g. ``switch_off``).

        Returns
        -------
        signal : float or numpy.ndarray
            Signal per band [ADU].
        variance : float or numpy.ndarray
            Variance per band [ADU^2].

        Raises
        ------
        ValueError
            If ``frame`` is neither ``"obs"`` nor ``"rest"``.

        """
        # accepts multiple ranges
        lbda_range = np.atleast_2d(lbda_range) # [[wmin, wmax]]

        if frame not in ["obs", "rest"]:
            raise ValueError(f"Unknown wavelength frame {frame!r}.")

        if frame == "rest" and (redshift := self.get_parameter("redshift", as_dict=False)) is not None:
            lbda_range = lbda_range * (1 + redshift)


        _, signal, variance = self.get_spectrum(**kwargs) # (2, lbda) in [ADU, ADU²]

        band_sigs, band_vars = [], []
        for wmin, wmax in lbda_range:
            sel = ((self.spectrograph.lbda >= wmin) &
                   (self.spectrograph.lbda <= wmax))

            band_sigs.append(statistic(signal[sel], axis=0))
            band_vars.append(statistic(variance[sel], axis=0))

        if len(lbda_range) == 1 and squeeze:
            return band_sigs[0], band_vars[0]
        else:
            return np.asarray(band_sigs), np.asarray(band_vars)

    def get_band_snr(self, lbda_range, frame="obs",
                     statistic=np.nanmean,
                     at_saturation=np.inf,
                     **kwargs):
        """Compute the signal-to-noise ratio over a spectral domain.

        The SNR is the band signal over the square root of the band
        variance, as computed by `Simulation.get_band_flux`.

        Parameters
        ----------
        lbda_range : tuple of float or list of tuple
            (wmin, wmax) test wavelength range (or list of) [Angstrom].
        frame : str, optional
            Wavelength frame of ``lbda_range`` (``"obs"`` or ``"rest"``).
            Default is ``"obs"``.
        statistic : callable, optional
            Function applied on the test domain (see
            `Simulation.get_band_flux`). Default is `numpy.nanmean`.
        at_saturation : float, optional
            Value returned if any detector pixel is saturated.
            Default is ``numpy.inf``.
        **kwargs
            Goes to `Simulation.get_band_flux`, then to
            `Simulation.get_spectrum` (e.g. ``switch_off``).

        Returns
        -------
        float or numpy.ndarray
            Signal-to-noise ratio (array if several ranges are given).

        """
        signal, variance = self.get_band_flux(lbda_range, frame,
                                              statistic=statistic,
                                              **kwargs)
        if self.detector.nsaturated_detpx >0:
            return at_saturation

        return signal / variance**0.5

    def get_times(self):
        """Get the detector times of the simulation.

        Returns
        -------
        dict
            Times [s], with keys ``"integration_time"``,
            ``"exposure_time"``, ``"tframe"``, ``"tgroup"`` (detector,
            per ramp) and ``"total_exptime"`` (see
            `Simulation.observing_time`).

        """
        # detector ones
        times = {k: getattr(self.detector, k) for k in ["integration_time","exposure_time", "tframe", "tgroup"]}
        # effective one
        times["total_exptime"] = self.observing_time # incl nrampss
        return times

    def get_band_variance_contribution(self, lbda_range, frame="rest",
                                       statistic=np.nanmean):
        """Estimate the contributions to the total variance over a band.

        Loops over `Simulation.variance_sources` (detector effects, scene
        elements and thermal emission), switching each off in turn, to
        estimate its contribution to the total band variance.

        Parameters
        ----------
        lbda_range : tuple of float or list of tuple
            (wmin, wmax) test wavelength range (or list of) [Angstrom].
        frame : str, optional
            Wavelength frame of ``lbda_range`` (``"obs"`` or ``"rest"``).
            Default is ``"rest"``.
        statistic : callable, optional
            Function applied on the test domain (see
            `Simulation.get_band_flux`). Default is `numpy.nanmean`.

        Returns
        -------
        dict
            Dictionary with keys ``"snr"``, ``"exptime"`` and ``"inttime"``
            (per ramp) and ``"obstime"`` (total) [s], ``"signal"`` [ADU],
            ``"variance"`` [ADU^2], and for each variance source ``source``,
            its variance contribution (key ``source``, [ADU^2]) and its
            fraction of the total variance (key ``frac_<source>``).

        """

        prop = dict(statistic=statistic, frame=frame)
        signal, variance = self.get_band_flux(lbda_range, **prop)

        # Detector elements
        # Dark
        variance_contribution = {}
        for source in self.variance_sources:
            _, variance_nosource = self.get_band_flux( lbda_range,
                                                       switch_off=[source],
                                                       **prop)
            variance_contribution[source] = variance - variance_nosource
            variance_contribution[f"frac_{source}"] = variance_contribution[source]/variance


        return {"snr": signal / variance**0.5,
                "exptime": self.detector.exposure_time,     # [s] (total per exp)
                "inttime": self.detector.integration_time,  # [s] (between extrema groups per exp)
                "obstime": self.observing_time,             # [s] (total observing time)
                "signal": signal,                           # [ADU]
                "variance": variance,                       # [ADU²]
                } | variance_contribution


    def get_variance_contribution(self, as_dataframe=True):
        """Estimate the noise contributions to the total spectrum variance.

        Each source of `Simulation.variance_sources` is switched off in turn
        to estimate its contribution per wavelength bin.

        Parameters
        ----------
        as_dataframe : bool, optional
            If True, return a `pandas.DataFrame`, otherwise a dict.
            Default is True.

        Returns
        -------
        pandas.DataFrame or dict
            Columns ``"lbda"`` [Angstrom], ``"flux"`` [ADU],
            ``"variance"`` [ADU^2] and one column per variance source
            (variance contribution [ADU^2]).

        """
        lbda, flux, variance = self.get_spectrum()
        estimates = {"lbda": lbda, "flux": flux, "variance": variance}

        for effect in self.variance_sources:
            _, _, variance_noeffect = self.get_spectrum( switch_off=[effect] )
            estimates[effect] = variance - variance_noeffect

        if as_dataframe:
            return pandas.DataFrame(estimates)

        return estimates # dict

    def get_data_volume(self, units="GB", per_ramp=False, **kwargs):
        """Get the total data volume of the simulation.

        Parameters
        ----------
        units : str, optional
            Units of the data volume (e.g. ``"GB"``, ``"MB"``). Default is ``"GB"``.
        per_ramp : bool, optional
            If True, return the data volume per ramp instead of the total
            data volume (i.e., not multiplied by the number of ramps).
            Default is False.
        **kwargs
            Goes to `~slicersim.detector.Detector.get_data_volume`.

        Returns
        -------
        float
            Data volume in the requested units.

        """

        data_volume_per_ramp = self.detector.get_data_volume(units=units, **kwargs)
        if per_ramp:
            return data_volume_per_ramp

        nramps = self.get_parameter("nramps")
        return data_volume_per_ramp * nramps

    # ----------- #
    #  Conversion #
    # ----------- #
    def convert_units(self, units_in, units_out, flux_in=1):
        """Convert flux units.

        Conversions are only implemented between ``"adu"`` (total signal
        integrated on the detector, over all ramps) and one of:

        - ``"flambda"`` (or ``"erg/s/a/cm2"``) [erg/s/cm^2/A];
        - ``"rate"`` (or ``"adu/s"``) [ADU/s];
        - ``"framerate"`` (or ``"adu/frame"``) [ADU/frame];
        - ``"fphoton"`` (or ``"ph/s"``) [ph/s].

        Parameters
        ----------
        units_in : str
            Input units.
        units_out : str
            Output units.
        flux_in : float or array_like, optional
            Input flux. Default is 1 (returns the conversion factor).

        Returns
        -------
        float or numpy.ndarray
            Converted flux (wavelength dependent, shape (nlbda,), for the
            ``"flambda"`` and ``"fphoton"`` conversions).

        Raises
        ------
        NotImplementedError
            If the requested conversion is not implemented.

        """
        NAME_CONVERTION = {"ph/s": "fphoton",
                           "erg/s/a/cm2": "flambda",
                           "adu/frame": "framerate",
                           "adu/s": "rate"}

        # allowing other names
        units_in = NAME_CONVERTION.get(units_in, units_in)
        units_out = NAME_CONVERTION.get(units_out, units_out)


        if units_in == units_out:
            coefs = 1

        # ADU
        # adu <=> flambda
        elif units_in == "adu" and units_out == "flambda":
            _, transmission = self.get_effective_transmission()
            coefs = 1 / (transmission * self.get_parameter("nramps"))

        elif units_in == "flambda" and units_out == "adu":
            coefs = 1/self.convert_units("adu", "flambda")

        # adu <=> rate
        elif units_in == "adu" and units_out == "rate":
            coefs = 1/self.observing_time
        elif units_in == "rate" and units_out == "adu":
            coefs = 1/self.convert_units("adu", "rate")

        # adu <=> frame-rate
        elif units_in == "adu" and units_out == "framerate":
            # adu
            coefs = self.detector.tframe/self.observing_time
        elif units_in == "framerate" and units_out == "adu":
            coefs =  1/self.convert_units("adu", "framerate")

        # adu <=> photons
        elif units_in == "adu" and units_out == "fphoton":
            coefs = 1/ (self.detector.photonflux_to_adu(self.spectrograph.lbda) * self.get_parameter("nramps"))
        elif units_in == "fphoton" and units_out == "adu":
            coefs = (self.detector.photonflux_to_adu(self.spectrograph.lbda) * self.get_parameter("nramps"))

        # not implemented
        else:
            raise NotImplementedError(f"{units_in} to {units_out} not implemented")

        return coefs*flux_in

    # ---------- #
    #  Fetching  #
    # ---------- #
    def fetch_snr(self, target_snr,
                      max_group=None,
                      nframe_per_group=None,
                      nframe_per_group_small=4,
                      small_ngroup_range=[32, 8],
                      allow_smaller_ramps=True,
                      too_large=1.,
                      ndrop=None,
                      guess=None,
                      use_cache=True,
                      lbda_range=[4000, 6800], frame="rest",
                      statistic=np.nanmean,
                      reset_param=True,
                      maxiter=30, tol=0.5, iterstep=1):
        """Find the exposure configuration reaching a target SNR.

        The read-out configuration ``nmd`` (number of groups, frames per
        group, dropped frames) and the number of ramps ``nramps`` are
        varied to reach ``target_snr`` (see `Simulation.get_band_snr`):

        - if a single ramp with ``max_group`` groups is enough, a single
          ramp is used and its number of groups is optimised (using
          ``nframe_per_group_small`` frames per group, or even single-frame
          groups, for bright targets);
        - otherwise, full ramps of ``max_group`` groups are used and the
          number of ramps (at least 2) is optimised. If
          ``allow_smaller_ramps``, the number of groups is then fine-tuned
          for that number of ramps.

        Parameters
        ----------
        target_snr : float
            Target signal-to-noise ratio.
        max_group : int or None, optional
            Maximum number of groups per ramp. If None,
            ``detector.max_group`` is used. Default is None.
        nframe_per_group : int or None, optional
            Number of frames per group for full ramps. If None, the current
            one is used. Default is None.
        nframe_per_group_small : int, optional
            Number of frames per group used for single short ramps (bright
            targets). Default is 4.
        small_ngroup_range : list of int, optional
            Range of number of groups tested for single short ramps.
            Default is ``[32, 8]``.
        allow_smaller_ramps : bool, optional
            If N+1 ramps give a too large SNR (see ``too_large``) while N
            ramps are not enough, reduce the number of groups of the
            N+1 ramps to reach the target SNR. Default is True.
        too_large : float, optional
            SNR excess (absolute) above ``target_snr`` triggering the
            reduction of the number of groups. Ignored if
            ``allow_smaller_ramps`` is False. Default is 1.
        ndrop : int or None, optional
            Number of dropped frames. If None, the current one is used.
            Default is None.
        guess : int or None, optional
            Initial guess for the free parameter (number of groups or of
            ramps). If None, a default guess is used. Default is None.
        use_cache : bool, optional
            If True, cache the projected scene while iterating (only the
            read-out configuration changes). The cache is cleared at the
            end. Default is True.
        lbda_range : list of float, optional
            (wmin, wmax) test wavelength range [Angstrom].
            Default is ``[4000, 6800]``.
        frame : str, optional
            Wavelength frame of ``lbda_range`` (``"obs"`` or ``"rest"``).
            Default is ``"rest"``.
        statistic : callable, optional
            Function applied on the test domain to compute the SNR (see
            `Simulation.get_band_flux`). Default is `numpy.nanmean`.
        reset_param : bool, optional
            If True, the simulation is set back to its input ``nmd`` and
            ``nramps``. If False, it keeps the configuration reaching the
            SNR. Default is True.
        maxiter : int, optional
            Maximum number of iterations. Default is 30.
        tol : float, optional
            Tolerance on the SNR. Default is 0.5.
        iterstep : int, optional
            Initial step of the iteration. Default is 1.

        Returns
        -------
        config : dict
            Configuration reaching the SNR, with keys ``"nmd"`` and
            ``"nramps"`` (to be given to `Simulation.update`).
        snr : float
            Reached SNR.
        observing_time : float
            Total observing time of this configuration [s] (see
            `Simulation.observing_time`).

        """
        if max_group is None:
            max_group = self.detector.max_group

        # store current ramp and nmd
        input_nmd = self.get_parameter("nmd")
        input_nramps = self.get_parameter("nramps")

        # default values are these from the current config.
        if nframe_per_group is None:
            nframe_per_group = int(input_nmd[1] + 0.) # copy

        if ndrop is None:
            ndrop = input_nmd[2]

        # How do we compute the snr
        prop_fetch = dict(lbda_range=lbda_range, frame=frame,
                          statistic=statistic,
                          tol=tol)

        # this is one max ramp
        full_singleramp_config = {"nmd": (max_group, nframe_per_group, ndrop),
                                  "nramps": 1}
        self.update(**full_singleramp_config)

        # who the snr is computed
        snr_prop = {"lbda_range": lbda_range,
                    "frame": frame,
                    "statistic": statistic,
                    "cached": use_cache,
                    # handles staturation SNR.
                    "at_saturation": np.inf}

        # compute the snr for 1 ramp.
        single_fullramp_snr = self.get_band_snr(**snr_prop)

        # is one ramp enought ?
        if (single_fullramp_snr >= (target_snr-tol) ):
            # yes ? Check if small ramp ok ?
            self.update( nramps = 1, nmd=(np.max(small_ngroup_range), nframe_per_group_small, 0) ) # e.g., 1* (n, 4, 0)
            single_ramp_smallgroup_snr = self.get_band_snr(**snr_prop)

            # do you reach the SNR with `np.max(small_ngroup_range)` "small ramps"
            if (single_ramp_smallgroup_snr <= (target_snr-tol)):
                # no ? use larger groups
                if guess is None:
                    guess = int(max_group/2)

                self.update( nramps = 1, nmd=(guess, nframe_per_group, 0) ) # e.g., 1* (n, 8, 0)
            else:
                # yes ? then it's a bright target.
                # => Is that so bright that (np.min(small_ngroup_range), 4, 0) would do the jobs ?
                self.update( nramps = 1, nmd=(np.min(small_ngroup_range), nframe_per_group_small, 0) ) # e.g., 1* (8, 4, 0)
                single_framegroup_snr = self.get_band_snr(**snr_prop)

                if (single_framegroup_snr >= (target_snr-tol)):
                    # yes ? then super bright, let's move to signel frame ramps starting from np.max(small_ngroup_range)
                    self.update( nramps = 1, nmd=(np.max(small_ngroup_range), 1, 0) ) # e.g., 1* (n, 1, 0)
                else:
                    # no ? then not that bright, let's stay in the 4 group ramps
                    self.update( nramps = 1, nmd=( int(np.mean(small_ngroup_range)), nframe_per_group_small, 0) ) # e.g., 1* (n, 4, 0)

            # In any case, you do 1 ramp, so loop over ngroup
            free_parameter = "ngroup"

        # is one ramp not enought:
        else:
             # no ? fix nmd at fullramp and  loop over ngroup
            free_parameter = "nramps"
            if guess is None:
                guess = 4 # we expect less than, say, 20 ramps and at least 2.

            self.update( nramps=guess, nmd=(max_group, nframe_per_group, ndrop) )
            prop_fetch |= {"min_value": 2}

        read_config, snr, integration_time = self._fetch_snr(target_snr,
                                                             free_parameter=free_parameter,
                                                             iterstep=iterstep, maxiter=maxiter,
                                                             use_cache=use_cache,
                                                             **prop_fetch)
        # fine tune ramps
        if free_parameter == "nramps" and allow_smaller_ramps and \
            ((snr - too_large > target_snr) or (snr + too_large < target_snr)):
            # number of ramp that leads to too high snr. nramps-1 is too low by design here.
            if snr - too_large > target_snr:
                nramps = read_config["nramps"]
            else:
                nramps = read_config["nramps"] + 1

            nmd = list(read_config["nmd"]) # this is too much, copy and change.
            nmd[0] -= 10  # let's start lower as it is currently at max_group.
            nmd[1] = nframe_per_group

            # fix the number of ramp, start -10 group as initial guess.
            self.update( nramps = nramps, nmd=nmd)
            free_parameter = "ngroup" # vary the number of groups.
            prop_fetch |= {"max_value": max_group} # make sure max_group is respected.
            read_config, snr, integration_time = self._fetch_snr(target_snr,
                                                             free_parameter=free_parameter,
                                                             iterstep=iterstep, maxiter=maxiter,
                                                             use_cache=use_cache,
                                                             **prop_fetch)
        if reset_param:
            self.update(nmd = input_nmd, nramps=input_nramps)


        if use_cache:
            # just to make sure it is all clean.
            self._reset_cached()

        # return what you where looking for.
        return read_config, snr, integration_time

    def _reset_cached(self):
        """ force reset any existing cached parameter. """
        self._cube_cached = None
        self.detector._cached_thermal = None


    def _fetch_snr(self, target_snr, free_parameter,
                   lbda_range=[4000, 6800], frame="rest", statistic=np.nanmean,
                   use_cache=False,
                   min_value=None, max_value=None,
                   maxiter=100, tol=0.5, iterstep=1):
        """Vary the free_parameter to reach the target SNR.

        Internal function with a fixed free parameter; see
        `Simulation.fetch_snr`.

        Parameters
        ----------
        target_snr : float
            Target signal to noise ratio.
        free_parameter : str
            Parameter to vary ('ngroup', 'nramps').
        lbda_range : list, optional
            (wmin, wmax) test wavelength range [Å]. Default is [4000, 6800].
        frame : str, optional
            Wavelength frame ('obs' or 'rest'). Default is "rest".
        statistic : callable, optional
            Function to apply on test domain to compute the snr. Default is
            `numpy.nanmean`.
        use_cache : bool, optional
            If True, use the cached projected scene. Default is False.
        min_value : int, optional
            Minimum value for the free parameter. Default is None.
        max_value : int, optional
            Maximum value for the free parameter. Default is None.
        maxiter : int, optional
            Maximum number of iterations. Default is 100.
        tol : float, optional
            Tolerance for the SNR. Default is 0.5.
        iterstep : int, optional
            Step for the iteration. Default is 1.

        Returns
        -------
        config : dict
            Reached configuration (``"nmd"`` and ``"nramps"`` keys).
        snr : float
            Reached SNR.
        observing_time : float
            Total observing time [s].

        """
        # minimal values (including these)
        minimal_values = {"ngroup": 2, "nramps": 1, 'nframe':2}
        if min_value is None:
            min_value = minimal_values.get(free_parameter)

        snr_prop = dict(lbda_range=lbda_range,
                        frame=frame,
                        statistic=statistic,
                        cached=use_cache)

        # internal function that perform the fit steps
        def change(value, current_snr, iterstep):
            """Update value in the given direction to approach target SNR."""

            if current_snr >= target_snr: # going down.
                was_high = True
                coefs = -1
            else: # going up
                was_high = False
                coefs = +1

            new_value = value + coefs*iterstep
            if new_value <=0 :
                new_value = 1
                # warnings.warn(f"requested value lower than 0: old value {value} + iterstep {iterstep}")

            new_value = max(new_value, min_value)
                # warnings.warn(f"requested value lower than min_value: old value {value} + iterstep {iterstep}")
            if max_value is not None and (new_value > max_value):
                new_value = max_value

            _ = self.update(**{free_parameter: new_value})
            new_snr = self.get_band_snr(**snr_prop)

            # need to go in the same direction
            if (new_snr >= target_snr and was_high) or (new_snr < target_snr and not was_high):
                new_iterstep = int(iterstep*2)
            # need to come back
            else:
                new_iterstep = 1

            return new_value, new_snr, new_iterstep

        # ---------------------- #
        # Parsing input uptions  #
        # ---------------------- #
        if free_parameter not in ["ngroup", "nramps", 'nframe']:
            raise ValueError(f"free_parameter should be 'ngroup', 'nramps' or 'nframe' {free_parameter} given.")


        # nframe supposed to change the macc mode to (1,1,0)
        if free_parameter == "nframe":
            self.update(nmd=(min_value, 1, 0)) # start at min value
            free_parameter = "ngroup" # 1 frame per group, so ngroup=nframe

        # initial state
        current_value = self.get_parameter(free_parameter)
        current_snr = self.get_band_snr(**snr_prop)
        break_at_next = False
        # while loop
        counter = 0
        value_history = []
        while np.abs(current_snr - target_snr) > tol and counter < maxiter:
            value_history.append(current_value)
            if np.isnan(current_value) or current_value<=min_value: # moving to too small value
                current_value = min_value # reset to minimum
                self.update(**{free_parameter: current_value })
                current_snr = self.get_band_snr(**snr_prop) # and get corresponding SNR

            if break_at_next or (current_value<=min_value and current_snr>=target_snr): # means lowest is already enough
                break

            current_value, current_snr, iterstep = change(current_value, current_snr, iterstep)

            if len(value_history)>3 and value_history[-2] == current_value:
                # we are stuck in a loop,
                if current_value > value_history[-1]: # we are above the target SNR
                    break
                else:
                    # do one more round.
                    break_at_next = True

            counter += 1

        # So this is what is used.
        used_config = {"nmd": self.get_parameter("nmd"),
                       "nramps": self.get_parameter("nramps")}

        total_exptime = self.observing_time # includes nrampss
        return used_config, current_snr, total_exptime

    # ---------- #
    #  Plotting  #
    # ---------- #
    def show_config(self, axes=None, lbda_units=None, colors=["#194D80", "#A8B7C7"] ):
        """Show configuration (resolving power and throughput).

        Parameters
        ----------
        axes : tuple of matplotlib.axes.Axes or None, optional
            (axr, axt) axes for the resolving power and the throughput. If
            None, a new figure is created. Default is None.
        lbda_units : str or None, optional
            Units (understood by `astropy.units`) to convert the wavelength
            to. If None, Angstrom is used. Default is None.
        colors : list of str, optional
            Colors used for the plot. Default is ``["#194D80", "#A8B7C7"]``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure.

        """

        if axes is None:
            import matplotlib.pyplot as plt
            fig, (axr, axt) = plt.subplots(nrows=2, gridspec_kw={"hspace": 0.05, "top":0.95})
        else:
            axr, axt = axes
            fig = axr.figure

        if lbda_units is not None:
            from astropy import units
            lbda = self.spectrograph.lbda * units.Angstrom.to(lbda_units)
        else:
            lbda = self.spectrograph.lbda


        # resolution power
        respow = self.spectrograph.get_resolving_power()
        axr.plot(lbda, respow, lw=2, color=colors[0])

        # throughput
        throughput = self.spectrograph.get_throughput(lbda=lbda)
        axt.fill_between(lbda, throughput,
                         facecolor=colors[1],
                         edgecolor="None", label="optics")

        # and qe
        qe = self.detector.get_qe(lbda=lbda)
        axt.fill_between(lbda, throughput*qe,
                         facecolor=colors[0], edgecolor="None",
                        label="+qe")

        axt.set_ylim(0, 1)

        axr.set_ylim(1)
        axr.set_xlim(*axt.get_xlim())
        axr.set_xticklabels([])

        axt.legend(fontsize="small", ncols=2, frameon=False)
        axr.set_ylabel("Resolving power $R$", fontsize="large")
        axt.set_ylabel("Transmission", fontsize="large")
        axt.set_xlabel("Wavelength", fontsize="large")

        [ax_.set_facecolor("w") for ax_ in fig.axes]
        fig.set_facecolor("None")

        return fig

    def show_spectrum(self, ax=None, switch_off=[], snr=False, **kwargs): # pragma: no cover
        """Plot the detected spectrum and its error.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure is created.
            Default is None.
        switch_off : list of str, optional
            Variance sources to switch off (see `Simulation.get_cube`).
            Default is ``[]``.
        snr : bool, optional
            If True, add the SNR curve on a twin y-axis. Default is False.
        **kwargs
            Goes to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure.

        """

        # data [ADU]
        _, sig, var = self.get_spectrum(switch_off=switch_off)
        dsig = np.where(var >= 0, var ** 0.5, np.Inf)

        # Axes
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=[7, 4])

        # Actual plot
        lbda_mu = self.spectrograph.lbda / 1e4
        line, = ax.plot(lbda_mu, sig, ds='steps-mid', **kwargs)
        ax.fill_between(lbda_mu, sig - dsig, sig + dsig,
                        step='mid', color=line.get_color(), alpha=0.2)
        ax.set(xlabel="Obsframe wavelength [µm]", ylabel="Signal [ADU]")
        if snr:
            ax2 = ax.twinx()
            ax2.plot(lbda_mu, sig / dsig, ds='steps-mid', c='C03')
            ax2.set_ylabel("SNR", color='C03')
            ax2.tick_params(axis='y', labelcolor='C03')

        if self.detector.nsaturated_detpx:
            npx = self.detector.nsaturated_detpx
            ax.annotate(f"WARNING: {npx} px saturated",
                        (0.05, 0.05), xycoords='axes fraction', c='r')

        return fig

    def show_cube(self, in_photons=True, switch_off=[], spec_prop={},
                      psf_profile="default", cached=False, **kwargs): # pragma: no cover
        """Display the cube generated by `Simulation.get_projected_scene`.

        The figure has two panels:

        - left: total spectrum (cube summed over spaxels);
        - right: white image (cube summed over wavelengths).

        Parameters
        ----------
        in_photons : bool, optional
            If True, the cube is in [ph/s], otherwise in [erg/s/cm^2/A].
            Default is True.
        switch_off : list of str, optional
            Components to discard (see `Simulation.get_projected_scene`).
            Default is ``[]``.
        spec_prop : dict, optional
            Spectrum plot options (goes to `matplotlib.axes.Axes.plot`).
            Default is ``{}``.
        psf_profile : str, optional
            PSF profile to use. Default is ``"default"``.
        cached : bool, optional
            If True, use the cached projected scene. Default is False.
        **kwargs
            Goes to `matplotlib.axes.Axes.imshow` (image panel).

        Returns
        -------
        matplotlib.figure.Figure
            2-axis figure.

        """

        cube = self.get_projected_scene(in_photons=in_photons,
                                            switch_off=switch_off,
                                            psf_profile=psf_profile,
                                            cached=cached)

        unit = "ph/s" if in_photons else "erg/cm²/Å/s"

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=[10, 3], tight_layout=True)
        grid = GridSpec(1, 5)
        ax_spec = fig.add_subplot(grid[0, :-2])
        ax_img = fig.add_subplot(grid[0, -2:])

        ax_spec.plot(self.spectrograph.lbda / 1e4, cube.sum(axis=(1, 2)),
                     ds='steps-mid', **spec_prop)
        ax_spec.set(xlabel="Wavelength [µm]", ylabel=f"Total signal [{unit}]")

        default = dict(cmap='Blues', origin="lower", aspect="equal")
        img = ax_img.imshow(cube.sum(axis=0),
                            extent=self.spectrograph.spaxel_extent,
                            **{**default, **kwargs})
        ax_img.set(xlabel="x [spx]", ylabel="y [spx]")
        fig.colorbar(img, ax=ax_img, label="Integrated signal")

        return fig

    def show_scene(self, incl_error=True, fig=None,
                   rest_lbda_range = [4000, 7000],
                   obs_lbda_ranges = [[4000, 6000], [9000, 11_000], [14_000, 16_000]],
                   psf_profile="default", oversampling=None, prop_slice={},
                  **kwargs): # pragma: no cover
        """Show images of the observed scene.

        The top panel shows the image in a rest-frame wavelength range, the
        bottom panels the images in observed-frame wavelength ranges (see
        `Simulation.get_slice`).

        Parameters
        ----------
        incl_error : bool, optional
            If True, the images are randomly scattered following their
            errors. Default is True.
        fig : matplotlib.figure.Figure or None, optional
            Figure to draw on. If None, a new one is created.
            Default is None.
        rest_lbda_range : list of float, optional
            Rest-frame wavelength range [Angstrom].
            Default is ``[4000, 7000]``.
        obs_lbda_ranges : list of list of float, optional
            Observed-frame wavelength ranges [Angstrom] (at most 3).
            Default is ``[[4000, 6000], [9000, 11_000], [14_000, 16_000]]``.
        psf_profile : str, optional
            PSF profile to use. Default is ``"default"``.
        oversampling : int or None, optional
            If given, show oversampled images with this oversampling factor.
            Default is None.
        prop_slice : dict, optional
            Goes to `Simulation.get_slice`. Default is ``{}``.
        **kwargs
            Goes to `matplotlib.axes.Axes.imshow`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure.

        """

        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(7,5))

        ncols = len(obs_lbda_ranges)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=2, ncols=ncols, figure=fig,
                      height_ratios=(ncols,1),
                      hspace=0.0, wspace=0.05)

        ax1 = fig.add_subplot(gs[0, :])
        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))

        if oversampling is not None:
            prop_slice["oversampling"] = oversampling
            prop_slice["as_oversampled"] = True

        flux, _ = self.get_slice(obs_lbda_ranges, frame="obs",
                                         incl_error=incl_error, psf_profile=psf_profile,
                                        **prop_slice)
        flux_rest,_ = self.get_slice(rest_lbda_range, frame="rest",
                                         incl_error=incl_error, psf_profile=psf_profile,
                                         **prop_slice)
        # Rest frame top panel
        ax1.imshow(flux_rest[0],
                        cmap="cividis",
                        **kwargs)
        ax1.text(1, 1, r"$\lambda \in$"+f"[{rest_lbda_range[0]}, {rest_lbda_range[1]}] rest",
                 va="top", ha="right",
                 transform=ax1.transAxes, color="w")

        # Obs frame bottom panel
        for i, (obs_, cmap_) in enumerate(zip(flux, ["Blues", "Greens", "Reds"])):
            ax_ = fig.add_subplot(gs[1, i])
            ax_.imshow(obs_, cmap=cmap_, **kwargs)
            ax_.text(1, 1,
                     r"$\lambda \in$"+f"[{obs_lbda_ranges[i][0]}, {obs_lbda_ranges[i][1]}] obs",
                     va="top", ha="right",fontsize="small",
                     transform=ax_.transAxes, color="k")

        [ax_.set_yticks([]) for ax_ in fig.axes]#[axb,axg, axr]]
        [ax_.set_xticks([]) for ax_ in fig.axes]

        return fig


    def show_nea_fwhm(self, figsize=(4,7)): # pragma: no cover
        """Show the noise equivalent areas (NEA) and the FWHM.

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size. Default is (4, 7).

        Returns
        -------
        matplotlib.figure.Figure
            Figure.

        """
        import matplotlib.pyplot as plt

        fig, (axnea, axneaspatial, axfwhm) = plt.subplots(ncols=1, nrows=3, figsize=figsize,
                                                         gridspec_kw={"hspace":0.05})
        self.spectrograph.show_nea(ax=axnea)
        axnea.set_xticklabels([])

        self.spectrograph.show_nea_spatial(ax=axneaspatial, legend=False)
        axneaspatial.set_xticklabels([])

        self.spectrograph.show_fwhm(ax=axfwhm, legend=False)
        return fig

    def show_variance_sources(self, variance_contrib=None, flux_calibrated=True,
                                  normalize_variances=True,
                                  lbda_nm=False,
                                  figsize=(7, 7), gridspec={}): # pragma: no cover
        """Summary figure showing various variance contributions.

        Parameters
        ----------
        variance_contrib : pandas.DataFrame or None, optional
            Variance contributions, as returned by
            ``self.get_variance_contribution(as_dataframe=True)``
            (see `Simulation.get_variance_contribution`). If None, it is
            computed. Default is None.
        flux_calibrated : bool, optional
            If True, show the flux-calibrated spectrum [erg/s/cm^2/A],
            otherwise in [ADU]. Default is True.
        normalize_variances : bool, optional
            If True, the variance contributions are normalised by their sum
            (fractions summing to 1), otherwise by the total variance.
            Default is True.
        lbda_nm : bool, optional
            If True, show wavelength in nanometers, otherwise in Angstrom.
            Default is False.
        figsize : tuple of float, optional
            Figure size. Default is (7, 7).
        gridspec : dict, optional
            Goes to ``gridspec_kw`` of `matplotlib.pyplot.subplots`.
            Default is ``{}``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure.

        """
        if variance_contrib is None:
            variance_contrib = self.get_variance_contribution(as_dataframe=True)

        if flux_calibrated:
            _, norm = self.get_effective_transmission()
        else:
            norm = 1

        # Figure definition
        import matplotlib.pyplot as plt
        fig, (ax, axsnr, axv) = plt.subplots(3, 1, figsize=figsize,
                                             gridspec_kw={"hspace":0.1} | gridspec)

        # Data of interest
        flux = variance_contrib["flux"]/norm
        variance = variance_contrib["variance"]
        if normalize_variances:
            variance_eff = variance_contrib[self.variance_sources].sum(axis=1)
        else:
            variance_eff = variance
        noise = np.sqrt(variance)/norm
        snr = flux/noise


        # wavelength
        if lbda_nm:
            lbda = variance_contrib["lbda"] /10  # convert to nm
        else:
            lbda = variance_contrib["lbda"] # Angstroms (default)


        # Main plot
        ax.plot(lbda, flux, lw=1, color="k")
        ax.fill_between(lbda, flux+noise, flux-noise, alpha=0.3,
                       color="0.5", lw=0)
        ax.axhline(0, color="0.5", lw=1, zorder=1)

        # Loop over effects
        in_effect = []
        base = 0
        for new_effect in self.variance_sources:
            in_effect = in_effect+[new_effect]

            spectra_contrib = variance_contrib[in_effect].sum(axis=1)/variance_eff
            axsnr.fill_between(lbda,
                                 base*snr,
                                 spectra_contrib*snr,
                                 facecolor=COLORS[new_effect],
                                 edgecolor="0.5", lw=0.,
                                 alpha=1)

            axv.fill_between(lbda,
                             base,
                             spectra_contrib,
                             facecolor=COLORS[new_effect],
                             edgecolor="0.5", lw=0.,
                             label=new_effect)

            base = spectra_contrib

        axsnr.plot(lbda, snr, color="0.5", lw=1)
        axsnr.plot(lbda, base*snr, color="k")

        # Fancy
        ax.set_xlim(lbda.values[0], lbda.values[-1])
        axsnr.set_xlim(*ax.get_xlim())
        axv.set_xlim(*ax.get_xlim())
        if normalize_variances:
            axv.set_ylim(0, 1)
        else:
            axv.set_ylim(0)


        ax.set_xticklabels([])
        axsnr.set_xticklabels([])
        axsnr.set_ylim(0)


        if lbda_nm:
            axv.set_xlabel("Wavelength [nm]", fontsize="large")
        else:
            axv.set_xlabel("Wavelength [A]", fontsize="large")

        if flux_calibrated:
            ax.set_ylabel("Flux [erg/s/cm2/A]", fontsize="large")
        else:
            ax.set_ylabel("Flux [ADU]", fontsize="large")
        axsnr.set_ylabel("Signal / Noise", fontsize="large")
        axv.set_ylabel("variance contrib.", fontsize="large")


        # reordering the labels
        handles, labels = axv.get_legend_handles_labels()
        axv.legend(handles=handles[::-1],
                   labels=labels[::-1],
                    loc=[0.01, 1.15], fontsize="small", frameon=False)

        title = f"z={self.get_parameter('redshift', as_dict=False)}"
        # SALT parameters, if any
        if (c := self.get_parameter('c', as_dict=False)) is not None and (x1 := self.get_parameter('x1', as_dict=False)) is not None:
            title += f" | c={c}, x1={x1}"

        ax.set_title(title + f" | t={self.get_times()['total_exptime']/60:.1f} min",
                    color="k", fontsize="small", loc="right")
        return fig

    # ================= #
    #   Properties      #
    # ================= #
    @property
    def meta(self):
        """Configuration of all elements (aka. meta), as a dict."""
        return self._in_meta | {"scene": self.scene.meta,
                                "telescope": self.telescope.meta,
                                "spectrograph": self.spectrograph.meta,
                                "detector": self.detector.meta,
                                "extraction":self.extraction}
    @property
    def cube_shape(self):
        """Shape of the generated 3d-cube (nlbda, ny, nx)."""
        return (self.spectrograph.nlbda,
                *self.spectrograph.spx_shape) # y, x

    @property
    def observing_time(self):
        """Total observing time [s], i.e. ``exposure_time * nramps``."""
        return self.detector.exposure_time * self.extraction["nramps"]

    @property
    def mutable_parameters(self):
        """List of the parameter names accepted by `Simulation.update`."""
        extra = [] + list(self.extraction.keys())
        scene_ = self.scene.mutable_parameters
        telescope_ = self.telescope.mutable_parameters
        spectro_ = self.spectrograph.mutable_parameters
        detector_ = [f"detector.{k}" for k in self.detector.mutable_parameters]
        all_mutables =  scene_+ telescope_ + spectro_ + detector_ + extra
        # remove lbda that could be in scene, as this is based on spectrograph
        if "lbda" in all_mutables:
            all_mutables.remove("lbda")

        return all_mutables

    @property
    def _elements(self): # test structure
        """Internal list of elements."""
        return ["scene", "telescope", "spectrograph", "detector", "extraction"]

    @property
    def variance_sources(self):
        """List of the variance source names (see `Simulation.get_cube`)."""
        return self.VARIANCE_SOURCES
