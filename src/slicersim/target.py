"""Targets, the observation-level view of a simulation.

A target wraps a `~slicersim.simulation.Simulation` and exposes what one
actually asks of an observation: exposure time and signal-to-noise
calculations, reconfiguration of the detector and of the spectrograph, and
access to the simulated spectra and cubes.

`VirtualTarget` holds that generic interface and is not meant to be used
directly. The concrete targets differ only by the scene they build:
`Supernova` for SN Ia models, `Kilonova` for kilonova models, `CalSpec` for
HST CalSpec standard stars, `BlackBody` and `PowerLaw` for analytic spectra,
and `Target` for an arbitrary input spectrum. Instrument-specific flavours are
defined in `~slicersim.lazuli`, and calibration exposures in
`~slicersim.calibration`.
"""

import numpy as np

from .simulation import Simulation
from .iotools import get_config
from .scene import get_scene


class VirtualTarget:
    """Base class of all targets (see child classes).

    This class provides a generic, observation-level interface to a
    `~slicersim.simulation.Simulation`. It is not intended to be used
    directly, but rather to be inherited by classes that define specific
    targets.

    Parameters
    ----------
    simulation : slicersim.simulation.Simulation, optional
        The simulation the target wraps. Default is None.
    """
    _DEFAULT_CONFIG = {"instrument": None}

    def __init__(self, simulation=None):
        """Initialize the VirtualTarget.

        Parameters
        ----------
        simulation : slicersim.simulation.Simulation, optional
            The simulation the target wraps. Default is None.
        """
        # set it.
        self._simulation = simulation


    @classmethod
    def _source_to_simulation_(cls, source, host=None, instrument=None, **kwargs):
        """ Internal method to get the simulation associated to this Target (uses class attributes)"""
        if instrument is None and hasattr(cls,"_INSTRUMENT"):
            instrument = cls._INSTRUMENT

        scene = get_scene(source=source, host=host, **kwargs)
        config = get_config( **( cls._DEFAULT_CONFIG | {"scene": scene, "instrument": instrument}) )

        return Simulation.from_config(config)


    @classmethod
    def from_scene(cls, scene=None, instrument=None, **kwargs):
        """Load the instance from a scene configuration.

        Parameters
        ----------
        scene : dict or str, optional
            Scene configuration (see `~slicersim.scene.get_scene`), in the
            format::

                scene = {"scene": {"pointsource": {},  # PSF
                                   "background": {},   # spatially flat
                                   "host": {}}}        # structured background

            or the name of a scene configuration file. Default is None.
        instrument : str or dict, optional
            Configuration of the instrument, if any. If None, the class
            ``_INSTRUMENT`` is used when defined. Default is None.
        **kwargs
            Goes to `~slicersim.iotools.get_config`, overriding the
            ``scene`` and ``instrument`` configuration.

        Returns
        -------
        ~slicersim.target.VirtualTarget
            An instance of the class.
        """
        if instrument is None and hasattr(cls,"_INSTRUMENT"):
            instrument = cls._INSTRUMENT

        # create the simulator
        config = get_config( **(cls._DEFAULT_CONFIG | {"instrument": instrument, "scene": scene} | kwargs) )
        simulation = Simulation.from_config( config )
        return cls(simulation=simulation)


    @classmethod
    def from_simulation(cls, simulation):
        """Load the instance from an already-built simulation.

        Parameters
        ----------
        simulation : slicersim.simulation.Simulation
            The simulation the target wraps.

        Returns
        -------
        ~slicersim.target.VirtualTarget
            An instance of the class.

        See Also
        --------
        from_scene : Build the simulation from a scene configuration.
        """
        return cls(simulation=simulation)


    def to_image(self, mapper, sliceid, image=None, **kwargs):
        """Project the simulated cube onto a detector image.

        Parameters
        ----------
        mapper : slicersim.mapper.SlicerMapper
            Mapper describing where each slice falls on the detector.
        sliceid : int or array_like
            Identifier(s) of the slice(s) the cube is projected onto.
        image : numpy.ndarray, optional
            Existing detector image this projection is added to. If None,
            only the contribution of this target is returned.
            Default is None.
        **kwargs
            Goes to ``get_cube``. The generic
            `~slicersim.target.VirtualTarget.get_cube`
            accepts none; use them with subclasses such as
            `~slicersim.lazuli.VirtualLazuliTarget`.

        Returns
        -------
        numpy.ndarray
            2D detector image. When ``image`` is given, the sum of ``image``
            and of the projected cube; ``image`` itself is left unchanged.
        """
        cube, *_ = self.get_cube(**kwargs)

        this_image = mapper.project_slice(sliceid, cube, lbda=self.simulation.spectrograph.lbda)
        if image is not None:
            this_image += image

        return this_image


    # =============== #
    #   Methods       #
    # =============== #
    def change_detector(self, nmd=None, max_group=None, nramps=None):
        """Change the detector read-out configuration.

        Parameters set to None are left unchanged.

        Parameters
        ----------
        nmd : tuple of int, optional
            MACC mode ``(n_groups, n_frames_per_group, n_drops)``.
            Default is None.
        max_group : int, optional
            Maximum number of groups in a single ramp. Default is None.
        nramps : int, optional
            Number of ramps. Default is None.

        Returns
        -------
        None
        """
        if max_group is not None:
            self.simulation.update(max_group=max_group)

        if nmd is not None:
            self.simulation.update(nmd=nmd)

        if nramps is not None:
            self.simulation.update(nramps=nramps)


    def change_spectrograph(self, spatial_shape=None, spatial_scale=None):
        """Change the spectrograph configuration.

        Parameters
        ----------
        spatial_shape : tuple of int, optional
            Spaxel grid shape (e.g. ``(40, 40)``). If None, it is left
            unchanged. Default is None.
        spatial_scale : float or list of float, optional
            Spaxel size in arcsec. If a float, spaxels are square; if a list,
            ``(x, y)``. If None, it is left unchanged. Default is None.

        Returns
        -------
        None
        """
        new_config = {}

        # manual setting if any
        if spatial_shape is not None:
            new_config["spatial_shape"] = spatial_shape

        if spatial_scale is not None:
            new_config["spatial_scale"] = spatial_scale

        return self.simulation.update(**new_config)


    # SETTER
    def change_properties(self, **kwargs):
        """Change any mutable parameter of the simulation.

        Shortcut to `~slicersim.simulation.Simulation.update`.

        Parameters
        ----------
        **kwargs
            The parameters to change, as ``name=value`` (e.g.
            ``redshift=0.5`` or ``pointsource__phase=-2``).

        Returns
        -------
        None
        """
        _ = self.simulation.update(**kwargs)


    def setup_to_snr(self, snr, per_resolution=True,
                     lbda_range=[4000, 6800], frame='rest',
                     statistic=np.nanmean, inplace=True, **kwarg):
        """Find the detector read-out configuration reaching a target SNR.

        The MACC mode and number of ramps are varied until the
        signal-to-noise ratio over ``lbda_range`` reaches ``snr`` (see
        `~slicersim.simulation.Simulation.fetch_snr`).

        Parameters
        ----------
        snr : float
            Target signal-to-noise ratio.
        per_resolution : bool, optional
            If True, ``snr`` is given per spectral resolution element and is
            converted into a SNR per wavelength bin,
            ``snr / sqrt(dispersion_resolution)``, before being passed to
            `~slicersim.simulation.Simulation.fetch_snr`. If False, ``snr``
            is per wavelength bin. Default is True.
        lbda_range : list of float, optional
            Wavelength range ``[min, max]`` in Angstrom over which the SNR is
            computed. Default is ``[4000, 6800]``.
        frame : str, optional
            Reference frame of ``lbda_range``: ``"rest"`` (rest-frame) or
            ``"obs"`` (observer-frame). Default is ``"rest"``.
        statistic : callable, optional
            Function used to summarise the SNR over ``lbda_range``.
            Default is `numpy.nanmean`.
        inplace : bool, optional
            If True, the simulation is updated to the returned
            configuration. Default is True.
        **kwarg
            Goes to `~slicersim.simulation.Simulation.fetch_snr` (e.g.
            ``max_group`` or ``ndrop``). ``nframe_per_group`` is set to 8 and
            cannot be given.

        Returns
        -------
        config : dict
            Read-out configuration reaching ``snr``, with keys ``"nmd"``
            and ``"nramps"``.
        reached_snr : float
            SNR reached with ``config``, per resolution element if
            ``per_resolution`` is True, per wavelength bin otherwise.
        """
        if per_resolution:
            # get the snr per wavelength bin as expected by fetch_snr
            snr = snr / np.sqrt(self.simulation.spectrograph.dispersion_resolution)

        # get configuration to reach this SNR
        config, reached_snr, exptime = self.simulation.fetch_snr(target_snr=snr,
                                                                 lbda_range=lbda_range, frame=frame,
                                                                 statistic=statistic,
                                                                 nframe_per_group=8, # default guess
                                                                 **kwarg)
        # update simulation at this config
        if inplace:
            self.simulation.update(**config)

        if per_resolution:
            reached_snr = reached_snr * np.sqrt(self.simulation.spectrograph.dispersion_resolution)

        return config, reached_snr


    # GETTER
    def get_exposure_time(self, snr=None, full=False, **kwargs):
        """Get the exposure time of the current configuration.

        Parameters
        ----------
        snr : float, optional
            Currently has no effect on the returned time: `setup_to_snr` is
            called with ``inplace=False``, so the configuration used for the
            exposure time remains the current one. Default is None.
        full : bool, optional
            If True, return all timing details; otherwise only the total
            exposure time. Default is False.
        **kwargs
            Goes to `setup_to_snr`. Ignored if ``snr`` is None.

        Returns
        -------
        float or dict
            Total exposure time in seconds (all ramps included), or, if
            ``full`` is True, a dict of times in seconds with keys
            ``"integration_time"``, ``"exposure_time"`` (one ramp),
            ``"tframe"``, ``"tgroup"`` and ``"total_exptime"``.
        """
        if snr is not None:
            self.setup_to_snr(snr=snr, inplace=False, **kwargs)

        times = self.simulation.get_times()
        if full:
            return times

        return times["total_exptime"]


    def get_readout_config(self):
        """Get the current MACC mode and number of ramps.

        Returns
        -------
        dict
            Read-out configuration, with keys:

            - ``"nmd"``: ``(n_groups, n_frames_per_group, n_drops)``.
            - ``"nramps"``: number of ramps, each ramp following the
              ``nmd`` MACC mode.
        """
        # This should be generalized once other detector than HxRG have been implemented.
        return self.get_properties(["nmd", "nramps"])


    def get_data_volume(self, units="GB", per_ramp=False):
        """Get the data volume of the observation.

        Parameters
        ----------
        units : str, optional
            Unit of the returned volume (any astropy data unit, e.g.
            ``"MB"`` or ``"GB"``). Default is ``"GB"``.
        per_ramp : bool, optional
            If True, return the volume of a single ramp instead of the total
            volume over all ramps. Default is False.

        Returns
        -------
        float
            Data volume in ``units``.

        See Also
        --------
        ~slicersim.simulation.Simulation.get_data_volume : The underlying method.
        """
        return self.simulation.get_data_volume(units=units, per_ramp=per_ramp)


    def get_spectrum(self, unit="adu", incl_error=True, **kwargs):
        """Get the simulated spectrum of the point source.

        The spectrum is extracted from the simulation in ADU and converted to
        ``unit``.

        Parameters
        ----------
        unit : str, optional
            Unit of the returned flux (and of the variance, squared):

            - ``"adu"``: total ADU integrated over all ramps.
            - ``"flambda"``: erg/s/cm^2/A.
            - ``"fphoton"``: photon/s.
            - ``"rate"``: ADU/s.
            - ``"framerate"``: ADU/frame.

            Default is ``"adu"``.
        incl_error : bool, optional
            If True, the returned flux is randomly scattered according to the
            variance (hence realistic). If False, the noise-free flux is
            returned. Default is True.
        **kwargs
            Goes to `~slicersim.simulation.Simulation.get_spectrum` (e.g.
            ``switch_off``).

        Returns
        -------
        lbda : numpy.ndarray
            Wavelengths in Angstrom.
        flux : numpy.ndarray
            Flux in ``unit``.
        variance : numpy.ndarray
            Variance of the flux, in ``unit`` squared.
        """
        # get spectra [adu]
        lbda, flux, variance = self.simulation.get_spectrum(incl_error=incl_error, **kwargs)

        # change the unit
        coefs = self.simulation.convert_units(units_in="adu", units_out=unit)
        return lbda, flux*coefs, variance*coefs**2


    def get_band_snr(self, lbda_range, frame="obs", per_resolution=True,
                    statistic=np.nanmean, **kwargs):
        """Get the signal-to-noise ratio over a wavelength range.

        Parameters
        ----------
        lbda_range : tuple of float or list of tuple
            Wavelength range ``(min, max)`` in Angstrom (or a list of them).
        frame : str, optional
            Reference frame of ``lbda_range``: ``"rest"`` or ``"obs"``.
            Default is ``"obs"``.
        per_resolution : bool, optional
            If True, return the SNR per spectral resolution element,
            otherwise per wavelength bin. Default is True.
        statistic : callable, optional
            Function used to summarise the signal and variance over
            ``lbda_range``. Default is `numpy.nanmean`.
        **kwargs
            Goes to `~slicersim.simulation.Simulation.get_band_snr` (e.g.
            ``switch_off``).

        Returns
        -------
        float or numpy.ndarray
            The SNR, one per wavelength range if a list is given. If any
            detector pixel saturates, ``numpy.inf`` is returned.
        """
        snr = self.simulation.get_band_snr(lbda_range, frame=frame,
                                           statistic=statistic, **kwargs)
        if per_resolution:
            # not *= to allow broadcasting change.
            snr = snr*np.sqrt(self.simulation.spectrograph.dispersion_resolution)

        return snr


    def get_cube(self):
        """Get the simulated cube in the current configuration.

        Returns
        -------
        cube : numpy.ndarray
            Data cube in ADU, of shape ``(nlbda, ny, nx)``.
        varcube : numpy.ndarray
            Variance cube in ADU^2, same shape.

        See Also
        --------
        ~slicersim.simulation.Simulation.get_cube : The underlying method.
        """
        return self.simulation.get_cube()


    def get_detector_image(self, mapper, cubes=None, **kwargs):
        """Not implemented for the generic VirtualTarget class.

        Parameters
        ----------
        mapper : slicersim.mapper.SlicerMapper
            Mapper describing where each slice falls on the detector.
        cubes : tuple, optional
            Pre-computed cubes to project, to avoid regenerating them.
            Default is None.
        **kwargs
            Goes to the subclass implementation.

        Raises
        ------
        NotImplementedError
            Always raised; the projection depends on how the instrument lays
            its fields out on the detector, so instrument-specific subclasses
            must override this method. See e.g.
            `~slicersim.lazuli.VirtualLazuliTarget`.
        """
        raise NotImplementedError("This functionality is not implemented for the generic Target class. See e.g., LazuliTarget")


    def get_variance_contribution(self):
        """Get the contribution of each noise source to the variance.

        Returns
        -------
        pandas.DataFrame
            One row per wavelength, with columns ``"lbda"`` (Angstrom),
            ``"flux"`` and ``"variance"`` (ADU, ADU^2), and one column per
            variance source (see
            `~slicersim.simulation.Simulation.variance_sources`, e.g.
            ``"dark"``, ``"ron"``, ``"pointsource"``, ``"background"``,
            ``"thermal"``) giving its contribution to the variance in ADU^2.
        """
        return self.simulation.get_variance_contribution()


    def get_properties(self, which, default=None, as_dict=True):
        """Get parameter values of the target's simulation.

        Shortcut to `~slicersim.simulation.Simulation.get_parameter`.

        Parameters
        ----------
        which : str or list of str
            Name(s) of the parameter(s), e.g. ``"redshift"`` or
            ``["nmd", "nramps"]``.
        default : object, optional
            Value returned if a parameter is not found. Default is None.
        as_dict : bool, optional
            Only used when ``which`` is a single name that is not found: if
            True, ``{which: default}`` is returned instead of ``default``.
            Default is True.

        Returns
        -------
        object or dict
            The parameter value if ``which`` is a str, a dict
            ``{name: value}`` if it is a list.
        """
        return self.simulation.get_parameter(which=which, default=default, as_dict=as_dict)


    # ============== #
    #  Properties    #
    # ============== #
    @property
    def pointsource_properties(self):
        """Names of the mutable parameters of the point source (list of str)."""
        return [param_ for param_ in self.simulation.scene.mutable_parameters
                    if param_.startswith("pointsource.")]


    @property
    def simulation(self):
        """The `~slicersim.simulation.Simulation` wrapped by the target."""
        return self._simulation


# ============ #
#  Specifics   #
# ============ #
# # Generic object
class Target( VirtualTarget ):
    """Point source target with an arbitrary input spectrum.

    Parameters
    ----------
    lbda : array_like
        Wavelengths in Angstrom.
    flux : array_like
        Flux in erg/s/cm^2/A, or in arbitrary units when ``mag`` is given.
    mag : float, optional
        If given, ``flux`` is rescaled to this magnitude in ``band``.
        Default is None.
    band : str, optional
        Bandpass of ``mag`` (any sncosmo bandpass name).
        Default is ``"bessellb"``.
    **kwargs
        Goes to `~slicersim.scene.get_scene`, e.g. ``background``
        (default ``"zodi"``), ``host`` or point source parameters such as
        ``position``. ``instrument`` sets the instrument configuration.

    See Also
    --------
    slicersim.lazuli.LazuliTarget : The Lazuli flavour of this target.
    """
    def __init__(self, lbda, flux, mag=None, band="bessellb",
                    host=None, **kwargs):
        """Initialize the Target.

        Parameters
        ----------
        lbda : array_like
            Wavelengths in Angstrom.
        flux : array_like
            Flux in erg/s/cm^2/A, or in arbitrary units when ``mag`` is given.
        mag : float, optional
            If given, ``flux`` is rescaled to this magnitude in ``band``.
            Default is None.
        band : str, optional
            Bandpass of ``mag``. Default is ``"bessellb"``.
        **kwargs
            Goes to `~slicersim.scene.get_scene`; ``instrument`` sets the
            instrument configuration.
        """
        simulation = self._source_to_simulation_(source=[lbda, flux],
                                                 mag=mag, band=band,
                                                 host=host,
                                                 **kwargs)
        super().__init__(simulation=simulation)


    @classmethod
    def from_simulation(cls, simulation):
        """Load the instance from an already-built simulation.

        .. warning::
            This override is currently broken and always raises `TypeError`:
            ``super().__init__`` is called from a classmethod, so no instance
            is ever built. It shadows the working
            `~slicersim.target.VirtualTarget.from_simulation` for
            `~slicersim.target.Target` and its subclasses.

        Parameters
        ----------
        simulation : slicersim.simulation.Simulation
            The simulation the target wraps.

        Returns
        -------
        ~slicersim.target.Target
            An instance of the class, once the implementation is fixed.

        Raises
        ------
        TypeError
            Always, see the warning above.
        """
        return super().__init__(simulation=simulation)


# Supernova
class Supernova( VirtualTarget ):
    """Type Ia supernova point source target.

    Parameters
    ----------
    model : str, optional
        The supernova model, passed to `~slicersim.scene.get_scene` as
        ``source="snia-{model}"``:

        - ``"salt"``: SALT2-extended, parameters ``x1``, ``c`` and ``MBmax``
          [default]. Any sncosmo SALT source name (e.g. ``"salt3"``) also
          works.
        - ``"twin"``: Twins-Embedding, parameters ``xi1``, ``xi2``, ``xi3``
          and ``color``.
    **kwargs
        Goes to `~slicersim.scene.get_scene`, i.e. point source parameters
        (e.g. ``redshift``, ``phase``, ``position`` and the model
        parameters), ``background`` or ``host``. ``instrument`` sets the
        instrument configuration (default: the class ``_INSTRUMENT``, if
        any).

    See Also
    --------
    slicersim.lazuli.LazuliSupernova : The Lazuli flavour of this target.
    """
    def __init__(self, redshift, host=None, model="salt", **kwargs):
        """Initialize the Supernova.

        Parameters
        ----------
        model : str, optional
            The supernova model. Default is ``"salt"``.
        **kwargs
            Goes to `~slicersim.scene.get_scene`; ``instrument`` sets the
            instrument configuration.
        """
        simulation = self._source_to_simulation_(source=f"snia-{model}",
                                                 redshift=redshift, host=host,
                                                 **kwargs)
        super().__init__(simulation=simulation)


class Galaxy( VirtualTarget ):
    def __init__(self, redshift, model="sersic:bb-5000", **kwargs):
        """Initialize the Supernova.

        Parameters
        ----------
        model : str, optional
            The supernova model. Default is ``"salt"``.
        **kwargs
            Goes to `~slicersim.scene.get_scene`; ``instrument`` sets the
            instrument configuration.
        """
        simulation = self._source_to_simulation_(source=f"galaxy-{model}",
                                                 redshift=redshift,
                                                 **kwargs)
        super().__init__(simulation=simulation)

# Kilonova
class Kilonova( VirtualTarget ):
    """Kilonova point source target.

    The kilonova spectrum is computed from POSSIS radiative transfer models
    (Bulla 2019, 2023) for a given viewing angle, see
    `~slicersim.scene.sources.kilonova.get_kilonova_flux`.

    Parameters
    ----------
    model : str, optional
        The kilonova model, passed to `~slicersim.scene.get_scene` as
        ``source="kilonova-{model}"``: ``"bulla23"`` (POSSIS grid of
        Bulla 2023) [default] or ``"bulla19"`` (Bulla 2019).
    **kwargs
        Goes to `~slicersim.scene.get_scene`. Main point source parameters
        are:

        - ``redshift``: redshift of the kilonova (default 0.2).
        - ``phase``: days since merger (default 1.4).
        - ``theta``: viewing angle in degrees (default 0, i.e. pole-on).
        - ``magabs``: peak absolute magnitude in ``band`` (default -15.8).
        - ``mag``: peak observed magnitude in ``band`` (overrides
          ``magabs``).
        - ``band``: bandpass of the normalisation (default ``"sdssr"``).
        - ``position``: position in the IFU in spaxels (default
          ``[1, 0.5]``).

        ``instrument`` sets the instrument configuration.

    See Also
    --------
    slicersim.lazuli.LazuliKilonova : The Lazuli flavour of this target.
    """
    def __init__(self, model="bulla23", host=None, **kwargs):
        """Initialize the Kilonova.

        Parameters
        ----------
        model : str, optional
            The kilonova model. Default is ``"bulla23"``.
        **kwargs
            Goes to `~slicersim.scene.get_scene`; ``instrument`` sets the
            instrument configuration.
        """
        simulation = self._source_to_simulation_(source=f"kilonova-{model}",
                                                 host=host,
                                                 **kwargs)
        super().__init__(simulation=simulation)

# Kilonova
class BlackBody( VirtualTarget ):
    """Black body point source target.

    Parameters
    ----------
    temperature : float
        Black body temperature in Kelvin.
    **kwargs
        Goes to `~slicersim.scene.get_scene`, mainly the point source
        parameters ``mag`` and ``band`` (normalisation, e.g.
        ``band="sdssr"``), ``magsys`` (e.g. ``"ab"``) and ``position``
        (spaxels). ``instrument`` sets the instrument configuration.

    See Also
    --------
    slicersim.lazuli.LazuliBlackBody : The Lazuli flavour of this target.
    """
    def __init__(self, temperature, host=None, **kwargs):
        """Initialize the BlackBody.

        Parameters
        ----------
        temperature : float
            Black body temperature in Kelvin.
        **kwargs
            Goes to `~slicersim.scene.get_scene`; ``instrument`` sets the
            instrument configuration.
        """
        simulation = self._source_to_simulation_(source=f"blackbody-{temperature}",
                                                 host=host,
                                                 **kwargs)
        super().__init__(simulation=simulation)

# PowerLaw
class PowerLaw( VirtualTarget ):
    """Power-law point source target.

    Parameters
    ----------
    alpha : float, optional
        Power-law index, see
        `~slicersim.scene.sources.generic.get_powerlaw_flux`.
        Default is None.
    **kwargs
        Goes to `~slicersim.scene.get_scene`, e.g. ``mag`` and ``band``
        (normalisation), ``lambda_ref`` (Angstrom) and ``position``
        (spaxels). ``instrument`` sets the instrument configuration.

    See Also
    --------
    slicersim.lazuli.LazuliPowerLaw : The Lazuli flavour of this target.
    """
    from .scene.sources.calspec import calspecsource
    _SOURCES = calspecsource

    def __init__(self, alpha=None, host=None, **kwargs):
        """Initialize the PowerLaw.

        Parameters
        ----------
        alpha : float, optional
            Power-law index. Default is None.
        **kwargs
            Goes to `~slicersim.scene.get_scene`; ``instrument`` sets the
            instrument configuration.
        """
        simulation = self._source_to_simulation_(source=f"powerlaw-{alpha}",
                                                 host=host,
                                                 **kwargs)
        super().__init__(simulation=simulation)

# CalSpec Stars
class CalSpec( VirtualTarget ):
    """HST CalSpec spectrophotometric standard star target.

    Parameters
    ----------
    name : str
        Name (or short name) of the CalSpec star, e.g. ``"bd_17"`` or
        ``"gd_71"`` (see `source_names`).
    **kwargs
        Goes to `~slicersim.scene.get_scene`, e.g. ``position`` (spaxels)
        or ``background``. ``instrument`` sets the instrument configuration.

    See Also
    --------
    slicersim.lazuli.LazuliCalSpec : The Lazuli flavour of this target.
    """
    from .scene.sources.calspec import calspecsource
    _SOURCES = calspecsource

    def __init__(self, name, **kwargs):
        """Initialize the CalSpec.

        Parameters
        ----------
        name : str
            Name of the CalSpec star.
        **kwargs
            Goes to `~slicersim.scene.get_scene`; ``instrument`` sets the
            instrument configuration.
        """
        simulation = self._source_to_simulation_(source=f"calspec-{name}", **kwargs)
        super().__init__(simulation=simulation)

    @classmethod
    def from_name(cls, name, **kwargs):
        """Build the target from the name of the star.

        Equivalent to calling the class directly.

        Parameters
        ----------
        name : str
            Name of the CalSpec star.
        **kwargs
            Goes to the class initialisation.

        Returns
        -------
        ~slicersim.target.CalSpec
            An instance of the class.
        """
        # this is actually a wrapper of the init
        return cls(name, **kwargs)

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def source_names(self):
        """Names of the available CalSpec stars (array of str, one entry per spectrum)."""
        return self._SOURCES.source.index.values.astype(str)
