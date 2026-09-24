""" This module handles the targets, the observation-level view of a simulation.

A target wraps a `~slicersim.simulation.Simulation` and exposes what one
actually asks of an observation: exposure time and signal-to-noise
calculations, reconfiguration of the detector and of the spectrograph, and
access to the simulated spectra and cubes.

`VirtualTarget` holds that generic interface and is not meant to be used
directly. The concrete targets differ only by the scene they build:
`Supernova` for SN Ia models, `Kilonova` for kilonova models, `CalSpec` for HST CalSpec standard stars, and
`Target` for an arbitrary input spectrum. Instrument-specific flavours are
defined in `~slicersim.lazuli`, and calibration exposures in
`~slicersim.calibration`.
"""

import numpy as np

from .simulation import Simulation
from .iotools import get_config
from .scene import get_scene


class VirtualTarget:
    """A virtual class to build Lazuli Target (see child classes).

    This class provides a generic interface to the `slicersim.Simulation`
    object. It is not intended to be used directly, but rather to be inherited
    by other classes that define specific targets.

    Parameters
    ----------
    simulation : slicersim.Simulation, optional
        The simulation object. Default is None.

    """
    _DEFAULT_CONFIG = {"instrument": None}

    def __init__(self, simulation=None):
        """Initialize the VirtualLazuliTarget.

        Parameters
        ----------
        simulation : slicersim.Simulation, optional
            The simulation object. Default is None.
        """
        # set it.
        self._simulation = simulation


    @classmethod
    def _source_to_simulation_(cls, source, instrument=None, **kwargs):
        """ Internal method to get the simulation associated to this Target (uses class attributes)"""
        if instrument is None and hasattr(cls,"_INSTRUMENT"):
            instrument = cls._INSTRUMENT

        scene = get_scene(source=source, **kwargs)
        config = get_config( **( cls._DEFAULT_CONFIG | {"scene": scene, "instrument": instrument}) )

        return Simulation.from_config(config)

    @classmethod
    def from_scene(cls, scene=None, instrument=None, **kwargs):
        """Load the instance from a scene configuration.

        Parameters
        ----------
        scene : dict, optional
            Scene configuration with the given format (see
            `slicersim.scene.get_scene`)::

                scene = {"scene": {"pointsource": {},  # PSF
                                   "background": {},   # spatially flat
                                   "host": {}}}        # structured background

            Default is None.
        instrument : str or dict, optional
            Configuration of the instrument, if any. If None, the class
            ``_INSTRUMENT`` is used when defined. Default is None.
        **kwargs
            Goes to `iotools.get_config()` and updates the configuration.

        Returns
        -------
        VirtualTarget
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
        VirtualTarget
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
            Goes to `get_cube`.

        Returns
        -------
        numpy.ndarray
            2D detector image. When `image` is given, the sum of `image` and
            of the projected cube; `image` itself is left unchanged.
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
        """Change the detector configuration.

        Parameters
        ----------
        nmd : list, optional
            MACC mode (n, m, d).
            n: number of groups
            m: number of frames per group
            d: number of drops between groups
            Default is None.
        max_group : int, optional
            Maximum number of groups for a single ramp. Default is None.
        nramps : int, optional
            Number of ramps. Default is None.
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
        spatial_shape : tuple of float, optional
            Manually set the grid shape (e.g., (40, 40)).
            This is on top of `sampling` if any. Default is None.
        spatial_scale : float or list, optional
            Manually set spaxel size in arcsec.
            If float, this is assumed to be squared. If list, (x, y).
            This is on top of `sampling` if any. Default is None.
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
        """Shortcut to change any of the simulation properties.

        Parameters
        ----------
        **kwargs
            The properties to change. The keys should be the name of the
            property to change, and the values the new value.
        """
        _ = self.simulation.update(**kwargs)

    def setup_to_snr(self, snr, per_resolution=True,
                     lbda_range=[4000, 6800], frame='rest',
                     statistic=np.nanmean, inplace=True, **kwarg):
        """Set the simulation parameters to achieve a specified Signal-to-Noise Ratio (SNR).

        Parameters
        ----------
        snr : float
            The target Signal-to-Noise Ratio to achieve.
        per_resolution: bool
            Did you provide the snr per spectral resolution element (True) or
            per wavelength bins (False).
            If per resolution, this method will convert this into snr per
            wavelength bin (snr_lbdabin = snr / sqrt(dispersion_resolution))
            and feed this to the fetch_snr() Simulation methods that expects it
            per wavelength bin.
        lbda_range : list of int, optional
            The wavelength range in Angstroms over which to calculate the SNR.
            Default is [4000, 6800].
        frame : str, optional
            The reference frame for the wavelength range. Options are 'rest'
            (rest-frame) or 'obs' (observer-frame). Default is 'rest'.
        statistic : function, optional
            The statistical function to use for calculating the SNR.
            Default is `numpy.nanmean`.
        inplace : bool, optional
            If True, updates the instance configuration to match the requested SNR.
            Default is True.
        **kwargs
            Additional keyword arguments to pass to the `simulation.fetch_snr` method.

        Returns
        -------
        config : dict
            The configuration required to achieve the target SNR.
        reached_snr : float
            The actual SNR achieved with the returned configuration.
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
            Change the SNR of the target before calculating the exposure time.
            This does not change the configuration of the current instance
            (`inplace=False`). `kwargs` are used to update `setup_to_snr`
            arguments. Default is None.
        full : bool, optional
            If True, returns all timing details (t-frame, t-group, etc.)
            associated with the exposure. If False, only the total exposure
            time is returned. Default is False.
        **kwargs
            Goes to `setup_to_snr()`. Ignored if `snr` is None.

        Returns
        -------
        float or dict
            Exposure time. See `full`.
        """
        if snr is not None:
            self.setup_to_snr(snr=snr, inplace=False, **kwargs)

        times = self.simulation.get_times()
        if full:
            return times

        return times["total_exptime"]

    def get_readout_config(self):
        """Get the current MACC (nmd) mode and the number of ramps.

        Returns
        -------
        dict
            Readout configuration:

            - nmd: (ngroup, nframe_per_group, ndrop)
            - nramps: number of ramps (1-ramp = 1-nmd)
        """
        # This should be generalized once other detector than HxRG have been implemented.
        return self.get_properties(["nmd", "nramps"])

    def get_data_volume(self, units="GB", per_ramp=False):
        """ get the data volume associated to each observations """
        return self.simulation.get_data_volume(units=units, per_ramp=per_ramp)

    def get_spectrum(self, unit="adu", incl_error=True, **kwargs):
        """Get a realistic simulated spectrum given the current configurations.

        This function fetches the spectrum data from the simulation and converts the
        flux and variance to the specified unit.

        Parameters
        ----------
        unit : str, optional
            The unit to convert the spectrum flux and variance to.
            Available units are:

            - adu [total]
            - flambda [erg/s/a/cm2]
            - fphoton [ph/s]
            - rate [adu/s]
            - framerate [adu/frame]

            Default is "adu".
        incl_error : bool, optional
            If True, the returned flux is scattered from the true_flux given the
            variance. If False, the true_flux is returned. Default is True.
        **kwargs
            Additional keyword arguments to pass to the `simulation.get_spectrum` method.

        Returns
        -------
        lbda : array_like
            The wavelength values of the spectrum in Angstroms.
        flux : array_like
            The flux values of the spectrum, converted to the specified unit.
            If `incl_error` is True, this flux is scattered given the variance
            (hence realistic).
        variance : array_like
            The variance of the spectrum.
        """
        # get spectra [adu]
        lbda, flux, variance = self.simulation.get_spectrum(incl_error=incl_error, **kwargs)

        # change the unit
        coefs = self.simulation.convert_units(units_in="adu", units_out=unit)
        return lbda, flux*coefs, variance*coefs**2

    def get_band_snr(self, lbda_range, frame="obs", per_resolution=True,
                    statistic=np.nanmean, **kwargs):
        """Get the SNR for a given wavelength range.

        Parameters
        ----------
        lbda_range : tuple, optional
            The wavelength range to calculate the SNR for.
        frame : str, optional
            The frame of the wavelength range. Default is "obs".
        per_resolution : bool, optional
            Whether to calculate the SNR per resolution element. Default is True.
        statistic : function, optional
            Numpy function to apply on test domain. Default is `np.nanmean`.
        **kwargs
            Additional keyword arguments to pass to the `simulation.get_spectrum` method.

        Returns
        -------
        float
            The SNR for the given wavelength range.
        """
        snr = self.simulation.get_band_snr(lbda_range, frame=frame,
                                           statistic=statistic, **kwargs)
        if per_resolution:
            # not *= to allow broadcasting change.
            snr = snr*np.sqrt(self.simulation.spectrograph.dispersion_resolution)

        return snr


    def get_cube(self):
        """ returns the current cubes.

        Returns
        -------
        cube, variance cube
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
        """Get a dataframe detailing the variance contribution for each wavelength
        of each variance source.

        Returns
        -------
        pandas.DataFrame
            Variance contribution.
        """
        return self.simulation.get_variance_contribution()

    def get_properties(self, which, default=None, as_dict=True):
        """Get the properties of the target.

        Parameters
        ----------
        which : str or list of str
            The name of the property to get.
        default : any, optional
            The default value to return if the property is not found.
            Default is None.
        as_dict : bool, optional
            If True, return a dictionary of properties. Default is True.

        Returns
        -------
        any or dict
            The value of the property or a dictionary of properties.

        """
        return self.simulation.get_parameter(which=which, default=default, as_dict=as_dict)

    # ============== #
    #  Properties    #
    # ============== #
    @property
    def pointsource_properties(self):
        """Get mutable properties of the pointsource."""
        return [param_ for param_ in self.simulation.scene.mutable_parameters
                    if param_.startswith("pointsource.")]

    @property
    def simulation(self):
        """Core attribute containing simulation details."""
        return self._simulation

# ============ #
#  Specifics   #
# ============ #
# # Generic object
class Target( VirtualTarget ):
    """Lazuli class for generic targets.

    Parameters
    ----------
    lbda : array_like
        Wavelength array.
    flux : array_like
        Flux array.
    mag : float, optional
        Magnitude of the target. Default is None.
    band : str, optional
        Photometric band for the magnitude. Default is "bessellb".
    background : str, optional
        Background to use. Default is "zodi".
    **kwargs
        Goes to `simulation.Simulation.from_source()`.

    """
    def __init__(self, lbda, flux,
                     mag=None, band="bessellb",
                     **kwargs):
        """Initialize the LazuliTarget.

        Parameters
        ----------
        lbda : array_like
            Wavelength array.
        flux : array_like
            Flux array.
        mag : float, optional
            Magnitude of the target. Default is None.
        band : str, optional
            Photometric band for the magnitude. Default is "bessellb".
        background : str, optional
            Background to use. Default is "zodi".
        **kwargs
            Goes to `simulation.Simulation.from_source()`.
        """
        simulation = self._source_to_simulation_(source=[lbda, flux],
                                                 mag=mag, band=band,
                                                 **kwargs)
        super().__init__(simulation=simulation)

    @classmethod
    def from_simulation(cls, simulation):
        """Load the instance from an already-built simulation.

        .. warning::
            This override is currently broken and always raises `TypeError`:
            `super().__init__` is called from a classmethod, so `simulation`
            is bound to ``self`` and no instance is ever built. It shadows the
            working `VirtualTarget.from_simulation` for `Target` and its
            subclasses.

        Parameters
        ----------
        simulation : slicersim.simulation.Simulation
            The simulation the target wraps.

        Returns
        -------
        Target
            An instance of the class, once the implementation is fixed.

        Raises
        ------
        TypeError
            Always, see the warning above.
        """
        return super().__init__(simulation=simulation)

# Supernova
class Supernova( VirtualTarget ):
    """Target class for Supernovae.

    Parameters
    ----------
    instrument : str, optional
        Configuration of the instrument, if any. If None, the class
        ``_INSTRUMENT`` is used when defined. Default is None.
    model : str, optional
        The supernova model to use. Default is "salt".
    **kwargs
        Goes to `scene.get_scene()`.

    See Also
    --------
    slicersim.lazuli.LazuliSupernova : The Lazuli flavour of this target.
    """
    def __init__(self, model="salt", **kwargs):
        """Initialize the Supernova.

        Parameters
        ----------
        model : str, optional
            The supernova model to use (it is passed to `scene.get_scene` as
            ``source="snia-{model}"``):

            - "salt": SALT2-extended (parameters: x1, c, MBmax)
            - any sncosmo salt source name (e.g. "salt3")
            - "twin": Twins-Embedding

            Default is "salt".

        **kwargs
            - instrument: specify the instrument details.
            Rest goes to `scene.get_scene()`, i.e. to the point source
            configuration (e.g. redshift, phase, position, x1, c).
        """
        simulation = self._source_to_simulation_(source=f"snia-{model}", **kwargs)
        super().__init__(simulation=simulation)

# Kilonova
class Kilonova( VirtualTarget ):
    """Target class for Kilonovae.

    The kilonova spectrum is computed from POSSIS radiative transfer models
    (Bulla 2019, 2023) for a given viewing angle, see
    `~slicersim.scene.sources.kilonova.get_kilonova_flux`.

    Parameters
    ----------
    model : str, optional
        The kilonova model to use. Default is "bulla23".
    **kwargs
        Goes to `scene.get_scene()`.

    See Also
    --------
    slicersim.lazuli.LazuliKilonova : The Lazuli flavour of this target.
    """
    def __init__(self, model="bulla23", **kwargs):
        """Initialize the Kilonova.

        Parameters
        ----------
        model : str, optional
            The kilonova model to use (it is passed to `scene.get_scene` as
            ``source="kilonova-{model}"``):

            - "bulla23": POSSIS grid from Bulla (2023) [default]
            - "bulla19": POSSIS model from Bulla (2019)
        **kwargs
            - instrument: specify the instrument details.
            Goes to `scene.get_scene()`, i.e. to the point source
            configuration. Main parameters are (see
            `~slicersim.scene.sources.kilonova.get_kilonova_flux`):

            - redshift: redshift of the kilonova (default 0.2)
            - phase: days since merger (default 1.4)
            - theta: viewing angle in degrees (default 0, i.e. pole-on)
            - magabs: peak absolute magnitude in `band` (default -15.8)
            - magobs: peak observed magnitude in `band` (overrides `magabs`)
            - band: bandpass for the normalization (default "sdssr")
            - position: position in the IFU in spaxels (default [1, 0.5])
        """
        simulation = self._source_to_simulation_(source=f"kilonova-{model}", **kwargs)
        super().__init__(simulation=simulation)

# Kilonova
class BlackBody( VirtualTarget ):
    """Target class for BlackBody.

    Parameters
    ----------
    temperature : str, optional
        The blackbody temperature (in Kelvin)
    **kwargs
        Goes to `scene.get_scene()`.
        - mag
        - band
        - magsys

    See Also
    --------
    slicersim.lazuli.LazuliBlackBody : The Lazuli flavour of this target.
    """
    def __init__(self, temperature, **kwargs):
        """Initialize the Kilonova.

        Parameters
        ----------
        temperature : str, optional
            The blackbody temperature (in Kelvin)
        **kwargs
            Goes to `scene.get_scene()`.
            - mag: magnitude
            - band: band where the magnitude is computed (e.g. "sdssr")
            - magsys: "ab"
            - position: position in the IFU in spaxels (default [1, 0.5])
        """
        simulation = self._source_to_simulation_(source=f"blackbody-{temperature}", **kwargs)
        super().__init__(simulation=simulation)

# PowerLaw
class PowerLaw( VirtualTarget ):
    """Lazuli class for PowerLaw generic objects.

    Parameters
    ----------
    name : str
        Name of the CalSpec star.
    **kwargs
        Goes to `simulation.Simulation.from_source()`.
    """
    from .scene.sources.calspec import calspecsource
    _SOURCES = calspecsource

    def __init__(self, alpha=None, **kwargs):
        """Initialize the LazuliCalSpec.

        Parameters
        ----------
        name : str
            Name of the CalSpec star.
        background : str, optional
            Background to use. Default is "zodi".
        **kwargs
            Goes to `simulation.Simulation.from_source()`.
        """
        simulation = self._source_to_simulation_(source=f"powerlaw-{alpha}", **kwargs)
        super().__init__(simulation=simulation)

# CalSpec Stars
class CalSpec( VirtualTarget ):
    """Lazuli class for CalSpec stars.

    Parameters
    ----------
    name : str
        Name of the CalSpec star.
    **kwargs
        Goes to `simulation.Simulation.from_source()`.
    """
    from .scene.sources.calspec import calspecsource
    _SOURCES = calspecsource

    def __init__(self, name, **kwargs):
        """Initialize the LazuliCalSpec.

        Parameters
        ----------
        name : str
            Name of the CalSpec star.
        background : str, optional
            Background to use. Default is "zodi".
        **kwargs
            Goes to `simulation.Simulation.from_source()`.
        """
        simulation = self._source_to_simulation_(source=f"calspec-{name}", **kwargs)
        super().__init__(simulation=simulation)

    @classmethod
    def from_name(cls, name, **kwargs):
        """Build a `LazuliCalSpec` from the name of the star.

        Parameters
        ----------
        name : str
            Name of the CalSpec star.
        **kwargs
            Goes to `simulation.Simulation.from_source()`.

        Returns
        -------
        LazuliCalSpec
            An instance of the class.

        """
        # this is actually a wrapper of the init
        return cls(name, **kwargs)

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def source_names(self):
        """List of available CalSpec sources."""
        return self._SOURCES.source.index.values.astype(str)
