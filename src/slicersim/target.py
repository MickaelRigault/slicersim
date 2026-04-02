import numpy as np

from .simulation import Simulation


class VirtualTarget():
    """A virtual class to build Lazuli Target (see child classes).

    This class provides a generic interface to the `slicersim.Simulation`
    object. It is not intended to be used directly, but rather to be inherited
    by other classes that define specific targets.

    Parameters
    ----------
    simulation : slicersim.Simulation, optional
        The simulation object. Default is None.

    """
    _DEFAULT_CONFIG = {"instrument":'lazuli.toml'}
    
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
    def from_scene(cls, scene=None, instrument=None, **kwargs):
        """Load the instance from a scene configuration.

        Parameters
        ----------
        scene : dict, optional
            Scene configuration with the given format:
            `scene = {scene: {pointsource:{}, # PSF
                             background: {}, # spatially flat
                             host: {} }} # structured background`
            Default is None.
        **kwargs
            Goes to `iotools.get_config()` and updates the configuration.

        Returns
        ------
        VirtualLazuliTarget
            An instance of the class.
        """
        from .iotools import get_config
        if instrument is None and hasattr(cls,"_INSTRUMENT"):
            instrument = cls._INSTRUMENT

        # create the simulator
        config = get_config( **(cls._DEFAULT_CONFIG | {"instrument": instrument} | kwargs) )
        simulation = Simulation.from_config( config )
        return cls(simulation=simulation)

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
            self.simulation.detector.max_group = max_group
        
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

    def get_cube(self):
        """ returns the current cubes. 
        
        Returns
        -------
        cube, variance cube
        """
        return self.simulation.get_cube()

    def get_detector_image(self, mapper, cubes=None, **kwargs):
        """ """
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
                    if param_.startswith("pointsource__")]
        
    @property
    def simulation(self):
        """Core attribute containing simulation details."""
        return self._simulation

# ============ #
#  Specifics   #
# ============ #
# Supernova
class Supernova( VirtualTarget ):
    """Lazuli class for Supernovae.

    Parameters
    ----------
    model : str, optional
        The supernova model to use. Default is "salt".
    slicer : bool, optional
        Should the spectrograph assume slicer (True) or MLA (False).
        Default is True.
    **kwargs
        Goes to `scene.get_sn_scene()`.

    """
    def __init__(self, instrument=None, model="salt", **kwargs):
        """Initialize the LazuliSN.

        Parameters
        ----------
        model : str, optional
            The supernova model to use. Default is "salt".
        instrument: str, None, optional
            configuration of the instrument if any. (see self._INSTRUMENT for default)
        **kwargs
            Goes to `scene.get_sn_scene()`.
        """
        from .scene import get_sn_scene
        from .iotools import get_config
        if instrument is None and hasattr(self,"_INSTRUMENT"):
            instrument = self._INSTRUMENT

        
        scene = get_sn_scene(model=model, **kwargs)
        config = get_config( **( self._DEFAULT_CONFIG | {"scene": scene, "instrument": instrument}) )

        simulation = Simulation.from_config(config)
        
        super().__init__(simulation=simulation)

# CalSpec Stars
class CalSpec( VirtualTarget ):
    """Lazuli class for CalSpec stars.

    Parameters
    ----------
    name : str
        Name of the CalSpec star.
    background : str, optional
        Background to use. Default is "zodi".
    **kwargs
        Goes to `simulation.Simulation.from_source()`.

    """
    from .extra.calspec import calspecsource
    _SOURCES = calspecsource
    
    def __init__(self, name,
                     instrument=None,
                     background="zodi",
                 **kwargs):
        """Initialize the LazuliCalSpec.

        Parameters
        ----------
        name : str
            Name of the CalSpec star.
        instrument: str, None, optional
            configuration of the instrument if any. (see self._INSTRUMENT for default)
        background : str, optional
            Background to use. Default is "zodi".
        **kwargs
            Goes to `simulation.Simulation.from_source()`.
        """
        if instrument is None and hasattr(self,"_INSTRUMENT"):
            instrument = self._INSTRUMENT
        
        lbda, flux, _ = self._SOURCES.get_spectrum(name)
        simulation = Simulation.from_source(lbda, flux, background=background,
                                            instrument=instrument,
                                            **kwargs)
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

# Generic object
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
                     background="zodi",
                     instrument=None,
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
        if instrument is None and hasattr(self,"_INSTRUMENT"):
            instrument = self._INSTRUMENT
        
        simulation = Simulation.from_source(lbda, flux, background=background,
                                                mag=mag, band=band,
                                                instrument=instrument,
                                                **kwargs)
        super().__init__(simulation=simulation)
