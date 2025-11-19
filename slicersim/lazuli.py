import numpy as np
import warnings

from .simulation import Simulation

__all__ = ["lazuli_etc", "lazuli_sn_etc",
            "LazuliSN", "LazuliTarget", "LazuliCalSpec", "VirtualLazuliTarget"]


def lazuli_sn_etc(model, redshift, snr, phase=0, 
                   lbda_range=[4000, 6800], frame='rest',
                   statistic=np.nanmean,
                   max_group=None, nmd=None, time_details=False,
                   **kwargs):
    """Calculate the exposure time to achieve a specified Signal-to-Noise Ratio (SNR).

    This function creates a Supernova with a specified model, configures the
    detector read-out mode needed to achieve the desired SNR, 
    and calculates the exposure time .

    Parameters
    ----------
    model : str
        The kind of supernova requested. Specify parameters with kwargs.
        - salt: SN Ia - parameters: x1, c
        - twin: SN Ia - parameters: xi1, xi2, xi3, color
    redshift : float
        The redshift of the supernova.
    snr : float
        The target Signal-to-Noise Ratio to achieve.
    phase : float, optional
        The phase (with respect to manimum light) of the supernova. Default is 0.
    lbda_range : list of float, optional
        The wavelength range in Angstroms over which to calculate the SNR.
        Default is [4000, 6800].
    frame : str, optional
        The reference frame for lbda_range. Options are 'rest' (rest-frame) or
        'obs' (observer-frame). Default is 'rest'.
    statistic : function, optional
        The statistical function to use for calculating the SNR.
        Default is `numpy.nanmean`.
    max_group : int, optional
        Specify the maximum number of groups in a single ramp for the detector
        read-out mode. If None, default configuration parameters are used.
        Default is None.
    nmd : int, optional
        Specify the detector MACC mode (n, m, d).
        n: number of groups
        m: number of frames per group
        d: number of drops between groups
        Default is None.
    time_details : bool, optional
        If True, returns detailed exposure time information. Default is False.
    **kwargs
        Parameters of the SN model (see `model`).

    Returns
    -------
    exptime : float or dict
        The calculated exposure time required to achieve the target SNR.
        (see `time_details`).
    target : LazuliTarget
        The configured target object with the specified conditions.
    """
    # create a target of specified conditions
    target = LazuliSN(model=model, redshift=redshift, phase=phase, **kwargs)

    # specify the detector read-out mode | None are ignored.
    target.change_detector_mode(nmd=nmd, max_group=max_group)
    
    # setup the instrument to the requested signal to noise
    _ = target.setup_to_snr(snr, lbda_range=lbda_range, frame='rest',
                                statistic=statistic, inplace=True)
    
    # get the exposure time
    exptime = target.get_exposure_time(full=time_details)

    return exptime, target

def lazuli_etc(lbda, flux, snr,
                   mag=None, band="bessellb",
                   lbda_range=[4000, 6800], frame='rest',
                   statistic=np.nanmean,
                   max_group=None, nmd=None, time_details=False):
    """Calculate the exposure time to achieve a specified Signal-to-Noise Ratio (SNR).

    This function creates a target with specified spectral conditions, configures
    the detector read-out mode, and calculates the exposure time needed to
    achieve the desired SNR.

    Parameters
    ----------
    lbda : array_like
        The wavelength of the spectrum in Angstroms.
    flux : array_like
        The flux values of the spectrum. See `mag` for the unit.
    snr : float
        The target Signal-to-Noise Ratio to achieve.
    mag : float, optional
        Specify the desired magnitude of the target. If None, the input flux is
        assumed to be in erg/s/cm2/A. If given, the input flux will be
        multiplied to reach the desired magnitude in the given band (see `band`).
        Default is None.
    band : str, optional
        The photometric band for the magnitude. Ignored if `mag` is None.
        Default is "bessellb".
    lbda_range : list of float, optional
        The wavelength range in Angstroms over which to calculate the SNR.
        Default is [4000, 6800].
    frame : str, optional
        The reference frame for `lbda_range`. Options are 'rest' (rest-frame) or
        'obs' (observer-frame). Default is 'rest'.
    statistic : function, optional
        The statistical function to use for calculating the SNR.
        Default is `numpy.nanmean`.
    max_group : int, optional
        Specify the maximum number of groups in a single ramp for the detector
        read-out mode. If None, default configuration parameters are used.
        Default is None.
    nmd : int, optional
        Specify the detector MACC mode (n, m, d).
        n: number of groups
        m: number of frames per group
        d: number of drops between groups
        Default is None.
    time_details : bool, optional
        If True, returns detailed exposure time information. Default is False.

    Returns
    -------
    exptime : float or dict
        The calculated exposure time required to achieve the target SNR.
        (see `time_details`).
    target : LazuliTarget
        The configured target object with the specified conditions.
    """
    # create a target of specified conditions
    target = LazuliTarget(lbda, flux, mag=mag, band=band)

    # specify the detector read-out mode | None are ignored.
    target.change_detector_mode(nmd=nmd, max_group=max_group)
    
    # setup the instrument to the requested signal to noise
    _ = target.setup_to_snr(snr, lbda_range=lbda_range, frame='rest',
                                statistic=statistic, inplace=True)
    
    # get the exposure time
    exptime = target.get_exposure_time(full=time_details)

    return exptime, target



class VirtualLazuliTarget():
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
    def from_scene(cls, scene=None, slicer=True, **kwargs):
        """Load the instance from a scene configuration.

        Parameters
        ----------
        scene : dict, optional
            Scene configuration with the given format:
            `scene = {scene: {pointsource:{}, # PSF
                             background: {}, # spatially flat
                             host: {} }} # structured background`
            Default is None.
        slicer : bool, optional
            Should the spectrograph assume slicer (True) or MLA (False).
            Default is True.
        **kwargs
            Goes to `iotools.get_config()` and updates the configuration.

        Returns
        ------
        VirtualLazuliTarget
            An instance of the class.
        """
        from .iotools import get_config        
        # create the simulator
        config = get_config( **(cls._DEFAULT_CONFIG | kwargs) )
        simulation = Simulation.from_config(config, slicer=slicer)
        return cls(simulation=simulation)

    # =============== #
    #   Methods       #
    # =============== #
    def change_detector_mode(self, nmd=None, max_group=None, nramps=None):
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

    def change_spectrograph_mode(self, sampling=None, spatial_shape=None, spatial_scale=None):
        """Change the spectrograph configuration.

        Parameters
        ----------
        sampling : str, optional
            Sampling mode to use:
            - "fine": well-spatially sampled grid (~[58/2, 58] with 40mas spaxels)
            - "medium": coarser grid sampling (~[58/2, 58] with 80mas spaxels)
            Default is None.
        spatial_shape : tuple of float, optional
            Manually set the grid shape (e.g., (40, 40)).
            This is on top of `sampling` if any. Default is None.
        spatial_scale : float or list, optional
            Manually set spaxel size in arcsec.
            If float, this is assumed to be squared. If list, (x, y).
            This is on top of `sampling` if any. Default is None.
        """
        if sampling is not None:
            from .spectrograph import Spectrograph
            config = Spectrograph._SAMPLING.get(sampling, None)
            if config is None:
                raise ValueError(f"cannot parse the given sampling {sampling=} | {Spectrograph._SAMPLING} expected")
        else:
            config = {}

        # manual setting if any
        if spatial_shape is not None:
            config["spatial_shape"] = spatial_shape

        if spatial_scale is not None:
            config["spatial_scale"] = spatial_scale

        return self.simulation.update(**config)
        
    # SETTER
    def set_properties(self, **kwargs):
        """Shortcut to change the pointsource properties.

        This method allows to change the properties of the point source of the
        scene. The available properties are defined in the `pointsource_properties`
        property.

        Parameters
        ----------
        **kwargs
            The properties to change. The keys should be the name of the
            property to change, and the values the new value.

        """
        # list of mutable properties
        mutable_allowed = self.pointsource_properties

        for k in kwargs.keys():
            if k not in mutable_allowed:
                warnings.warn(f"{k} is not a pointsource property. It will be ignored.")
                
        # Adding 'pointsource__' as favored by the update method.
        updates = {f"pointsource__{k}":v for k,v in kwargs.items()}
        _ = self.simulation.update(**updates)
        
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
                                                    statistic=statistic, **kwarg)
        # update simulation at this config
        if inplace:
            self.simulation.update(**config)

        return config, reached_snr
        
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
        return self.get_properties(["nmd", "nramps"])

    def get_spectrograph_sampling(self):
        """Get the current spectrograph sampling configuration.


        Returns
        -------
        str
            The name of the sampling mode if it exists, otherwise "manual".
        dict
            The configuration of the sampling.

        """
        from .spectrograph import Spectrograph
        config = self.get_properties(["spatial_shape", "spatial_scale"])
        sampling = "manual"
        for this_sampling, this_config in Spectrograph._SAMPLING.items():
            if config == this_config:
                sampling = this_sampling
                break
            
        return sampling, config        

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

    def get_cubes(self, which="both", **kwargs):
        """ returns both cubes, one for each slicer. 
        
        Parameters
        ----------
        which: string
            which cube should be computed. 
            - both: both fine and medium field.
            - current: only current setup.
            - fine: only the fine field
            - medium: only the medium field
    
        **kwargs goes to simulation.get_cube(**kwargs)
        
        Returns
        -------
        (cube, varcube)[, (cube, varcube)]:
            cube and variance cube. 
            if which == "both":
                (cube_fine, varcube_fine), (cube_med, varcube_med)
            else:
                (cube, varcube)
        """
        # just get the current cube.
        if which == "current":
            return self.simulation.get_cube(**kwargs)
    
        # let's see which configuration you have
        current_sampling, current_config = self.get_spectrograph_sampling()
        original_position = self.simulation.get_parameter("position")
        if current_sampling not in ["fine", "medium"]:
            raise ValueError(f"sampling is neither fine nor medium ({current_sampling}). Only which='current' available. {which=}")
    
        # => you want the other one, or both.
        #    First, let's get the position in each sampling.
        pos_fine, pos_med = self._get_field_positions()
        
        # let's loop over fields, update the position 
        # to that of interest
        if which in ["both", "fine"]:
            self.change_spectrograph_mode("fine")
            self.simulation.update(position = pos_fine)
            cubes_fine = self.simulation.get_cube(**kwargs)
        else:
            cubes_fine = None
    
        if which in ["both", "medium"]:
            self.change_spectrograph_mode("medium")
            self.simulation.update(position = pos_med)
            cubes_medium = self.simulation.get_cube(**kwargs)
        else:
            cubes_medium = None
    
        # revert back to original config (could be )
        self.change_spectrograph_mode(**current_config)
        self.simulation.update(position = original_position)
    
        if which == "both":
            return cubes_fine, cubes_medium
        elif which == "fine":
            return cubes_fine
        elif which == "medium":
            return cubes_medium

    def get_detector_image(self, mapper, cubes=None, **kwargs):
        """ """
        # Generate cube
        if cubes is None:
            (cube_fine, _), (cube_medium, _) = self.get_cubes(**kwargs)
        else:
            cube_fine, cube_medium = cubes


        nslices_fine = cube_fine.shape[-2]
        nslices_medium = cube_medium.shape[-2]
        
        # Project to detector
        lbda = self.simulation.spectrograph.lbda
        # [::-1] as top <-> bottom definition inversion: 1 is top for Tim, 1 is first (lower) for me
        img_fine = mapper.project_slice(np.arange(1, nslices_fine+1)[::-1], cube_fine, lbda)
        img_med = mapper.project_slice(np.arange(nslices_fine+1, nslices_fine+1+nslices_medium)[::-1], cube_medium, lbda)
        img_ = np.sum([img_med, img_fine], axis=0)
        return img_

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

    def _get_field_positions(self, position=None, field=None):
        """ get the 'position' parameters for each of the two fields (fine and medium).
        
        These position are in the respective slicer units, and could directly be used 
        using the position= arguments one the corresponding field has been set.
        """
        if position is None:
            position = np.asarray(self.simulation.get_parameter("position"))
        if field is None:
            field, _ = self.get_spectrograph_sampling()
        
        anamorphe = np.asarray(self.simulation.spectrograph._ANAMORPHOSE)
        # information for the fine field
        nx_fine, ny_fine = self.simulation.spectrograph._SAMPLING["fine"]["spatial_shape"] / anamorphe
        
        # information for the fine field
        nx_med, ny_med = self.simulation.spectrograph._SAMPLING["medium"]["spatial_shape"] / anamorphe
    
        # multiplying factory between mid and large
        fine_to_med = 0.5
        
        # these are in unit of respective slice width
        # they share "x=0"
        fine_bottom = -ny_fine/2
        mid_top = +ny_fine/2
        
        if field == "fine":
            position_fine = position
            centroid_fine_for_med = np.asarray([0, mid_top - fine_bottom * fine_to_med])
            offset_from_centroid_for_med = position * fine_to_med
            # fine is on top
            position_medium = centroid_fine_for_med + offset_from_centroid_for_med
    
        elif field  == "medium":
            position_medium = position
            centroid_medium_for_fine = np.asarray([0, fine_bottom - mid_top / fine_to_med])
            offset_from_centroid_for_fine = position / fine_to_med
            # fine is on top
            position_fine = centroid_medium_for_fine + offset_from_centroid_for_fine
            
        else:
            raise ValueError(f"in_which should be either fine or medium ; {field=} given")
    
        return position_fine, position_medium

    @staticmethod
    def _get_fieldlayout_(fig=None, left=0.1, bottom=0.1, right=0.9, top=0.9):
        """ get matploblib's axes corresponding to the Lazuli layout.
        
        Parameters
        ----------
        fig: matplotlib.Figure
            the figure you want to axes to be generated into.
            If None, a new figure will be generated.

        Returns
        -------
        fig, (ax_fine, ax_medium)
        """
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8,8))

        width = right-left
        height = top-bottom

        width_med = width # full width
        height_med = height*0.66
        
        axfine = fig.add_axes([left+width/4, bottom+height_med, width_med/2, height_med/2])
        axmed = fig.add_axes([left, bottom, width_med, height_med])
        return fig, (axfine, axmed)

    def show_cube(self, cubes=None, lbda_range=400, cube_prop={},
                      axes=None, norm="PowerNorm", **kwargs):
        """ """
        from matplotlib import colors

        if axes is None:
            fig, (axfine, axmed) = self._get_fieldlayout_()
        else:
            axfine, axmed = axes
            fig = axfine.figure
            

        if cubes is None:
            (cube_fine, _), (cube_medium, _) = self.get_cubes(**cube_prop)
        else:
            cube_fine, cube_medium = cubes

        # cube to metaslice
        if lbda_range is None: # stack them all.
            metaslice_fine = np.nansum(cube_fine, axis=0)
            metaslice_medium = np.nansum(cube_medium, axis=0)
            
        elif len(np.atleast_1d(lbda_range)) == 1: # unique value
            lbda = self.simulation.spectrograph.lbda
            lbda_index = np.argmin( (lbda_range-lbda)**2 )
            metaslice_fine = cube_fine[lbda_index]
            metaslice_medium = cube_medium[lbda_index]
        else:
            lbda_min = np.argmin( (lbda_range[0]-lbda)**2 )
            lbda_max = np.argmin( (lbda_range[1]-lbda)**2 )            
            metaslice_fine = np.nansum(cube_fine[lbda_min:lbda_max], axis=0)
            metaslice_medium = np.nansum(cube_medium[lbda_min:lbda_max], axis=0)
            
        current_sampling, _ = self.get_spectrograph_sampling()

        # vmin, vmax        
        vmin, vmax = np.percentile(np.vstack([metaslice_fine, metaslice_medium]), [0.01, 99.99])
        vmin = kwargs.pop("vmin", vmin)
        vmax = kwargs.pop("vmax", vmax)
        
        norm = getattr(colors, norm)(0.1, vmin=vmin, vmax=vmax)

        # showing them.        
        axfine.imshow(metaslice_fine, norm=norm, origin="lower", **kwargs)
        axmed.imshow(metaslice_medium, norm=norm, origin="lower", **kwargs)
    
        [ax.set_yticks([]) for ax in fig.axes]
        [ax.set_xticks([]) for ax in fig.axes]
            
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def pointsource_properties(self):
        """Get mutable properties of the pointsource."""
        return [l for l in self.simulation.scene.mutable_parameters if l.startswith("pointsource__")]
        
    @property
    def simulation(self):
        """Core attribute containing simulation details."""
        return self._simulation

# ============ #
#  Specifics   #
# ============ #
# Supernovae
class LazuliSN( VirtualLazuliTarget ):
    """Lazuli class for Supernovae.

    Parameters
    ----------s
    model : str, optional
        The supernova model to use. Default is "salt".
    slicer : bool, optional
        Should the spectrograph assume slicer (True) or MLA (False).
        Default is True.
    **kwargs
        Goes to `scene.get_sn_scene()`.

    """
    def __init__(self, model="salt", slicer=True, **kwargs):
        """Initialize the LazuliSN.

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
        from .scene import get_sn_scene
        from .iotools import get_config
        scene = get_sn_scene(model=model, **kwargs)
        config = get_config( **( self._DEFAULT_CONFIG | {"scene": scene}) )
        simulation = Simulation.from_config(config, slicer=slicer)
        
        super().__init__(simulation=simulation)

# CalSpec Stars
class LazuliCalSpec( VirtualLazuliTarget ):
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
    def __init__(self, name, background="zodi", 
                 **kwargs):
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
        
        lbda, flux, _ = self._SOURCES.get_spectrum(name)
        simulation = Simulation.from_source(lbda, flux, background=background,
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
class LazuliTarget( VirtualLazuliTarget ):
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
    def __init__(self, lbda, flux, mag=None, band="bessellb",
                     background="zodi", 
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
        simulation = Simulation.from_source(lbda, flux, background=background,
                                                mag=mag, band=band,
                                                **kwargs)
        super().__init__(simulation=simulation)
