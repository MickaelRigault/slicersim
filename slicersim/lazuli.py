import numpy as np
from .simulation import Simulation

__all__ = ["lazuli_etc", "lazuli_sn_etc",
            "LazuliSN", "LazuliTarget", "LazuliCalSpec"]


def lazuli_sn_etc(model, redshift, snr, phase=0, 
                   lbda_range=[4000, 6800], frame='rest',
                   statistic=np.nanmean,
                   max_group=None, nmd=None, time_details=False,
                   **kwargs):
    """ calculate the exposure time required to achieve a specified Signal-to-Noise Ratio (SNR).
    
    This function creates a Supernovae with specified model, configures the detector
    read-out mode, and calculates the exposure time needed to achieve the desired SNR.
    
    Parameters
    ----------
    model : str
        kind of supernovae requested. Specify parameters with kwargs
        - salt: SN Ia - parameters: x1, c
        - twin: SN Ia - parameters: xi1, xi2, xi3, color

    snr : float
        The target Signal-to-Noise Ratio to achieve.

    mag : float
        Specify the desire magnitude of the target. 
        If None, the input flux is assumed to be in erg/s/cm2/A.
        If given, the input flux will be multiply to reach the desire magnitude
        in the given band (see band).

    band : str
        = ignored if mag=None =
        The photometric band for the magnitude. 

    lbda_range : list of float
        The wavelength range in Angstroms over which to calculate the SNR.

    frame : str, optional
        The reference frame for lbda_range. 
        Available options 'rest' (rest-frame) or 'obs' (observer-frame).

    statistic : function, optional
        The statistical function to use for calculating the SNR. 

    max_group : int, None
        Specify the maximum number of groups in a single ramp for the detector read-out mode.
        If None, default configuration parameters are used.

    nmd : int, optional
        Specify the detector MACC mode (n, m, d). 
        n: number of groups ;  m: number of frames per group; d: number of drops between groups

    time_details : bool
        If True, returns detailed exposure time information. 
    
    **kwargs parameters of the SN model (see model)

    Returns
    -------
    exptime : float, dict
        The calculated exposure time required to achieve the target SNR.
        (see time_details)

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
    """ calculate the exposure time required to achieve a specified Signal-to-Noise Ratio (SNR).
    
    This function creates a target with specified spectral conditions, configures the detector
    read-out mode, and calculates the exposure time needed to achieve the desired SNR.
    
    Parameters
    ----------
    lbda: array
        The wavelength [angtrom] of the spectrum.

    flux : array
        The flux values of the spectrum. See mag for the unit.

    snr : float
        The target Signal-to-Noise Ratio to achieve.

    mag : float
        Specify the desire magnitude of the target. 
        If None, the input flux is assumed to be in erg/s/cm2/A.
        If given, the input flux will be multiply to reach the desire magnitude
        in the given band (see band).

    band : str
        = ignored if mag=None =
        The photometric band for the magnitude. 

    lbda_range : list of float
        The wavelength range in Angstroms over which to calculate the SNR.

    frame : str, optional
        The reference frame for lbda_range. 
        Available options 'rest' (rest-frame) or 'obs' (observer-frame).

    statistic : function, optional
        The statistical function to use for calculating the SNR. 

    max_group : int, None
        Specify the maximum number of groups in a single ramp for the detector read-out mode.
        If None, default configuration parameters are used.

    nmd : int, optional
        Specify the detector MACC mode (n, m, d). 
        n: number of groups ;  m: number of frames per group; d: number of drops between groups

    time_details : bool
        If True, returns detailed exposure time information. 
    
    Returns
    -------
    exptime : float, dict
        The calculated exposure time required to achieve the target SNR.
        (see time_details)

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



class _LazuliScene_():
    """ """
    _DEFAULT_CONFIG = {"instrument":'lazuli.toml'}
    
    def __init__(self, simulation=None):
        
        """ """
        # set it.
        self._simulation = simulation

    @classmethod
    def from_scene(cls, scene=None, slicer=True, **kwargs):
        """ load the instance from a scene configuration 

        Parameters
        ----------
        scene: dict
            scene configuration with the given format
            scene = {scene: {point_source:{}, # PSF
                             background: {}, # spatially flat
                             host: {} }} # structured background

        slicer: bool
            should the spectrograph assume slicer (True) or MLA (False)

        **kwargs goes to iotools.get_config() and update the configuration
        
        Returns
        -------
        instance
        """
        from .iotools import get_config        
        # create the simulator
        config = get_config( **(cls._DEFAULT_CONFIG | kwargs) )
        simulation = Simulation.from_config(config, slicer=slicer)
        return cls(simulation=simulation)

    # =============== #
    #   Methods       #
    # =============== #
    def change_detector_mode(self, nmd=None, max_group=None):
        """ change the configuration of the detector 

        Parameters
        ----------
        nmd: list
            MACC mode (n, m, d)
            n: number of groups
            m: number of frames per group
            d: number of drops between groups

        max_group: int
            maximum number of group for a single ramp.

        Returns
        -------
        None
        """
        if max_group is not None:
            self.simulation.detector.max_group = max_group
        
        if nmd is not None:
            self.simulation.update(nmd=nmd)
            
    def set_properties(self, **kwargs):
        """ shortcut to change the pointsource properties """
        # list of mutable properties
        mutable_allowed = self.target_properties
        
        # Adding 'target__' as favored by the update method.
        updates = {f"target__{k}":v for k,v in kwargs.items()}
        _ = self.simulation.update(**updates)
        
    def setup_to_snr(self, snr,
                     lbda_range=[4000, 6800], frame='rest',
                     statistic=np.nanmean, inplace=True, **kwarg):
        """ set the simulation parameters to achieve a specified Signal-to-Noise Ratio (SNR).

    
        Parameters
        ----------
        snr : float
            The target Signal-to-Noise Ratio to achieve.

        lbda_range : list of int, optional
            The wavelength range in Angstroms over which to calculate the SNR.

        frame : str, optional
            The reference frame for the wavelength range. Options include 'rest'
            (rest-frame) or 'obs' (observer-frame).

        statistic : function, optional
            The statistical function to use for calculating the SNR. 
            Default is `numpy.nanmean`.

        inplace : bool, optional
            If True, updates the instance configuration to be that matching the requested
            snr.

        **kwarg : dict
            Additional keyword arguments to pass to the `simulation.fetch_snr` method.
    
        Returns
        -------
        config : dict
            The configuration required to achieve the target SNR.

        reached_snr : float
            The actual SNR achieved with the returned configuration.
        """
        # get configuration to reach this SNR
        config, reached_snr, exptime = self.simulation.fetch_snr(target_snr=snr, 
                                                    lbda_range=lbda_range, frame=frame,
                                                    statistic=statistic, **kwarg)
        # update simulation at this config
        if inplace:
            self.simulation.update(**config)

        return config, reached_snr
        
    def get_exposure_time(self, snr=None, full=False, **kwargs):
        """ get the exposure time of the current configuration.

        Parameters
        ----------
        snr: None, float
            Change the snr of the target prior calculating the exposure time.
            This does not change the configuration of the current instance 
            (inplace=False). kwargs are used to update setup_to_snr arguments.

        full: bool
            Shall this retunrs all timing details (dict) associated to the 
            exposure ? (t-frame, t-group, etc.)
            If False, only the "total_exposure" is returned.

        **kwargs goes to setup_to_snr() | ignored if snr=None

        Returns
        -------
        float, dict
            See full=bool.
        """
        if snr is not None:
            self.setup_to_snr(snr=snr, inplace=False, **kwargs)

        times = self.simulation.get_times()
        if full:
            return times
        
        return times["total_exptime"]

    def get_readout_config(self):
        """ get the current macc (nmd) mode and the number of ramps 
        
        Returns
        -------
        dict:
           nmd: (ngroup, nframe_per_group, ndrop)
           nramp: number of ramps (1-ramp = 1-nmd)
        """
        return self.get_properties(["nmd", "nramp"])

    def get_spectrum(self, unit="adu", incl_error=True, **kwargs):
        """ get a realistic simulated spectrum given the current configurations.
        
        This function fetches the spectrum data from the simulation and converts the
        flux and variance to the specified unit.
        
        Parameters
        ----------
        unit : str, optional
            The unit to convert the spectrum flux and variance to.
            available units are:
            - adu [total]
            - flambda [erg/s/a/cm2]
            - fphoton [ph/s]
            - rate [adu/s]
            - framerate [adu/frame]

        incl_error : bool, optional
            If True, returned flux is scattered from the true_flux given the variance. 
            If not, the true_flux is returned.

        **kwargs : dict
            Additional keyword arguments to pass to the `simulation.get_spectrum` method.
        
        Returns
        -------
        lbda : array
            The wavelength values of the spectrum [angstrom]

        flux : array
            The flux values of the spectrum, converted to the specified unit.
            If incl_error=True, this flux is scattered given the variance (hence realistic).

        variance : array
            The variance of the spectrum,
        """
        # get spectra [adu]
        lbda, flux, variance = self.simulation.get_spectrum(incl_error=incl_error, **kwargs)

        # change the unit
        coefs = self.simulation.convert_units(units_in="adu", units_out=unit)
        return lbda, flux*coefs, variance*coefs**2

    def get_variance_contribution(self):
        """ obtain a dataframe detailing the variance contribution for each wavelength 
        of each variance source.
        
        Returns
        -------
        variance_contribution: pandas.DataFrame
        """
        return self.simulation.estimate_variance_contribution_spectra()

    def get_properties(self, which, default=None, as_dict=True):
        """ change the properties of the target. """
        return self.simulation.get_parameter(which=which, default=default, as_dict=as_dict)
    
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def target_properties(self):
        """ get mutable properties of the target """
        return [l for l in self.simulation.scene.mutable_parameters if l.startswith("target__")]
        
    @property
    def simulation(self):
        """ core attribute containing simulation details """
        return self._simulation

# ============ #
#  Specifics   #
# ============ #
# Supernovae
class LazuliSN( _LazuliScene_ ):
    """ """
    def __init__(self, model="salt", slicer=True, **kwargs):
        """ """
        from .scene import get_sn_scene
        from .iotools import get_config
        scene = get_sn_scene(model=model, **kwargs)
        config = get_config( **( self._DEFAULT_CONFIG | {"scene": scene}) )
        simulation = Simulation.from_config(config, slicer=slicer)
        
        super().__init__(simulation=simulation)

# CalSpec Stars
class LazuliCalSpec( _LazuliScene_ ):
    """ """
    from .extra.calspec import calspecsource
    _SOURCES = calspecsource
    def __init__(self, name, background="zodi", 
                 **kwargs):
        """ """
        
        lbda, flux, _ = self._SOURCES.get_spectrum(name)
        simulation = Simulation.from_source(lbda, flux, background=background,
                                                **kwargs)
        super().__init__(simulation=simulation)

    @classmethod
    def from_name(cls, name, **kwargs):
        """ Construct the CalSpec Lazuli object from a CalSpec name """
        # this is actually a wrapper of the init
        return cls(name, **kwargs)

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def source_names(self):
        """ list of available calspec sources """
        return self._SOURCES.source.index.values.astype(str)

# Generic object
class LazuliTarget( _LazuliScene_ ):
    """ """
    def __init__(self, lbda, flux, mag=None, band="bessellb",
                     background="zodi", 
                 **kwargs):
        """ """
        simulation = Simulation.from_source(lbda, flux, background=background,
                                                mag=mag, band=band,
                                                **kwargs)
        super().__init__(simulation=simulation)
