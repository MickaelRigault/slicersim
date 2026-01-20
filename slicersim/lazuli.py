import numpy as np
import warnings

from .simulation import Simulation
from .target import Supernova, CalSpec, Target

__all__ = ["lazuli_etc", "lazuli_sn_etc",
            "LazuliSupernova", "LazuliTarget", "LazuliCalSpec"]


SPECTROGRAPH_SAMPLING = {"fine": {'spatial_shape': [29, 58], 'spatial_scale': 0.04,
                            "throughput__noptics__coating": 11},
                         "medium": {'spatial_shape': [29, 58], 'spatial_scale': 0.08,
                                "throughput__noptics__coating": 9}
                             }


    
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
    target = LazuliSupernova(model=model, redshift=redshift, phase=phase, **kwargs)

    # specify the detector read-out mode | None are ignored.
    target.change_detector(nmd=nmd, max_group=max_group)
    
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
    target.change_detector(nmd=nmd, max_group=max_group)
    
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
    _INSTRUMENT = 'lazuli.toml'
    
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
    def from_scene(cls, scene=None, **kwargs):
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
    def change_spectrograph(self, sampling=None, spatial_shape=None, spatial_scale=None):
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
        # overwrites the current one.
        if sampling is not None:
            config = SPECTROGRAPH_SAMPLING.get(sampling, None)
            if config is None:
                raise ValueError(f"cannot parse the given sampling {sampling=} | {SPECTROGRAPH_SAMPLING} expected")
        else:
            config = {}

        # manual setting if any
        if spatial_shape is not None:
            config["spatial_shape"] = spatial_shape

        if spatial_scale is not None:
            config["spatial_scale"] = spatial_scale

        return self.simulation.update(**config)
        
        
    # GETTER
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
        keys_to_check = ["spatial_shape", "spatial_scale"]
        current_config = self.get_properties(keys_to_check)
        sampling = "manual"
        for this_sampling, this_config in SPECTROGRAPH_SAMPLING.items():
            sampling_config = {k: this_config.get(k, None) for k in keys_to_check}
            if sampling_config == current_config:
                sampling = this_sampling
                break
            
        return sampling, current_config      

    def get_cube(self, which="both", **kwargs):
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
            self.change_spectrograph("fine")
            self.simulation.update(position = pos_fine)
            cubes_fine = self.simulation.get_cube(**kwargs)
        else:
            cubes_fine = None
    
        if which in ["both", "medium"]:
            self.change_spectrograph("medium")
            self.simulation.update(position = pos_med)
            cubes_medium = self.simulation.get_cube(**kwargs)
        else:
            cubes_medium = None
    
        # revert back to original config (could be )
        self.change_spectrograph(**current_config)
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
            (cube_fine, _), (cube_medium, _) = self.get_cube(which="both", **kwargs)
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
        nx_fine, ny_fine = SPECTROGRAPH_SAMPLING["fine"]["spatial_shape"] / anamorphe
        
        # information for the fine field
        nx_med, ny_med = SPECTROGRAPH_SAMPLING["medium"]["spatial_shape"] / anamorphe
    
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
            (cube_fine, _), (cube_medium, _) = self.get_cube( which="both", **cube_prop)
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
            

# ============ #
#  Specifics   #
# ============ #
# Supernovae
class LazuliSupernova( VirtualLazuliTarget, Supernova ):
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
class LazuliCalSpec( VirtualLazuliTarget, CalSpec  ):
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
class LazuliTarget( VirtualLazuliTarget, Target  ):
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
