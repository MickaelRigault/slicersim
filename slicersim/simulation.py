import warnings
warnings.simplefilter('always', UserWarning)

import pprint

import numpy as np
import pandas
from .scene import Scene
from .spectrograph import Spectrograph, MLASpectrograph, SlicerSpectrograph
from .detector import Detector
from .telescope import Telescope

__all__ = ["Simulation"]

COLORS = {# detector
          "dark": "#25283C",
          "thermal_dark": "#005D8F",
          "ron": "#80B056", 
          # scene
          "pointsource": "#C2C1B0", 
          "background": "#FFBC42",
          "host":"#E56C10",
          # thermal
          "thermal":"#8C2B2B"
          }

class Simulation():
    """Main class to simulate an observation.

    This class is the main entry point to the simulation. It contains all the
    necessary information to simulate an observation, from the scene to the
    detector.

    Parameters
    ----------
    scene : slicersim.Scene, optional
        The scene to observe. Default is None.
    spectrograph : slicersim.Spectrograph, optional
        The spectrograph used for the observation. Default is None.
    detector : slicersim.Detector, optional
        The detector used for the observation. Default is None.
    telescope : slicersim.Telescope, optional
        The telescope used for the observation. Default is None.
    extraction : dict, optional
        A dictionary of parameters for the extraction. Default is {}.
    meta : dict, optional
        A dictionary of metadata. Default is {}.

    Attributes
    ----------
    scene : slicersim.Scene
        The scene to observe.
    spectrograph : slicersim.Spectrograph
        The spectrograph used for the observation.
    detector : slicersim.Detector
        The detector used for the observation.
    telescope : slicersim.Telescope
        The telescope used for the observation.
    extraction : dict
        A dictionary of parameters for the extraction.
    meta : dict
        A dictionary of metadata.

    """
    VARIANCE_SOURCES = ["dark", "thermal_dark", "ron", "pointsource", "background", "host", "thermal"]
    
    def __init__(self,
                 scene=None,
                 spectrograph=None,
                 detector=None,
                 telescope=None,                 
                 extraction={},
                 meta={}):
        """Initialize the simulation.

        Parameters
        ----------
        scene : slicersim.Scene, optional
            The scene to observe. Default is None.
        spectrograph : slicersim.Spectrograph, optional
            The spectrograph used for the observation. Default is None.
        detector : slicersim.Detector, optional
            The detector used for the observation. Default is None.
        telescope : slicersim.Telescope, optional
            The telescope used for the observation. Default is None.
        extraction : dict, optional
            A dictionary of parameters for the extraction. Default is {}.
        meta : dict, optional
            A dictionary of metadata. Default is {}.

        """
        self.scene = scene                #: Scene instance
        self.spectrograph = spectrograph  #: Spectrograph instance
        self.detector = detector          #: Detector instance
        self.extraction = extraction      #: Extraction parameters
        self.telescope = telescope        #: telescope
        self._in_meta = meta              #: Meta-parameters

    def __str__(self):

        w = 50

        s = ""
        if self.scene:
            s += (" Scene ".center(w, '-') + '\n' +
                  str(self.scene))
        if self.spectrograph:
            s += ('\n' + " Spectrograph ".center(w, '-') + '\n' +
                  str(self.spectrograph))
        if self.detector:
            s += ('\n' + " Detector ".center(w, '-') + '\n' +
                  str(self.detector))
                
        if self.telescope:
            s += ('\n' + " Telescope ".center(w, '-') + '\n' +
                  str(self.telescope))
                
        if self.extraction:
            s += ('\n' + " Extraction ".center(w, '-') + '\n' +
                  pprint.pformat(self.extraction, sort_dicts=False))
        if self.meta:
            s += ('\n' + " Meta ".center(w, '-') + '\n' +
                  pprint.pformat(self.meta, sort_dicts=False))

        return s

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_source(cls, lbda, flux, mag=None,
                        band="bessellb", position=(0, 0),
                        instrument="lazuli.toml",
                        background="zodi",
                        host=None,
                        snr=None, lbda_range=[4000, 6800], frame='obs',
                        **kwargs
                     ):
        """Load the simulation observing a known source (like a star).

        Parameters
        ----------
        lbda : array
            (High resolution) wavelength of the source (in Angstrom).
        flux : array
            Flux of the input source.
        mag : float, optional
            Requested magnitude of the source (overwrite flux's amplitude).
            Default is None.
        band : str, optional
            Name of the band used to compute the magnitude (should be known by
            sncosmo). Default is "bessellb".
        position : tuple, optional
            Location of the pointsource within the IFU. Default is (0, 0).
        instrument : str, dict, list, optional
            Configuration of the instrument. Default is "lazuli.toml".
        background : str, dict, list, optional
            Configuration of the (spatially flat) scene background (e.g.
            "zodi"). Default is "zodi".
        host : str, dict, list, optional
            Configuration of the (spatially structured) scene background (e.g.
            "zodi"). Default is None.
        snr : float, optional
            If not None, the simulation will try to reach this SNR per
            wavelength element while loading. Default is None.
        lbda_range : list, optional
            Wavelength range for averaging to SNR. Ignored if `snr` is None.
            Default is [4000, 6800].
        frame : str, optional
            Frame of the `lbda_range` ('obs' or 'rest'). Ignored if `snr` is
            None. Default is 'obs'.
        **kwargs
            Goes to :meth:`update`.

        Returns
        -------
        Simulation

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
                       instrument='lazuli.toml',
                       slicer=True,
                       **kwargs):
        """Load the simulation setting the config to acquire the pointed
        pointsource.

        Parameters
        ----------
        redshift : float, optional
            Redshift of the pointsource. Default is None.
        snr : float, optional
            Targeted average SNR per wavelength bin. (See `lbda_range` and
            `frame`). If None, no parameter update to reach a SNR. Default is
            20.
        lbda_range : list, optional
            (wmin, wmax) wavelength range where the SNR will be estimated [Å].
            Default is [4000, 6800].
        frame : str, optional
            Wavelength frame ('obs', 'rest'). Default is 'rest'.
        scene : str, optional
            Scene configuration file. Default is 'supernova.toml'.
        instrument : str, optional
            Instrument configuration file. Default is 'lazuli.toml'.
        slicer : bool, optional
            If True, use a slicer spectrograph. Default is True.
        **kwargs
            Goes to :meth:`update` to change any default configuration.

        Returns
        -------
        Simulation

        """
        
        from .iotools import get_config
        
        # create the simulator
        config = get_config(instrument=instrument, scene=scene)
        this = cls.from_config(config, slicer=slicer)

        # update to expecting target
        if redshift is not None:
            this.update(pointsource__redshift=redshift)
            
        if kwargs:
            this.update(**kwargs)

        if snr is not None:
            # fetch the config respecting the target SNR
            new_config, snr, integration_time = this.fetch_snr(snr, lbda_range=lbda_range, frame=frame)
            this.update(**new_config)
        
        return this
        
    @classmethod
    def from_config(cls, config, slicer=True):
        """Initiate simulation from config nested dictionary.

        Parameters
        ----------
        config : dict
            Top level configuration containing configurations for scene,
            spectrograph, detector and extraction parameters (by default: use
            default config from :func:`mlaperf.iotools.get_config`).
        slicer : bool, optional
            If True, use a slicer spectrograph. Default is True.

        Returns
        -------
        Simulation

        """

        # First initialize spectrograph to set wavelengths, then other elements
        # using spectrograph wavelengths.
        
        # Initialize the telescope (mirror)
        telescope = Telescope.from_config(config["telescope"])

        # Initialize the spectrograph from config
        if slicer:
            spectrograph = SlicerSpectrograph.from_config(config["spectrograph"], telescope=telescope)
        else:
            spectrograph = MLASpectrograph.from_config(config["spectrograph"], telescope=telescope)

        # Initialize the scene from config (wavelength from spectrograph)
        scene = Scene.from_config(config["scene"], lbda=spectrograph.lbda)


        # Initialize the detector from config
        detector = Detector.from_config(config["detector"],
                                        lbda=spectrograph.lbda,
                                        thermaloptics=spectrograph.optics)

        # Initialize extraction parameters from config
        extraction = config["extraction"]

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
            The key to search for in the mutable parameters.

        Returns
        -------
        list or None
            A list of matching mutable parameters, or None if no match is found.

        """
        mutable_ = [l for l in self.mutable_parameters if key in l]
        if len(mutable_)==0:
            return None
        
        return mutable_

    def change_pointsource(self, pointsource):
        """Override the considered pointsource.

        Parameters
        ----------
        pointsource : slicersim.PointSource
            New pointsource to be used.

        Returns
        -------
        None

        """
        self.scene._pointsource = pointsource
    
    def update(self, reset_others=False, **kwargs):
        """Update any mutable parameter of the simulation.

        For convinience, the update method respects the django '__' format,
        such that, e.g. 'pointsource__phase' is understood as
        'pointsource.phase'. This way, one can do:

        >>> self.update(pointsource__phase = -1)

        For convinience, you can specify a shorten name, like "phase". If the
        correspondance to a mutable_parameters is uniquen this will accept it.

        >>> self.update(phase = -1)

        For convenience and backward compatiblity, you can use "target__" in
        place of "pointsource__".

        Parameters
        ----------
        reset_others : bool, optional
            If True, reset other parameters to their default values. Default is
            False.
        **kwargs
            Parameters to update.

        """
        updates_scene = {"reset_others": reset_others}
        updates_detector = {"reset_others": reset_others}
        updates_spectrograph = {"reset_others": reset_others}
        updates_extraction = {}
        
        for k, v in kwargs.items():
            k = k.replace("__", ".") # django like 
            if k not in self.mutable_parameters:
                k = self._fetch_mutable_parameters(k)
                if k is None:
                    warnings.warn(f"Parameter {k!r} is not mutable.")
                    continue
                
                if len(k) == 1:
                    k = k[0]
                else:
                    warnings.warn(f"Parameter {k} is not well defined. several correspondance {k}.")
                    continue
            
            if v is None:       # Nothing to do
                continue
            
            if k in self.scene.mutable_parameters:
                updates_scene[k] = v
                
            elif k.startswith("detector."):
                updates_detector[k.replace("detector.","")] = v
                
            elif k in self.spectrograph.mutable_parameters:
                updates_spectrograph[k] = v
                
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
        """Reset the simulation element at their initial config value.

        Careful, this creates a new element element and erases the current one.

        Parameters
        ----------
        which : str, list, optional
            Element to reset. could be '*'/'all' or a key of
            :attr:`Simulation._elements`. Default is "*".

        """

        if which not in ['all', '*'] + list(self._elements):
            raise ValueError(f"cannot parse the '{which}' reset.")

        if which in ['all', '*']:
            which = self._elements  # Reset all elements

        if "spectrograph" in which:
            self.spectrograph = Spectrograph.from_config(
                self.meta.get("spectrograph"))

        if "scene" in which:
            self.scene = Scene.from_config(
                self.meta.get("scene"), self.spectrograph.lbda)

        if "detector" in which:
            self.detector = Detector.from_config(
                self.meta.get("detector"), self.spectrograph.lbda)

        if "extraction" in which:
            self.extraction = self.meta.get("extraction")

    # -------- #
    #  GETTER  #
    # -------- #
    def get_lbda(self):
        """ Get the wavelength array
        
        This is a shortcut to self.spectrograph.lbda
        """
        return self.spectrograph.lbda
    
    def get_input_spectrum(self, which="pointsource"):
        """Get the input spectrum.

        This is a shortcut to self.scene.get_element_spectrum

        Parameters
        ----------
        which : str, optional
            Which input do you want:
            - individual elements: 'pointsource', 'background', 'host'
            - all merged: 'stacked'
            Default is "pointsource".

        Returns
        -------
        lbda : array
            Wavelength in Angstrom.
        flux : array
            Input flux (erg/s/cm^2/A).

        """
        if which == "stacked":
            lbda, specs = self.scene.get_stacked_spectra()
            flux = np.sum(specs, axis=0)
        else:
            lbda, flux = self.scene.get_element_spectrum(which)
            
        return lbda, flux

    def get_effective_transmission(self):
        """Effective total transmission of the spectrograph.

        Product of the spectroscopic throughput: spectrograph.flambda2photon
        and the detector efficiency: detector.photonflux_to_adu

        Returns
        -------
        lbda : array
            Wavelength in Angstrom.
        throughput : array
            Effective throughput (ADU/ (erg/s/cm^2/A)).

        """
        lbda = self.spectrograph.lbda # make sure it is up to date.
        return lbda, self.spectrograph.flambda2photon * self.detector.photonflux_to_adu(lbda)

    def get_effective_waveresolution(self, npx=2, sigma=None):
        """Effective wavelength resolution.

        R &= \frac{2}{n \delta\lambda} \\
        with
        \delta\lambda &= \max(1, \sigma) \times \Delta\lambda
        and 
        `\Delta\lambda` the spectral step [Å] 
        and
        `\sigma` is the spectral resolution [px].

        Parameters
        ----------
        npx : float, optional
            n-px resolution (i.e. n px per spectral elements). Default is 2.
        sigma : float, optional
            Spectral PSF stddev override. Default is None.

        Returns
        -------
        lbda : array
            Wavelength in Angstrom.
        waveresolution : array
            Effective wavelength resolution.

        """
        return self.spectrograph.lbda, \
          self.spectrograph.effective_resolution(npx=npx, sigma=sigma)

    def get_pixel_variance(self, flux=0):
        """Get the variance associated with a single pixel on the detector.

        Parameters
        ----------
        flux : float, optional
            The flux in the pixel in ADU. Default is 0.

        Returns
        -------
        float
            The variance of the pixel in ADU^2.

        """
        lbda = self.spectrograph.lbda
        _, pixel_var = self.detector.estimate_pixel_signal(flux, lbda=self.spectrograph.lbda)
        pixel_var *= self.extraction["nramps"] # incl. multi ramp approach.
        return pixel_var
        
    def get_nea(self, nea_spatial=None, nea_pixels=None):
        """Get the Noise Equivalent Area (NEA).

        The NEA is the product of the spatial NEA and the pixel NEA.

        Parameters
        ----------
        nea_spatial : float or array_like, optional
            The spatial NEA in spaxels^2. If None, it is computed.
            Default is None.
        nea_pixels : float or array_like, optional
            The pixel NEA in pixels. If None, it is computed.
            Default is None.

        Returns
        -------
        float or array_like
            The total NEA.

        """
        return self.spectrograph.get_nea(position = self.scene.pointsource.position,
                                          nea_spatial=nea_spatial, nea_pixels=nea_pixels)
            
    def get_nea_variance(self, spectrum=None, nea_spatial=None, nea_pixels=None):
        """Get the variance estimated from the Noise Equivalent Area.

        Parameters
        ----------
        spectrum : array_like, optional
            The input spectrum in erg/s/cm^2/A. Default is None.
        nea_spatial : float or array_like, optional
            The spatial NEA in spaxels^2. If None, it is computed.
            Default is None.
        nea_pixels : float or array_like, optional
            The pixel NEA in pixels. If None, it is computed.
            Default is None.

        Returns
        -------
        float or array_like
            The estimated variance.

        """
        nea = self.get_nea(nea_spatial=nea_spatial, nea_pixels=nea_pixels)
    
        # "background" variance of a single pixels
        pixel_var = self.get_pixel_variance(0)
        variance_pixels = nea * pixel_var

        # adding spectrum. But very inclear if correct...
        if spectrum is not None: # 
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
        """Shortcut to get simulation parameter(s).

        Should be an attribute of one of the element of the simulation,
        e.g. `gain` from `Simulation.detector.gain`.

        Parameters
        ----------
        which : str or list of str, optional
            Parameter name (or list of). Default is None (get all).
        default : any, optional
            Default value. Default is None.
        as_dict : bool, optional
            Return `{which: value}` rather than `value`. Default is True.

        Returns
        -------
        any
            Parameter value.

        """
        if which is None:       # Get all of them
            which = [item
                      for sublist in self._elements
                      for item in sublist ]

        if np.ndim(which):  # If a list, loop over elements
            return {param: self.get_parameter(param, as_dict=False)
                     for param in which }

        # allowing to provide origin___{bla}
        
        # Extraction parameter
        if which.startswith("extraction__") or which in self.extraction:
            return self.extraction[ which.replace("extraction__", "") ]

        # Short cuts
        if which in ["detector__ngroup", "ngroup"]:
            return self.detector.nmd[0]

        # Short cuts
        if which in ["detector__nframe", "nframe"]:
            return self.detector.nmd[0]

        # Generic source__value term
        if "__" in which:
            try:
                source, value = which.split("__")
                instance = getattr(self, source)
                return getattr(instance, value)
            except:
                pass # failed

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

            if hasattr(instance, which):            # self.which
                return getattr(instance, which)

            if which in getattr(instance, "meta"):  # self.meta['which']
                return getattr(instance, "meta")[which]

            if element == "scene": # for scene: self.meta['pointsource']['which']
                meta_pointsource = getattr(instance, "meta")["pointsource"]
                if which in meta_pointsource:
                    return meta_pointsource[which]

        return {which: default} if as_dict else default

    def get_background_spectrum(self, unit="adu", skyarea=None, per_ramp=False, apply_lsf=True):
        """Get the (flat) background flux.

        Parameters
        ----------
        unit : str, optional
            Unit of the output flux. Can be "adu", "flambda", "ph", "photon",
            "photons". Default is "adu".
        skyarea : float, optional
            Sky area in arcsec^2. If None, it is computed from the
            spectrograph spaxel scale. Default is None.
        per_ramp : bool, optional
            If True, return the flux per ramp. Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.

        Returns
        -------
        array
            Background flux.

        """
        if skyarea is None:
            skyarea = self.spectrograph.spx_spatial_scale**2
            
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
        """Get the scene cubes.

        Parameters
        ----------
        unit : str, optional
            Unit of the output flux. Can be "adu", "flambda", "ph", "photon",
            "photons". Default is "adu".
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        oversampling : int, optional
            Oversampling factor. Default is None.
        as_dict : bool, optional
            If True, return a dictionary of cubes. Default is True.
        per_ramp : bool, optional
            If True, return the flux per ramp. Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to :meth:`spectrograph.generate_pointsource`.

        Returns
        -------
        dict or array
            Dictionary of cubes or stacked array of cubes.

        """
        lbda, (pointsource, host, background) = self.scene.get_stacked_spectra(fillna=0)

        # by default spectrograph. give flux in ph/s.
        
        # pointsource
        pointsource_cube = self.spectrograph.generate_pointsource(pointsource,
                                                            position=self.scene.pointsource_position,
                                                            psf_profile=psf_profile,
                                                            oversampling=oversampling,
                                                            as_oversampled=as_oversampled,
                                                            apply_lsf=apply_lsf, 
                                                            **kwargs)
        # background
        background_cube = self.spectrograph.generate_background(background, oversampling=oversampling,
                                                                    apply_lsf=apply_lsf)

        # host | empty
        host_cube = np.zeros( (self.spectrograph.nlbda, *self.spectrograph.get_spectrograph_shape(oversampling=oversampling)) )  # (nlbda, ny, nx)
        
        # thermal
        thermal_cube = self.spectrograph.generate_thermal_signal(as_cube=True, oversampling=oversampling, apply_lsf=apply_lsf) # [ph/s]

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
                                apply_lsf=True,
                                **kwargs):
        """Project the scene through spectrograph and get flux cube [ph or
        flambda].

        Parameters
        ----------
        in_photons : bool, optional
            If True, return the cube in photons/s. Default is True (flambda).
        switch_off : list, optional
            List of discarded scene elements (pointsource, host, background) +
            thermal. Default is [].
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        oversampling : int, optional
            Oversampling factor. Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to :meth:`spectrograph.generate_pointsource`.

        Returns
        -------
        array
            (nlbda, ny, nx) cube.

        """
        if oversampling is None:
            oversampling = 3

        #if not as_oversampled:
        #    oversampling = None
        
            

        cube = self.spectrograph.get_empty_cube(filled=0, oversampling=None if not as_oversampled else oversampling)
        
        # spectra (3, nlbda):
        # * point source spectrum [erg/s/cm²/Å]
        # * host (not yet implemented)
        # * background spectrum [erg/s/cm²/Å/arcsec²]
        lbda = self.spectrograph.lbda # make sure this is up to date
        _, (pointsource, host, background) = self.scene.get_stacked_spectra(lbda=lbda, fillna=0)

        # Fill the cube with scene elements in photons/s/spx
        if "pointsource" not in switch_off:
            cube += self.spectrograph.generate_pointsource(pointsource,
                                                            position=self.scene.pointsource_position,
                                                            psf_profile=psf_profile,
                                                            oversampling=oversampling,
                                                            as_oversampled=as_oversampled,
                                                            apply_lsf=False, # applied once, at the end
                                                            **kwargs)

        if "host" not in switch_off:      
            if np.any(host):
                warnings.warn("Host cube not implemented.")

        if "background" not in switch_off:                    
            cube += self.spectrograph.generate_background(background, oversampling=None if not as_oversampled else oversampling,
                                                              apply_lsf=False)

        if "thermal" not in switch_off:
            cube += self.spectrograph.generate_thermal_signal(as_cube=True, oversampling=None if not as_oversampled else oversampling,
                                                                apply_lsf=False) # [ph/s]

        if not in_photons:      # Convert back to flambda
            cube /= self.spectrograph.flambda2photon[:, np.newaxis, np.newaxis]

        # apply_lsf once
        if apply_lsf:
            cube = self.spectrograph.apply_line_spread_function(cube)
            
        return cube  # (nlbda, ny, nx) | [ph/s] or [erg/cm²/Å/s]
    
    def get_cube(self, switch_off=[], psf_profile="default",
                     as_oversampled=False, oversampling=None,
                     per_ramp=False, apply_lsf=True,
                     **kwargs):
        """Get data cube as extracted from exposure [ADU].

        Parameters
        ----------
        switch_off : list, optional
            Name of variance sources to switch off. (dark, ron, pointsource,
            host, background, thermal). Default is [].
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        oversampling : int, optional
            Oversampling factor. Default is None.
        per_ramp : bool, optional
            If True, return the flux per ramp. Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to :meth:`_get_mla_cube_` or :meth:`_get_slicer_cube_`.

        Returns
        -------
        cube_signal, cube_variance
            (nlbda, ny, nx) in [ADU].

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

        #
        # Get cubes
        #
        cube_prop = dict(psf_profile=psf_profile,
                          switch_off=switch_off,
                          as_oversampled=as_oversampled, oversampling=oversampling,
                          per_ramp=per_ramp, apply_lsf=apply_lsf
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
                           per_ramp=False, apply_lsf=True, 
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
        per_ramp : bool, optional
            If True, return the flux per ramp. Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to :meth:`get_projected_scene`.

        Returns
        -------
        sig_cube, var_cube
            (nlbda, ny, nx) in [ADU].

        """
        # scene split per slices   
        cube = self.get_projected_scene(in_photons=True,
                                        switch_off=switch_off,
                                        psf_profile=psf_profile,
                                        as_oversampled=as_oversampled,
                                        oversampling=oversampling,
                                        apply_lsf=apply_lsf,
                                        **kwargs)  # (nlbda, nslices_with_anamorphose, npixels)
                                        
        # slicers projected onto the detector: 1 lbda for 1 slice corresponds to 1 pixel
        lbda = self.spectrograph.lbda # make sure it is up to date
        sig_cube, var_cube = self.detector.estimate_pixel_signal( cube, lbda=lbda) # expects input in [ph/...] 
        
        return sig_cube, var_cube
        
    def _get_mla_cube_(self, psf_profile="default", switch_off=[],
                           as_oversampled=False, oversampling=None,
                           per_ramp=False, apply_lsf=True,
                           **kwargs):
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
        per_ramp : bool, optional
            If True, return the flux per ramp. Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to :meth:`get_projected_scene`.

        Returns
        -------
        sig_cube, var_cube
            (nlbda, ny, nx) in [ADU, ADU^2].

        """

        # Scene effect
        # takes 65.1 ms
        cube = self.get_projected_scene(in_photons=True,
                                        switch_off=switch_off,
                                        psf_profile=psf_profile,
                                        as_oversampled=as_oversampled,
                                        oversampling=oversampling,
                                        apply_lsf=apply_lsf, 
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
                  apply_lsf=True,
                  **kwargs):
        """Get slices of the cube.

        Parameters
        ----------
        lbda_range : tuple or list of tuples
            (lbda_min, lbda_max) to use [n-slice=1] or ((lbda_min, lbda_max),
            (lbda_min, lbda_max), ...) [n-slice]. Unit: that of
            self.spectrograph.lbda.
        frame : str, optional
            Wavelength frame ('obs' or 'rest'). Default is "obs".
        switch_off : list, optional
            List of discarded scene elements (pointsource, host, background) +
            thermal. Default is [].
        incl_error : bool, optional
            If True, include error in the plot. Default is False.
        squeeze : bool, optional
            If True, squeeze the output array. Default is False.
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        oversampling : int, optional
            Oversampling factor. Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to :meth:`get_cube`.

        Returns
        -------
        signal, variance
            (nslice, ny, nx) signal [ADU] and variance [ADU^2].

        """
        lbda_range = np.atleast_2d(lbda_range) # [[wmin, wmax]]
    
        if frame not in ["obs", "rest"]:
            raise ValueError(f"Unknown wavelength frame {frame!r}.")
    
        if frame == "rest" and self.scene.pointsource.redshift is not None:
            lbda_range = lbda_range * (1 + self.scene.pointsource.redshift)

            
        cube_signal, cube_variance = self.get_cube(switch_off=switch_off,
                                                   psf_profile=psf_profile,
                                                   as_oversampled=as_oversampled,
                                                   oversampling=oversampling,
                                                   apply_lsf=apply_lsf,
                                                   **kwargs)
        
        band_sigs = self.spectrograph.cube_to_slice(cube_signal, lbda_range, func=np.nansum, squeeze=squeeze)
        band_vars = self.spectrograph.cube_to_slice(cube_variance, lbda_range, func=np.nansum, squeeze=squeeze)
    
        if incl_error:
            band_sigs = np.random.normal(loc=band_sigs, scale=np.sqrt(band_vars)) # normal noise
            
        return band_sigs, band_vars        
    
    def get_spectrum(self, switch_off=[], incl_error=False, psf_profile="default",
                         apply_lsf=True):
        """Get the pointsource signal and variance [ADU].

        Parameters
        ----------
        switch_off : list, optional
            List of discarded scene elements (pointsource, host, background) +
            thermal. Default is [].
        incl_error : bool, optional
            If True, scatter the signal by the error. Default is False.
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.

        Returns
        -------
        lbda, signal, variance
            (nlbda,) signal and variance [ADU, ADU**2].

        """
        # (lbda, nx, ny) [ADU]
        sig_cube, var_cube = self.get_cube(switch_off=switch_off,
                                               per_ramp=True, # see later.
                                               psf_profile=psf_profile,
                                               apply_lsf=apply_lsf)

        try:
            radius = self.extraction["aperture_radius"]  # [spx/slicers]
        except KeyError:
            # See Spectrograph.rescale_parameters
            radius = (self.extraction["aperture_radius_insigma"] *
                      self.spectrograph.psf_sigma_spectral /  # [arcsec]
                      self.spectrograph.spx_spatial_scale)    # [arcsec/spx]
            # radius is used to limit where PSF and variance are considered, see pointsource_variance()
            
        spec_variance = self.spectrograph.pointsource_variance(
            var_cube, position=self.scene.pointsource_position, radius=radius,
            psf_profile=psf_profile)

        # Assume the spectrum is perfectly extracted
        if "pointsource" not in switch_off:
            # the input spectrum *is not* derived from the cube.
            lbda = self.spectrograph.lbda
            _, pointsource_phflux = self.scene.get_element_spectrum('pointsource', lbda=lbda) * self.spectrograph.flambda2photon
            spec_signal = pointsource_phflux * self.detector.photonflux_to_adu(lbda)
            if apply_lsf:
                # apply LSF on *true* spectrum.
                spec_signal = self.spectrograph.apply_line_spread_function(spec_signal)
                
        else:
            spec_signal = np.zeros_like(self.spectrograph.lbda)

        if (nexp := self.extraction["nramps"]) > 1:  # Nb of exposures
            spec_signal *= nexp
            spec_variance *= nexp

        if incl_error:
            spec_signal += np.random.normal(loc=0, scale=np.sqrt(spec_variance))
            
        return self.spectrograph.lbda, spec_signal, spec_variance  # (nlbda,) [ADU, ADU²]

    def get_snr(self, switch_off=[]):
        """Get signal to noise spectrum.

        See :meth:`get_spectrum`.

        Returns
        -------
        array
            Signal to noise ratio spectrum.

        """
        _, signal, variance = self.get_spectrum(switch_off=switch_off)
        return signal / variance**0.5

    def get_band_flux(self, lbda_range, frame="obs",
                        statistic=np.nanmean,
                        squeeze=True, **kwargs):
        """Estimate mean signal and variance over a spectral domain.

        The photometry is computed assuming a top-hat filter and simply
        considering wavelength bins within the spectral domain (no
        interpolation nor weighting).

        Parameters
        ----------
        lbda_range : tuple or list of tuples
            (wmin, wmax) test wavelength range [Å] (or list of).
        frame : str, optional
            Wavelength frame ('obs' or 'rest'). Default is "obs".
        statistic : function, optional
            Numpy function to apply on test domain. Default is `np.nanmean`.
        squeeze : bool, optional
            Squeeze final df (for single-band). Default is True.
        **kwargs
            Goes to :meth:`get_spectrum` (e.g. `switch_off`).

        Returns
        -------
        signal, variance
            Signal and variance [ADU, ADU^2].

        """
        # accepts multiple ranges
        lbda_range = np.atleast_2d(lbda_range) # [[wmin, wmax]]

        if frame not in ["obs", "rest"]:
            raise ValueError(f"Unknown wavelength frame {frame!r}.")

        if frame == "rest" and self.scene.pointsource.redshift is not None:
            lbda_range = lbda_range * (1 + self.scene.pointsource.redshift)

            
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
                     statistic=np.nanmean, **kwargs):
        """Compute mean signal to noise ratio over a spectral domain.
        (see get_band_flux)

        Parameters
        ----------
        lbda_range : tuple or list of tuples
            (wmin, wmax) test wavelength range [Å] (or list of).
        frame : str, optional
            Wavelength frame ('obs' or 'rest'). Default is "obs".
        statistic : function, optional
            Numpy function to apply on test domain. Default is `np.nanmean`.
        **kwargs
            Goes to :meth:`get_band_flux` -> :meth:`get_spectrum` (e.g.
            `switch_off`).

        Returns
        -------
        float or array
            Signal to noise ratio (depending of `lbda_range` input).

        """
        signal, variance = self.get_band_flux(lbda_range, frame,
                                              statistic=statistic,
                                              **kwargs)
        return signal / variance**0.5

    def get_times(self):
        """Dict of the simulation detector times [in sec].

        Returns
        -------
        dict
            Dictionary of times.

        """
        # detector ones
        times = {k: getattr(self.detector, k) for k in ["integration_time","exposure_time", "tframe", "tgroup"]}
        # effective one
        times["total_exptime"] = self.observing_time # incl nrampss
        return times

    def estimate_variance_contribution(self, *args, **kwargs):
        """DEPRECATED, use get_band_variance_contribution()"""
        warnings.warn("DEPRECATED, use get_band_variance_contribution instead of estimate_variance_contribution")
        return self.get_band_variance_contribution(args, **kwargs)
        
    def get_band_variance_contribution(self, lbda_range, frame="rest",
                                       statistic=np.nanmean):
        """Estimate different contributions to total variance.

        Loops over dark and RoN (for detector), background and pointsource
        (scene) and thermal to estimate their relative contribution of the total
        observed variance.

        Parameters
        ----------
        lbda_range : tuple or list of tuples
            Test wavelength domain [Å].
        frame : str, optional
            Wavelength frame ('obs' or 'rest'). Default is "rest".
        statistic : function, optional
            Numpy function to apply on test domain. Default is `np.nanmean`.

        Returns
        -------
        dict
            Fractional variance contribution.

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

    def estimate_variance_contribution_spectra(self, as_dataframe=True):
        """DEPRECATED, use get_variance_contribution()"""
        warnings.warn("DEPRECATED, use get_variance_contribution instead of estimate_variance_contribution_spectra")
        return self.get_variance_contribution(as_dataframe=True)

    def get_variance_contribution(self, as_dataframe=True):    
        """Estimate different noise contributions to the total variance.

        Parameters
        ----------
        as_dataframe : bool, optional
            If True, return a pandas DataFrame. Default is True.

        Returns
        -------
        pandas.DataFrame or dict
            DataFrame or dictionary with the variance contributions.

        """
        lbda, flux, variance = self.get_spectrum()
        estimates = {"lbda": lbda, "flux": flux, "variance": variance}
        
        for effect in self.variance_sources:
            _, _, variance_noeffect = self.get_spectrum( switch_off=[effect] )
            estimates[effect] = variance - variance_noeffect
    
        if as_dataframe:
            return pandas.DataFrame(estimates)
        
        return estimates # dict

    # ----------- #
    #  Conversion #
    # ----------- #
    def convert_units(self, units_in, units_out, flux_in=1):
        """Convert units.

        Units:
        - adu [total integrated on detector]
        <=>
        - flambda [erg/s/a/cm2]
        - rate [adu/s]
        - framerate [adu/frame]
        - fphoton [ph/s]

        Parameters
        ----------
        units_in : str
            Input units.
        units_out : str
            Output units.
        flux_in : float, optional
            Input flux. Default is 1.

        Returns
        -------
        float
            Converted flux.

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
            coefs = 1/self.detector.photonflux_to_adu(self.spectrograph.lbda)
        elif units_in == "fphoton" and units_out == "adu":
            coefs = self.detector.photonflux_to_adu(self.spectrograph.lbda)
    
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
                      ndrop=None,
                      guess=None,
                      fitter="native",
                      #
                      lbda_range=[4000, 6800], frame="rest",
                      statistic=np.nanmean,
                      reset_param=True, 
                      maxiter=30, tol=0.5, iterstep=1):
        """Vary the free_parameter to reach the target SNR.

        Parameters
        ----------
        target_snr : float
            Target signal to noise ratio.
        max_group : int, optional
            Ignored if `free_parameter` is not 'default'. Larger number of
            group accepted. Default is None.
        nframe_per_group : int, optional
            Number of frames per group. Default is None.
        ndrop : int, optional
            Number of dropped frames. Default is None.
        guess : int, optional
            Initial guess for the free parameter. Default is None.
        fitter : str, optional
            Fitter to use ("native" or "scipy"). Default is "native".
        lbda_range : list, optional
            (wmin, wmax) test wavelength range [Å]. Default is [4000, 6800].
        frame : str, optional
            Wavelength frame ('obs' or 'rest'). Default is "rest".
        statistic : function, optional
            Function to apply on test domain to compute the snr. Default is
            `np.nanmean`.
        reset_param : bool, optional
            If True, the input simulation is back to initial value. If False,
            it is that of the reached snr. Default is True.
        maxiter : int, optional
            Maximum number of iterations. Default is 30.
        tol : float, optional
            Tolerance for the SNR. Default is 0.5.
        iterstep : int, optional
            Step for the iteration. Default is 1.

        Returns
        -------
        int, float, float
            - Number of frame/group (see free_parameter)
            - Reached SNR.
            - Integration_time

        """
        if max_group is None:
            max_group = self.detector.max_group
            
        # store current ramp and nmd
        input_nmd = self.get_parameter("nmd")
        input_nramps = self.get_parameter("nramps")
        
        # default values are these from the current config.
        if nframe_per_group is None:
            nframe_per_group = input_nmd[1]
            
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

        # compute the snr for 1 ramp.
        single_fullramp_snr = self.get_band_snr(lbda_range=lbda_range,
                                                frame=frame,
                                                statistic=statistic)

        
        # is one ramp enought ?
        if single_fullramp_snr >= (target_snr-tol):
            # yes ? Check if small ramp ok ?
            self.update( nramps = 1, nmd=(np.max(small_ngroup_range), nframe_per_group_small, 0) ) # e.g., 1* (n, 4, 0)
            single_ramp_smallgroup_snr = self.get_band_snr(lbda_range=lbda_range,
                                                                   frame=frame,
                                                                   statistic=statistic)
            # do you reach the SNR with `np.max(small_ngroup_range)` "small ramps"
            if single_ramp_smallgroup_snr <= (target_snr-tol):
                # no ? use larger groups
                if guess is None:
                    guess = int(max_group/2)
                    
                self.update( nramps = 1, nmd=(guess, nframe_per_group, 0) ) # e.g., 1* (n, 8, 0)    
            else:
                # yes ? then it's a bright target.
                # => Is that so bright that (np.min(small_ngroup_range), 4, 0) would do the jobs ?
                self.update( nramps = 1, nmd=(np.min(small_ngroup_range), nframe_per_group_small, 0) ) # e.g., 1* (8, 4, 0)
                single_framegroup_snr = self.get_band_snr(lbda_range=lbda_range,
                                                                   frame=frame,
                                                                   statistic=statistic)
                if single_framegroup_snr >= (target_snr-tol):
                    # yes ? then super bright, let's move to signel frame ramps starting from np.max(small_ngroup_range)
                    self.update( nramps = 1, nmd=(np.max(small_ngroup_range), 1, 0) ) # e.g., 1* (n, 1, 0)
                else:
                    # no ? then not that bright, let's stay in the 4 group ramps
                    self.update( nramps = 1, nmd=( int(np.mean(small_ngroup_range)), nframe_per_group_small, 0) ) # e.g., 1* (n, 4, 0)    
                    
            # In any case, you do 1 ramp, so loop over ngroup                
            free_parameter = "ngroup"
            
        else:
             # no ? fix nmd at fullramp and  loop over ngroup
            free_parameter = "nramps"
            if guess is None:
                guess = 4 # we expect less than, say, 20 ramps and at least 2.
            self.update( nramps=guess, nmd=(max_group, nframe_per_group, ndrop) )
            prop_fetch |= {"min_value": 2}

        if fitter == "native":
            read_config, snr, integration_time = self._fetch_snr(target_snr,
                                                             free_parameter=free_parameter,
                                                             iterstep=iterstep, maxiter=maxiter,
                                                             **prop_fetch)
        elif fitter == "scipy":
            read_config, snr, integration_time = self._fetch_snr_minimize(target_snr, free_parameter=free_parameter,
                                                                              x0=guess, **prop_fetch)
            
        # reset back to initial nmd
        if reset_param:
            self.update(nmd = input_nmd, nramps=input_nramps)

        # return what you where looking for.
        return read_config, snr, integration_time

    def _fetch_snr_minimize(self, target_snr, free_parameter, x0,
                                lbda_range=[4000, 6800], frame="rest", statistic=np.nanmean,
                                min_value=None, as_int=True,
                                tol=0.3, **kwargs):
        """Fetch SNR using scipy.optimize.minimize.

        Parameters
        ----------
        target_snr : float
            Target signal to noise ratio.
        free_parameter : str
            Parameter to vary ('ngroup', 'nramps', 'nframe').
        x0 : float
            Initial guess.
        lbda_range : list, optional
            (wmin, wmax) test wavelength range [Å]. Default is [4000, 6800].
        frame : str, optional
            Wavelength frame ('obs' or 'rest'). Default is "rest".
        statistic : function, optional
            Function to apply on test domain to compute the snr. Default is
            `np.nanmean`.
        min_value : int, optional
            Minimum value for the free parameter. Default is None.
        as_int : bool, optional
            If True, round the result to the nearest integer. Default is True.
        tol : float, optional
            Tolerance for the optimization. Default is 0.3.
        **kwargs
            Goes to `scipy.optimize.minimize`.

        Returns
        -------
        dict, float, float
            - Used configuration
            - Reached SNR
            - Total exposure time

        """
        from scipy import stats, optimize
        
        minimal_values = {"ngroup": self.detector.min_group, "nramps": 1, 'nframe':2}
        if min_value is None:
            min_value = minimal_values.get(free_parameter)
        
        # ---------------------- #
        # Parsing input uptions  #
        # ---------------------- #
        if free_parameter not in ["ngroup", "nramps", 'nframe']:
            raise ValueError(f"free_parameter should be 'ngroup', 'nramps' or 'nframe' {free_parameter} given.")
        
        prop_snr = dict(lbda_range=lbda_range, 
                        frame=frame, 
                        statistic=statistic)

        # nframe supposed to change the macc mode to (1,1,0)
        if free_parameter == "nframe":
            input_nmd = self.get_parameter("nmd")
            self.update(nmd=(min_value, 1, 0)) # start at min value
            free_parameter = "ngroup" # 1 frame per group, so ngroup=nframe
        else:
            input_nmd = None

        # ---------------------- #
        # Fitting                #
        # ---------------------- #
        def to_minimize(value):
            """ """
            self.update(**{free_parameter: value})
            current_snr = self.get_band_snr(**prop_snr)
            return (current_snr - target_snr)**2

        fit_result = optimize.minimize(to_minimize, x0=x0, tol=tol, **kwargs)
        if as_int:
            bestfit = round(fit_result.x[0])
            self.update(**{free_parameter: bestfit})
            current_snr = self.get_band_snr(**prop_snr)
        else:
            # as fit_result.fun = (current_snr - target_snr)**2
            current_snr = np.sqrt(fit_result.fun) + target_snr

        # So this is what is used.
        used_config = {"nmd": self.get_parameter("nmd"),
                       "nramps": self.get_parameter("nramps")}
        
        total_exptime = self.observing_time # includes nrampss
        return used_config, current_snr, total_exptime
        
    def _fetch_snr(self, target_snr, free_parameter, 
                   lbda_range=[4000, 6800], frame="rest", statistic=np.nanmean,
                   min_value=None,
                   maxiter=100, tol=0.5, iterstep=1):
        """Vary the free_parameter to reach the target SNR.

        = internal function that has fixed free_parameters; see self.fetch_snr() =

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
        statistic : function, optional
            Function to apply on test domain to compute the snr. Default is
            `np.nanmean`.
        min_value : int, optional
            Minimum value for the free parameter. Default is None.
        maxiter : int, optional
            Maximum number of iterations. Default is 100.
        tol : float, optional
            Tolerance for the SNR. Default is 0.5.
        iterstep : int, optional
            Step for the iteration. Default is 1.

        Returns
        -------
        int, float
            - Number of frame/group (see free_parameter)
            - Reached SNR.

        """
        # minimal values (including these)
        minimal_values = {"ngroup": 2, "nramps": 1, 'nframe':2}
        if min_value is None:
            min_value = minimal_values.get(free_parameter)

        # internal function that perform the fit steps
        def change(value, current_snr, iterstep):
            """ """    
            if current_snr >= target_snr: # going down.
                was_high = True
                coefs = -1
                condition = np.less
            else: # going up
                was_high = False
                coefs = +1
                condition = np.greater

            new_value = value + coefs*iterstep
            if new_value <=0 :
                new_value = 1
                # warnings.warn(f"requested value lower than 0: old value {value} + iterstep {iterstep}")                

            if new_value < min_value:
                new_value = min_value
                # warnings.warn(f"requested value lower than min_value: old value {value} + iterstep {iterstep}")
                
            _ = self.update(**{free_parameter: new_value})
            new_snr = self.get_band_snr(**prop_snr)
            
            # need to go in the same direction
            if (new_snr >= target_snr and was_high):
                new_iterstep = int(iterstep*2)
            elif (new_snr < target_snr and not was_high):
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
        
        prop_snr = dict(lbda_range=lbda_range, 
                        frame=frame, 
                        statistic=statistic)

        # nframe supposed to change the macc mode to (1,1,0)
        if free_parameter == "nframe":
            input_nmd = self.get_parameter("nmd")
            self.update(nmd=(min_value, 1, 0)) # start at min value
            free_parameter = "ngroup" # 1 frame per group, so ngroup=nframe
        else:
            input_nmd = None

        # initial state
        current_value = self.get_parameter(free_parameter)
        current_snr = self.get_band_snr(**prop_snr)
        break_at_next = False
        # while loop
        counter = 0
        value_history = []
        while np.abs(current_snr - target_snr) > tol and counter < maxiter:
            value_history.append(current_value)
            if np.isnan(current_value) or current_value<=min_value: # moving to too small value
                current_value = min_value # reset to minimum 
                self.update(**{free_parameter: current_value })
                current_snr = self.get_band_snr(**prop_snr) # and get corresponding SNR
                
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
    def show_spectrum(self, ax=None, switch_off=[], snr=False, **kwargs):
        """Plot the detected spectrum.

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Axes. Default is None.
        switch_off : list, optional
            List of discarded scene elements (pointsource, host, background).
            Default is [].
        snr : bool, optional
            Add SNR curve on top. Default is False.
        **kwargs
            Propagated to plotting function.

        Returns
        -------
        matplotlib.Axes
            Axes.

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
        l, = ax.plot(lbda_mu, sig, ds='steps-mid', **kwargs)
        ax.fill_between(lbda_mu, sig - dsig, sig + dsig,
                        step='mid', color=l.get_color(), alpha=0.2)
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

        return ax

    def show_cube(self, in_photons=True, switch_off=[], spec_prop={},
                      psf_profile="default", **kwargs):
        """Display the cube generated by :meth:`get_projected_scene`.

        The figure has two panels:
        - left: total spectrum (cube summed over spaxels)
        - right: white image (cube summed over wavelengthes)

        Parameters
        ----------
        in_photons : bool, optional
            Cube in photon/s (default: flambda). Default is True.
        switch_off : list, optional
            List of discarded scene elements (pointsource, host, background) +
            thermal. Default is [].
        spec_prop : dict, optional
            Spectrum plot options. Default is {}.
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        **kwargs
            Cube `imshow` options.

        Returns
        -------
        matplotlib.figure.Figure
            2-axis figure.

        """

        cube = self.get_projected_scene(in_photons=in_photons,
                                            switch_off=switch_off,
                                            psf_profile=psf_profile)

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
                            extent=self.spectrograph.mla_extent,
                            **{**default, **kwargs})
        ax_img.set(xlabel="x [spx]", ylabel="y [spx]")
        fig.colorbar(img, ax=ax_img, label="Integrated signal")

        return fig

    def show_scene(self, incl_error=True, fig=None,
                   rest_lbda_range = [4000, 7000],
                   obs_lbda_ranges = [[4000, 6000], [9000, 11_000], [14_000, 16_000]],
                   psf_profile="default", oversampling=None, prop_slice={},
                  **kwargs):
        """Show the scene.

        Parameters
        ----------
        incl_error : bool, optional
            If True, include error in the plot. Default is True.
        fig : matplotlib.figure.Figure, optional
            Figure. Default is None.
        rest_lbda_range : list, optional
            Rest-frame wavelength range. Default is [4000, 7000].
        obs_lbda_ranges : list, optional
            Observed-frame wavelength ranges. Default is [[4000, 6000],
            [9000, 11_000], [14_000, 16_000]].
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        oversampling : int, optional
            Oversampling factor. Default is None.
        prop_slice : dict, optional
            Properties for the slice. Default is {}.
        **kwargs
            Goes to `imshow`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure.

        """

        if fig is None:
            import matplotlib.pyplot as plt            
            fig = plt.figure(figsize=(7,5))
    
        ncols = len(obs_lbda_ranges)
        # 
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=2, ncols=ncols, figure=fig, 
                      height_ratios=(ncols,1),
                      hspace=0.0, wspace=0.05)
        
        ax1 = fig.add_subplot(gs[0, :])
        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))

        if oversampling is not None:
            prop_slice["oversampling"] = oversampling
            prop_slice["as_oversampled"] = True
            
        flux, variance = self.get_slice(obs_lbda_ranges, frame="obs", 
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
        [ax_.set_xticks([]) for ax_ in fig.axes]#

        return fig
            

    def show_nea_fwhm(self, figsize=(4,7)):
        """Show NEA and FWHM.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (4, 7).

        Returns
        -------
        matplotlib.figure.Figure
            Figure.

        """
        import matplotlib.pyplot as plt
        
        fig, (axnea, axneaspatial, axfwhm) = plt.subplots(ncols=1, nrows=3, figsize=figsize,
                                                         gridspec_kw={"hspace":0.05})
        self.spectrograph.show_nea(ax=axnea);
        axnea.set_xticklabels([])
        self.spectrograph.show_nea_spatial(ax=axneaspatial, legend=False);
        axneaspatial.set_xticklabels([])
        self.spectrograph.show_fwhm(ax=axfwhm, legend=False);

        return fig
    
    def show_variance_sources(self, variance_contrib=None, flux_calibrated=True,
                                  figsize=(7, 7), gridspec={}):
        """Summary figure showing various variance contributions.

        Parameters
        ----------
        variance_contrib : pandas.DataFrame, optional
            DataFrame containing the variance contributions.
            `variance_contrib = self.get_variance_contribution(as_dataframe=True)`
            If None, this grabs it. Default is None.
        flux_calibrated : bool, optional
            If True, show spectra flux calibrated. Default is True.
        figsize : tuple, optional
            Figure size. Default is (7, 7).
        gridspec : dict, optional
            GridSpec keywords. Default is {}.

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
        noise = np.sqrt(variance)/norm
        snr = flux/noise
        
        # Main plot
        ax.plot(variance_contrib["lbda"], flux, lw=1, color="k")
        ax.fill_between(variance_contrib["lbda"], flux+noise, flux-noise, alpha=0.3, 
                       color="0.5", lw=0)
        ax.axhline(0, color="0.5", lw=1, zorder=1)

        # Loop over effects
        in_effect = []
        base = 0
        for new_effect in self.variance_sources:
            in_effect = in_effect+[new_effect]
            
            spectra_contrib = variance_contrib[in_effect].sum(axis=1)/variance
            axsnr.fill_between(variance_contrib["lbda"], 
                                 base*snr,
                                 spectra_contrib*snr, 
                                 facecolor=COLORS[new_effect],
                                 edgecolor="0.5", lw=0., 
                                 alpha=0.5)
            
            axv.fill_between(variance_contrib["lbda"], 
                             base,
                             spectra_contrib, 
                             facecolor=COLORS[new_effect],
                             edgecolor="0.5", lw=0.,
                             label=new_effect)
            
            base = spectra_contrib

        axsnr.plot(variance_contrib["lbda"], snr, color="0.5", lw=1)
        axsnr.plot(variance_contrib["lbda"], base*snr, color="k")
        
        # Fancy
        ax.set_xlim(variance_contrib["lbda"].values[0], variance_contrib["lbda"].values[-1])
        axsnr.set_xlim(*ax.get_xlim())
        axv.set_xlim(*ax.get_xlim())
        axv.set_ylim(0)
        
        ax.set_xticklabels([])
        axsnr.set_xticklabels([])
        axsnr.set_ylim(0)
        
        axv.set_xlabel("Wavelength [A]", fontsize="large")
        if flux_calibrated:
            ax.set_ylabel("Flux [erg/s/cm2/A]", fontsize="large")
        else:
            ax.set_ylabel("Flux [ADU]", fontsize="large")
        axsnr.set_ylabel("Signal / Noise", fontsize="large")
        axv.set_ylabel("variance contrib.", fontsize="large")
        axv.legend(loc=[0.01, 1.3], fontsize="small", frameon=False)
        
        ax.set_title(f"z={self.get_parameter('redshift')} | c={self.get_parameter('c')}, x1={self.get_parameter('x1')} | t={self.get_times()['total_exptime']/60:.1f} min",
                    color="k", fontsize="small", loc="right")
        return fig
        
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def meta(self):
        """Concatenation of all element configurations (aka. meta)."""
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
        """Total observing time, i.e. `exptime * nramps`."""
        return self.detector.exposure_time * self.extraction["nramps"]    

    @property
    def mutable_parameters(self):
        """List of mutable parameters."""
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
        """List of variance sources."""
        return self.VARIANCE_SOURCES
