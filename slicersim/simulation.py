"""
Top-level simulation tools.

.. autosummary::

   Simulation
"""

__authors__ = "Mickaël Rigault <m.rigault@ipnl.in2p3.fr>, " \
    "Yannick Copin <y.copin@ipnl.in2p3.fr>"

import warnings
warnings.simplefilter('always', UserWarning)

import pprint

import numpy as np
import pandas
from .scene import Scene
from .spectrograph import Spectrograph
from .detector import Detector

__all__ = ["Simulation"]

COLORS = {"target": "#283C48", 
          "host":"#6E441E", 
          "ron": "#80886D", 
          "background": "#B08630", 
          "dark": "#616B62", 
          "thermal":"#662515"
          }


    
class Simulation:
    """ Simulation setup.

    A simulation enables you to interact with the scene, the
    spectrograph and the detector to study their relative impact on
    the resulting target Signal-to-Noise Ratio.
    """
    def __init__(self,
                 scene=None,
                 spectrograph=None,
                 detector=None,
                 extraction={},
                 meta={}):
        """
        A simulation is made of 4 elements:

        1. a scene (what Mother Nature provides)
        2. a spectrograph, incl. telescope (how the scene is observed)
        3. a detector (how the signal is recorded)
        4. some extraction parameters (how the signal is extracted)

        :param Scene scene: input scene
        :param Spectrograph spectrograph: input spectrograph
        :param Detector detector: input detector
        :param dict extraction: input extraction parameters
        :param dict meta: meta-data
        :param dict kwargs: elements to be added to meta-data
        """
        self.scene = scene                #: Scene instance
        self.spectrograph = spectrograph  #: Spectrograph instance
        self.detector = detector          #: Detector instance
        self.extraction = extraction      #: Extraction parameters
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
    def from_config(cls, config):
        """ Initiate simulation from config nested dictionary.

        :param dict config: top level configuration containing
                            configurations for scene, spectrograph, detector
                            and extraction parameters (by default:
                            use default config from
                            :func:`mlaperf.iotools.get_config`)
        :return: scene instance
        """

        # First initialize spectrograph to set wavelengths, then other elements
        # using spectrograph wavelengths.

        # Initialize the spectrograph from config
        spectrograph = Spectrograph.from_config(config["spectrograph"])

        # Initialize the scene from config (wavelength from spectrograph)
        scene = Scene.from_config(config["scene"], lbda=spectrograph.lbda)

        # Initialize the detector from config
        detector = Detector.from_config(config["detector"], lbda=spectrograph.lbda)

        # Initialize extraction parameters from config
        extraction = config["extraction"]

        return cls(scene=scene,
                   spectrograph=spectrograph,
                   detector=detector,
                   extraction=extraction,
                   meta=config)  # Store config dict in meta
                   
    # =============== #
    #   Methods       #
    # =============== #
    def _fetch_mutable_parameters(self, key):
        """ """
        mutable_ = [l for l in self.mutable_parameters if key in l]
        if len(mutable_)==0:
            return None
        
        return mutable_

    def change_target(self, pointsource):
        """ override the considered pointsource 
        
        Parameters
        ----------
        pointsource: slicersim.PointSource
            new pointsource to be used.

        Returns
        -------
        None
        """
        self.scene._target = pointsource
    
    def update(self, reset_others=False, **kwargs):
        """ Update any mutable parameter of the simulation.

        for convinience, the update method respects the django '__' format, 
        such that, e.g. 'target__phase' is understood as 'target.phase'.
        This way, one can do:
        >>> self.update(target__phase = -1)

        for convinience, you can specify a shorten name, like "phase". 
        If the correspondance to a mutable_parameters is uniquen this will accept it.
        >>> self.update(phase = -1)

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
        self.spectrograph.update(**updates_spectrograph)
        self.scene.update(**updates_scene)
        self.detector.update(**updates_detector)

        # Convert extraction parameters in relative units to absolute units        
        self.extraction.update(**updates_extraction)
        # Do NOT update meta, so reset still works as expected.
        
    def reset(self, which="*"):
        """ reset the simulation element at their initial config value.

        Careful, this creates a new element element and erases the current one.

        Parameters
        ----------
        which: str, list
            element to reset. could be '*'/'all' or a key of
            :attr:`Simulation._elements`.
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
    def get_input_spectrum(self, which="target"):
        """ get the input spectrum. 

        This is a shortcut to self.scene.get_element_spectrum

        Parameters
        ----------
        which: string
            which input do you want:
            - individual elements: 'target', 'background', 'host' 
            - all merged: 'stacked'

        Returns
        -------
        lbda: array
            wavelength in Angstrom

        flux: array
            input flux (erg/s/cm^2/A)

        """
        if which == "stacked":
            lbda, specs = sim.scene.get_stacked_spectra()
            flux = np.sum(specs, axis=0)
        else:
            lbda, flux = self.scene.get_element_spectrum(which)
            
        return lbda, flux

    def get_effective_transmission(self):
        """ Effective total transmission of the spectrograph
        
        Product of the spectroscopic throughput: spectrograph.flambda2photon
        and the detector efficiency: detector.photonflux2ADU


        Returns
        -------
        lbda: array
            wavelength in Angstrom

        throughput: array
            effective throughput (ADU/ (erg/s/cm^2/A) )

        """
        return self.spectrograph.lbda, \
          self.spectrograph.flambda2photon * self.detector.photonflux2ADU

    def get_effective_waveresolution(self, npx=2, sigma=None):
        """ effective wavelength resolution

        R &= \frac{2}{n \delta\lambda} \\
        with
        \delta\lambda &= \max(1, \sigma) \times \Delta\lambda
        and 
        `\Delta\lambda` the spectral step [Å] 
        and
        `\sigma` is the spectral resolution [px].

        Parameters
        ----------
        npx: float
            n-px resolution (i.e. n px per spectral elements)
                    
        sigma: 
            spectral PSF stddev override
            
        Returns
        -------
        lbda: array
            wavelength in Angstrom

        waveresolution: array
            effective wavelength resolution 

        """
        return self.spectrograph.lbda, \
          self.spectrograph.effective_resolution(npx=npx, sigma=sigma)

    def get_pixel_variance(self, flux=0):
        """ get variance associated to 1 pixel on the detector. """
        _, pixel_var = self.detector.estimate_pixel_signal(flux)
        pixel_var *= self.extraction["nramp"] # incl. multi ramp approach.
        return pixel_var
        
    def get_nea(self, nea_spatial=None, nea_pixels=None):
        """ Noise effective area  (nea_spatial * nea_pixel) """
        return self.spectrograph.get_nea(position = self.scene.target.position,
                                          nea_spatial=nea_spatial, nea_pixels=nea_pixels)
            
    def get_nea_variance(self, spectrum=None, nea_spatial=None, nea_pixels=None):
        """ get variance estimatation from the noise equivalent area. 

        Parameters
        ----------
        spectrum: None
            * limited to test config...*
            Provide a spectrum in [erg/s/cm2/A]. 
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
                
            spec_adu = self.flambda_to_adu(spectrum)
            # This is an approximation that seem to work, not clear why... (so the above warning)
            poisson_adu_to_variance = np.sqrt(self.spectrograph.get_nea_spatial()  / self.spectrograph.get_nea_pixels() )
            var_source = spec_adu * poisson_adu_to_variance
        else:
            var_source = 0
        
        return variance_pixels + var_source
    
    def get_parameter(self, which=None, default=None, as_dict=True):
        """ shortcut to get simulation parameter(s).

        Should be an attribute of one of the element of the simulation,
        e.g. `gain` from `Simulation.detector.gain`.

        :param str which: parameter name (or list of)
        :param default: default value
        :param bool as_dict: return `{which: value}` rather than `value`
        :return: parameter value
        """
        if which is None:       # Get all of them
            which = [ item
                      for sublist in self._elements
                      for item in sublist ]

        if np.ndim(which):  # If a list, loop over elements
            return { param: self.get_parameter(param, as_dict=False)
                     for param in which }

        # Extraction parameter
        if which in self.extraction:
            return self.extraction[which]

        # Short cuts
        if which == "ngroup":
            return self.detector.nmd[0]

        # Short cuts
        if which == "nframe":
            return self.detector.nmd[0]

        # Otherwise, look at individual elements
        elements = ["scene", "spectrograph", "detector"]
        for element in elements:
            instance = getattr(self, element)

            if '.' in which:    # Composed key: key1.key2
                k1, k2 = which.split('.')
                if hasattr(instance, k1):
                    instance = getattr(instance, k1)
                    which = k2

            if hasattr(instance, which):            # self.which
                return getattr(instance, which)

            if which in getattr(instance, "meta"):  # self.meta['which']
                return getattr(instance, "meta")[which]

            if element == "scene": # for scene: self.meta['target']['which']
                meta_target = getattr(instance, "meta")["target"]
                if which in meta_target:
                    return meta_target[which]

        return {which: default} if as_dict else default

    def get_projected_scene(self, in_photons=True, switch_off=[]):
        """ project the scene through spectrograph and get flux cube [ph or flambda].

        :param bool in_photons: cube in photon/s (default: flambda)
        :param list switch_off: list of discarded scene elements
                                (target, host, background) + thermal
        :return: (nlbda, ny, nx) cube
        """
        cube = np.zeros(self.cube_shape)  # (nlbda, ny, nx)

        # spectra (3, nlbda):
        # * point source spectrum [erg/s/cm²/Å]
        # * host (not yet implemented)
        # * background spectrum [erg/s/cm²/Å/arcsec²]
        lbda, (target, host, background) = self.scene.get_stacked_spectra(fillna=0)

        # Fill the cube with scene elements in photons/s/spx
        if "target" not in switch_off:
            cube += self.spectrograph.generate_point_source(
                target, position=self.scene.target_position)

        if "host" not in switch_off:      
            if np.any(host):
                warnings.warn("Host cube not implemented.")

        if "background" not in switch_off:                    
            cube += self.spectrograph.generate_background(background)

        if "thermal" not in switch_off:
            cube += self.spectrograph.get_thermal_signal(as_cube=True) # [ph/s]

        if not in_photons:      # Convert back to flambda
            cube /= self.spectrograph.flambda2photon[:, np.newaxis, np.newaxis]
            
        return cube  # (nlbda, ny, nx) | [ph/s] or [erg/cm²/Å/s]

    def get_detected_cube(self, switch_off=[]):
        """ DEPRECATED, use get_cube instead
        """
        warnings.warn("DEPRECATED: get_detected_cube is deprecated, use get_cube() instead")
        return self.get_cube(switch_off=switch_off)
    
    def get_cube(self, switch_off=[]):
        """ get data cube as extracted from exposure [ADU].

        Parameters
        ----------
        switch_off: list
            name of variance sources to switch off.
            (dark, ron, target, host, background, thermal)

        Returns:
        --------
        cube_signal, cube_variance: 
            (nlbda, ny, nx) in [ADU].
        """
        AVAILABLE_SWITCHOFF = ["dark", "ron", "target", "host", "background","thermal"]
        switch_off = np.atleast_1d(switch_off).tolist() # as list
        
        if np.any(effect_bool := [effect not in AVAILABLE_SWITCHOFF for effect in switch_off]):
            unknown_effect = np.asarray(switch_off)[effect_bool]
            warnings.warn(f"unknown switch off effect(s) {unknown_effect}.")
            
        # Detector effect
        if "dark" in switch_off:
            current_dark = self.get_parameter("dark")
            self.update(detector__dark=0)            # Switch off dark
        else:
            current_dark = None
            
        if "ron" in switch_off:
            current_ron = self.get_parameter("ron")
            self.update(detector__ron=0)             # Switch off ron
        else:
            current_ron = None

        # Scene effect
        cube = self.get_projected_scene(in_photons=True,
                                        switch_off=switch_off)  # (nlbda, ny, nx)
        sigmas = self.spectrograph.get_xdisp_sigma_spectral()   # (nlbda,)

        try:
            width = self.extraction["xdisp_width"]
        except KeyError:
            # See Spectrograph.rescale_parameters
            width = round(self.extraction["xdisp_width_insigma"] *
                          self.spectrograph.xdisp_sigma_spectral)

        # This assumes optimal extraction, only the variance depends
        # on detector parameters.
        sig_cube, var_cube = self.detector.estimate_spx_spectrum(
            cube,               # (nlbda, ny, nx) [ph/s]
            sigma=sigmas,       # (nlbda,) [px]
            width=width)

        #
        # switch back in detector effects
        #
        if current_dark is not None:
            self.update(detector__dark=current_dark)
            
        if current_ron is not None:
            self.update(detector__ron=current_ron)
            
        # output
        return sig_cube, var_cube  # (nlbda, ny, nx) [ADU, ADU²]


    def get_slice(self, lbda_range, frame="obs", switch_off=[], incl_error=False, 
                  squeeze=False,
                  **kwargs):
        """ get slices of the cube.
        
        Parameters
        ----------
        lbda_range: (float, float), or list of.
            lbda_range to use (lbda_min, lbda_max) [n-slice=1] or 
            ((lbda_min, lbda_max), (lbda_min, lbda_max), ...)  [n-slice].
            unit: that of self.spectrograph.lbda.
            
        Returns
        -------
        signal, variance: (nslice, ny, nx)
            signal [ADU] and variance [ADU^2]
            
        """
        lbda_range = np.atleast_2d(lbda_range) # [[wmin, wmax]]
    
        if frame not in ["obs", "rest"]:
            raise ValueError(f"Unknown wavelength frame {frame!r}.")
    
        if frame == "rest" and self.scene.target.redshift is not None:
            lbda_range = lbda_range * (1 + self.scene.target.redshift)
    
        cube_signal, cube_variance = self.get_cube(switch_off=switch_off, **kwargs)
    
        band_sigs, band_vars = [], []
        for wmin, wmax in lbda_range:
            flag_lbda = ((self.spectrograph.lbda >= wmin) & (self.spectrograph.lbda <= wmax))
            band_sigs.append( np.nansum(cube_signal[flag_lbda], axis=0) )
            band_vars.append( np.nansum(cube_variance[flag_lbda], axis=0) )
    
        band_sigs = np.asarray(band_sigs)
        band_vars = np.asarray(band_vars) 
        if len(lbda_range) == 1 and squeeze:
            band_sigs = band_sigs[0] 
            band_vars = band_vars[0]
    
        if incl_error:
            band_sigs = np.random.normal(loc=band_sigs, scale=np.sqrt(band_vars)) # normal noise
            
        return band_sigs, band_vars        
    
    def get_spectrum(self, switch_off=[], incl_error=False):
        """ get the target signal and variance [ADU].

        :param list switch_off: list of discarded scene elements
                                (target, host, background) + thermal
        :param bool incl_error: should the signal be scattered by the error?
        :return: (nlbda,) signal and variance [ADU]
        """
        # (lbda, nx, ny) [ADU]
        sig_cube, var_cube = self.get_cube(switch_off=switch_off)

        try:
            radius = self.extraction["aperture_radius"]  # [spx]
        except KeyError:
            # See Spectrograph.rescale_parameters
            radius = (self.extraction["aperture_radius_insigma"] *
                      self.spectrograph.psf_sigma_spectral /  # [arcsec]
                      self.spectrograph.spx_spatial_scale)    # [arcsec/spx]

        spec_variance = self.spectrograph.point_source_variance(
            var_cube, position=self.scene.target_position, radius=radius)

        # Assume the spectrum is perfectly extracted
        if "target" not in switch_off:
            _, target_phflux = self.scene.get_element_spectrum('target') * self.spectrograph.flambda2photon
            spec_signal = target_phflux * self.detector.photonflux2ADU
        else:
            spec_signal = np.zeros_like(self.spectrograph.lbda)

        if (nexp := self.extraction["nramp"]) > 1:  # Nb of exposures
            spec_signal *= nexp
            spec_variance *= nexp

        if incl_error:
            spec_signal += np.random.normal(loc=0, scale=np.sqrt(spec_variance))
            
        return self.spectrograph.lbda, spec_signal, spec_variance  # (nlbda,) [ADU, ADU²]

    def get_snr(self, switch_off=[]):
        """ get signal to noise spectrum.

        See :meth:`get_spectrum`.
        Returns
        -------
        array
        """
        _, signal, variance = self.get_spectrum(switch_off=switch_off)
        return signal / variance**0.5

    def get_band_flux(self, lbda_range, frame="obs",
                        statistic=np.nanmean,
                        squeeze=True, **kwargs):
        """ Estimate mean signal and variance over a spectral domain.

        The photometry is computed assuming a top-hat filter and simply
        considering wavelength bins within the spectral domain (no
        interpolation nor weighting).

        :param lbda_range: (wmin, wmax) test wavelength range [Å] (or list of)
        :param str frame: wavelength frame ('obs' or 'rest')
        :param statistic: numpy function to apply on test domain
        :param bool squeeze: squeeze final df (for single-band)
        :param kwargs: goes to get_spectrum (e.g. switch_off)
        :return: signal and variance [ADU, ADU²]
        """
        # accepts multiple ranges
        lbda_range = np.atleast_2d(lbda_range) # [[wmin, wmax]]

        if frame not in ["obs", "rest"]:
            raise ValueError(f"Unknown wavelength frame {frame!r}.")

        if frame == "rest" and self.scene.target.redshift is not None:
            lbda_range = lbda_range * (1 + self.scene.target.redshift)

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
        """ Compute mean signal to noise ratio over a spectral domain.
        (see get_band_flux)

        Parameters
        ----------
        lbda_range: (float, float)
            (wmin, wmax) test wavelength range [Å] (or list of)

        frame: str
            wavelength frame ('obs' or 'rest')
        
        statistic: func
            numpy.function to apply on test domain
            
        **kwargs goes to get_band_flux->get_spectrum (e.g. switch_off)

        Returns
        -------
        signal_to_noise:
            float or array (depending of lbda_range input)
        """
        signal, variance = self.get_band_flux(lbda_range, frame,
                                              statistic=statistic,
                                              **kwargs)
        return signal / variance**0.5


    def get_times(self):
        """ dict of the simulation detector times [in sec] """
        # detector ones
        times = {k: getattr(self.detector, k) for k in ["integration_time","exposure_time", "tframe", "tgroup"]}
        # effective one
        times["total_exptime"] = self.observing_time # incl nramps
        return times
    
    def estimate_variance_contribution(self, lbda_range, frame="rest",
                                       statistic=np.nanmean):
        """
        Estimate different contributions to total variance.

        Loops over dark and RoN (for detector), background and target
        (scene) and thermal to estimate their relative contribution of the total
        observed variance.

        :param lbda_range: test wavelength domain [Å]
        :param str frame: wavelength frame ('obs' or 'rest')
        :param statistic: numpy function to apply on test domain
        :return dict: fractional variance contribution.
        """

        prop = dict(statistic=statistic, frame=frame)
        signal, variance = self.get_band_flux(lbda_range, **prop)
        
        # Detector elements
        # Dark
        _, variance_nodark = self.get_band_flux(
            lbda_range,
            switch_off=["dark"], **prop)
        dark_contrib = variance - variance_nodark


        # RoN
        _, variance_noron = self.get_band_flux(
            lbda_range,
            switch_off=["ron"], **prop)
        ron_contrib = variance - variance_noron

        # Scene elements
        # Background
        _, variance_nobkgd = self.get_band_flux(
            lbda_range,
            switch_off=["background"], **prop)
        bkgd_contrib = variance - variance_nobkgd

        # Target
        _, variance_notarget = self.get_band_flux(
            lbda_range,
            switch_off=["target"], **prop)
        target_contrib = variance - variance_notarget

        # Thermal
        _, variance_nothermal = self.get_band_flux(
            lbda_range,
            switch_off=["thermal"], **prop)
        thermal_contrib = variance - variance_nothermal

        return {
            "snr": signal / variance**0.5,
            "exptime": self.detector.exposure_time,     # [s] (total per exp)
            "inttime": self.detector.integration_time,  # [s] (between extrema groups per exp)
            "obstime": self.observing_time,             # [s] (total observing time)
            "signal": signal,                           # [ADU]
            "variance": variance,                       # [ADU²]
            "frac_dark": dark_contrib / variance,       # Detector dark
            "frac_ron": ron_contrib / variance,         # Detector RoN
            "frac_background": bkgd_contrib / variance, # Background
            "frac_target": target_contrib / variance,   # Point source
            "frac_thermal": thermal_contrib / variance, # Thermal
        }

    def estimate_variance_contribution_spectra(self, as_dataframe=True):
        """ Estimate different noise contributions to the total variance 

        Returns
        -------
        DataFrame
        """
        lbda, flux, variance = self.get_spectrum()
        estimates = {"lbda": lbda, "flux": flux, "variance": variance}
        
        for effect in ["dark", "ron", "target", "background", "thermal"]:
            _, _, variance_noeffect = self.get_spectrum( switch_off=[effect] )
            estimates[effect] = variance - variance_noeffect
    
        if as_dataframe:
            return pandas.DataFrame(estimates)
        
        return estimates # dict

    # ----------- #
    #  Conversion #
    # ----------- #
    def flambda_to_adu(self, flux=1.):
        """ convert input spectrum in [erg/s/cm2/A] into ADU (including nramp) """
        _, transmission = self.get_effective_transmission()
        return flux * transmission * self.get_parameter("nramp")

    def adu_to_flambda(self, flux=1.):
        """ convert input spectrum in ADU (including nramp) into [erg/s/cm2/A]"""
        _, transmission = self.get_effective_transmission()
        return flux_adu / (transmission * self.get_parameter("nramp"))
    # ---------- #
    #  Fetching  #
    # ---------- #
    def fetch_snr(self, target_snr,
                      free_parameter="default",
                      #
                      max_group=64,
                      nmd_ramp=(64,8,0),
                      allow_bypass=True,
                      restart_ramp=True,
                      #
                      lbda_range=[4000, 6800], frame="rest",
                      statistic=np.nanmean,
                      reset_param=True, guess=None,
                      maxiter=100, tol=0.5, iterstep=1):
        """ vary the free_parameter to reach the target SNR.
    
        Parameters
        ----------
        target_snr: float
            target signal to noise ratio.
    
        free_parameter: str
            if None or default, this will follow the expecting
            observation strategy: ngroup up to 
            30min exposures, nramp for more.
            otherwise, forces single free_parameter: nframe, ngroup, nramp
        
        max_group: int
            = ignored if free_parameter is not 'default' =
            larger number of group accepted.
            
        lbda_range: list
            (wmin, wmax) test wavelength range [Å]
    
        frame: str
            wavelength frame ('obs', 'rest')
        
        statistic: func
            function to apply on test domain to compute the snr.
        
        reset_param: bool
            should the intput simulation be back to initial value (True)
            or that of the reached snr (False)
            
        restart_ramp: bool
            if free_parameter is not nramp, should this restart from nramp=1 to move the freeparameter ?
            if not, it will assume current nramp.


        Returns
        -------
        int, float
            - number of frame/group (see free_parameter)
            - reached SNR.
            - integration_time
        """
        prop_fetch = dict(lbda_range=lbda_range, frame=frame,
                          statistic=statistic,
                          reset_param=reset_param, guess=guess,
                          maxiter=maxiter,
                          tol=tol, iterstep=iterstep)

        if free_parameter is None:
            free_parameter = "default"

        if free_parameter == "default": # observation strategy
            used_free_parameter = "ngroup" # start by moving groups
            check_if_too_long = True
        else:
            used_free_parameter = free_parameter
            check_if_too_long = False

        # fast check:
        bypass = False # do not bypass 
        if check_if_too_long and allow_bypass:
#            print("check_if_too_long and allow_bypass")
            test_config = {"nmd": nmd_ramp, "nramp": 2}
            input_config = self.get_parameter( list(test_config.keys()) )
            self.update(**test_config)
            tworamp_snr = self.get_band_snr(lbda_range=lbda_range,
                                            frame=frame, statistic=statistic)
            if tworamp_snr<target_snr:
                bypass=True
                
            self.update(**input_config)
                
        if not bypass:
#            print("no bypass")
            if used_free_parameter != "nramp" and restart_ramp:
                self.update( nramp = 1 )
                
            read_config, snr, integration_time = self._fetch_snr(target_snr,
                                                            # This is changing.
                                                            free_parameter=used_free_parameter,
                                                            #
                                                            **prop_fetch)
                                                            
            # single ramp too long: let's loop over ramps.
            new_ngroup = read_config.get("nmd")[0]
            is_too_long = new_ngroup>max_group
            
        if bypass or (check_if_too_long and is_too_long):
#            print("scanning nramp")
            used_free_parameter = "nramp"
            input_nmd = self.get_parameter("nmd")
            self.update(nmd = nmd_ramp)
            read_config, snr, integration_time = self._fetch_snr(target_snr,
                                                                # This is changing.
                                                                free_parameter=used_free_parameter,
                                                                # make sure then at least 2 ramps.
                                                                **(prop_fetch | {"min_value":2})
                                                                )
            # reset back to initial nmd
            self.update(nmd = input_nmd)
        
        return read_config, snr, integration_time

        
        
    def _fetch_snr(self, target_snr, free_parameter="ngroup", 
                   lbda_range=[4000, 6800], frame="rest", statistic=np.nanmean,
                   reset_param=True, guess=None, min_value=None,
                   maxiter=100, tol=0.5, iterstep=1,
                   verbose=False):
        """ vary the free_parameter to reach the target SNR.
    
        = internal function that has fixed free_parameters; see self.fetch_snr() = 

        Parameters
        ----------
        target_snr: float
            target signal to noise ratio.
    
        free_parameter: str
            parameters to vary (ngroup, nramp)
        
        lbda_range: list
            (wmin, wmax) test wavelength range [Å]
    
        frame: str
            wavelength frame ('obs', 'rest')
        
        statistic: func
            function to apply on test domain to compute the snr.
        
        reset_param: bool
            should the intput simulation be back to initial value (True)
            or that of the reached snr (False)
            
        Returns
        -------
        int, float
            - number of frame/group (see free_parameter)
            - reached SNR.
        """
        # minimal values (including these)
        minimal_values = {"ngroup": 2, "nramp": 1, 'nframe':2}
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
        if free_parameter not in ["ngroup", "nramp", 'nframe']:
            raise ValueError(f"free_parameter should be 'ngroup', 'nramp' or 'nframe' {free_parameter} given.")
        
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


        # initial values
        init_value = self.get_parameter(free_parameter)
        
        if guess is None:
            current_value = init_value
            current_snr = self.get_band_snr(**prop_snr)
        else:
            current_value = guess
            self.update( **{free_parameter: current_value} )
            current_snr = self.get_band_snr(**prop_snr)

        #
        # while loop
        #
        counter = 0        
        while np.abs(current_snr-target_snr)>tol and counter<maxiter:
            if np.isnan(current_value) or current_value<=min_value: # moving to too small value
                current_value = min_value # reset to minimum 
                self.update(**{free_parameter: current_value })
                current_snr = self.get_band_snr(**prop_snr) # and get corresponding SNR
                
            if current_value<=min_value and current_snr>=target_snr: # means lowest is already enough
                break

            current_value, current_snr, iterstep = change(current_value, current_snr, iterstep)
            if verbose:
                print(f"{current_value=}, {current_snr=}, {iterstep=}")
            counter += 1      


        # return used nmd
        used_config = {"nmd": self.get_parameter("nmd"),
                       "nramp": self.get_parameter("nramp")}
            
        total_exptime = self.observing_time # includes nramps
        if reset_param: # reset if needed
            if input_nmd is not None:
                self.update(nmd = input_nmd)
            else:
                self.update(**{free_parameter: init_value})
    
        return used_config, current_snr, total_exptime
    
    # ---------- #
    #  Plotting  #
    # ---------- # 
    def show_spectrum(self, ax=None, switch_off=[], snr=False, **kwargs):
        """
        Plot the detected spectrum.

        :param matplotlib.Axes ax: axes
        :param list switch_off: list of discarded scene elements
                                (target, host, background)
        :param bool snr: add SNR curve on top
        :param dict kwargs: propagated to plotting function
        :return: axes
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

    def show_cube(self, in_photons=True, switch_off=[], spec_prop={}, **kwargs):
        """
        Display the cube generated by :meth:`get_projected_scene`.

        The figure has two panels:

        - left: total spectrum (cube summed over spaxels)
        - right: white image (cube summed over wavelengthes)

        :param bool in_photons: cube in photon/s (default: flambda)
        :param list switch_off: list of discarded scene elements
                                (target, host, background) + thermal
        :param dict spec_prop: spectrum plot options
        :param kwargs: cube `imshow` options
        :return: 2-axis figure
        """

        cube = self.get_projected_scene(in_photons=in_photons,
                                  switch_off=switch_off)

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
                  **kwargs):
        """ 
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
        
        
        flux, variance = self.get_slice(obs_lbda_ranges, frame="obs", 
                                         incl_error=incl_error)
        flux_rest,_ = self.get_slice(rest_lbda_range, frame="rest", 
                                         incl_error=incl_error)
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

    def show_nea_fwhm(self, figsize=(4,7)):
        """ """
        import matplotlib.pyplot as plt
        
        fig, (axnea, axneaspatial, axfwhm) = plt.subplots(ncols=1, nrows=3, figsize=figsize,
                                                         gridspec_kw={"hspace":0.05})
        self.spectrograph.show_nea(ax=axnea);
        axnea.set_xticklabels([])
        self.spectrograph.show_nea_spatial(ax=axneaspatial, legend=False);
        axneaspatial.set_xticklabels([])
        self.spectrograph.show_fwhm(ax=axfwhm, legend=False);

        return fig
    
    def show_variance_sources(self, variance_contrib=None, flux_calibrated=True):
        """ summary figure showing various variance contributions.

        Parameters
        ----------
        variance_contrib: pandas.DataFrame
            dataframe containing the variance contributions. 
            variance_contrib = self.estimate_variance_contribution_spectra(as_dataframe=True)
            If None, this grabs it.

        flux_calibrated: bool
            should spectra be shown flux calibrated of not ?

        Returns
        -------
        fig
        """
        if variance_contrib is None:
            variance_contrib = self.estimate_variance_contribution_spectra(as_dataframe=True)

        if flux_calibrated:
            _, norm = self.get_effective_transmission()
        else:
            norm = 1

        # Figure definition
        import matplotlib.pyplot as plt
        fig, (ax, axsnr, axv) = plt.subplots(3,1, figsize=[7,7], 
                                             gridspec_kw={"hspace":0.1})    
        # Data of interest
        flux = variance_contrib["flux"]/norm
        variance = variance_contrib["variance"]
        noise = np.sqrt(variance)/norm
        snr = flux/noise
        # Main plot
        ax.plot(variance_contrib["lbda"], flux, lw=1, color=COLORS["target"])
        ax.fill_between(variance_contrib["lbda"], flux+noise, flux-noise, alpha=0.3, 
                       color=COLORS["target"], lw=0)
        ax.axhline(0, color="0.5", lw=1, zorder=1)

        # Loop over effects
        in_effect = []
        base = 0
        for new_effect in ["dark", "ron", "target", "background", "thermal"]:
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
        axv.set_ylabel("variance contrib.", fontsize="medium")
        axv.legend(loc=[0.01, 1.5], fontsize="small")
        
        ax.set_title(f"z={self.get_parameter('redshift')} | c={self.get_parameter('c')}, x1={self.get_parameter('x1')} | t={self.get_times()['total_exptime']/60:.1f} min",
                    color="k", fontsize="small", loc="right")
        return fig
        
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def meta(self):
        """ concatenation of all element configurations (aka. meta) """
        return self._in_meta | {"scene": self.scene.meta,
                                "spectrograph": self.spectrograph.meta,
                                "detector": self.detector.meta,
                                "extraction":self.extraction}
    @property
    def cube_shape(self):
        """ Shape of the generated 3d-cube (nlbda, ny, nx). """
        return (self.spectrograph.nlbda,
                *self.spectrograph.spx_shape) # y, x

    @property
    def observing_time(self):
        """ Total observing time, i.e. `exptime * nramp`. """
        return self.detector.exposure_time * self.extraction["nramp"]    

    @property
    def mutable_parameters(self):
        """ list of mutable parameters """
        extra = [] + list(self.extraction.keys())
        scene_ = self.scene.mutable_parameters
        spectro_ = self.spectrograph.mutable_parameters
        detector_ = [f"detector.{k}" for k in self.detector.mutable_parameters]
        all_mutables =  scene_+spectro_+detector_ + extra
        # remove lbda that could be in scene, as this is based on spectrograph
        if "lbda" in all_mutables: 
            all_mutables.remove("lbda")
            
        return all_mutables
                
    @property
    def _elements(self): # test structure
        """ internal list of elements """
        return ["scene", "spectrograph", "detector", "extraction"]
