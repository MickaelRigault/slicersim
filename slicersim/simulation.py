"""
Top-level simulation tools.

.. autosummary::

   Simulation
"""

__authors__ = "Mickaël Rigault <m.rigault@ipnl.in2p3.fr>, " \
    "Yannick Copin <y.copin@ipnl.in2p3.fr>"

import warnings
import pprint

import numpy as np
from .scene import Scene
from .spectrograph import Spectrograph
from .detector import Detector

# Use tqdm (fancy progress bar) if available
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = lambda arg, **kwargs: arg


__all__ = ["Simulation"]

    
class Simulation:
    """
    Simulation setup.

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
        updates_extraction = self.spectrograph.rescale_parameters(
            **updates_extraction,
            spatial_sigma=self.spectrograph.spatial_sigma,
            spectral_sigma=self.spectrograph.spectral_sigma)
        
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
        eff_throughput = self.spectrograph.flambda2photon * self.detector.photonflux2ADU
        
        return self.spectrograph.lbda, eff_throughput
            
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
            cube += self.spectrograph.generate_thermal() # [ph/s]

        if not in_photons:      # Convert back to flambda
            cube /= self.spectrograph.flambda2photon[:, np.newaxis, np.newaxis]

        return cube  # (nlbda, ny, nx) | [ph/s] or [erg/cm²/Å/s]

    def get_detected_cube(self, switch_off=[]):
        """
        Get data cube as extracted from exposure [ADU].

        :param list switch_off: list of discarded scene elements
                                (target, host, background) + thermal
        :return: (nlbda, ny, nx) cube signal and variance [ADU]
        """

        cube = self.get_projected_scene(in_photons=True,
                                  switch_off=switch_off)  # (nlbda, ny, nx)
        sigmas = self.spectrograph.get_spectral_sigma()   # (nlbda,)

        try:
            width = self.extraction["xdisp_width"]
        except KeyError:
            # See Spectrograph.rescale_parameters
            width = round(self.extraction["xdisp_width_insigma"] *
                          self.spectrograph.spectral_sigma)

        # This assumes optimal extraction, only the variance depends
        # on detector parameters.
        sig_cube, var_cube = self.detector.estimate_spx_spectrum(
            cube,               # (nlbda, ny, nx) [ph/s]
            sigma=sigmas,       # (nlbda,) [px]
            width=width)

        return sig_cube, var_cube  # (nlbda, ny, nx) [ADU, ADU²]

    def get_spectrum(self, switch_off=[], incl_error=False):
        """
        Get the target signal and variance (in ADU).

        :param list switch_off: list of discarded scene elements
                                (target, host, background) + thermal
        :param bool incl_error: should the signal be scattered by the error?
        :return: (nlbda,) signal and variance [ADU]
        """

        # (lbda, nx, ny) [ADU]
        sig_cube, var_cube = self.get_detected_cube(switch_off=switch_off)

        try:
            radius = self.extraction["aperture_radius"]  # [spx]
        except KeyError:
            # See Spectrograph.rescale_parameters
            radius = (self.extraction["aperture_radius_insigma"] *
                      self.spectrograph.spatial_sigma /  # [arcsec]
                      self.spectrograph.spatial_scale)   # [arcsec/spx]

        target_variance = self.spectrograph.point_source_variance(
            var_cube, position=self.scene.target_position, radius=radius)

        # Assume the spectrum is perfectly extracted
        if "target" not in switch_off:
            _, target_phflux = self.scene.get_element_spectrum('target') * self.spectrograph.flambda2photon
            target_signal = target_phflux * self.detector.photonflux2ADU
        else:
            target_signal = np.zeros_like(self.spectrograph.lbda)

        if (nexp := self.extraction["nramp"]) > 1:  # Nb of exposures
            target_signal *= nexp
            target_variance *= nexp

        if incl_error:
            target_signal += np.random.normal(loc=0, scale=np.sqrt(target_variance))

        return self.spectrograph.lbda, target_signal, target_variance  # (nlbda,) [ADU, ADU²]

    def get_snr(self, switch_off=[]):
        """
        Compute SNR spectrum.

        See :meth:`get_spectrum`.
        """

        _, signal, variance = self.get_spectrum(switch_off=switch_off)

        return signal / variance**0.5

    def get_band_flux(self, lbda_range, frame="obs", statistic=np.nanmean,
                        squeeze=True, **kwargs):
        """
        Estimate mean signal and variance over a spectral domain.

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
        """
        Compute mean SNR over a spectral domain.

        See :meth:`get_band_flux`.
        """

        signal, variance = self.get_band_flux(lbda_range, frame,
                                                statistic=statistic, **kwargs)

        return signal / variance**0.5


        # df = pandas.concat(subdfs).reset_index(drop=True)
        # df.attrs = self.meta    # Add meta-data

        # return df

    def get_times(self):
        """ dict of the simulation detector times [in sec] """
        return  {k: getattr(self.detector, k) for k in ["integration_time","exposure_time", "tframe", "tgroup"]}
    
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
        current_dark = self.detector.dark
        self.detector.update(dark=0)             # Switch off dark
        _, variance_nodark = self.get_band_flux(lbda_range, **prop)
        self.detector.update(dark=current_dark)  # Switch it back
        dark_contrib = variance - variance_nodark

        # RoN
        current_ron = self.detector.ron
        self.detector.update(ron=0)
        _, variance_noron = self.get_band_flux(lbda_range, **prop)
        self.detector.update(ron=current_ron)
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

    # ---------- #
    #  Fetching  #
    # ---------- #
    def fetch_snr(self, target_snr, free_parameter="ngroup", 
                  lbda_range=[4000, 6800], frame="rest", statistic=np.nanmean,
                  reset_param=True, guess=None,
                  maxiter=100):
        """ vary the free_parameter to reach the target SNR.
    
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
        if free_parameter not in ["ngroup", "nramp", 'nframe']:
            raise ValueError(f"free_parameter should be 'ngroup', 'nramp' or 'nframe' {free_parameter} given.")
        
        prop_snr = dict(lbda_range=lbda_range, 
                        frame=frame, 
                        statistic=statistic)

        # nframe supposed to change the macc mode to (1,1,0)
        if free_parameter == "nframe":
            input_nmd = self.get_parameter("nmd")
            self.update(nmd=(minimal_values.get("ngroup"), 1, 0)) # start at min value
            free_parameter = "ngroup" # 1 frame per group, so ngroup=nframe
        else:
            input_nmd = None

        # used to reset the simu as its initial condition.
        input_value = self.get_parameter(free_parameter)
        new_snr = self.get_band_snr(**prop_snr)
        if new_snr>=target_snr: # going down.
            iterstep = -1
            condition = np.less
        else: # going up
            iterstep = +1
            condition = np.greater

        # while loop
        counter = 0
        current_value = input_value if guess is None else guess            
        while counter < maxiter and current_value+ iterstep>=minimal_values.get(free_parameter):
        
            new_value = current_value + iterstep
            self.update(**{free_parameter: new_value})
            new_snr =  self.get_band_snr(**prop_snr)
            if condition(new_snr, target_snr):
                if iterstep>0:  
                    current_value = new_value
                    self.update(**{free_parameter: new_value})
                    new_snr =  self.get_band_snr(**prop_snr)
                else:
                    self.update(**{free_parameter: new_value-iterstep})
                    new_snr = self.get_band_snr(**prop_snr)
                break
                
            counter += 1
            current_value = new_value

        integration_time = self.detector.get_integration_time()
        if reset_param: # reset if needed
            if input_nmd is not None:
                self.update(nmd = input_nmd)
            else:
                self.update(**{free_parameter: input_value})
    
        return current_value, new_snr, integration_time
    
    # ---------- #
    #  Plotting  #
    # ---------- # 
    def plot_spectrum(self, ax=None, switch_off=[], snr=False, **kwargs):
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

    def plot_cube(self, in_photons=True, switch_off=[], spec_prop={}, **kwargs):
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

    # ================= #
    #   Properties      #
    # ================= #
    @property
    def meta(self):
        """ """
        return self._in_meta | {"scene": self.scene.meta,
                                "spectrograph": self.spectrograph.meta,
                                "detector": self.detector.meta,
                                "extraction":self.extraction}
    @property
    def cube_shape(self):
        """ Shape of the generated 3d-cube (nlbda, ny, nx). """
        return (self.spectrograph.nlbda,
                self.spectrograph.ny,
                self.spectrograph.nx)

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
