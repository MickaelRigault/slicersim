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

from .iotools import get_config
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

    #: Mutable parameters (dict by origin)
    simu_mutable_parameters = {
        "scene": Scene.mutable_parameters,                # scene parameters
        "spectrograph": Spectrograph.mutable_parameters,  # spectrograph parameters
        "detector": Detector.mutable_parameters,          # detector parameters
        "extraction": (                                   # extraction parameters
            'xdisp_width', 'xdisp_width_insigma',
            'aperture_radius', 'aperture_radius_insigma',
            "nramp",
        )}

    #: Mutable parameters (concatenated list)
    mutable_parameters = (simu_mutable_parameters["scene"] +
                          simu_mutable_parameters["spectrograph"] +
                          simu_mutable_parameters["detector"] +
                          simu_mutable_parameters["extraction"])

    def __init__(self,
                 scene=None,
                 spectrograph=None,
                 detector=None,
                 extraction={},
                 meta={}, **kwargs):
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
        self.meta = {**meta, **kwargs}    #: Meta-parameters

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

    @classmethod
    def from_config(cls, config=None):
        """
        Initiate simulation from config nested dictionary.

        :param dict config: top level configuration containing
                            configurations for scene, spectrograph, detector
                            and extraction parameters (by default:
                            use default config from
                            :func:`mlaperf.iotools.get_config`)
        :return: scene instance
        """

        if not config:
            config = get_config()  # Use default config

        # First initialize spectrograph to set wavelengths, then other elements
        # using spectrograph wavelengths.

        # Initialize the spectrograph from config
        spectrograph = Spectrograph.from_config(config["spectrograph"])

        # Initialize the scene from config (wavelength from spectrograph)
        scene = Scene.from_config(config["scene"], spectrograph.lbda)

        # Initialize the detector from config
        detector = Detector.from_config(config["detector"], spectrograph.lbda)

        # Initialize extraction parameters from config
        extraction = config["extraction"]

        return cls(scene=scene,
                   spectrograph=spectrograph,
                   detector=detector,
                   extraction=extraction,
                   meta=config)  # Store config dict in meta

    @property
    def cube_shape(self):
        """
        Shape of the generated 3d-cube (nlbda, ny, nx).
        """

        return (self.spectrograph.nlbda,
                self.spectrograph.ny,
                self.spectrograph.nx)

    @property
    def observing_time(self):
        """
        Total observing time, i.e. `exptime * nramp`.
        """

        return self.detector.exposure_time * self.extraction["nramp"]

    def update(self, **kwargs):
        """
        Update any mutable parameter of the simulation.

        This will guess the origin of the parameters from
        :attr:`simu_mutable_parameters` origin-indexed dict.

        :param dict kwargs: parameter names and values
        """

        updates_scene = {}
        updates_detector = {}
        updates_spectrograph = {}
        updates_extraction = {}

        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue
            if v is None:       # Nothing to do
                continue
            if k in self.simu_mutable_parameters["scene"]:
                updates_scene[k] = v
            elif k in self.simu_mutable_parameters["detector"]:
                updates_detector[k] = v
            elif k in self.simu_mutable_parameters["spectrograph"]:
                updates_spectrograph[k] = v
            elif k in self.simu_mutable_parameters["extraction"]:
                updates_extraction[k] = v
            else:
                raise ValueError(f"Unknown simulation parameter {k!r}.")

        # Update spectrograph 1st because it sets the wavelengths
        self.spectrograph.update(**updates_spectrograph)

        if ({'spectral_range', 'spectral_resolution'} &
            set(updates_spectrograph)):  # Updates of spectral quantities
            updates_scene["lbda"] = self.spectrograph.lbda
            updates_detector["lbda"] = self.spectrograph.lbda

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
        """
        Reset the simulation component at their initial config value.

        Careful, this creates a new component element and erases the current one.

        :param str or list which: component to reset.
            could be '*'/'all' or a key of
            :attr:`Simulation.simu_mutable_parameters`.
        """

        if which not in ['all', '*'] + list(self.simu_mutable_parameters.keys()):
            raise ValueError(f"cannot parse the '{which}' reset.")

        if which in ['all', '*']:
            which = self.simu_mutable_parameters.keys()  # Reset all components

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

    def get_parameter(self, which=None, default=None, as_dict=True):
        """
        Get the value of a parameter of the simulation.

        Should be an attribute of one of the component of the simulation,
        e.g. `gain` from `Simulation.detector.gain`.

        :param str which: parameter name (or list of)
        :param default: default value
        :param bool as_dict: return `{which: value}` rather than `value`
        :return: parameter value
        """

        if which is None:       # Get all of them
            which = [ item
                      for sublist in self.simu_mutable_parameters.values()
                      for item in sublist ]

        if np.ndim(which):      # If a list, loop over elements
            return { param: self.get_parameter(param, as_dict=False)
                     for param in which }

        # Extraction parameter
        if which in self.extraction:
            return self.extraction[which]

        # Short cuts
        if which == "ngroup":
            return self.detector.nmd[0]

        # Otherwise, look at individual components
        components = ["scene", "spectrograph", "detector"]
        for comp in components:
            instance = getattr(self, comp)

            if '.' in which:    # Composed key: key1.key2
                k1, k2 = which.split('.')
                if hasattr(instance, k1):
                    instance = getattr(instance, k1)
                    which = k2

            if hasattr(instance, which):            # self.which
                return getattr(instance, which)

            if which in getattr(instance, "meta"):  # self.meta['which']
                return getattr(instance, "meta")[which]

            if comp == "scene": # for scene: self.meta['target']['which']
                meta_target = getattr(instance, "meta")["target"]
                if which in meta_target:
                    return meta_target[which]

        return {which: default} if as_dict else default

    def project_scene(self, in_photons=True, switch_off=[]):
        """
        Project the scene through spectrograph and get flux cube [ph or flambda].

        :param bool in_photons: cube in photon/s (default: flambda)
        :param list switch_off: list of discarded scene components
                                (target, host, background) + thermal
        :return: (nlbda, ny, nx) cube
        """

        cube = np.zeros(self.cube_shape)  # (nlbda, ny, nx)

        # spectra (3, nlbda):
        # * point source spectrum [erg/s/cm²/Å]
        # * host (not yet implemented)
        # * background spectrum [erg/s/cm²/Å/arcsec²]
        target, host, background = self.scene.get_component_spectra(fillna=0)

        # Fill the cube with scene components in photons/s/spx
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

        :param list switch_off: list of discarded scene components
                                (target, host, background) + thermal
        :return: (nlbda, ny, nx) cube signal and variance [ADU]
        """

        cube = self.project_scene(in_photons=True,
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

        :param list switch_off: list of discarded scene components
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
            target_phflux = self.scene.target * self.spectrograph.flambda2photon
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

    def get_band_signal(self, lbda_range, frame="obs", statistic=np.nanmean,
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

        if frame == "rest" and self.scene.redshift is not None:
            lbda_range = lbda_range * (1 + self.scene.redshift)

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

        See :meth:`get_band_signal`.
        """

        signal, variance = self.get_band_signal(lbda_range, frame,
                                                statistic=statistic, **kwargs)

        return signal / variance**0.5


        # df = pandas.concat(subdfs).reset_index(drop=True)
        # df.attrs = self.meta    # Add meta-data

        # return df

    def estimate_variance_contribution(self,
                                       lbda_range, frame="rest",
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
        signal, variance = self.get_band_signal(lbda_range, **prop)

        # Detector components
        # Dark
        current_dark = self.detector.dark
        self.detector.update(dark=0)             # Switch off dark
        _, variance_nodark = self.get_band_signal(lbda_range, **prop)
        self.detector.update(dark=current_dark)  # Switch it back
        dark_contrib = variance - variance_nodark

        # RoN
        current_ron = self.detector.ron
        self.detector.update(ron=0)
        _, variance_noron = self.get_band_signal(lbda_range, **prop)
        self.detector.update(ron=current_ron)
        ron_contrib = variance - variance_noron

        # Scene components
        # Background
        _, variance_nobkgd = self.get_band_signal(
            lbda_range,
            switch_off=["background"], **prop)
        bkgd_contrib = variance - variance_nobkgd

        # Target
        _, variance_notarget = self.get_band_signal(
            lbda_range,
            switch_off=["target"], **prop)
        target_contrib = variance - variance_notarget

        # Thermal
        _, variance_nothermal = self.get_band_signal(
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

    def plot_spectrum(self, ax=None, switch_off=[], snr=False, **kwargs):
        """
        Plot the detected spectrum.

        :param matplotlib.Axes ax: axes
        :param list switch_off: list of discarded scene components
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
        Display the cube generated by :meth:`project_scene`.

        The figure has two panels:

        - left: total spectrum (cube summed over spaxels)
        - right: white image (cube summed over wavelengthes)

        :param bool in_photons: cube in photon/s (default: flambda)
        :param list switch_off: list of discarded scene components
                                (target, host, background) + thermal
        :param dict spec_prop: spectrum plot options
        :param kwargs: cube `imshow` options
        :return: 2-axis figure
        """

        cube = self.project_scene(in_photons=in_photons,
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
