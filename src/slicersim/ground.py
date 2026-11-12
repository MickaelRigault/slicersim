"""Ground-based slicer IFS: simulations, targets and exposure time calculator.

A ground-based observation differs from a space one by the atmosphere
(`~slicersim.atmosphere.Atmosphere`): seeing-limited PSF, atmospheric
transmission and sky emission. This module combines it with the
`~slicersim.ground_spectrograph.GroundSlicerSpectrograph` (Moffat PSF,
1 pixel per voxel) and the `~slicersim.detector.CCDDetector` (NDIT exposures
of DIT seconds).

The instrument configuration needs an ``[atmosphere]`` section, see
`~slicersim.atmosphere.Atmosphere.from_config`, in addition to the usual
``[telescope]``, ``[spectrograph]``, ``[detector]`` and ``[extraction]``
sections. Its data files (throughput, dispersion law, sky table) are looked
for next to the configuration file (see `~slicersim.iotools.read_config`),
so instrument packages can ship their own configurations. Use `GroundTarget`
for any spectrum and `ground_etc` for the exposure time.

.. autosummary::

   GroundSimulation
   GroundTarget
   ground_etc
"""

import warnings

import numpy as np

from .atmosphere import Atmosphere
from .detector import CCDDetector
from .ground_spectrograph import GroundSlicerSpectrograph
from .iotools import get_config
from .scene import Scene, get_scene
from .scene.background import Background
from .simulation import Simulation
from .target import VirtualTarget
from .telescope import Telescope

__all__ = ["GroundSimulation", "GroundTarget", "ground_etc"]


class GroundSimulation(Simulation):
    """Simulation of a ground-based slicer IFS observation.

    It differs from `~slicersim.simulation.Simulation` by:

    - an `~slicersim.atmosphere.Atmosphere` (``[atmosphere]`` configuration
      section), shared by the spectrograph (seeing, transmission) and the
      scene, whose background is the sky emission (any configured background
      is replaced). The observing conditions are mutable parameters:
      ``update(seeing=0.6, airmass=1.2, moon="greysky")``;
    - a `~slicersim.ground_spectrograph.GroundSlicerSpectrograph` and a
      `~slicersim.detector.CCDDetector`; ``ndit`` is an alias of ``nramps``
      and ``dit`` the exposure time;
    - the LSF applied to the sources *before* the atmospheric transmission
      (tabulated at the instrument resolution), and no LSF applied to the
      sky emission (idem);
    - a ``"box"`` extraction method (square aperture);
    - a CCD exposure time calculator (`fetch_snr`).

    See `~slicersim.simulation.Simulation` for the parameters; build it with
    `from_config` (or `GroundTarget`).
    """
    VARIANCE_SOURCES = ["dark", "ron", "pointsource", "background", "host"]

    @classmethod
    def from_config(cls, config, slicer=True):
        """Initialize the simulation from a nested configuration dictionary.

        Parameters
        ----------
        config : dict
            Top level configuration with the ``"atmosphere"``,
            ``"telescope"``, ``"spectrograph"``, ``"detector"`` and
            ``"extraction"`` entries, and optionally ``"scene"``.
        slicer : bool, optional
            Must be True (slicer spectrograph). Default is True.

        Returns
        -------
        GroundSimulation
            The loaded simulation.

        Raises
        ------
        NotImplementedError
            If ``slicer`` is False.
        """
        if not slicer:
            raise NotImplementedError("only slicer spectrographs are implemented for ground-based simulations.")

        atmosphere = Atmosphere.from_config(config["atmosphere"])
        telescope = Telescope.from_config(config["telescope"])
        spectrograph = GroundSlicerSpectrograph.from_config(config["spectrograph"],
                                                            telescope=telescope,
                                                            atmosphere=atmosphere)
        detector = CCDDetector.from_config(config["detector"], lbda=spectrograph.lbda,
                                           thermaloptics=spectrograph.optics)
        scene = Scene.from_config(config["scene"], lbda=spectrograph.lbda) if "scene" in config else None

        this = cls(scene=scene, telescope=telescope, spectrograph=spectrograph,
                   detector=detector, extraction=dict(config["extraction"]), meta=config)
        this._attach_sky()
        return this

    def _attach_sky(self):
        """Use the atmospheric sky emission as the scene background."""
        if self.scene is not None:
            self.scene._background = Background(self.atmosphere.get_emission,
                                                meta={"name": "sky"})
        self._reset_cached()

    def reset(self, which="*"):
        """Reset simulation elements to their initial configuration.

        Parameters
        ----------
        which : str or list of str, optional
            Elements to reset, among ``"scene"``, ``"telescope"``,
            ``"spectrograph"`` (atmosphere included), ``"detector"`` and
            ``"extraction"``; ``"*"`` for all. Default is ``"*"``.
        """
        which = self._elements if which == "*" else np.atleast_1d(which).tolist()
        fresh = self.__class__.from_config(self._in_meta)
        if "spectrograph" in which or "telescope" in which:
            self.telescope, self.spectrograph = fresh.telescope, fresh.spectrograph
        for element in ("scene", "detector", "extraction"):
            if element in which:
                setattr(self, element, getattr(fresh, element))
        self._attach_sky()

    # ---------- observing conditions ---------- #
    @property
    def atmosphere(self):
        """The `~slicersim.atmosphere.Atmosphere` of the observation."""
        return self.spectrograph.atmosphere

    @property
    def observing_conditions(self):
        """Current observing conditions (``seeing``, ``airmass``, ``moon``)."""
        return {k: self.atmosphere.meta[k] for k in self.atmosphere.mutable_parameters}

    def update(self, reset_others=False, **kwargs):
        """Update any mutable parameter (see `~slicersim.simulation.Simulation.update`).

        ``seeing``, ``airmass`` and ``moon`` change the observing conditions,
        ``ndit`` is an alias of ``nramps``.
        """
        if "ndit" in kwargs:
            ndit = kwargs.pop("ndit")
            kwargs["nramps"] = None if ndit is None else int(ndit)
        super().update(reset_others=reset_others, **kwargs)

    def get_parameter(self, which=None, default=None, as_dict=True):
        """Get a parameter (see `~slicersim.simulation.Simulation.get_parameter`).

        Also accepts ``seeing``, ``airmass``, ``moon``, ``ndit``, ``dit`` and
        the Moffat PSF parameters (``optical_fwhm``, ``moffat_beta``,
        ``moffat_truncation``).
        """
        if which in self.atmosphere.mutable_parameters:
            value = self.atmosphere.meta[which]
            return {which: value} if as_dict else value
        if which in self.spectrograph.PSF_KEYS:
            value = self.spectrograph.spatial_psf[which]
            return {which: value} if as_dict else value
        if which == "ndit":
            which = "nramps"
        return super().get_parameter(which=which, default=default, as_dict=as_dict)

    # ---------- scene projection ---------- #
    def get_projected_scene(self, in_photons=True, switch_off=[], psf_profile="default",
                            as_oversampled=False, oversampling=None, apply_lsf=True,
                            cached=False, **kwargs):
        """Project the scene through the spectrograph and get the flux cube.

        See `~slicersim.simulation.Simulation.get_projected_scene`. The
        astrophysical sources are convolved by the LSF *before* applying the
        atmospheric transmission, and the sky emission is not convolved: both
        are tabulated at the instrument resolution. In [erg/s/cm^2/A]
        (``in_photons=False``), the sources are at the top of the atmosphere.
        """
        if cached and getattr(self, "_cube_cached", None) is not None:
            return self._cube_cached

        switch_off = list(np.atleast_1d(switch_off))
        self.spectrograph._atmosphere_off = True
        try:
            cube = super().get_projected_scene(in_photons=in_photons,
                                               switch_off=switch_off + ["background"],
                                               psf_profile=psf_profile, as_oversampled=as_oversampled,
                                               oversampling=oversampling, apply_lsf=apply_lsf,
                                               cached=False, **kwargs)
        finally:
            self.spectrograph._atmosphere_off = False

        if in_photons:
            cube = cube * self.spectrograph.get_atmospheric_transmission()[:, None, None]

        if "background" not in switch_off and self.scene.has_element("background"):
            sky = self.scene.background.get_spectrum(self.spectrograph.lbda)[1]
            sky_cube = self.spectrograph.generate_background(
                sky, oversampling=oversampling if as_oversampled else None, apply_lsf=False)
            if not in_photons:
                sky_cube /= self.spectrograph.flambda2photon[:, None, None]
            cube = cube + sky_cube

        if cached:
            self._cube_cached = cube
        return cube

    # ---------- extraction ---------- #
    def get_spectrum(self, switch_off=[], incl_error=False, of_host=False, method=None,
                     psf_profile="default", apply_lsf=True, cached=False, box_hsize=None):
        """Get the extracted spectrum and its variance.

        See `~slicersim.simulation.Simulation.get_spectrum`, plus
        ``method="box"``: plain summation of the signal and variance cubes
        within a square aperture of ``(2*box_hsize+1)²`` spaxels centered on
        the point source (or on the host aperture position).

        Parameters
        ----------
        box_hsize : int or None, optional
            Half size of the box in spaxels; if None, the ``box_hsize``
            extraction parameter (default 2). Default is None.

        Returns
        -------
        lbda, signal, variance : numpy.ndarray
            Wavelength [Angstrom], signal [ADU] and variance [ADU^2].
        """
        if method is None:
            method = self.extraction.get("method", "default")
        if method != "box":
            return super().get_spectrum(switch_off=switch_off, incl_error=incl_error,
                                        of_host=of_host, method=method, psf_profile=psf_profile,
                                        apply_lsf=apply_lsf, cached=cached)

        if box_hsize is None:
            box_hsize = int(self.extraction.get("box_hsize", 2))
        which = "host" if (of_host or not self.scene.has_element("pointsource")) else "pointsource"
        position = (self.scene.pointsource_position if which == "pointsource"
                    else self.get_aperture()[0])

        _, var_cube = self.get_cube(switch_off=switch_off, per_ramp=True,
                                    psf_profile=psf_profile, apply_lsf=apply_lsf, cached=cached)
        if which not in switch_off:
            others = [k for k in ["pointsource", "host", "background", "thermal"] if k != which]
            cube = self.get_projected_scene(in_photons=True, switch_off=others,
                                            psf_profile=psf_profile, apply_lsf=apply_lsf)
            cube *= np.reshape(self.detector.photonflux_to_adu(self.spectrograph.lbda), (-1, 1, 1))
        else:
            cube = np.zeros_like(var_cube)

        box = self.spectrograph.get_box_aperture(position=position, hsize=box_hsize)
        nexp = self.extraction["nramps"]
        signal = cube[:, box].sum(axis=-1) * nexp
        variance = var_cube[:, box].sum(axis=-1) * nexp
        if incl_error:
            signal = signal + np.random.normal(loc=0, scale=np.sqrt(variance))
        return self.spectrograph.lbda, signal, variance

    # ---------- exposure time ---------- #
    def fetch_snr(self, target_snr, lbda_range=[4000, 6800], frame="rest",
                  statistic=np.nanmean, dit=None, ndit=None, dit_range=(1., 3600.),
                  reset_param=True, **kwargs):
        """Find the (DIT, NDIT) configuration reaching a target SNR.

        The SNR is the band SNR per wavelength bin (see
        `~slicersim.simulation.Simulation.get_band_snr`):

        - if ``ndit`` is None: smallest NDIT reaching the SNR with exposures
          of ``dit`` seconds (default: current DIT);
        - otherwise: DIT (within ``dit_range``) reaching the SNR with ``ndit``
          exposures. A saturated configuration is handled as too long.

        Parameters
        ----------
        target_snr : float
            Target SNR per wavelength bin.
        lbda_range : list of float, optional
            (wmin, wmax) wavelength range [Angstrom]. Default is [4000, 6800].
        frame : str, optional
            Frame of ``lbda_range`` (``"obs"`` or ``"rest"``). Default is
            ``"rest"``.
        statistic : callable, optional
            See `~slicersim.simulation.Simulation.get_band_flux`. Default is
            `numpy.nanmean`.
        dit, ndit : float, int or None, optional
            Fixed DIT [s] or NDIT, see above. Default is None.
        dit_range : tuple of float, optional
            Allowed DIT range [s]. Default is (1, 3600).
        reset_param : bool, optional
            If True, the simulation is set back to its input DIT and NDIT.
            Default is True.
        **kwargs
            Ignored (MACC-specific parameters of
            `~slicersim.simulation.Simulation.fetch_snr`).

        Returns
        -------
        config : dict
            ``{"dit": float, "nramps": int}`` (to be given to `update`).
        snr : float
            Reached SNR.
        observing_time : float
            Total exposure time [s].

        Warns
        -----
        UserWarning
            If the SNR cannot be reached (saturation or ``dit_range``).
        """
        from scipy.optimize import brentq

        input_config = {"dit": self.detector.dit, "nramps": self.extraction["nramps"]}
        snr_prop = dict(lbda_range=lbda_range, frame=frame, statistic=statistic, cached=True)

        if ndit is None:
            dit = self.detector.dit if dit is None else float(dit)
            self.update(dit=dit, nramps=1)
            snr1 = self.get_band_snr(**snr_prop)
            if not np.isfinite(snr1):
                warnings.warn(f"saturation with DIT={dit} s: reduce the DIT.")
            self.update(nramps=max(1, int(np.ceil((target_snr / snr1) ** 2))))
        else:
            self.update(nramps=int(ndit))

            def snr_offset(logdit):
                self.update(dit=10 ** logdit)
                return self.get_band_snr(**snr_prop) - target_snr

            lo, hi = np.log10(dit_range)
            if snr_offset(lo) > 0:
                dit = dit_range[0]
            elif np.isfinite(offset := snr_offset(hi)) and offset < 0:
                dit = dit_range[1]
            else:
                dit = 10 ** brentq(snr_offset, lo, hi, xtol=1e-4)
            self.update(dit=dit)

        snr = self.get_band_snr(**snr_prop)
        if snr < 0.99 * target_snr:
            warnings.warn(f"SNR={target_snr} not reached (SNR={snr:.2f}, DIT={self.detector.dit:.0f} s, "
                          f"NDIT={self.extraction['nramps']}): saturation or DIT range.")
        config = {"dit": self.detector.dit, "nramps": self.extraction["nramps"]}
        total = self.observing_time
        if reset_param:
            self.update(**input_config)
        self._reset_cached()
        return config, snr, total


class GroundTarget(VirtualTarget):
    """Any spectrum observed with a ground-based slicer IFS, as a point source.

    Parameters
    ----------
    lbda : array_like
        Wavelengths in Angstrom. Should cover the instrument domain.
    flux : array_like
        Flux in erg/s/cm^2/A (top of the atmosphere), or in arbitrary units
        when ``mag`` is given.
    instrument : str, list or dict
        Instrument configuration (see `~slicersim.iotools.get_config`), e.g.
        the path to a TOML file.
    mag : float, optional
        If given, ``flux`` is rescaled to this magnitude in ``band``.
        Default is None.
    band : str, optional
        Bandpass of ``mag`` (any sncosmo bandpass name). Default is
        ``"bessellb"``.
    seeing, airmass, moon : optional
        Observing conditions (see `~slicersim.atmosphere.Atmosphere`); None
        keeps the configured values. Default is None.
    host : str or dict, optional
        Host galaxy (see `~slicersim.scene.get_scene`). Default is None.
    **kwargs
        Goes to `~slicersim.scene.get_scene` (e.g. ``redshift``,
        ``position``).
    """
    _SIMULATION = GroundSimulation

    def __init__(self, lbda, flux, instrument, mag=None, band="bessellb",
                 seeing=None, airmass=None, moon=None, host=None, **kwargs):
        """Initialize the target (see the class docstring)."""
        scene = get_scene(source=[lbda, flux], mag=mag, band=band, host=host, **kwargs)
        simulation = self._SIMULATION.from_config(get_config(scene=scene, instrument=instrument))
        simulation.update(seeing=seeing, airmass=airmass, moon=moon)
        super().__init__(simulation=simulation)

    def get_readout_config(self):
        """Get the current exposure configuration.

        Returns
        -------
        dict
            ``{"dit": float, "ndit": int}``.
        """
        return {"dit": self.simulation.detector.dit, "ndit": self.simulation.extraction["nramps"]}


def ground_etc(target, snr, lbda_range=None, per_resolution=False, dit=1800., ndit=None,
               statistic=np.nanmean):
    """Set a ground-based target to the exposure reaching a given SNR.

    Parameters
    ----------
    target : GroundTarget
        The target (instrument, spectrum and observing conditions). It is
        modified in place.
    snr : float
        Target SNR, per wavelength bin, or per resolution element if
        ``per_resolution``.
    lbda_range : list of float, optional
        (wmin, wmax) wavelength range [Angstrom] of the SNR; if None, the
        central 400 A of the instrument domain. Default is None.
    per_resolution : bool, optional
        If True, ``snr`` is per resolution element. Default is False.
    dit : float, optional
        Fixed DIT [s] (if ``ndit`` is None). Default is 1800.
    ndit : int, optional
        Fixed NDIT, the DIT being then adjusted. Default is None.
    statistic : callable, optional
        See `~slicersim.simulation.Simulation.get_band_flux`. Default is
        `numpy.nanmean`.

    Returns
    -------
    exptime : float
        Total exposure time (NDIT x DIT) in seconds.
    target : GroundTarget
        The target, set to the configuration reaching ``snr``.
    """
    sim = target.simulation
    if lbda_range is None:
        lmid = np.mean(sim.spectrograph.lbda[[0, -1]])
        lbda_range = [lmid - 200, lmid + 200]
    if per_resolution:
        snr = snr / np.sqrt(sim.spectrograph.dispersion_resolution)
    config, _, exptime = sim.fetch_snr(snr, lbda_range=lbda_range, frame="obs",
                                       statistic=statistic, dit=dit, ndit=ndit)
    sim.update(**config)
    return exptime, target
