"""WST-specific simulations, targets and exposure time calculator.

The Wide-field Spectroscopic Telescope (WST) is a 12 m ground-based telescope
with an integral field spectrograph (IFS) in two channels, ``"blue"``
(3700-6100 A) and ``"red"`` (6000-9600 A). The instrument description and the
site approximations follow the WST ETC ``pyetc`` (``pyetc/wst.py``,
``pyetc/etc.py``):

- seeing-limited Moffat PSF (`~slicersim.atmosphere.Atmosphere` for the seeing,
  `~slicersim.ground_spectrograph.GroundSlicerSpectrograph` for the optical part);
- Paranal sky emission and transmission (ESO SkyCalc tables computed by pyetc);
- CCD detector (`~slicersim.detector.CCDDetector`), NDIT exposures of DIT seconds.

The configurations are ``wst_ifs_blue.toml`` and ``wst_ifs_red.toml``. Use
`WSTTarget` for any spectrum, and `wst_etc` for the exposure time.

.. autosummary::

   WSTSimulation
   WSTTarget
   wst_etc
   make_wst_config_files
"""

import warnings
from pathlib import Path

import numpy as np

from .atmosphere import Atmosphere
from .detector import CCDDetector
from .iotools import PACKAGE_PATH, get_config
from .scene import Scene, get_scene
from .scene.background import Background
from .simulation import Simulation
from .ground_spectrograph import GroundSlicerSpectrograph
from .target import VirtualTarget
from .telescope import Telescope

__all__ = ["WST_INSTRUMENTS", "WSTSimulation", "WSTTarget", "wst_etc"]

#: Instrument configuration of each WST IFS channel
WST_INSTRUMENTS = {"blue": "wst_ifs_blue.toml",
                   "red": "wst_ifs_red.toml"}


class WSTSimulation(Simulation):
    """Simulation of a ground-based slicer IFS observation (WST).

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
    - a ``"box"`` extraction method (square aperture, pyetc ``square_fixed``);
    - a CCD exposure time calculator (`fetch_snr`).

    See `~slicersim.simulation.Simulation` for the parameters; build it with
    `from_config` (or `WSTTarget`).
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
        WSTSimulation
            The loaded simulation.

        Raises
        ------
        NotImplementedError
            If ``slicer`` is False.
        """
        if not slicer:
            raise NotImplementedError("only slicer spectrographs are implemented for WST.")

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
        are tabulated at the instrument resolution (as in pyetc). In
        [erg/s/cm^2/A] (``in_photons=False``), the sources are at the top of
        the atmosphere.
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
        the point source (or on the host aperture position), as pyetc
        ``square_fixed``.

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


class WSTTarget(VirtualTarget):
    """Any spectrum observed with a WST IFS channel, as a point source.

    Parameters
    ----------
    lbda : array_like
        Wavelengths in Angstrom. Should cover the channel.
    flux : array_like
        Flux in erg/s/cm^2/A (top of the atmosphere), or in arbitrary units
        when ``mag`` is given.
    mag : float, optional
        If given, ``flux`` is rescaled to this magnitude in ``band``.
        Default is None.
    band : str, optional
        Bandpass of ``mag`` (any sncosmo bandpass name). Default is
        ``"bessellb"``.
    channel : str, optional
        IFS channel, ``"blue"`` or ``"red"``. Default is ``"blue"``.
    instrument : str or dict, optional
        Instrument configuration, overriding ``channel``. Default is None.
    seeing, airmass, moon : optional
        Observing conditions (see `~slicersim.atmosphere.Atmosphere`); None
        keeps the configured values. Default is None.
    host : str or dict, optional
        Host galaxy (see `~slicersim.scene.get_scene`). Default is None.
    **kwargs
        Goes to `~slicersim.scene.get_scene` (e.g. ``redshift``,
        ``position``).

    Examples
    --------
    >>> lbda = np.linspace(3000, 11000, 4000)
    >>> target = WSTTarget(lbda, np.ones_like(lbda), mag=22, band="bessellv",
    ...                    channel="blue", seeing=0.7)  # doctest: +SKIP
    """
    def __init__(self, lbda, flux, mag=None, band="bessellb", channel="blue",
                 instrument=None, seeing=None, airmass=None, moon=None, host=None, **kwargs):
        """Initialize the target (see the class docstring)."""
        if instrument is None:
            instrument = WST_INSTRUMENTS[channel]
        scene = get_scene(source=[lbda, flux], mag=mag, band=band, host=host, **kwargs)
        simulation = WSTSimulation.from_config(get_config(scene=scene, instrument=instrument))
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


def wst_etc(lbda, flux, snr, channel="blue", mag=None, band="bessellb",
            lbda_range=None, per_resolution=False, dit=1800., ndit=None,
            seeing=None, airmass=None, moon=None, statistic=np.nanmean):
    """Exposure time needed to reach a given SNR with a WST IFS channel.

    Parameters
    ----------
    lbda, flux : array_like
        Input spectrum (see `WSTTarget`).
    snr : float
        Target SNR, per wavelength bin, or per resolution element if
        ``per_resolution``.
    channel : str, optional
        ``"blue"`` or ``"red"``. Default is ``"blue"``.
    mag, band : optional
        Normalization of the spectrum (see `WSTTarget`).
    lbda_range : list of float, optional
        (wmin, wmax) wavelength range [Angstrom] of the SNR; if None, the
        central 400 A of the channel. Default is None.
    per_resolution : bool, optional
        If True, ``snr`` is per resolution element. Default is False.
    dit : float, optional
        Fixed DIT [s] (if ``ndit`` is None). Default is 1800.
    ndit : int, optional
        Fixed NDIT, the DIT being then adjusted. Default is None.
    seeing, airmass, moon : optional
        Observing conditions. Default is None (configured values).
    statistic : callable, optional
        See `~slicersim.simulation.Simulation.get_band_flux`. Default is
        `numpy.nanmean`.

    Returns
    -------
    exptime : float
        Total exposure time (NDIT x DIT) in seconds.
    target : WSTTarget
        The target, set to the configuration reaching ``snr``.

    Examples
    --------
    >>> lbda = np.linspace(3000, 11000, 4000)
    >>> exptime, target = wst_etc(lbda, np.ones_like(lbda), snr=10, mag=22,
    ...                           band="bessellv")  # doctest: +SKIP
    """
    target = WSTTarget(lbda, flux, mag=mag, band=band, channel=channel,
                       seeing=seeing, airmass=airmass, moon=moon)
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


# =========================================================================== #
#  Maintenance: configuration files from pyetc                                #
# =========================================================================== #
#: WST IFS parameters, from pyetc/wst.py (e6e351e)
_PYETC_WST = {
    "telescope": dict(area=100.0, diameter=12.0, ifs_obscuration=0.18,
                      desc="Cass design", version="Busyweek Oct 2023"),
    "blue": dict(desc="Inspired from BlueMUSE throughput", version="0.1 10/02/2023",
                 iq_fwhm=0.10, iq_beta=2.50, spaxel_size=0.25,
                 dlbda=0.64, lbda1=3700, lbda2=6100, lsfpix=2.5, ron=3.0, dcurrent=3.0),
    "red": dict(desc="Inspired from MUSE throughput", version="0.1 10/02/2023",
                iq_fwhm=0.10, iq_beta=2.50, spaxel_size=0.25,
                dlbda=0.97, lbda1=6000, lbda2=9600, lsfpix=2.5, ron=3.0, dcurrent=3.0),
}

_TOML_TEMPLATE = '''# WST IFS {channel} channel
# Generated by slicersim.wst.make_wst_config_files from pyetc ({pyetc_version}):
# pyetc/wst.py ({desc}, version {version}) and pyetc/data/WST tables.

# ============= #
#  Atmosphere   #
# ============= #
[atmosphere]
site = "Paranal"
sky_table = "wst_ifs_{channel}_sky.ecsv"    # ESO SkyCalc (via pyetc): transmission & emission
seeing = 0.8                    # Seeing FWHM [arcsec] at 5000 A and zenith (seeing-induced PSF)
airmass = 1.0                   # Airmass (tabulated: 1.0, 1.2, 1.5, 2.0)
moon = "darksky"                # Sky condition: darksky, greysky, brightsky

# ============= #
#  Telescope    #
# ============= #
[telescope]
diameter_ext = {diameter:.1f}             # Primary mirror external diameter [m] (seeing model)
diameter_int = 0.0
surface = {surface:.1f}                 # Collecting area [m2] = {area} m2 x (1 - {obscuration} IFS obscuration)
temperature = 0                 # No thermal emission in the optical
emissivity = 0

# ============= #
# Spectrograph  #
# ============= #
[spectrograph]
name = "wst_ifs_{channel}"
spectral_range = [{lbda1}, {lbda2}]     # Wavelength domain [A]
dispersion_law = "wst_ifs_{channel}_dispersion.ecsv"  # Linear dispersion: {dlbda} A/px
dispersion_resolution = {lsfpix}       # LSF FWHM [px]
dispersion_scale = {dispersion_scale}            # = 2/dispersion_resolution, keeps {dlbda} A/px
spatial_scale = {spaxel_size}            # Spaxel size [arcsec]
spatial_shape = [25, 25]        # Simulated sub-field [spaxels] (point sources)
lbda_ref = 5000                 # Reference wavelength [A] (Gaussian fallback PSF)

[spectrograph.throughput]
throughput = "wst_ifs_{channel}_throughput.csv"  # Telescope + instrument + QE (pyetc TOT)

[spectrograph.optics]           # No thermal emission in the optical
temperature = [0]
emissivity = [0]
nelements = [1]
fratio = [[1, 1]]
geometry = ["circular"]
dispersed = [false]

[spectrograph.psf.spatial]
psf_profile = "moffat"          # Seeing-limited Moffat PSF
optical_fwhm = {iq_fwhm}             # Optical (telescope + instrument) FWHM [arcsec], added in quadrature to the seeing
moffat_beta = {iq_beta}              # Moffat shape parameter
moffat_truncation = 0           # 0: exact Moffat; 2: pyetc truncation (box of ~2 FWHM, renormalized)
psf_sigma_spectral = 0.3        # [arcsec] Gaussian fallback profile only
guiding_sigma = 0.0             # Included in the seeing

# ============= #
#  Detector     #
# ============= #
[detector]
shape = [6144, 6144]
QE = 1.0                        # Included in the throughput
gain = 1.0                      # Gain [ADU/e-]
saturation = 65535              # Saturation limit [ADU]
ron = {ron}                       # Read-out noise per read [e-]
dark = {dark:.6g}            # Dark current [e-/s/px] = {dcurrent} e-/h
pixel_size = 15.0               # Pixel physical size [um]
dit = 1800.0                    # Exposure time of a single exposure (DIT) [s]

# ============= #
#  Extraction   #
# ============= #
[extraction]
nramps = 1                      # Number of exposures (NDIT)
aperture_radius = 10            # Point-source extraction aperture radius [spx]
box_hsize = 2                   # Half size of the "box" extraction aperture [spx]
'''


def make_wst_config_files(pyetc_data, outdir=None, channels=("blue", "red")):
    """Write the WST IFS configuration files from the pyetc data (maintenance).

    For each channel, it writes ``wst_ifs_<channel>.toml`` and:

    - ``wst_ifs_<channel>_throughput.csv``: pyetc ``ifs_<channel>_noatm.fits``
      (``TOT``: telescope + instrument + QE) linearly resampled on the
      instrument wavelength grid, as pyetc does;
    - ``wst_ifs_<channel>_dispersion.ecsv``: linear dispersion law;
    - ``wst_ifs_<channel>_sky.ecsv``: transmission (per airmass) and sky
      emission (per moon condition and airmass, converted from
      ph/s/m^2/um/arcsec^2 to erg/s/cm^2/A/arcsec^2) of pyetc
      ``ifs_<channel>_<moon>_<airmass>.fits`` (ESO SkyCalc, Paranal).

    Parameters
    ----------
    pyetc_data : str or pathlib.Path
        pyetc ``pyetc/data/WST`` directory.
    outdir : str or pathlib.Path, optional
        Output directory. Default is the slicersim configuration directory.
    channels : tuple of str, optional
        Channels to process. Default is ``("blue", "red")``.

    Returns
    -------
    list of pathlib.Path
        The configuration (TOML) files.
    """
    from astropy.table import Table

    try:
        from pyetc import __version__ as pyetc_version
    except ImportError:
        pyetc_version = "data files"

    pyetc_data = Path(pyetc_data)
    outdir = PACKAGE_PATH.joinpath("config") if outdir is None else Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tel = _PYETC_WST["telescope"]
    hc = 1.9864459e-08  # [erg A]

    tomls = []
    for channel in channels:
        ins = _PYETC_WST[channel]
        npx = round((ins["lbda2"] - ins["lbda1"]) / ins["dlbda"]) + 1
        lbda = ins["lbda1"] + ins["dlbda"] * np.arange(npx)

        # throughput
        tab = Table.read(pyetc_data / f"ifs_{channel}_noatm.fits", unit_parse_strict="silent")
        tput = np.interp(lbda, np.asarray(tab["WAVE"]) * 10, np.asarray(tab["TOT"]))
        with open(outdir / f"wst_ifs_{channel}_throughput.csv", "w") as f:
            f.write(f"wavelength,wst_ifs_{channel}\n")
            f.writelines(f"{l:.3f},{t:.6f}\n" for l, t in zip(lbda, tput))

        # dispersion law
        nodes = np.linspace(ins["lbda1"], ins["lbda2"], 11)
        dtab = Table({"wavelength": nodes / 1e4, "offset": (nodes - ins["lbda1"]) / ins["dlbda"]})
        dtab["wavelength"].unit, dtab["offset"].unit = "micron", "pix"
        dtab["offset"].description = f"Linear dispersion law, {ins['dlbda']} A/px"
        dtab.meta["comments"] = [f"WST IFS {channel} (pyetc dlbda={ins['dlbda']} A/px)"]
        dtab.write(outdir / f"wst_ifs_{channel}_dispersion.ecsv", format="ascii.ecsv", overwrite=True)

        # sky: transmission and emission
        sky = None
        for airmass in (1.0, 1.2, 1.5, 2.0):
            for moon in ("darksky", "greysky", "brightsky"):
                tab = Table.read(pyetc_data / f"ifs_{channel}_{moon}_{airmass:.1f}.fits",
                                 unit_parse_strict="silent")
                wave = np.asarray(tab["lam"], dtype=float) * 10  # nm -> A
                if sky is None:
                    sky = Table({"wavelength": np.round(wave, 3)})
                    sky["wavelength"].unit = "Angstrom"
                    sky.meta["comments"] = [
                        f"WST IFS {channel}: ESO SkyCalc model (Paranal, pwv=3.5 mm) via pyetc",
                        f"convolved by a Gaussian LSF of FWHM {ins['lsfpix']} px, vacuum wavelengths"]
                if f"transmission_{airmass:.1f}" not in sky.colnames:
                    sky[f"transmission_{airmass:.1f}"] = np.round(np.asarray(tab["trans"], dtype=float), 6)
                name = f"emission_{moon}_{airmass:.1f}"
                emission = np.asarray(tab["flux"], dtype=float) * (hc / wave) * 1e-8
                sky[name] = [float(f"{e:.5e}") for e in emission]  # 5 significant digits
                sky[name].unit = "erg / (s cm2 Angstrom arcsec2)"
        sky.write(outdir / f"wst_ifs_{channel}_sky.ecsv", format="ascii.ecsv", overwrite=True)

        toml = outdir / f"wst_ifs_{channel}.toml"
        toml.write_text(_TOML_TEMPLATE.format(
            channel=channel, pyetc_version=pyetc_version, desc=ins["desc"], version=ins["version"],
            diameter=tel["diameter"], area=tel["area"], obscuration=tel["ifs_obscuration"],
            surface=tel["area"] * (1 - tel["ifs_obscuration"]),
            lbda1=ins["lbda1"], lbda2=ins["lbda2"], dlbda=ins["dlbda"], lsfpix=ins["lsfpix"],
            dispersion_scale=2 / ins["lsfpix"], spaxel_size=ins["spaxel_size"],
            iq_fwhm=ins["iq_fwhm"], iq_beta=ins["iq_beta"],
            ron=ins["ron"], dark=ins["dcurrent"] / 3600, dcurrent=ins["dcurrent"]))
        tomls.append(toml)

    return tomls
