"""Detector module.

This module simulates detector effects such as readout noise and dark current,
and Multiple Accumulated Sampling (MACC) readout modes.
"""

__author__ = "Mickael Rigault <m.rigault@ip2i.in2p3.fr>, Yannick Copin <y.copin@ip2i.in2p3.fr>"

import warnings
from copy import deepcopy

import numpy as np
from astropy import units
from astropy.utils.exceptions import AstropyWarning

from .utils import complete_dims


class SaturationWarning(AstropyWarning):
    """Warning issued when detector pixels saturate."""

class Detector:
    """Near-infrared detector read out in MACC (up-the-ramp) mode.

    This class converts incident photon fluxes into integrated detector
    signals and estimates the associated variance, accounting for the
    quantum efficiency, gain, read-out noise, dark current, ROIC glow,
    thermal emission from the surrounding optics and saturation.

    Parameters
    ----------
    tframe : float
        Frame (single read) time in seconds.
    dark : float
        Detector dark current in e-/s/pixel.
    ron : float
        Read-out noise per frame in e- (RMS).
    qe : float or callable
        Quantum efficiency in e-/ph, either as a constant or as a function of
        wavelength (in Angstrom).
    pixel_size : float
        Physical pixel size in micrometers.
    gain : float, optional
        Detector gain in ADU/e-. Default is 1.
    saturation : int, optional
        Saturation level in ADU. Default is 65535.
    roic_glow : float, optional
        ROIC glow in e-/pixel/frame. Default is 0.
    nmd : tuple of int, optional
        MACC read-out mode ``(ngroups, nframes_per_group, ndrops)``.
        Default is ``(64, 8, 0)``.
    lbda_range : array_like or None, optional
        Wavelength range ``[lmin, lmax]`` seen by the detector, in Angstrom.
        If None, the first and last values of `lbda` are used.
        Default is None.
    min_group : int, optional
        Minimum number of groups per ramp. Default is 2.
    max_group : int, optional
        Maximum number of groups per ramp. Default is 64.
    lbda : float or array_like, optional
        Wavelength(s) in Angstrom at which chromatic quantities (e.g. the QE)
        are evaluated. Default is 10000.
    variance_model : str, optional
        Variance model used for noise estimation, ``"rauscher07"`` or
        ``"kubik16"``. Default is ``"rauscher07"``.
    thermaloptics : `~slicersim.thermal.ThermalOptics` or None, optional
        Optics whose thermal emission reaches the detector (thermal dark).
        Default is None (no thermal dark).
    ron_floor : float, optional
        Read-out noise floor in e-, added in quadrature to the effective
        read-out noise of the ramp. Default is 0.
    shape : list of int, optional
        Detector shape in pixels ``[ny, nx]``. Default is ``[4096, 4096]``.
    meta : dict, optional
        Additional metadata. The detector parameters above are stored in it.
        Default is an empty dict.

    Attributes
    ----------
    mutable_parameters : list of str
        Parameters that can be changed after initialization with `update`.
    """
    #: Mutable parameters (list)
    mutable_parameters = ['ngroup', "nframe_per_group",
                          'nmd', 'tframe', "max_group",
                          'ron', 'gain', 'qe', 'dark',
                          'saturation', 'variance_model',
                          "ron_floor", "roic_glow"]

    def __init__(self, tframe, dark, ron, qe, pixel_size,
                     gain=1, saturation=65_535,
                     roic_glow=0,
                     nmd=(64, 8, 0), lbda_range=None,
                     min_group=2, max_group=64,
                     lbda=10_000.,
                     variance_model="rauscher07",
                     thermaloptics=None,
                     ron_floor=0,
                     shape=[4096, 4096],
                     meta={} ):
        """Initialize the Detector (see the class docstring for parameters)."""
        # from other components
        self._lbda = lbda
        self._thermaloptics = thermaloptics

        # internal structure is meta-based.
        # detector
        meta["shape"] = shape
        meta["tframe"] = tframe
        meta["dark"] = dark
        meta["roic_glow"] = roic_glow # e-/read
        meta["ron"] = ron
        meta["ron_floor"] = ron_floor
        meta["qe"] = qe
        meta["pixel_size"] = pixel_size
        meta["gain"] = gain
        meta["nmd"] = nmd

        # modelling
        meta["variance_model"] = variance_model

        # extra
        meta["max_group"] = max_group
        meta["min_group"] = min_group
        meta["saturation"] = saturation
        meta["lbda_range"] = lbda_range

        self._meta_in = deepcopy(meta)                  #: Meta-parameters as given
        self._meta = deepcopy(meta)                     #: current Meta-parameters

    @classmethod
    def from_config(cls, config, lbda=10_000., thermaloptics=None):
        """Initialize the detector from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing detector parameters
            (e.g. the ``[detector]`` section of a configuration file).
            It must contain:

            - ``"nmd"`` or ``"readout_mode"``: MACC read-out mode,
            - ``"ron"`` or ``"readout_noise"``: read-out noise per frame [e-],
            - ``"dark"``: dark current [e-/s/pixel],
            - ``"tframe"``: frame time [s],
            - ``"QE"``: quantum efficiency, either a float or the name of an
              ECSV file with ``wavelength`` and ``qe`` columns,
            - ``"pixel_size"``: pixel size [micrometer].

            Optional keys are ``"shape"``, ``"max_group"``, ``"min_group"``,
            ``"roic_glow"`` [e-/pixel/frame], ``"gain"`` [ADU/e-] and
            ``"variance_model"``. Other keys are ignored.
        lbda : float or array_like, optional
            Wavelength(s) in Angstrom. Default is 10000.
        thermaloptics : `~slicersim.thermal.ThermalOptics` or None, optional
            Optics whose thermal emission reaches the detector.
            Default is None.

        Returns
        -------
        Detector
            The detector instance.

        Warns
        -----
        UserWarning
            If both ``"nmd"`` and ``"readout_mode"`` are given (``"readout_mode"``
            is then ignored and removed from ``config``), or if neither is given.
        """
        from . import iotools

        dict_init = {}

        dict_init["shape"] = config.get("shape", [4096, 4096])

        #
        # READOUT
        #
        ## readout_mode
        if "nmd" in config and "readout_mode" in config:
            warnings.warn("readout_mode ignore as nmd found in config")
            _ = config.pop("readout_mode")

        elif "nmd" not in config and "readout_mode" not in config:
            warnings.warn("not readout mode found (neither 'readout_mode' nor 'nmd' in config. This will create problem.")

        dict_init["nmd"] = config.get("nmd", config.get("readout_mode", None))
        ### max group in readout_mode allowed
        if "max_group" in config:
            dict_init["max_group"] = int(config.get("max_group"))     #: maximum number of group per ramp

        if "min_group" in config:
            dict_init["min_group"] = int(config.get("min_group"))     #: maximum number of group per ramp

        ## readout_noise aka, ron
        dict_init["ron"] = float(config.get("ron", config.get("readout_noise")) ) #: Read-Out Noise per frame [e-]

        ## dark
        dict_init["dark"] = float(config.get("dark"))                 #: Dark current [e-/s]

        ## roic gow
        dict_init["roic_glow"] = float(config.get("roic_glow", 0))    #: Roic glow [e-/frame]

        ## gain
        dict_init["gain"] = float(config.get("gain", 1))              #: Gain [ADU/e-]

        ## readout frame time
        dict_init["tframe"] = float(config.get("tframe"))             #: Frame time [s]

        #
        # Detector properties
        #
        ## pixel size
        dict_init["pixel_size"] = float(config.get("pixel_size"))            #: Pixel size [µm]

        ## quantum efficiency | as float or as func
        qe_ = config.get("QE")
        try:
            qe = float(qe_)          # Constant QE
        except ValueError:                         # QE is a filename
            tab = iotools.read_ecsv(qe_,
                                    colnames=["wavelength", "qe"],
                                    description="")
            # qe is a function now (interp)
            qe = iotools.chromatic_interpolator(
                                 tab["wavelength"].to(units.AA).value,
                                 tab["qe"],
                                 k=1, ext='zeros') # k=1 to avoid artificial-wiggles.

        dict_init["qe"] = qe # float or func

        ## variance model
        if "variance_model" in config:
            dict_init["variance_model"] = config.get("variance_model") #: Variance model

        # build the instance given this configuration
        return cls(**dict_init, lbda=lbda, thermaloptics=thermaloptics)

    # ============== #
    #   Methods      #
    # ============== #
    def update(self, reset_others=False, **kwargs):
        """Change any mutable parameter of the detector.

        See `mutable_parameters` for the list of accepted names.
        ``readout_noise`` is accepted as an alias of ``ron``; ``ngroup`` and
        ``nframe_per_group`` only change the corresponding element of `nmd`.
        ``lbda`` is also accepted and passed to `update_lbda`.

        Parameters
        ----------
        reset_others : bool, optional
            If True, parameters not given in ``kwargs`` are reset to their
            initial values. Default is False.
        **kwargs
            Mutable parameters to update. Parameters set to None are ignored.

        Warns
        -----
        UserWarning
            If a given parameter is not mutable (it is then ignored).
        """
        lbda = kwargs.pop("lbda", None)  # removes it from the kwargs

        ALLOWED = {"readout_noise": "ron"}

        updates = {}
        for k, v in kwargs.items():
            # allow alternative names for parameters.

            k = ALLOWED.get(k, k)
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            if v is None:        # Skip
                continue

            # only change the number N of groups (M and D remain the same)
            if k == 'ngroup':
                n, m, d = self.nmd  # NMD = (ngroup, ngroups, ndrops)
                k, v = 'nmd', (v, m, d)

            if k in ["nframe_per_group", "nframes"]:
                n, m, d = self.nmd  # NMD = (ngroup, nframe_per_group, ndrops)
                k, v = 'nmd', (n, v, d)

            #setattr(self, k, v)  # Update
            updates[k] = v

        # Update the lbda *after* all the parameters have been updated.
        if lbda is not None:
            self.update_lbda(lbda)  # updates all chromatic components

        # update the metadata
        if reset_others:
            self._meta = self._meta_in | updates
        else:
            self._meta = self._meta | updates

    def update_lbda(self, lbda):
        """Update the wavelength(s) used by chromatic components.

        Parameters
        ----------
        lbda : float or array_like
            Wavelength(s) in Angstrom.
        """
        self._lbda = lbda

    # ======= #
    #  GETTER  #
    # ======= #
    def get_data_volume(self, units="GB", nbit_record=16):
        """Get the data volume produced by a single ramp.

        Only the recorded groups contribute: within a group the ``m`` frames
        are co-added on board and the ``d`` dropped frames are never read out,
        so the volume scales with the number of groups ``n`` of the MACC mode
        and with the number of detector pixels (`shape`).

        Parameters
        ----------
        units : str, optional
            Unit of the returned volume. Must be a known astropy data unit
            (e.g. ``"bit"``, ``"byte"``, ``"MB"``, ``"GB"``). Default is ``"GB"``.
        nbit_record : int, optional
            Number of bits recorded per pixel and per group. Default is 16.

        Returns
        -------
        float
            The data volume of one ramp, in the requested unit.

        See Also
        --------
        slicersim.target.VirtualTarget.get_data_volume : Volume for a full exposure.
        """
        from astropy import units as u
        npixels = np.prod(self.shape)
        (ngroups, nframe, ndrops) = self.nmd
        volume_in_bit = (npixels * nbit_record * ngroups) * u.bit
        return volume_in_bit.to(units).value

    def get_qe(self, lbda=None):
        """Get the quantum efficiency.

        Parameters
        ----------
        lbda : float or array_like or None, optional
            Wavelength(s) in Angstrom. If None, `lbda` is used. Ignored if
            the QE is a constant. Default is None.

        Returns
        -------
        float or numpy.ndarray
            Quantum efficiency in e-/ph.
        """
        # as float
        if not callable(self.qe):
            return self.qe

        # as function
        if lbda is None:
            lbda = self.lbda

        return self.qe(lbda)

    def get_pixel_size(self, unit="micrometer"):
        """Get the pixel size in the specified unit.

        Parameters
        ----------
        unit : str, optional
            Unit for the pixel size. Must be a known astropy length unit.
            Default is ``"micrometer"``.

        Returns
        -------
        float
            Pixel size in the specified unit.
        """
        # self.pixel_size is in micrometer
        return self.pixel_size * units.micrometer.to(unit)

    def get_thermal_dark(self, thermaloptics=None, as_sum=True, units="ph/s", lbda_range=None,
                             cached=False):
        """Get the thermal dark current of a detector pixel.

        This is the thermal emission of the non-dispersed optical elements of
        ``thermaloptics`` (those flagged False in its ``meta["dispersed"]``),
        integrated over ``lbda_range`` and received by one pixel.

        Parameters
        ----------
        thermaloptics : `~slicersim.thermal.ThermalOptics` or None, optional
            Optics emitting the thermal signal. If None, `thermaloptics` is
            used. Default is None.
        as_sum : bool, optional
            If True, return the sum of the contributions of all optical
            elements. Otherwise, return one value per element.
            Default is True.
        units : str, optional
            Output units, either ``"ph/s"`` or ``"e/s"`` (alias ``"e-/s"``).
            In the latter case, the QE is applied. Default is ``"ph/s"``.
        lbda_range : array_like or None, optional
            Wavelength range ``[lmin, lmax]`` in Angstrom. If None,
            `lbda_range` is used. Default is None.
        cached : bool, optional
            If True, store the result (per unit) and reuse it in later calls
            made with ``cached=True`` and ``as_sum=True``; the cache is not
            invalidated when parameters change. Default is False.

        Returns
        -------
        float or numpy.ndarray
            Thermal dark current per pixel, in ``units``. 0 if no thermal
            optics are available.

        Raises
        ------
        ValueError
            If ``units`` is not one of the accepted values.
        """
        if as_sum and cached and hasattr(self, "_cached_thermal") and self._cached_thermal is not None:
            # cached by unit
            if (cached := self._cached_thermal.get(units, None)) is not None:
                return cached

        # which thermaloptics to consider.
        if thermaloptics is None:
            thermaloptics = self.thermaloptics

        # nothing given ? returns 0, i.e. no thermal dark.
        if thermaloptics is None:
            return 0

        # select the coefficient to match the request output units.
        if units == "ph/s":
            qe = None

        elif units in ["e/s", "e-/s"]:
            qe = self.qe # float or func
        else:
            raise ValueError(f"could not parse requested units: ph/s or e/s accepted, {units} given")

        if lbda_range is None:
            lbda_range = self.lbda_range

        # the collective area of pixels in m
        pixel_area = self.get_pixel_size("m")**2 # pixel_size is in micro

        # the contribution from thermal emission of optcs, in ph/s
        signals = thermaloptics.get_signal(lbda_bin = lbda_range, # expectedin in [A]
                                            area = pixel_area, # Collecting area [m²]
                                            qe=qe) # qe could be float or func
        # only take the undispersed one
        signals = signals[ ~np.asarray(thermaloptics.meta["dispersed"]).astype(bool) ]


        # sum over 1 element if only 1 temperature
        if as_sum:
            signals = np.sum(signals, axis=0)

        if cached:
            if not hasattr(self, "_cached_thermal") or self._cached_thermal is None:
                self._cached_thermal = {}

            self._cached_thermal[units] = signals

        return signals

    def get_effective_ron(self, nmd=None, variance_model="default", ron=None, as_variance=False):
        """Compute the effective read-out noise of a ramp.

        Following Rauscher et al. (2007), the read-out noise variance of the
        up-the-ramp fit is ``12 (n - 1) / (m n (n + 1)) * ron**2``, to which
        ``ron_floor**2`` is added.

        Parameters
        ----------
        nmd : tuple of int or None, optional
            MACC read-out mode ``(ngroups, nframes_per_group, ndrops)``.
            If None, `nmd` is used. Default is None.
        variance_model : str, optional
            Variance model to use. Only the Rauscher model (``"rauscher07"``)
            is implemented. If ``"default"``, `variance_model` is used.
            Default is ``"default"``.
        ron : float or None, optional
            Read-out noise per frame in e-. If None, `ron` is used.
            Default is None.
        as_variance : bool, optional
            If True, return the variance (e-^2) instead of the standard
            deviation (e-). Default is False.

        Returns
        -------
        float
            Effective read-out noise of the ramp in e- (or its variance in
            e-^2 if ``as_variance=True``). 0 if ``ron`` is 0.

        Raises
        ------
        NotImplementedError
            If ``variance_model`` is not a Rauscher model.
        """
        # which ron
        if ron is None:
            ron = self.ron

        if ron == 0: # switched off.
            return 0

        # which macc mode to use
        if nmd is None:
            nmd = self.nmd

        n, m, d = nmd

        # which variance model to use ?
        if variance_model == "default":
            variance_model = self.variance_model


        # compute the effective read-out noise estimator.
        if variance_model.lower() in ["rauscher07", "rauscher10", "rauscher+07"]:
            rauscher_ron_variance = 12 * (n - 1) / (m * n * (n + 1)) * ron**2
            effective_ron_variance = rauscher_ron_variance + self.meta["ron_floor"]**2
        else:
            raise NotImplementedError(f"effective ron has only been implemented for the Rauscher variance model ; {variance_model=} given. ")

        if as_variance:
            return effective_ron_variance

        return np.sqrt(effective_ron_variance)

    def get_effective_dark(self, cached_thermal=False, incl_thermal=True):
        """Get the total effective dark current.

        This is the sum of the detector dark current (`dark`), the ROIC glow
        (`roic_glow` / `tframe`) and, optionally, the thermal dark from the
        surrounding optics (`get_thermal_dark`).

        Parameters
        ----------
        cached_thermal : bool, optional
            Passed as ``cached`` to `get_thermal_dark`. Default is False.
        incl_thermal : bool, optional
            If True, include the thermal dark. Default is True.

        Returns
        -------
        float
            Total effective dark current in e-/s/pixel.
        """
        roic_glow_e_per_sec = self.roic_glow / self.tframe # e-/frame => e-/s

        dark = self.dark + roic_glow_e_per_sec
        if incl_thermal:
            dark += self.get_thermal_dark(units="e-/s", cached=cached_thermal)

        return dark

    def estimate_pixel_signal(self, flux, lbda=None,
                                  withdark=False,
                                  variance_model="default",
                                  saturation="default",
                                  cached_thermal=False,
                                  npix_per_spaxel=1):
        """Estimate the measured signal and variance from an incident flux.

        The flux of each voxel is spread over ``npix_per_spaxel`` detector
        pixels; the signal, effective dark (see `get_effective_dark`, thermal
        dark included), variance and saturation are computed per pixel, then
        summed back per voxel. Signals are integrated over `integration_time`.
        The number of saturated detector pixels is stored in the
        ``nsaturated_detpx`` attribute.

        Parameters
        ----------
        flux : array_like
            Incident flux in ph/s/voxel (arbitrary shape, e.g.
            ``(nlbda, ny, nx)``).
        lbda : array_like or None, optional
            Wavelengths in Angstrom used to evaluate the QE, along the first
            axis of ``flux``. If None, `lbda` is used. Default is None.
        withdark : bool, optional
            If True, keep the dark contribution in the output signal (it is
            always included in the variance and in the saturation test).
            Default is False.
        variance_model : str, optional
            Variance model to use (``"rauscher07"`` or ``"kubik16"``).
            If ``"default"``, `variance_model` is used. Default is ``"default"``.
        saturation : float or str or None, optional
            Saturation level in ADU per pixel. Saturated pixels get a NaN
            signal and an infinite variance. If ``"default"``, `saturation` is
            used. If None, saturation is not tested. Default is ``"default"``.
        cached_thermal : bool, optional
            Passed as ``cached`` to `get_thermal_dark`. Default is False.
        npix_per_spaxel : float, optional
            Number of detector pixels over which the flux of one voxel is
            spread. Default is 1.

        Returns
        -------
        signal : numpy.ndarray or float
            Integrated signal in ADU/voxel.
        variance : numpy.ndarray or float
            Signal variance in ADU^2/voxel.

        Raises
        ------
        NotImplementedError
            If ``variance_model`` is unknown.
        """
        if variance_model == "default":
            variance_model = self.variance_model

        if saturation == "default":
            saturation = self.saturation

        flux_pixel = flux/npix_per_spaxel  # ph/s/pixel

        # Variance estimators works with input flux in e-/s (not ph/s)
        qe = complete_dims(self.get_qe(lbda=lbda), -np.ndim(flux_pixel))
        flux_e = flux_pixel * qe  # [e-/s]

        # Variance estimate in [ADU²]
        ## gain, ron, dark etc. are in there.
        ## thermal dark is computed twice, for the variance and for the signal.
        variance = self.estimate_variance(flux=flux_e, model=variance_model, incl_thermal=True,
                                            cached_thermal=cached_thermal)

        # actual signal registered, including darks for staturation tests.
        effective_dark = self.get_effective_dark(cached_thermal=cached_thermal, incl_thermal=True)
        signal = (flux_e + effective_dark) * self.electronpers_to_adu()  # Total signal [ADU]

        # Detect and mask saturated pixels
        if saturation is not None:
            saturated = signal > saturation
            self.nsaturated_detpx = np.count_nonzero(saturated)
            if self.nsaturated_detpx > 0:
                #warnings.warn(
                #    f"{self.nsaturated_detpx} detector px above {self.saturation} ADU.",
                #    SaturationWarning)
                if np.ndim(signal):
                    signal[saturated] = np.nan
                    variance[saturated] = np.inf
                else:
                    signal, variance = np.nan, np.inf
        else:
            self.nsaturated_detpx = None

        if not withdark:  # Remove dark contribution
            signal -= effective_dark * self.electronpers_to_adu()  # [ADU]

        # now recompute in ADU/spaxel
        signal *= npix_per_spaxel
        variance *= npix_per_spaxel

        return signal, variance  # [ADU/voxel], [ADU²/voxel]

    def estimate_variance(self, flux, model=None, incl_thermal=True, cached_thermal=False):
        """Estimate the variance of the integrated signal of a pixel.

        The variance accounts for the photon noise of the input flux and of
        the effective dark current (see `get_effective_dark`), and for the
        read-out noise, given the current MACC mode `nmd`.

        Parameters
        ----------
        flux : array_like
            Incident flux in e-/s/pixel.
        model : str or None, optional
            Variance model, ``"rauscher07"`` (Rauscher et al. 2007, corrected
            in 2010) or ``"kubik16"`` (Kubik et al. 2016). If None,
            `variance_model` is used. Default is None.
        incl_thermal : bool, optional
            If True, include the thermal dark in the dark current.
            Default is True.
        cached_thermal : bool, optional
            Passed as ``cached`` to `get_thermal_dark`. Default is False.

        Returns
        -------
        numpy.ndarray or float
            Variance in ADU^2.

        Raises
        ------
        NotImplementedError
            If ``model`` is unknown.
        """
        if model is None:
            model = self.variance_model

        effective_dark = self.get_effective_dark(incl_thermal=incl_thermal, cached_thermal=cached_thermal)

        variance_input = dict(nmd=self.nmd, tframe=self.tframe,
                              ron=self.ron, dark=effective_dark,
                              gain=self.gain)

        if model.lower() in ["rauscher07", "rauscher10", "rauscher+07"]: # allowing old format
            return self._estimate_variance_rauscher07(flux, **variance_input)

        elif model.lower() in ["kubik16", "kubik16"]:
            return self._estimate_variance_kubik16(flux, **variance_input)
        else:
            raise NotImplementedError(f"unknown variance model {model=} ; rauscher07 or kubik16 available.")

    # - conversion
    def photonflux_to_adu(self, lbda=None):
        """Get the conversion factor from photon flux [ph/s] to signal [ADU].

        This is `integration_time` [s] * QE [e-/ph] * `gain` [ADU/e-].

        Parameters
        ----------
        lbda : float or array_like or None, optional
            Wavelength(s) in Angstrom used to evaluate the QE (see `get_qe`).
            If None, `lbda` is used. Default is None.

        Returns
        -------
        float or numpy.ndarray
            Conversion factor in ADU/(ph/s).
        """
        qe = self.get_qe(lbda=lbda) # applied the lbda if needed.
        return self.integration_time * qe * self.gain

    # could be propery but not to consistancy with photonflux_to_adu
    def electronpers_to_adu(self):
        """Get the conversion factor from electron flux [e-/s] to signal [ADU].

        This is `integration_time` [s] * `gain` [ADU/e-].

        Returns
        -------
        float
            Conversion factor in ADU/(e-/s).
        """
        return self.integration_time * self.gain

    # - Internal
    def _estimate_variance_rauscher07(self, flux, nmd, tframe, ron, dark, gain):
        """Variance from Rauscher+2007 (corrected in Rauscher+2010).

        Parameters
        ----------
        flux : array_like
            Flux in e-/s.
        nmd : tuple of int
            MACC(N=#group, M=#frame, D=#drop).
        tframe : float
            Frame time in seconds.
        ron : float
            Read noise per frame in e-.
        dark : float
            Dark current in e-/s.
        gain : float
            Gain in ADU/e-.

        Returns
        -------
        numpy.ndarray or float
            Total variance in ADU^2.

        Notes
        -----
        In Rauscher+07, the flux estimate (eqs 2, 3 & 4) is unbiased and
        naturally falls back to the incident flux in the noiseless limit.
        """
        n, m, d = nmd
        if (n == 0) or (m == 0):
            return np.full_like(flux, np.nan)

        signal = dark + flux    # [e-/s]
        tgroup = tframe * (m + d)

        # Rauscher works in e-
        term1 = self.get_effective_ron(variance_model="rauscher07", ron=ron, nmd=nmd,  as_variance=True)
        term2 = 6 * (n**2 + 1) / (5 * n * (n + 1)) * (n - 1) * tgroup * signal
        term3 = 2 * (m**2 - 1) * (n - 1) / (m * n * (n + 1)) * tframe * signal
        var = term1 + term2 - term3 # [e-²]

        return var / gain**2        # [ADU²]

    @staticmethod
    def _estimate_variance_kubik16(flux, nmd, tframe, ron, dark, gain):
        """Variance from Kubik+2016.

        Parameters
        ----------
        flux : array_like
            Flux in e-/s.
        nmd : tuple of int
            MACC(N=#group, M=#frame, D=#drop).
        tframe : float
            Frame time in seconds.
        ron : float
            Read noise per frame in e-.
        dark : float
            Dark current in e-/s.
        gain : float
            Gain in ADU/e-.

        Returns
        -------
        numpy.ndarray or float
            Total variance in ADU^2.
        """
        n, m, d = nmd

        # Kubik16 works in ADU
        tint = Detector._get_integration_time(nmd, tframe)  # [s]
        signal = (flux + dark) * gain * tint    # e-/s * ADU/e- * s [ADU]

        gghat = signal / (n - 1)                # g*ghat in K+16 [ADU]
        alpha = (1 - m**2) / (3 * m * (m + d))  # [1]
        gamma = 2 * gain**2 * ron**2 / m        # [ADU²]
        beta = gamma / (1 + alpha)
        s_beta = gghat + beta / gain
        var = ( ((n-1) + alpha) * gghat + gamma / gain ) * gain * 4 * s_beta**2
        var /= (1 + alpha)**2 * gain**2 + 4 * s_beta**2  # [ADU²]

        return var              # [ADU²]


    # ================ #
    #   Properties     #
    # ================ #
    # - Core
    @property
    def lbda(self):
        """Wavelength(s) in Angstrom at which chromatic quantities are evaluated."""
        return self._lbda

    @property
    def thermaloptics(self):
        """`~slicersim.thermal.ThermalOptics` producing the thermal dark, or None."""
        return self._thermaloptics

    @property
    def meta(self):
        """Metadata dictionary holding the current detector parameters."""
        # info: self._meta_in contains the input meta.
        return self._meta

    #
    # from meta
    #
    @property
    def name(self):
        """Name of the detector (``"unknown"`` if not set in `meta`)."""
        return self.meta.get("name", "unknown")

    @property
    def shape(self):
        """Detector shape in pixels ``[ny, nx]``."""
        return self.meta.get("shape", None)


    @property
    def variance_model(self):
        """Name of the variance model (``"rauscher07"`` or ``"kubik16"``)."""
        return self.meta.get("variance_model")

    @property
    def lbda_range(self):
        """Wavelength range ``[lmin, lmax]`` seen by the detector, in Angstrom.

        If not specified at initialization, the first and last values of
        `lbda` are used.
        """
        lbda_range = self.meta.get("lbda_range", None)
        if lbda_range is None:
            lbda_range = self.lbda[[0, -1]]
        return lbda_range

    # - pixels properties
    @property
    def dark(self):
        """Detector dark current in e-/s/pixel.

        The total effective dark current, including the ROIC glow and the
        thermal dark, is given by `get_effective_dark`.
        """
        return self.meta.get("dark")

    @property
    def roic_glow(self):
        """ROIC glow in e-/pixel/frame.

        This is the additional signal induced by each read of the ROIC.
        """
        return self.meta.get("roic_glow", 0)

    @property
    def ron(self):
        """Detector read-out noise per frame in e- (RMS)."""
        return self.meta.get("ron")

    @property
    def qe(self):
        """Quantum efficiency in e-/ph, as a float or a callable of wavelength.

        Use `get_qe` to evaluate it at the current wavelength(s) `lbda`.
        """
        return self.meta.get("qe")

    @property
    def pixel_size(self):
        """Detector pixel size in micrometers."""
        return self.meta.get("pixel_size")

    @property
    def saturation(self):
        """Saturation level in ADU."""
        return self.meta.get("saturation")

    @property
    def gain(self):
        """Gain in ADU/e-."""
        return self.meta.get("gain")

    @property
    def tframe(self):
        """Frame time in seconds."""
        return self.meta.get("tframe")

    # - read-out model and co.
    @property
    def nmd(self):
        """MACC read-out mode ``(n, m, d)``.

        ``n`` is the number of groups, ``m`` the number of frames per group and
        ``d`` the number of dropped frames between groups.
        """
        return self.meta.get("nmd")

    @property
    def macc(self):
        """MACC read-out mode as a ``"N:M:D"`` string."""
        return ':'.join([ str(_) for _ in self.nmd ])

    @property
    def min_group(self):
        """Minimum allowed number of groups per ramp."""
        return self.meta.get("min_group")

    @property
    def max_group(self):
        """Maximum allowed number of groups per ramp."""
        return self.meta.get("max_group")

    @property
    def nframes(self):
        """Total number of frames in the ramp, ``n * (m + d) - d``."""
        n, m, d = self.nmd
        return (n * (m + d) - d)

    @property
    def tgroup(self):
        """Time between the starts of two groups in seconds, ``(m + d) * tframe``."""
        n, m, d = self.nmd
        return (m + d) * self.tframe

    @property
    def integration_time(self):
        """Effective integration time in seconds, ``(n - 1) * tgroup``.

        This is the time between the first and last groups, used to convert
        fluxes into integrated signals.
        """
        n, m, d = self.nmd
        return (n - 1) * (m + d) * self.tframe  # = (n - 1) * tgroup

    @property
    def exposure_time(self):
        """Total duration of the ramp in seconds, ``nframes * tframe``."""
        return self.nframes * self.tframe
