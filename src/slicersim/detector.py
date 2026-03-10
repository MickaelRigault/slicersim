"""Detector module.

This module simulates detector effects such as readout noise and dark current,
and Multiple Accumulated Sampling (MACC) readout modes.
"""

__author__ = "Mickael Rigault <m.rigault@ip2i.in2p3.fr>, Yannick Copin <y.copin@ip2i.in2p3.fr>"

import warnings
import numpy as np
from copy import deepcopy

from astropy import units
from astropy.utils.exceptions import AstropyWarning

from .utils import complete_dims


class SaturationWarning(AstropyWarning):
    """ Base class for saturation warnings. """
    
class Detector():
    """Simulates a detector for a spectrograph.

    This class handles signal and variance estimation from input flux and provides
    methods for optimal extraction of spectra.

    Parameters
    ----------
    tframe : float
        Frame time in seconds.
    dark : float
        Dark current in electrons per second.
    ron : float
        Read-out noise per frame in electrons.
    qe : float or callable
        Quantum efficiency, either as a constant or a function of wavelength.
    pixel_size : float
        Size of each pixel in micrometers.
    gain : float, optional
        Gain in ADU per electron. Default is 1.
    saturation : int, optional
        Saturation level in ADU. Default is 65635.
    nmd : tuple, optional
        Number of groups, frames per group, and drops. Default is (64, 8, 0).
    lbda_range : array_like, optional
        Wavelength range accepted by the detector in Angstroms.
    min_group : int, optional
        Minimal number of groups per ramp. Default is 2.
    max_group : int, optional
        Maximum number of groups per ramp. Default is 64.
    lbda : float, optional
        Wavelength in Angstroms. Default is 10000.
    variance_model : str, optional
        Variance model to use. Default is "rauscher07".
    thermaloptics : object, optional
        Object handling thermal optics.
    meta : dict, optional
        Metadata dictionary.

    Attributes
    ----------
    mutable_parameters : list
        List of parameters that can be changed after initialization.
    """
    #: Mutable parameters (list)
    mutable_parameters = ['ngroup', "nframe_per_group",
                          'nmd', 'tframe',
                          'ron', 'gain', 'qe', 'dark',
                          'saturation', 'variance_model']

    def __init__(self, tframe, dark, ron, qe, pixel_size,
                     gain=1, saturation=65_635,
                     nmd=(64, 8, 0), lbda_range=None,
                     min_group=2, max_group=64,
                     lbda=10_000.,
                     variance_model="rauscher07",
                     thermaloptics=None,
                     shape=[4096, 4096],
                     meta={} ):
        # from other components
        self._lbda = lbda
        self._thermaloptics = thermaloptics

        # internal structure is meta-based.
        # detector
        meta["shape"] = shape
        meta["tframe"] = tframe
        meta["dark"] = dark
        meta["ron"] = ron
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
            Configuration dictionary containing detector parameters.
            It must contain at least:
            - nmd / readout_mode
            - ron
            - dark
            - tframe
            - qe
            - pixel_size
        lbda : float or array_like, optional
            Wavelength in Angstroms. Default is 10000.
        thermaloptics : object, optional
            Thermal optics object.

        Returns
        -------
        Detector
            An instance of the Detector class.
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
            
        ## readout_noise
        dict_init["ron"] = float(config.get("readout_noise"))         #: Read-Out Noise per frame [e-]

        ## dark
        dict_init["dark"] = float(config.get("dark"))                 #: Dark current [e-/s]

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
        """Change any mutable attribute of the detector.

        (see self.mutable_parameters)

        Parameters
        ----------
        reset_others : bool, optional
            If True, reset other parameters to their initial values. Default is False.
        **kwargs : dict
            Keyword arguments representing the mutable attributes to update.
        """
        lbda = kwargs.pop("lbda", None)  # removes it from the kwargs
        
        updates = {}
        for k, v in kwargs.items():
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
        """Update chromatic components.

        Parameters
        ----------
        lbda : array_like or float
            Wavelength in Angstroms.
        """
        self._lbda = lbda

    # ======= #
    #  GETTER  #
    # ======= #
    def get_data_volume(self, units="GB", nbit_record=16):
        """ """
        from astropy import units as u
        npixels = np.prod(self.shape)
        (ngroups, nframe, ndrops) = self.nmd
        volume_in_bit = (npixels * nbit_record * ngroups) * u.bit
        return volume_in_bit.to(units).value
        
    def get_qe(self, lbda=None):
        """Get the quantum efficiency.

        Parameters
        ----------
        lbda : array_like or float, optional
            Wavelength in Angstroms. If None, use the current wavelength.

        Returns
        -------
        float or array_like
            Quantum efficiency.
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
            Unit for the pixel size. Must be a known astropy unit.
            Default is "micrometer".

        Returns
        -------
        float
            Pixel size in the specified unit.
        """
        # self.pixel_size is in micrometer
        return self.pixel_size * units.micrometer.to(unit)

    def get_thermal_dark(self, thermaloptics=None, as_sum=True, units="ph/s", lbda_range=None):
        """Get the thermal dark current.

        This is based on the `thermaloptics` object, if provided.

        Parameters
        ----------
        thermaloptics : object, optional
            Thermal optics object. Uses self.thermaloptics is not provided.
        as_sum : bool, optional
            If True, return the sum of signals. Default is True.
        units : str, optional
            Units for the output, either "ph/s" or "e/s". Default is "ph/s".
        lbda_range : array_like, optional
            Wavelength range in Angstroms. If None, use the current wavelength range.

        Returns
        -------
        float or array_like
            Thermal dark current (see units)
        """
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
                                          
        # sum over 1 element if only 1 temperature
        if as_sum:
            signals = np.sum(signals, axis=0)
        
        return signals


    def get_effective_ron(self, nmd=None, variance_model="default", ron=None):
        """ compute the effective impact of the read-out noise (ron) 

        Parameters
        ----------
        nmd: list, None
            MACC model (ngroups, nframes_per_group, ndrops). If None, self.nmd used.
        ron: float, None
            read-out noise per frame to use. If None, self.ron used.
        variance_model : str, optional
            Variance model to use (e.g., 'rauscher07', 'kubik16').
            If "default", `self.variance_model` is used. Default is "default".
        
        Returns
        -------
        ron: float
            effective ron up the ramp.

        """
        # which ron
        if ron is None:
            ron = self.ron
        
        # which macc mode to use
        if nmd is None:
            nmd = self.nmd

        n, m, d = nmd

        # which variance model to use ?        
        if variance_model == "default":
            variance_model = self.variance_model


        # compute the effective read-out noise estimator.
        if variance_model.lower() in ["rauscher07", "rauscher10" "rauscher+07"]:
            effective_ron = np.sqrt(12 * (n - 1) / (m * n * (n + 1)) * ron**2)
        else:
            raise NotImplementedError(f"effective ron has only been implemented for the Rauscher variance model ; {variance_model=} given. ")

        return effective_ron
            
    
    def estimate_pixel_signal(self, flux, lbda=None,
                                  withdark=False,
                                  variance_model="default",
                                  saturation="default"):
        """Estimate measured signal and variance from incident flux.

        Parameters
        ----------
        flux : array_like
            Incident flux in ph/s/px (arbitrary shape, e.g., (nlbda, ny, nx)).
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, `self.lbda` is used.
            Default is None.
        withdark : bool, optional
            Include dark contribution to output signal. Default is False.
        variance_model : str, optional
            Variance model to use (e.g., 'rauscher07', 'kubik16').
            If "default", `self.variance_model` is used. Default is "default".
        saturation : int or str, optional
            Saturation level in ADU. If given, this attempts to detect saturation.
            If "default", `self.saturation` is used.
            If None, no saturation is applied. Default is "default".

        Returns
        -------
        tuple of array_like
            - Pixel signal [ADU].
            - Variance [ADU²].
        """
        if variance_model == "default":
            variance_model = self.variance_model

        if saturation == "default":
            saturation = self.saturation
            
        # Variance estimators works with input flux in e-/s (not ph/s)
        qe = complete_dims(self.get_qe(lbda=lbda), -np.ndim(flux))
        flux_e = flux * qe  # [e-/s]

        # Variance estimate in [ADU²]
        ## gain, ron, dark etc. are in there.
        variance = self.estimate_variance(flux=flux_e, model=variance_model, incl_thermal=True) 

        # actual signal registered, including darks for staturation tests.
        effective_dark = self.dark + self.get_thermal_dark(units="e-/s")
        signal = (flux_e + effective_dark) * self.electronpers_to_adu()  # Total signal [ADU]

        # Detect and mask saturated pixels
        if saturation is not None:
            saturated = signal > saturation
            self.nsaturated_detpx = np.count_nonzero(saturated)
            if self.nsaturated_detpx > 0:
                warnings.warn(
                    f"{self.nsaturated_detpx} detector px above {self.saturation} ADU.",
                    SaturationWarning)
                if np.ndim(signal):
                    signal[saturated] = np.nan
                    variance[saturated] = np.inf
                else:
                    signal, variance = np.nan, np.inf
        else:
            self.nsaturated_detpx = None

        if not withdark:  # Remove dark contribution
            signal -= effective_dark * self.electronpers_to_adu()  # [ADU]

        return signal, variance  # [ADU], [ADU²]

    def estimate_variance(self, flux, model=None, incl_thermal=True):
        """Estimate the variance associated with the input flux.

        Parameters
        ----------
        flux : array_like
            Incident flux in e-/s.
        model : str, optional
            Variance model to use. If None, `self.variance_model` is used.
            Default is None.
        incl_thermal : bool, optional
            Include thermal contribution. Default is True.

        Returns
        -------
        array_like
            Variance in ADU².
        """
        if model is None:
            model = self.variance_model

        if incl_thermal:
            effective_dark = self.dark + self.get_thermal_dark(units="e-/s")
        else:
            effective_dark = self.dark
            
        variance_input = dict(nmd=self.nmd, tframe=self.tframe,
                              ron=self.ron, dark=effective_dark,
                              gain=self.gain)
                              
        if model.lower() in ["rauscher07", "rauscher10" "rauscher+07"]: # allowing old format
            return self._estimate_variance_rauscher07(flux, **variance_input)
            
        elif model.lower() in ["kubik16", "kubik16"]:
            return self._estimate_variance_kubik16(flux, **variance_input)
        else:
            raise NotImplementedError(f"unknown variance model {model=} ; rauscher07 or kubik16 available.")

    # - conversion
    def photonflux_to_adu(self, lbda=None):
        """Conversion factor from photon flux [ph/s] to integrated signal [ADU].

        This is effective exposure time [s] * QE [e-/ph] * gain [ADU/e-].

        Parameters
        ----------
        lbda : array_like, optional
            used to estimated the qe (see.get_qe())
            Wavelength array in Angstrom. If None, `self.lbda` is used.
            Default is None.

        Returns
        -------
        adu: array
        """
        qe = self.get_qe(lbda=lbda) # applied the lbda if needed.
        return self.integration_time * qe * self.gain

    # could be propery but not to consistancy with photonflux_to_adu
    def electronpers_to_adu(self):
        """Conversion factor from electron/s to integrated signal [ADU].

        This is effective exposure time [s] * gain [ADU/e-].
        """
        return self.integration_time * self.gain
        
    # - Internal
    @staticmethod
    def _estimate_variance_rauscher07(flux, nmd, tframe, ron, dark, gain):
        """Variance from Rauscher+2007 (corrected in Rauscher+2010).

        Note
        ----
        In Rauscher+07, the flux estimate (eqs 2, 3 & 4) is unbiased and
        naturally falls back to the incident flux in the noiseless limit.

        Parameters
        ----------
        flux : array_like
            Flux in e-/s.
        nmd : tuple
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
        array_like
            Total variance in ADU².
        """
        n, m, d = nmd
        if (n == 0) or (m == 0):
            return np.full_like(flux, np.nan)

        signal = dark + flux    # [e-/s]
        tgroup = tframe * (m + d)

        # Rauscher works in e-
        term1 = 12 * (n - 1) / (m * n * (n + 1)) * ron**2
        term2 = 6 * (n**2 + 1) / (5 * n * (n + 1)) * (n - 1) * tgroup * signal
        term3 = 2 * (m**2 - 1) * (n - 1) / (m * n * (n + 1)) * tframe * signal
        var = term1 + term2 - term3 # [e-²]

        return var / gain**2        # [ADU²]

    @staticmethod
    def _estimate_variance_kubik16(flux, nmd, tframe, ron, dark, gain):
        """Variance from Kubik+2016 (as updated by ).

        Parameters
        ----------
        flux : array_like
            Flux in e-/s.
        nmd : tuple
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
        array_like
            Total variance in ADU².
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
        """Wavelength in Angstroms."""
        return self._lbda

    @property
    def thermaloptics(self):
        """ThermalOptics object producing signal onto the detector."""
        return self._thermaloptics

    @property
    def meta(self):
        """Metadata containing the current detector parameters."""
        # info: self._meta_in contains the input meta.
        return self._meta

    #
    # from meta
    #
    @property
    def name(self):
        """Name of the detector, if any."""
        return self.meta.get("name", "unknown")
    
    @property
    def shape(self):
        """ shape of the detector """
        return self.meta.get("shape", None)

    
    @property
    def variance_model(self):
        """Variance model name."""
        return self.meta.get("variance_model")
        
    @property
    def lbda_range(self):
        """Wavelength range accepted by the detector.

        This is the first and last value of `self.lbda` if not specified.
        """
        lbda_range = self.meta.get("lbda_range", None)
        if lbda_range is None:
            lbda_range = self.lbda[[0, -1]]
        return lbda_range

    # - pixels properties
    @property
    def dark(self):
        """Detector dark current in e-/s.

        Note
        ----
        The total effective dark current is `self.dark` + `self.get_thermal_dark()`.
        """
        return self.meta.get("dark")
    
    @property
    def ron(self):
        """Detector Read-Out Noise per frame in e-."""
        return self.meta.get("ron")
        
    @property
    def qe(self):
        """Quantum efficiency (float, array, or function).

        Note
        ----
        Use `self.get_qe()` to get the actual QE for the input `self.lbda`.
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
        """Read-out mode (n, m, d).

        n: number of groups
        m: frames per group
        d: dropped frames
        """
        return self.meta.get("nmd")

    @property
    def macc(self):
        """Read-out mode in MACC description 'N:M:D'."""
        return ':'.join([ str(_) for _ in self.nmd ])

    @property
    def min_group(self):
        """Minimal allowed number of groups per ramp (n_min, m, d)."""
        return self.meta.get("min_group")
    
    @property
    def max_group(self):
        """Maximum allowed number of groups per ramp (n_max, m, d)."""
        return self.meta.get("max_group")

    @property
    def nframes(self):
        """Total number of individual frames in the ramp."""
        n, m, d = self.nmd
        return (n * (m + d) - d)
    
    @property
    def tgroup(self):
        """Time between two groups, including drops."""
        n, m, d = self.nmd
        return (m + d) * self.tframe

    @property
    def integration_time(self):
        """Effective integration time for flux computations."""
        n, m, d = self.nmd
        return (n - 1) * (m + d) * self.tframe  # = (n - 1) * tgroup        

    @property
    def exposure_time(self):
        """Total integration time for sequences."""
        return self.nframes * self.tframe
