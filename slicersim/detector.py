"""
Detector module, to simulate:
* various detector effects (readout noise, dark current)
* Multiple Accumulated Sampling (MACC) readout modes
* Optimal cross-dispersion summation

.. autosummary::

   Detector
"""

__author__ = "Mickael Rigault <m.rigault@ip2i.in2p3.fr>, Yannick Copin <y.copin@ip2i.in2p3.fr>"

import warnings
import numpy as np
from astropy import units
from astropy.utils.exceptions import AstropyWarning

from . import iotools
from .utils import integ_gaussian1D_erf, complete_dims



class SaturationWarning(AstropyWarning):
    """ Base class for saturation warnings. """
    
class Detector:
    """
    Detector simulation.

    * Simulate px-level signal [ADU/px] and variance from (stationary)
      input flux [photon/s/px],
    * Simulate spx spectrum from optimal extraction (i.e. inverse-variance
      weighted cross-dispersion profile fit) of spectrograms on detector.
    """

    #: Mutable parameters (list)
    mutable_parameters = ['ngroup',"nframe_per_group",
                          'nmd', 'tframe',
                          'ron', 'gain', 'qe', 'dark', 'thermal_dark',
                          'saturation', 'variance_model']
    # Do not mutate px_size, fixed to 10 µm in spectrograph

    def __init__(self, config, lbda=10_000., verbose=False):
        """ Initialize the detector properties from `config` dictionary.

        :param dict config: detector configuration dictionary
        :param lbda: wavelengths [Å]
        :param bool verbose: verbose mode
        :return: detector instance
        """
        if "readout_mode" in config:
            if "nmd" in config:
                warnings.warn("readout_mode ignore as nmd found in config")
            config["nmd"] = config.pop("readout_mode")
            
        self.lbda = lbda                   #: Wavelengths [Å]
        self.name = config["name"]         #: Detector name
        self.nmd = config.get("nmd") #: MACC(N=#group, M=#frame, D=#drop)
        self.max_group = int(config.get("max_group", 64))     #: maximum number of group per ramp
        
        self.tframe = float(config["tframe"])      #: Frame time [s]
        self.dark = float(config["dark"])          #: Dark current [e-/s]
        self.thermal_dark = float(config.get("thermal_dark", 0) )  #: thermal induced dark current [e-/s]
        self.ron = float(config["readout_noise"])  #: Read-Out Noise per frame [e-]
        self.gain = float(config["gain"])          #: Gain [ADU/e-]

        try:
            self.qe = float(config["QE"])          # Constant QE
            self.qe_name = self.qe_interp = None
        except ValueError:                         # QE is a filename
            self.qe_name = config["QE"]            #: QE filename
            wname, qname = "wavelength", "qe"
            tab = iotools.read_ecsv(self.qe_name,
                                    colnames=[wname, qname],
                                    description='QE' if verbose else '')
            #: QE interpolator (wavelengths in Å)
            self.qe_interp = iotools.chromatic_interpolator(
                tab[wname].to(units.AA).value, tab[qname], ext='zeros')
            #: Quantum efficiency [e-/photon]
            self.qe = self.qe_interp(self.lbda)
        self.saturation = int(config["saturation"])     #: Saturation limit [ADU]
        self.px_size = float(config["px_size"])         #: Pixel size [µm]
        self.variance_model = config["variance_model"]  #: Variance model

        self._meta_in = config.copy()                   #: Meta-parameters
        self._meta = config.copy()                      #: Meta-parameters
        
    def __str__(self):

        s = f"Detector {self.name!r}:"
        s += f"\n  {self.px_size:.0f} µm px, dark: {self.dark:.3f} e-/s, thermal_dark: {self.thermal_dark:.3f} e-/s, RoN: {self.ron:.0f} e-/frame"
        if self.qe_name:
            s += f"\n  QE: {self.qe_name!r} (~{self.qe.mean():.2f} e-/ph)"
        else:
            s += f"\n  QE: {self.qe:.2f} e-/ph"
        sat = f"{self.saturation} ADU" if self.saturation else "none"
        s += f"\n  Gain: {self.gain:.1f} ADU/e-, Saturation: {sat}"
        s += f"\n  NMD={self.macc} (total: {self.nframes} frames), Tframe: {self.tframe:.2f} s"
        s += f"\n    Effective integration time: {self.integration_time:.2f} s"
        s += f"\n    Total exposure time:        {self.exposure_time:.2f} s"
        s += f"\n  Variance model: {self.variance_model}"

        return s

    @classmethod
    def from_config(cls, config, lbda=10_000., verbose=False):
        """
        Initialize from detector config.

        .. Note:: added for consistency between classes.

        :param dict config: detector configuration dictionary
        :param bool verbose: verbose mode
        :param lbda: wavelengths [Å]
        """
        return cls(config, lbda, verbose=verbose)

    # ============== #
    #   Methods      #
    # ============== #
    def update(self, reset_others=False, **kwargs):
        """ Change any mutable attribute of the detector. """
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
                
            setattr(self, k, v)  # Update
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
        """ Update chromatic components.
        
        Parameters
        ----------
        lbda: Array, float
            Wavelength [Å]

        Returns
        -------
        None
        """
        self.lbda = lbda
        # Update QE if needed
        if self.qe_interp is not None:
            self.qe = self.qe_interp(self.lbda)
    
    def get_exposure_time(self, nmd=None, tframe=None):
        """ Total integration time (to be used for sequences).

        Parameters
        ----------
        nmd: list, None
            provide the nmd (aka MACC mode): #group, #frame/group, #drop
            if None, self.nmd is used

        tframe: float, None
            the time between two frames. 
            if None, self.tframe is used

        Returns
        -------
        float
            total integration time [s]
        """
        return self.nframes * self.tframe
    
    def get_integration_time(self, nmd=None, tframe=None):
        """ Effective integration time (to be used for flux computations).

        :param nmd: (#group, #frame/group, #drop)
        :param float tframe: frame time [s]
        :return: effective integration time [s]
        """
        if nmd is None:
            nmd = self.nmd

        if tframe is None:
            tframe = self.tframe

        return self._get_integration_time(nmd=nmd, tframe=tframe)
    
    @staticmethod
    def _get_integration_time(nmd, tframe):
        """ internal function for the integration time """
        n, m, d = nmd
        return (n - 1) * (m + d) * tframe  # = (n - 1) * tgroup

    def estimate_slice_spectrum(self, flux):
        """ 
        [slicer mode]

        Parameters
        ----------
        flux: Array
            incident flux (arbitrary shape, e.g. (nlbda, nlices, npixels)) [ph/s/px]
        
        Returns
        -------
        Array, Array
            - slice spectra [ADU] 
            - variance [ADU²]
        """
        # Reshape photonflux2adu (() or (nlbda,)) to match flux's shape
        photonflux2ADU = complete_dims(self.photonflux2ADU, -np.ndim(flux))

        # Detector signal [ADU], shape is (width,) + flux.shape  | take ~200ms
        signal_at_detector, variance = self.estimate_pixel_signal(signal, withdark=False)
        
        # "optimal" extraction | signal_at_detector ignored then.
        signal = flux * photonflux2ADU   # Total incident signal [ADU]
        
        return signal, variance
        
    # - estimators
    def estimate_spx_spectrum(self, flux, sigma=1, width=5):
        """ Estimate extracted signal and variance from incident flux [ph/s].

        [mla mode] 

        It is evaluated from an *optimal* extraction, i.e. inverse-variance
        weighted least-square fit of the cross-dispersion profile (assumed
        Gaussian).  It is assumed the optimal extraction would do a perfect job
        on the signal; only the variance depends on detector parameters.

        If saturation, px in output spectrum [ADU] including a saturated
        detector px will have a NaN value and infinite variance.

        Parameters
        ----------
        flux: Array
            incident flux (arbitrary shape, e.g. (nlbda, ny, nx)) [ph/s/px]

        sigma: Array float
            cross-dispersion PSF width [px] (() or (nlbda,))

        width: int
            cross-dispersion aperture width [px]

        Returns
        -------
        Array, Array
            - spx spectra [ADU] 
            - variance [ADU²]
        """
        # Reshape photonflux2adu (() or (nlbda,)) to match flux's shape
        photonflux2ADU = complete_dims(self.photonflux2ADU, -np.ndim(flux))

        # Cross-dispersion weighted least-square ("optimal") extraction
        signal = flux * photonflux2ADU   # Total incident signal [ADU]

        # Normalized cross-dispersion profile, flux.shape + (width,)
        sigma = complete_dims(sigma, -np.ndim(flux))
        p = self._xdisp_profile(sigma=sigma, width=width)

        # Incident flux [ph/s], flux.shape + (width,) | take~50ms
        profiles = p * complete_dims(flux, 1)

        # Detector signal [ADU], shape is (width,) + flux.shape  | take ~200ms
        sig, var = self.estimate_pixel_signal(profiles, withdark=False)

        # One should have signal almost equal to sig.sum(axis=0)
        # (up to aperture corrections)
        # Saturated px with infinite variance will not contribute
        variance = 1 / (p**2 / var).sum(axis=-1)  # Variance on signal [ADU²]

        return signal, variance                  # flux.shape [ADU]
        
    def estimate_pixel_signal(self, flux, withdark=False,
                                  variance_model="default",
                                  saturation="default"):
        """ Estimate measured signal and variance from incident flux [ph/s].

        Signal includes incident flux and dark contribution if needed.
        Variance systematically includes incident flux and dark contributions,
        as well as impact of read-out noise and MACC mode.

        If saturation, detector px above saturation limit will have infinite
        variance and NaN signal.
        
        Parameters
        ----------
        flux: Array
            incident flux (arbitrary shape, e.g. (nlbda, ny, nx)) [ph/s/px]
        
        withdark: bool
            include dark contribution to output signal

        variance_model: bool
            variance model to use. Rauscher07 or Kubik20.
            If "default", self.variance_model is used.

        saturation: int
            if given, this attent to detect saturation. 
            if "default", self.saturation used.
            If None, no saturation
            
        Returns
        -------
        Array, Array
            - pixel signal [ADU]
            - variance [ADU²]
        """
        if variance_model == "default":
            variance_model = self.variance_model

        if saturation == "default":
            saturation = self.saturation
            
        # Reshape qe (() or (nlbda,)) to match flux's shape (nlbda, ..., width)
        qe = complete_dims(self.qe, -np.ndim(flux))

        # Variance estimators works with input flux in e-/s (not ph/s)
        flux_e = flux * qe  # [e-/s]
        variance_prop = dict(flux=flux_e, nmd=self.nmd, tframe=self.tframe,
                             ron=self.ron, dark=self.effective_dark, gain=self.gain)

        # Variance estimate in [ADU²]
        if  variance_model in ["Rauscher07", "Rauscher+07"]:
            variance = self._estimate_variance_rauscher07(**variance_prop)
            
        elif variance_model == "Kubik20":
            variance = self._estimate_variance_kubik20(**variance_prop)
        else:
            raise NotImplementedError(
                f"Unknown variance model {self.variance_model!r}.")

        signal = (flux_e + self.effective_dark) * self.electronpers2ADU  # Total signal [ADU]

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
            signal -= self.effective_dark * self.electronpers2ADU  # [ADU]

        return signal, variance  # [ADU], [ADU²]

        
    # - Internal
    @staticmethod
    def _estimate_variance_rauscher07(flux, nmd, tframe, ron, dark, gain):
        """
        Variance from Rauscher+2007 (corrected in Rauscher+2010).

        :param float flux: Flux in units of [e-/s]
        :param nmd: MACC(N=#group, M=#frame, D=#drop)
        :param float tframe: Frame time [s]
        :param float ron: Read noise per frame in units of [e-]
        :param float dark: Dark current in [e-/s]
        :param float gain: Gain [ADU/e-]
        :return: total variance [ADU²]

        .. Note:: in Rauscher+07, flux estimate (eqs 2, 3 & 4) is
           unbiased and naturally falls back to incident flux in
           the noiseless limit.
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
    def _estimate_variance_kubik20(flux, nmd, tframe, ron, dark, gain):
        """
        Variance from Kubik2020 (private communication).

        The variance is estimated from error propagation, less biased
        than likelihood estimate in Kubik+16 (Kubik 2020, private
        communication).

        :param float flux: Flux in units of [e-/s]
        :param nmd: MACC(N=#group, M=#frame, D=#drop)
        :param float tframe: Frame time [s]
        :param float ron: Read noise per frame in units of [e-]
        :param float dark: Dark current in [e-/s]
        :param float gain: Gain [ADU/e-]
        :return: total variance [ADU²]
        """
        n, m, d = nmd

        # Kubik20 works in ADU
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

    @staticmethod
    def _xdisp_profile(sigma=1, width=5, xdims=0):
        """
        Get normalized cross-dispersion profile.

        This uses the exact 1D Gaussian PSF integration over the px.

        If needed, the 1D profile can be embedded in a N-dim array of
        shape `(width,) + (1,)*xdims`.

        :param sigma: cross-dispersion PSF width [px]
        :param int width: cross-dispersion aperture width [px]
        :param int xdims: extra dimensions to be appended
        :return: normalized (embedded) profile
        """
        assert int(width) == width, \
            f"Non-integer x-disp. width not supported: {width}, {type(width)}."

        y_edges = np.r_[-width/2:+width/2:(width+1)*1j]  # (width+1,)
        sigma = complete_dims(sigma, 1)                  # sigma.shape + (1,)
        # sigma.shape + (width,)
        p = integ_gaussian1D_erf(y_edges, sigma, normed=True)
        if xdims:                   # N-dim. embedding: (width,) + (1, ...)
            p = p.reshape(p.shape + (1,) * xdims)

        return p
        
    # ================ #
    #   Properties     #
    # ================ #
    @property
    def effective_dark(self):
        """ sum of the detector dark current and that induced by thermal radiation in the detector vicinity  """
        return self.dark + self.thermal_dark
    
    @property
    def meta(self):
        """ metadata of the instance """
        return self._meta
    
    @property
    def macc(self):
        """ MACC description 'N:M:D'. (from self.nmd) """
        return ':'.join([ str(_) for _ in self.nmd ])
        
    @property
    def nframes(self):
        """ Total number of individual frames in the ramp. """
        n, m, d = self.nmd
        return (n * (m + d) - d)
    
    @property
    def tgroup(self):
        """ Time between two groups (including drops). """
        n, m, d = self.nmd
        return (m + d) * self.tframe

    @property
    def integration_time(self):
        """ Effective integration time (to be used for flux computations). """
        return self.get_integration_time()

    @property
    def exposure_time(self):
        """ Total integration time (to be used for sequences). """
        return self.get_exposure_time()

    @property
    def photonflux2ADU(self):
        """ Conversion factor from photon flux [ph/s] to integrated signal [ADU].

        This is effective exposure time [s] * QE [e-/ph] * gain [ADU/e-].
        """
        return self.integration_time * self.qe * self.gain

    @property
    def electronpers2ADU(self):
        """ Conversion factor from electron/s to integrated signal [ADU].

        This is effective exposure time [s] * gain [ADU/e-].
        """
        return self.integration_time * self.gain
