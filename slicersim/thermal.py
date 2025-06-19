"""
Computation of thermal signal (solid angles, black body spectrum).
"""
from scipy.integrate import quad_vec
from astropy import constants
import numpy as np


def get_detector_dark_current(lbda_cutoff, temperature, px_size, type='HgCdTe'):
    """ Calculate the dark current (thermal noise) for a detector.

    This function computes the dark current based on the detector's cutoff wavelength,
    temperature, pixel size, and material type. The calculation follows models described
    in the literature for HgCdTe and InAs detectors.

    Parameters
    ----------
    lbda_cutoff : float
        Detector cut-off (upper) wavelength in micrometers [µm].

    temperature : float
        Detector temperature in Kelvin [K].

    px_size : float
        Pixel size in micrometers [µm].

    type : str, optional
        Detector material type, either 'HgCdTe' or 'InAs' (default is 'HgCdTe').

    Returns
    -------
    float
        Dark current in electrons per second per pixel [e/s/px].

    References
    ----------
    Tennant et al. 2008, Journal of Electronic Materials, 37, 1406T
    Tennant, 2010, Journal of Electronic Materials, 39, 1030T
    O'Loughlin, PhD Thesis, 2020
    """

    if type == 'HgCdTe':
        # Tennant+08 original parameters for HgCdTe
        J0 = 8367.00001853855              # [A/cm²]
        Pwr = 0.544071281108481
        C = -1.16239134096245
        lamb_scale = 0.200847413564122     # [µm]
        lamb_threshold = 4.63513642316149  # [µm]
        
    elif type == 'InAs':
        # O'Loughlin parameters for InAs
        J0 = 5315.034051              # [A/cm²]
        Pwr = 0.569561634
        C = -1.140270099
        lamb_scale = 0.21507853       # [µm]
        lamb_threshold = 4.843406496  # [µm]
    else:
        raise NotImplementedError(f"Unknown detector type {type!r}.")

    kB = 1.380649e-23             # Boltzmann constant [J/K]
    q = 1.602176634e-19           # electron charge [C]
    amp2e = 6.2415091e+18         # A to e/s

    apx = (px_size * 1e-4)**2     # [cm²]

    lamb_e = np.where(
        lbda_cutoff >= lamb_threshold,
        lbda_cutoff,
        lbda_cutoff / (1 - (lamb_scale/lbda_cutoff - lamb_scale/lamb_threshold)**Pwr))
    J = J0 * np.exp((C * 1.24 * q / (lamb_e * kB * temperature)))  # [A/cm²]

    return J * apx * amp2e      # [e/s/px]




class ThermalRadiation():
    """A class to simulate the thermal radiation of telescope and instrument components.

    This class calculates the thermal radiation signal based on the black body radiation
    principles. It allows for the computation of photon flux within specified wavelength
    ranges, taking into account the temperature and emissivity of the components.

    Attributes
    ----------
    temperature : float, numpy.ndarray
        The temperature of the components in Kelvin.
        
    emissivity : float, numpy.ndarray
        The emissivity of the components.
    """
    def __init__(self, temperature, emissivity):
        """ Initialize the ThermalRadiation class with temperature and emissivity.

        Parameters
        ----------
        temperature : float, numpy.ndarray
            The temperature of the components in Kelvin.
        
        emissivity : float, numpy.ndarray
            The emissivity of the components.
        """
        self._temperature = np.atleast_1d(temperature)[:, None].astype(float)
        self._emissivity = np.atleast_1d(emissivity)[:, None].astype(float)

    def get_signal(self, solid_angle, area, lbda_bin):
        """ Calculate the thermal radiation signal for given parameters.

        Parameters
        ----------
        solid_angle : float
            The solid angle in steradians.
            
        area : float
            The area in square meters.
            
        lbda_bin : array-like
            The wavelength bin(s) in Angtrom. Can be a 1D or 2D array.
            - 1D: [lbda_min, lbda_max]
            - 2D: [[lbda_min, lbda_max],[lbda_min, lbda_max],...]

        Returns
        -------
        float or numpy.ndarray
            The calculated flux signal in photons per second, integrated over the bandwidth.
        """
        # allows [[lbda_min, lbda_max], [lbda_min, lbda_max], ...]
        int_flux = self.get_integrated_blackbody_photonflux(lbda_bin)

        # int_flux in [photon/s / sr /m²]
        return int_flux * solid_angle * area * self.emissivity  # [photon/s] integrated over bandwidth

    def get_blackbody_photonflux(self, lbda):
        """ Calculate the blackbody photon flux for a given wavelength.

        Parameters
        ----------
        lbda : float, numpy.array
            The wavelength in Angstrong.

        Returns
        -------
        float, numpy.array
            The blackbody photon flux in [photon/s/sr/m²/A]
        """
        return self._blackbody_photonflux(lbda, self.temperature)
        
    def get_integrated_blackbody_photonflux(self, lbda_bin, allow_trapez=True):
        """ Calculate the blackbody photon flux intergrated over given wavelength bins.
        
        Parameters
        ----------
        lbda_bin : array-like
            The wavelength bin(s) in Angtrom. Can be a 1D or 2D array.
            - 1D: [lbda_min, lbda_max]
            - 2D: [[lbda_min, lbda_max],[lbda_min, lbda_max],...]
            
        allow_trapez : bool
            Whether to allow the use of trapez to approximate the integral.
            This is used only for 2D-lbda_bin to significantly speed the code.
            (per-mil level approximation error).
        
        Returns
        -------
        float, numpy.array
            The blackbody photon flux integrated in given band in [photon/s/sr/m²]
        """
        # 1d-boundaries: let's use exact method.
        if np.ndim(lbda_bin) == 1: # 
            int_flux = self._get_flux1d(self.temperature, *lbda_bin)
            
        elif np.ndim(lbda_bin) == 2:
            # trapeze method: muuuuch faster. Correct at the per-mil level.
            if allow_trapez:
                # approximated method, but vectorized.
                delta_bin = np.diff(lbda_bin).squeeze()
                lbda_mid = np.mean(lbda_bin, axis=-1)
                int_flux = self.get_blackbody_photonflux(lbda_mid) * delta_bin
                
            else:
                # exact method, but require for loop
                int_flux = np.hstack( [self._get_flux1d(self.temperature, lbda_min, lbda_max)
                                           for lbda_min, lbda_max in lbda_bin] )

        else:
            raise ValueError(f"ndim of input lbda_bin must be 1 or 2 {np.ndim(lbda_bin)=} ")
        
        return int_flux

    # ------------ #
    #  Internal    #
    # ------------ #    
    @classmethod
    def _get_flux1d(cls, temperature, lbda_min, lbda_max):
        """ Calculate the flux within a specified wavelength range.

        Parameters
        ----------
        temperature : float
            The temperature in Kelvin.
            
        lbda_min : float
            The minimum wavelength in Angstrong.
            
        lbda_max : float
            The maximum wavelength in Angstrong.

        Returns
        -------
        float
            The calculated flux within the specified wavelength range. [photon/s/sr/m²]
        """
        # quad_vec is the vectorized version of quad. Needed as temperature could be an array
        flux, _ = quad_vec(cls._blackbody_photonflux, lbda_min, lbda_max, args=(temperature,))
        return flux

    @staticmethod
    def _blackbody_photonflux(lbda, temperature):
        """ Calculate the blackbody photon flux for a given wavelength and temperature.

        Parameters
        ----------
        lbda : float
            The wavelength in Angstrong.
        temperature : float
            The temperature in Kelvin.

        Returns
        -------
        float
            The blackbody photon flux in photons per second per steradian per square meter per micrometer.
        """
        c = constants.c.value     # [m/s]
        hc_over_kB = ( (constants.h*constants.c) / (constants.k_B) ).value  # [K.m]
        
        l = 2 * c / ((lbda * 1e-10)**4 *
                (np.exp(hc_over_kB / (lbda*1e-10 * temperature)) - 1))  # [ph/s/sr/m²/m]
        return l * 1e-10  # [photon/s/sr/m²/A]

    # ============ #
    #  Properties  #
    # ============ #
    @property
    def temperature(self):
        """Get the temperature [in K] of the components """
        return self._temperature

    @property
    def emissivity(self):
        """Get the emissivity of the components. """
        return self._emissivity


    
