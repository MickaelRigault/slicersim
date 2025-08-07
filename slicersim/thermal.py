"""
Computation of thermal signal (solid angles, black body spectrum).
"""
from scipy.integrate import quad_vec
from astropy import constants
import numpy as np
from copy import deepcopy


def fratio_to_solidangle(fratio, geometry="circular"):
    """ Convert f-ratio to solid angle.

    Parameters
    ----------
    fratio : float or array-like
        The f-ratio of the optical system. If a single value is given, it is
        assumed to be the same for both x and y directions. If an array-like
        of two values is given, it is assumed to be (fratio_y, fratio_x).

    geometry: str
        shape of the optical element. Could be square or circular. This
        affects the normalisation coefficient. 

    Returns
    -------
    float
        The solid angle in steradians.
    """
    
    if np.ndim(fratio) == 0:
        fratio_y = fratio_x = fratio
    else:
        fratio_y, fratio_x = fratio

    # will match the fratio shape if possible. Break if not.
    geometry = np.broadcast_to(geometry, fratio_y.shape)
    if not np.all([geom in ["circular", "square"] for geom in geometry]):
        raise NotImplementedError("only 'circular' or 'square' geometries implemented: {geometry=} given.")

    # square by default
    coefs = np.ones(fratio_y.shape) # square
    coefs[geometry=="circular"] = np.pi / 4
    
    return coefs / (fratio_y*fratio_x)


class ThermalOptics():
    """Class to compute the thermal radiation of optical elements.

    Parameters
    ----------
    temperature : float or array-like
        The temperature of the optical elements in Kelvin.
    emissivity : float or array-like
        The emissivity of the optical elements.
    fratio : float or array-like, optional
        The f-ratio of the optical system. If a single value is given, it is
        assumed to be the same for both x and y directions. If an array-like
        of two values is given, it is assumed to be (fratio_y, fratio_x).
        Default is None.
    solid_angle : float or array-like, optional
        The solid angle in steradians. If not given, it is computed from the
        f-ratio. Default is None.
    nelements : int, optional
        The number of optical elements. Default is 1.
    geometry : str, optional
        The geometry of the optical elements. Can be "circular" or "square".
        Default is "circular".
    meta : dict, optional
        A dictionary of metadata. Default is {}.

    """
    mutable_parameters = ['temperature', 'emissivity'] # lbda

    def __init__(self, temperature, emissivity, fratio=None, solid_angle=None,
                     nelements=1, geometry="circular",
                     meta={}):
        """ """
        self._temperature = np.atleast_1d(temperature)[:, None].astype(float)
        self._emissivity = np.atleast_1d(emissivity)[:, None].astype(float)
        self._nelements = np.atleast_1d(nelements)[:, None].astype(int)
        if fratio is not None:
            self._fratio = np.atleast_2d(fratio).astype(float)
        else:
            self._fratio = None

        if solid_angle is not None:
            self._solid_angle = np.atleast_1d(solid_angle).astype(float)
        else:
            self._solid_angle = None
        
        self._geometry = np.atleast_1d(geometry).astype(str)
        self._meta = deepcopy(meta)
    
    @classmethod
    def from_config(cls, config, no_solidangle_ok=True):
        """Create a `ThermalOptics` object from a configuration dictionary.

        Parameters
        ----------
        config : dict
            A dictionary of configuration parameters.
        no_solidangle_ok : bool, optional
            If True, do not raise a warning if both fratio and solid_angle are
            None. Default is True.

        Returns
        -------
        ThermalOptics
            A `ThermalOptics` object.

        """
        
        temperature = config.get("temperature", None)        
        emissivity = config.get("emissivity", None)
        if temperature is None or emissivity is None:
            raise ValueError(f" one of {temperature=} and/or {emissivity=} is None. They are mandatory.")

        fratio = config.get("fratio", None)
        solid_angle = config.get("solid_angle", None)
        
        if not no_solidangle_ok and (fratio is None and solid_angle is None):
            warnings.warn("Either fratio or solid_angle must be given. Both are None, some computations will not work.")

        if fratio is not None and solid_angle is not None:
            warnings.warn("both fratio and solid_angle are given, solid_angle will then *not* be derived from input fratio")
        
        nelements = config.get("nelements", 1)
        geometry = config.get("geometry", "circular")
        return cls(temperature=temperature, emissivity=emissivity, 
                   fratio=fratio, solid_angle=solid_angle,
                   nelements=nelements, geometry=geometry,
                  meta=config)

    def update(self, reset_thermal=True, **kwargs):
        """Update the mutable parameters of the object.

        Parameters
        ----------
        reset_thermal : bool, optional
            If True, reset the thermal radiation object. Default is True.
        **kwargs
            The mutable parameters to update.

        """
        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue
            if k in ["temperature", "emissivity"]:
                setattr(self, f"_{k}", np.atleast_2d(v).astype(float) )
            else:
                warnings.warn(f"update not implemented for {k}. *Ignored* ")

        if reset_thermal:
            self._thermal = None
            
    # ============= #
    #   Methods     #
    # ============= #
    def get_signal(self, lbda_bin, area, solid_angle=None, qe=None):
        """ Calculate the thermal radiation signal for given parameters.
        
        Parameters
        ----------
        lbda_bin : array-like
            The wavelength bin(s) in Angtrom. Can be a 1D or 2D array.
            - 1D: [lbda_min, lbda_max]
            - 2D: [[lbda_min, lbda_max],[lbda_min, lbda_max],...]

        area : float
            The area in square meters.

        solid_angle : float, None
            if None, self.solid_angle is used
            The solid angle in steradians ; it corresponds to the angular area [sr] of the reception 
            (e.g. spaxel_size in rad*rad)

        Returns
        -------
        float or numpy.ndarray
            The calculated flux signal in photons per second, integrated over the bandwidth.
        """
        if solid_angle is None:
            solid_angle = self.solid_angle

        solid_angle = np.atleast_1d(solid_angle)[:,None].astype(float)
        return self.thermal.get_signal(lbda_bin, area, solid_angle=solid_angle, qe=qe)
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def temperature(self):
        """The temperature of the components [K]."""
        return self._temperature

    @property
    def emissivity(self):
        """The emissivity of the components."""
        return self._emissivity

    @property
    def nelements(self):
        """The number of elements at a given temperature and emissivity."""
        return self._nelements

    @property
    def fratio(self):
        """The f-ratio of the optical system."""
        return self._fratio

    @property
    def geometry(self):
        """The geometry of the optical elements."""
        return self._geometry
        
    @property
    def solid_angle(self):
        """The solid angle of the optical system [sr].

        If not provided at initialization, it is computed from the f-ratio.
        """
        if self._solid_angle is None:
            if self.fratio is None: 
                raise ValueError("both fratio and solid_angle are None. Cannot derive or get solid_angle")
            self._solid_angle = fratio_to_solidangle( fratio = self.fratio.T,
                                                geometry = self._geometry)
            
        return self._solid_angle

    @property
    def thermal(self):
        """The thermal radiation object."""
        if not hasattr(self, "_thermal") or self._thermal is None:
            self._thermal = ThermalRadiation(self.temperature, self.emissivity, self.nelements)
            
        return self._thermal

class ThermalRadiation():
    """A class to simulate the thermal radiation of telescope and instrument components.

    This class calculates the thermal radiation signal based on the black body radiation
    principles. It allows for the computation of photon flux within specified wavelength
    ranges, taking into account the temperature and emissivity of the components.

    Parameters
    ----------
    temperature : float, numpy.ndarray
        The temperature of the components in Kelvin.
    emissivity : float, numpy.ndarray
        The emissivity of the components.
    nelements : int, optional
        The number of elements. Default is 1.

    Attributes
    ----------
    temperature : float, numpy.ndarray
        The temperature of the components in Kelvin.
        
    emissivity : float, numpy.ndarray
        The emissivity of the components.
    """
    def __init__(self, temperature, emissivity, nelements=1):
        """ Initialize the ThermalRadiation class with temperature and emissivity.

        Parameters
        ----------
        temperature : float, numpy.ndarray
            The temperature of the components in Kelvin.
        
        emissivity : float, numpy.ndarray
            The emissivity of the components.
        """
        self._temperature = np.atleast_2d(temperature)
        self._emissivity = np.atleast_2d(emissivity)
        self._nelements = np.atleast_2d(nelements)
        
    def get_signal(self, lbda_bin, area, solid_angle, qe=None):
        """ Calculate the thermal radiation signal for given parameters.
        
        Parameters
        ----------
        lbda_bin : array-like
            The wavelength bin(s) in Angtrom. Can be a 1D or 2D array.
            - 1D: [lbda_min, lbda_max]
            - 2D: [[lbda_min, lbda_max],[lbda_min, lbda_max],...]

        area : float
            The area in square meters.

        solid_angle : float
            The solid angle in steradians ; it corresponds to the angular area [sr] of the reception 
            (e.g. spaxel_size in rad*rad)

        qe: None, float, func
            Quantum efficiency to convert ph->e-

        Returns
        -------
        float or numpy.ndarray
            The calculated flux signal in photons (or e- if qe!=1) per second, integrated over the bandwidth.
        """
        # allows [[lbda_min, lbda_max], [lbda_min, lbda_max], ...]
        int_flux = self.get_integrated_blackbody_flux(lbda_bin, qe=qe)

        # int_flux in [{photon,e-}/s / sr /m²]
        return int_flux * solid_angle * area * self.emissivity * self.nelements  # [{photon,e-}/s] integrated over bandwidth

    def get_blackbody_flux(self, lbda, qe=1):
        """ Calculate the blackbody photon flux for a given wavelength.

        Parameters
        ----------
        lbda : float, numpy.array
            The wavelength in Angstrong.

        qe: None, float, array, func
            Quantum efficiency to convert ph->e-
            - None: returned unit: ph/s
            - function: qe <= qe(lbda)
            - float or array: should broadcast with lbda.shape
            
        Returns
        -------
        float, numpy.array
            The blackbody photon (or e-) flux in [{photon, e-}/s/sr/m²/A]
        """
        return self._blackbody_flux(lbda, self.temperature, qe=qe)
        
    def get_integrated_blackbody_flux(self, lbda_bin, allow_trapez=True, qe=None):
        """ Calculate the blackbody photon (or e-) flux intergrated over given wavelength bins.
        
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

        qe: None, float, array, func
            Quantum efficiency to convert ph->e-
            - None: returned unit: ph/s
            - function: qe <= qe(lbda)
            - float or array: should broadcast with lbda.shape
        
        Returns
        -------
        float, numpy.array
            The blackbody photon (or e- if qe!=1) flux integrated in given band in [{photon,e-}/s/sr/m²]
        """
        # 1d-boundaries: let's use exact method.
        if np.ndim(lbda_bin) == 1: # 
            int_flux = self._get_flux1d(self.temperature, *lbda_bin, qe=qe)
            
        elif np.ndim(lbda_bin) == 2:
            # trapeze method: muuuuch faster. Correct at the per-mil level.
            if allow_trapez:
                # approximated method, but vectorized.
                delta_bin = np.diff(lbda_bin).squeeze()
                lbda_mid = np.mean(lbda_bin, axis=-1)
                int_flux = self.get_blackbody_flux(lbda_mid, qe=qe) * delta_bin
                
            else:
                # exact method, but require for loop
                int_flux = np.hstack( [self._get_flux1d(self.temperature, lbda_min, lbda_max, qe=qe)
                                           for lbda_min, lbda_max in lbda_bin] )

        else:
            raise ValueError(f"ndim of input lbda_bin must be 1 or 2 {np.ndim(lbda_bin)=} ")
        
        return int_flux

    # ------------ #
    #  Internal    #
    # ------------ #    
    @classmethod
    def _get_flux1d(cls, temperature, lbda_min, lbda_max, qe=None):
        """ Calculate the flux within a specified wavelength range.

        Parameters
        ----------
        temperature : float
            The temperature in Kelvin.
            
        lbda_min : float
            The minimum wavelength in Angstrong.
            
        lbda_max : float
            The maximum wavelength in Angstrong.

        qe: None, float, array, func
            Quantum efficiency to convert ph->e-
            - None: returned unit: ph/s
            - function: qe <= qe(lbda)
            - float: unit qe accross all wavelengths

        Returns
        -------
        float
            The calculated flux within the specified wavelength range. [{photon, e-}/s/sr/m²]
        """
        # quad_vec is the vectorized version of quad. Needed as temperature could be an array
        flux, _ = quad_vec(cls._blackbody_flux, lbda_min, lbda_max, args=(temperature, qe))
        return flux

    
    @classmethod
    def _blackbody_flux(cls, lbda, temperature, qe=None):
        """ Calculate the blackbody photon flux for a given wavelength and temperature.
         
        Parameters
        ----------
        lbda : float
            The wavelength in Angstrong.

        temperature : float
            The temperature in Kelvin.

        qe: None, float, array, func
            Quantum efficiency to convert ph->e-
            - None: returned unit: ph/s
            - function: qe => qe(lbda)
            - float or array: should broadcast with lbda.shape

        Returns
        -------
        float
            The blackbody flux in photons (or e-) per second per steradian per square meter per micrometer.
        """
        flux_ph = cls._blackbody_photonflux(lbda, temperature)
        
        if qe is None:
            flux = flux_ph
            
        elif callable(qe):
            flux = flux_ph * qe(lbda)
            
        else: # must broadcast
            flux = flux_ph * qe
            
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
        """The temperature of the components [K]."""
        return self._temperature

    @property
    def emissivity(self):
        """The emissivity of the components."""
        return self._emissivity

    @property
    def nelements(self):
        """The number of elements at a given temperature and emissivity."""
        return self._nelements
    
