"""Thermal emission module.

This module computes the thermal (black body) emission of optical elements
received by a detector or a spaxel: black body photon fluxes, solid angles
from f-ratios and integrated thermal signals.
"""
import warnings
from copy import deepcopy

import numpy as np
from astropy import constants
from scipy.integrate import quad_vec


def get_source_radiation(lbda, temperature, emissivity=1, angular_coefficient=1):
    """Calculate the photon flux of a grey body.

    This is ``emissivity * blackbody_photonflux(lbda, temperature)``.

    Parameters
    ----------
    lbda : float or array_like
        Wavelength(s) in Angstrom.
    temperature : float
        Temperature in K.
    emissivity : float, optional
        Emissivity of the source, between 0 and 1. Default is 1.
    angular_coefficient : float, optional
        Multiplicative coefficient associated with the opening angle of the
        source. Currently not used. Default is 1.

    Returns
    -------
    float or numpy.ndarray
        Photon flux in ph/s/sr/m^2/A.

    Raises
    ------
    ValueError
        If ``emissivity`` is not between 0 and 1.

    See Also
    --------
    blackbody_photonflux : Black body photon flux.
    """
    if emissivity<0 or emissivity>1:
        raise ValueError(f"emissivity is defined in between 0 and 1,{emissivity=} given")

    return emissivity * blackbody_photonflux(lbda, temperature)

def blackbody_photonflux(lbda, temperature):
    """Calculate the black body photon flux (Planck's law in photon units).

    Parameters
    ----------
    lbda : float or array_like
        Wavelength(s) in Angstrom.
    temperature : float or array_like
        Temperature(s) in K. Must broadcast with ``lbda``.

    Returns
    -------
    float or numpy.ndarray
        Black body photon flux in ph/s/sr/m^2/A.
    """
    c = constants.c.value     # [m/s]
    hc_over_kB = ( (constants.h*constants.c) / (constants.k_B) ).value  # [K.m]
    
    bbflux_per_m = 2 * c / ((lbda * 1e-10)**4 *
                     (np.exp(hc_over_kB / (lbda*1e-10 * temperature)) - 1))  # [ph/s/sr/m²/m]
                     
    return bbflux_per_m * 1e-10  # [photon/s/sr/m²/A] | 1e-10 for A

def fratio_to_solidangle(fratio, geometry="circular"):
    """Convert f-ratios into solid angles.

    The solid angle is ``coef / (fratio_y * fratio_x)``, with ``coef = pi / 4``
    for a circular pupil and ``coef = 1`` for a square one.

    Parameters
    ----------
    fratio : numpy.ndarray
        F-ratio(s) of the optical elements. If 0-dimensional, the same value is
        used for both the y and x directions. Otherwise, the first axis must
        have length 2 and holds ``(fratio_y, fratio_x)``, e.g. an array of
        shape ``(2, nelements)``.
    geometry : str or array_like of str, optional
        Pupil shape of the optical element(s), ``"circular"`` or
        ``"square"``; must broadcast with ``fratio_y``.
        Default is ``"circular"``.

    Returns
    -------
    numpy.ndarray
        Solid angle(s) in steradian.

    Raises
    ------
    NotImplementedError
        If ``geometry`` contains values other than ``"circular"`` or
        ``"square"``.
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


class ThermalOptics:
    """Thermal emission of a set of optical elements.

    Each optical element (or group of identical elements) is described by a
    temperature, an emissivity, a number of elements and an f-ratio (or a
    solid angle). Array inputs define one entry per element.

    Parameters
    ----------
    temperature : float or array_like
        Temperature(s) of the optical elements in K.
    emissivity : float or array_like
        Emissivity of the optical elements, between 0 and 1.
    fratio : float or array_like or None, optional
        F-ratio(s) of the optical elements, used to compute `solid_angle`.
        Either one value per element (same in both directions) or one
        ``(fratio_y, fratio_x)`` pair per element. Default is None.
    solid_angle : float or array_like or None, optional
        Solid angle(s) in steradian. If None, it is computed from ``fratio``.
        Default is None.
    nelements : int or array_like of int, optional
        Number of optical elements sharing each temperature and emissivity.
        Default is 1.
    geometry : str or array_like of str, optional
        Pupil shape of the optical elements, ``"circular"`` or ``"square"``.
        Default is ``"circular"``.
    meta : dict, optional
        Metadata dictionary (e.g. the input configuration). Default is an
        empty dict.

    Attributes
    ----------
    mutable_parameters : list of str
        Parameters that can be changed after initialization with `update`.
    """
    mutable_parameters = ['temperature', 'emissivity', 'nelements'] # lbda

    def __init__(self, temperature, emissivity, fratio=None, solid_angle=None,
                     nelements=1, geometry="circular",
                     meta={}):
        """Initialize the ThermalOptics (see the class docstring for parameters)."""
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
        self._meta_in = deepcopy(self._meta)
    
    @classmethod
    def from_config(cls, config, no_solidangle_ok=True):
        """Create a `ThermalOptics` from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary. It must contain ``"temperature"`` and
            ``"emissivity"``, and may contain ``"fratio"``, ``"solid_angle"``,
            ``"nelements"`` and ``"geometry"`` (see `ThermalOptics`). The full
            dictionary is stored as `meta`.
        no_solidangle_ok : bool, optional
            If True, do not warn when both ``"fratio"`` and ``"solid_angle"``
            are missing. Default is True.

        Returns
        -------
        ThermalOptics
            The thermal optics instance.

        Raises
        ------
        ValueError
            If ``"temperature"`` or ``"emissivity"`` is missing.

        Warns
        -----
        UserWarning
            If both ``"fratio"`` and ``"solid_angle"`` are given (the solid
            angle is then not derived from the f-ratio), or if both are missing
            and ``no_solidangle_ok`` is False.
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
        """Update the mutable parameters.

        See `mutable_parameters` for the list of accepted names.

        Parameters
        ----------
        reset_thermal : bool, optional
            If True, reset the cached `thermal` object so that it is rebuilt
            with the new parameters. Default is True.
        **kwargs
            Mutable parameters to update.

        Warns
        -----
        UserWarning
            If a given parameter is not mutable (it is then ignored).
        """
        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue
            
            if k in ["temperature", "emissivity", "nelements"]:
                setattr(self, f"_{k}", np.atleast_2d(v).astype(float) )
            else:
                warnings.warn(f"update not implemented for {k}. *Ignored* ")

        if reset_thermal:
            self._thermal = None
            
    # ============= #
    #   Methods     #
    # ============= #
    def get_signal(self, lbda_bin, area, solid_angle=None, qe=None):
        """Calculate the thermal signal received, integrated over wavelength bins.

        Parameters
        ----------
        lbda_bin : array_like
            Wavelength bin(s) in Angstrom, either 1D ``[lbda_min, lbda_max]``
            or 2D ``[[lbda_min, lbda_max], [lbda_min, lbda_max], ...]``.
        area : float
            Collecting area in m^2.
        solid_angle : float or array_like or None, optional
            Solid angle(s) in steradian under which the emission is received
            (e.g. the spaxel angular area). If None, `solid_angle` is used.
            Default is None.
        qe : float or callable or None, optional
            Quantum efficiency in e-/ph, as a constant or a function of
            wavelength (in Angstrom). If None, the signal is in photons.
            Default is None.

        Returns
        -------
        numpy.ndarray
            Thermal signal in ph/s (or e-/s if ``qe`` is given) integrated over
            each bin, with one row per optical element.

        See Also
        --------
        ThermalRadiation.get_signal : Underlying computation.
        """
        if solid_angle is None:
            solid_angle = self.solid_angle

        solid_angle = np.atleast_1d(solid_angle)[:,None].astype(float)
        return self.thermal.get_signal(lbda_bin, area, solid_angle=solid_angle, qe=qe)
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def meta(self):
        """Metadata dictionary of the instance."""
        return self._meta
    
    @property
    def temperature(self):
        """Temperature of the optical elements in K, shape ``(n, 1)``."""
        return self._temperature

    @property
    def emissivity(self):
        """Emissivity of the optical elements, shape ``(n, 1)``."""
        return self._emissivity

    @property
    def nelements(self):
        """Number of elements sharing each temperature and emissivity."""
        return self._nelements

    @property
    def fratio(self):
        """F-ratio(s) of the optical elements, or None."""
        return self._fratio

    @property
    def geometry(self):
        """Pupil shape(s) of the optical elements (``"circular"`` or ``"square"``)."""
        return self._geometry
        
    @property
    def solid_angle(self):
        """Solid angle(s) of the optical elements in steradian.

        If not provided at initialization, it is computed from `fratio` and
        `geometry` (see `fratio_to_solidangle`). A `ValueError` is raised if
        neither is available.
        """
        if self._solid_angle is None:
            if self.fratio is None: 
                raise ValueError("both fratio and solid_angle are None. Cannot derive or get solid_angle")
            self._solid_angle = fratio_to_solidangle( fratio = self.fratio.T,
                                                geometry = self._geometry)
            
        return self._solid_angle

    @property
    def thermal(self):
        """`ThermalRadiation` object built from the current parameters."""
        if not hasattr(self, "_thermal") or self._thermal is None:
            self._thermal = ThermalRadiation(self.temperature, self.emissivity, self.nelements)
            
        return self._thermal

class ThermalRadiation:
    """Black body thermal radiation of telescope and instrument components.

    This class computes the thermal photon (or electron) flux of grey bodies,
    integrated over wavelength bins, given the temperature, emissivity and
    number of the emitting components.

    Parameters
    ----------
    temperature : float or numpy.ndarray
        Temperature(s) of the components in K.
    emissivity : float or numpy.ndarray
        Emissivity of the components, between 0 and 1.
    nelements : int or numpy.ndarray, optional
        Number of components sharing each temperature and emissivity.
        Default is 1.
    """
    def __init__(self, temperature, emissivity, nelements=1):
        """Initialize the ThermalRadiation (see the class docstring for parameters)."""
        self._temperature = np.atleast_2d(temperature)
        self._emissivity = np.atleast_2d(emissivity)
        self._nelements = np.atleast_2d(nelements)
        
    def get_signal(self, lbda_bin, area, solid_angle, qe=None):
        """Calculate the thermal signal received, integrated over wavelength bins.

        This is the integrated black body flux (see
        `get_integrated_blackbody_flux`) times ``solid_angle * area *
        emissivity * nelements``.

        Parameters
        ----------
        lbda_bin : array_like
            Wavelength bin(s) in Angstrom, either 1D ``[lbda_min, lbda_max]``
            or 2D ``[[lbda_min, lbda_max], [lbda_min, lbda_max], ...]``.
        area : float
            Collecting area in m^2.
        solid_angle : float or numpy.ndarray
            Solid angle(s) in steradian under which the emission is received
            (e.g. the spaxel angular area).
        qe : float or callable or None, optional
            Quantum efficiency in e-/ph, as a constant or a function of
            wavelength (in Angstrom). If None, the signal is in photons.
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            Thermal signal in ph/s (or e-/s if ``qe`` is given) integrated over
            each bin.
        """
        # allows [[lbda_min, lbda_max], [lbda_min, lbda_max], ...]
        int_flux = self.get_integrated_blackbody_flux(lbda_bin, qe=qe)

        # int_flux in [{photon,e-}/s / sr /m²]
        return int_flux * solid_angle * area * self.emissivity * self.nelements  # [{photon,e-}/s] integrated over bandwidth

    def get_blackbody_flux(self, lbda, qe=1):
        """Calculate the black body flux of the components' temperatures.

        Parameters
        ----------
        lbda : float or numpy.ndarray
            Wavelength(s) in Angstrom.
        qe : float or numpy.ndarray or callable or None, optional
            Quantum efficiency in e-/ph used to convert photons into
            electrons: None (flux in photons), a function of wavelength, or
            a float or array broadcasting with ``lbda``. Default is 1.

        Returns
        -------
        float or numpy.ndarray
            Black body photon (or electron) flux in ph/s/sr/m^2/A
            (or e-/s/sr/m^2/A).
        """
        return self._blackbody_flux(lbda, self.temperature, qe=qe)
        
    def get_integrated_blackbody_flux(self, lbda_bin, allow_trapez=True, qe=None):
        """Calculate the black body flux integrated over wavelength bins.

        Parameters
        ----------
        lbda_bin : array_like
            Wavelength bin(s) in Angstrom, either 1D ``[lbda_min, lbda_max]``
            or 2D ``[[lbda_min, lbda_max], [lbda_min, lbda_max], ...]``.
        allow_trapez : bool, optional
            If True, approximate the integral by the flux at the bin centre
            times the bin width (per-mil level accuracy, much faster). Only
            used for 2D ``lbda_bin``; 1D bins are always integrated exactly.
            Default is True.
        qe : float or numpy.ndarray or callable or None, optional
            Quantum efficiency in e-/ph used to convert photons into
            electrons: None (flux in photons), a function of wavelength, or
            a float or array broadcasting with the wavelengths.
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            Integrated black body photon (or electron) flux in ph/s/sr/m^2
            (or e-/s/sr/m^2).

        Raises
        ------
        ValueError
            If ``lbda_bin`` is neither 1D nor 2D.
        """
        # 1d-boundaries: let's use exact method.
        if np.ndim(lbda_bin) == 1:
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
            The minimum wavelength in Angstrom.
            
        lbda_max : float
            The maximum wavelength in Angstrom.

        qe : float or numpy.ndarray or callable or None, optional
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
            The wavelength in Angstrom.

        temperature : float
            The temperature in Kelvin.

        qe : float or numpy.ndarray or callable or None, optional
            Quantum efficiency to convert ph->e-
            - None: returned unit: ph/s
            - function: qe => qe(lbda)
            - float or array: should broadcast with lbda.shape

        Returns
        -------
        float
            The blackbody flux in photons (or e-) per second per steradian per square meter per Angstrom.
        """
        flux_ph = blackbody_photonflux(lbda, temperature)
        
        if qe is None:
            flux = flux_ph
            
        elif callable(qe):
            flux = flux_ph * qe(lbda)
            
        else: # must broadcast
            flux = flux_ph * qe
            
        return flux
    
    # ============ #
    #  Properties  #
    # ============ #
    @property
    def temperature(self):
        """Temperature of the components in K."""
        return self._temperature

    @property
    def emissivity(self):
        """Emissivity of the components."""
        return self._emissivity

    @property
    def nelements(self):
        """Number of components sharing each temperature and emissivity."""
        return self._nelements
    
