"""
Mirror module, to simulate:

* mirror collecting surface and thermal emission


.. autosummary::

   Mirror
"""

__author__ = "Mickael Rigault <m.rigault@ip2i.in2p3.fr>"

import warnings
import numpy as np
from astropy import units


class Mirror():
    """
    Mirror data class.
    """

    mutable_parameters = ['temperature', 'emissivity',
                          "diameter_ext", "diameter_int",
                          "surface"]

    def __init__(self, surface=None, temperature=0, emissivity=0, meta={}):
        """ """
        
        self._surface = surface
        # update a copy of the meta
        meta = meta.copy()
        # make sure temperature and emissivity are given.
        meta["temperature"] = temperature
        meta["emissivity"] = emissivity

        self._meta = meta.copy()
        self._meta_in = meta.copy()        

        if surface is None and np.any([k not in self.meta for k in ["diameter_ext", "diameter_int"]]):
            warnings.warn("Surface not given and 'diameter_ext' and/or 'diameter_int' are unknown.")
                                          
        
    @classmethod
    def from_config(cls, config):
        """ """
        surface = config.get("surface", None) # if None, diameter_ext and diameter_int used.
        temperature = config.get("temperature", 0)
        emissivity = config.get("emissivity", 0)

        return cls(surface=surface, temperature=temperature, emissivity=emissivity,
                    meta=config # core information stored in meta
                    )
    
    def __str__(self):

        s = f"Primary Mirror: {self.surface:.0f} m²"
        if self.nelements > 1:
            s += f": {self.nelements} mirrors (emissivity in %):"
            for temp, emmi in zip(self.temperature, self.emissivity):
                s += f" {temp:.0f}K ({emmi:.2%}), "      
        elif self.temperature:
            s += f" at {self.temperature:.0f} K, emissivity: {self.emissivity:.2f}"
        else:
            s += ", no thermal emission"

        return s
    
    # ================ #
    #   Methods        #
    # ================ #
    def update(self, reset_others=False, **kwargs):
        """ """
        updates = {}
        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"{k} is not a Mirror mutable parameter.")
                continue
        
            if k == "surface":
                self._surface = float(v)
                continue

            # keys that affect others
            if k in ["diameter_ext", "diameter_int"]:
                self._surface = None # reset since diameters given

            updates[k] = v
            
        if reset_others:
            self._meta = self._meta_in | updates
        else:
            self._meta = self._meta | updates


    # --------- #
    #  GETTER   #
    # --------- #
    def get_thermal_signal(self, domains, solid_angle,
                               temperature=None, emissivity=None):
        """ Mirror thermal signal [ph/s/spx/Δλ].

        Parameters
        ----------
        domains: array
            (nlbda, 2) list of spectral domains [Å] or spectral px by default

        solid_angle: float
            spaxel solid angle [sr].

        temperature: float
            mirror temperature [K], or default one
            
        emissivity: float
            mirror emissivity, or default one
            
        as_cube: bool
            output format (3d cube of float or float)

        Returns
        -------
        thermal signal in ph/s/spx/Δλ
        """

        from .thermal import thermal_signal

        if temperature is None:
            temperature = self.temperature  # Mirror temperature [K]
        temperature = np.atleast_1d(temperature)
        
        if emissivity is None:
            emissivity = self.emissivity    # Mirror emissivity
        emissivity = np.atleast_1d(emissivity)

        # sum over 1 element if only 1 temperature
        signal = np.sum([ thermal_signal(solid_angle,
                                    self.surface,    # Collecting area [m²]
                                    domains * 1e-4,  # Spectral bin [µm]
                                    temperature_,
                                    emissivity_)
                        for temperature_, emissivity_ in zip(temperature, emissivity)
                            ], axis=0)

        return signal                              # [ph/s/spx/Δλ]

    def get_airy_radius(self, lbda, norm_scale=1):
        """ returns the airy disk first radius caused by the primary mirror 
        
        radius [radian] = 1.22 * lambda [meter] / diameter [meter]
        
        Parameters
        ----------
        lbda: float, array
            wavelength in Angstrom.

        Returns
        -------
        radius: array
             radius of the first minimum in arcsec.
        """
        return 1.22 * lbda * units.angstrom.to("m") / self.diameter_ext * units.radian.to("arcsec") / norm_scale
    
    def get_nea_airy(self, lbda, norm_scale=1, padding=5, position=(0 ,0), **kwargs):
        """ 
        
        """
        from .nea import get_2dpsf_nea
        radius = self.get_airy_radius(lbda, norm_scale=norm_scale)[:,None, None] 
        padding_sigma = int(radius.mean()*padding)
        xx = f"{-padding_sigma}:{padding_sigma}:50j"
        yy = f"{-padding_sigma}:{padding_sigma}:50j"
        
        return get_2dpsf_nea("airy", radius=radius, xx=xx, yy=yy,
                                 position=position, **kwargs)
    
    def to_poppy(self, opticalsys=None):
        """ add component to a poppy opitcal system 
        
        Parameters
        ----------
        opticalsys: poppy.OpticalSystem
            optical system this optical element should be added to.

        Returns
        -------
        opticalsys (careful: inplace the input)
        """
        import poppy
        from astropy import units
        if opticalsys is None:
            opticalsys = poppy.OpticalSystem()

        primary_radius = self.diameter_ext * units.m /2 

        opticslist = []
        ap = poppy.CircularAperture(name="primary mirror", radius=primary_radius) # primary mirror
        # sec = poppy.SecondaryObscuration(secondary_radius=0.1, n_supports=3, support_width=0.05) # secondary with spiders
        mirror = poppy.CompoundAnalyticOptic( opticslist=[ap], name='telescope mirror')           # combine into one optic

        opticalsys.add_pupil( mirror )
        return opticalsys

    
    # ============= #
    #   Properties  #
    # ============= #
    @property
    def meta(self):
        """ metadata containing the mirror specifications """
        return self._meta
    
    @property
    def surface(self):
        """ Collecting area from outer and inner diameters [m^2] """
        if self._surface is None:
            return np.pi/4 * (self.diameter_ext ** 2 - self.diameter_int ** 2)
        
        return self._surface

    @property
    def diameter_ext(self):
        """ external diameter (assumed circular) """
        return self.meta.get("diameter_ext", None)

    @property
    def diameter_int(self):
        """ internal diameter (assumed circular) """
        return self.meta.get("diameter_int", None)
    
    @property
    def temperature(self):
        """ mirror temperature (in K) """
        return self.meta.get("temperature", None)
    
    @property
    def emissivity(self):
        """ Emissivity of the mirror  """
        return self.meta.get("emissivity", None)

    @property
    def nelements(self):
        """ numer of mirrors """
        return len( np.atleast_1d(self.temperature) )
    
