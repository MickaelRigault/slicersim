"""
Mirror module, to simulate:

* mirror collecting surface and thermal emission


.. autosummary::

   Telescope
"""

__author__ = "Mickael Rigault <m.rigault@ip2i.in2p3.fr>"

import warnings
import numpy as np
from astropy import units
from copy import deepcopy
from .thermal import ThermalOptics

class Telescope():
    """Telescope class."""

    mutable_parameters = ['temperature', 'emissivity',
                          "diameter_ext", "diameter_int",
                          "surface"]

    def __init__(self, surface=None, optics=None, meta={}):
        """Initialize the Telescope.

        Parameters
        ----------
        surface : float, optional
            Collecting area in m^2. If None, it is computed from the diameters.
            Default is None.
        optics : ThermalOptics, optional
            ThermalOptics object containing the thermal properties of the telescope.
            Default is None.
        meta : dict, optional
            Dictionary of metadata. Default is {}.
        """
        
        self._surface = surface
        self._optics = optics
        # update a copy of the meta
        
        # make sure temperature and emissivity are given.
        self._meta = deepcopy(meta)
        self._meta_in = deepcopy(meta)

        if surface is None and np.any([k not in self.meta for k in ["diameter_ext", "diameter_int"]]):
            warnings.warn("Surface not given and 'diameter_ext' and/or 'diameter_int' are unknown.")
                                          
    @classmethod
    def from_config(cls, config):
        """Build the class from a configuration file.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.

        Returns
        -------
        Telescope
            An instance of the Telescope class.
        """
        surface = config.get("surface", None) # if None, diameter_ext and diameter_int used.
        optics = ThermalOptics.from_config( config, no_solidangle_ok=True)
        return cls(surface=surface, optics=optics, meta=config) # core information stored in meta
                       
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
        """Update mutable parameters.

        Parameters
        ----------
        reset_others : bool, optional
            If True, parameters not given in `kwargs` are reset to their
            initial values. Default is False.
        **kwargs
            Parameters to update.
        """
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
    def get_thermal_signal(self, lbda_bin, solid_angle,
                               as_sum=True):
        """ Mirror thermal signal [ph/s/spx/Δλ].

        Parameters
        ----------
        lbda_bin : array-like
            The wavelength bin(s) in Angtrom. Can be a 1D or 2D array.
            - 1D: [lbda_min, lbda_max]
            - 2D: [[lbda_min, lbda_max],[lbda_min, lbda_max],...]

        solid_angle : float
            spaxel solid angle [sr].
        
        as_sum : bool
            if multiple mirrors, should this be the sum of all contribution (as_sum=True)
            or the list of (False).

        Returns
        -------
        float, array
            thermal signal(s) in ph/s/spx/Δλ.
        """
        signals = self.optics.get_signal(lbda_bin=lbda_bin,        # expectedin in [A]
                                         solid_angle=solid_angle,  # expectedin in [sr]
                                         area = self.surface,      # Collecting area [m²]
                                        )
        
        # sum over 1 element if only 1 temperature
        if as_sum:
            return np.sum(signals, axis=0)
        
        return signals                              # [ph/s/spx/Δλ]

    def get_airy_radius(self, lbda, norm_scale=1):
        """Return the Airy disk radius.

        The radius of the first minimum of the Airy disk is computed as:
        radius [rad] = 1.22 * lambda [m] / diameter [m]

        Parameters
        ----------
        lbda: float, array
            wavelength in Angstrom.

        Returns
        -------
        radius: array
             radius of the first minimum in arcsec.
        """
        return 1.22 * np.atleast_1d(lbda) * units.angstrom.to("m") / self.diameter_ext * units.radian.to("arcsec") / norm_scale
    
    def get_nea_airy(self, lbda, norm_scale=1, padding=5, position=(0, 0), **kwargs):
        """Get the Noise Equivalent Area of the Airy disk.

        Parameters
        ----------
        lbda : float or array_like
            Wavelength in Angstrom.
        norm_scale : float, optional
            Normalization scale. Default is 1.
        padding : int, optional
            Padding for the PSF grid. Default is 5.
        position : tuple, optional
            Position of the PSF. Default is (0, 0).
        **kwargs
            Additional arguments passed to `get_2dpsf_nea`.

        Returns
        -------
        array_like
            Noise Equivalent Area of the Airy disk.
        """
        from .nea import get_2dpsf_nea
        radius = self.get_airy_radius(lbda, norm_scale=norm_scale)[:,None, None] 
        padding_sigma = int(radius.mean()*padding)
        xx = f"{-padding_sigma}:{padding_sigma}:50j"
        yy = f"{-padding_sigma}:{padding_sigma}:50j"
        
        return get_2dpsf_nea("airy", radius=radius, xx=xx, yy=yy,
                                 position=position, **kwargs)

    def get_psfprofile(self, lbda, profile="airy", shape=None,
                           normal_scatter=None,
                           oversampling=10, **kwargs):
        """Get the PSF profile.

        Parameters
        ---------
        lbda: float, array
            wavelengths used to get the first airy radius

        profile: str
            - airy
            - gaussian: if so, sigma obtained from airy-radius equivalent.

        shape: (float, float)
            shape of the psf image (in arcsec)

        normal_scatter: float
            sigma of an additional normal convolution scatter (in arcsec)

        oversampling: int
            number of sub-pixel per pixel element (given by shape)
        
        Returns
        -------
        psf, centroid_in_imagecoord, pixels_sampling, oversampling
        """
        from .profiles import get_profilepsf
        radius = self.get_airy_radius(lbda)
        norm_pixels = 1/np.min(radius) # for corrected sampling
        if shape is None:
            shape = ( int(radius*15), int(radius*15) )
        
        shape = np.asarray(shape) * norm_pixels
        radius_eff = radius * norm_pixels
        if normal_scatter is not None:
            normal_scatter_eff = normal_scatter * norm_pixels
        else:
            normal_scatter_eff = None

        # properties of PSF
        prop = {"normal_scatter":normal_scatter_eff,
                "oversampling":oversampling,
                "shape": shape} | kwargs
            
        if profile in ["airy"]:
            prop["radius"] = radius_eff
        elif profile in ["gaussian", "gauss", "norm", "normal"]:
            from .profiles import airyradius_to_gaussiansigma
            prop["sigma"] = airyradius_to_gaussiansigma(radius_eff)

        psf, centroid, pixelarea = get_profilepsf(profile, **prop)
        arcsec_to_pixels = oversampling * norm_pixels
        return psf, pixelarea, arcsec_to_pixels, radius

    def get_encircled_energy(self, lbda, radius, profile="airy",
                             normal_scatter=None,
                             size=2, **kwargs):
        """Get the encircled energy.

        Parameters
        ----------
        lbda : float or array_like
            Wavelength in Angstrom.
        radius : float or array_like
            Radius or radii to compute the encircled energy for.
        profile : str, optional
            PSF profile to use. Default is "airy".
        normal_scatter : float, optional
            Sigma of an additional normal convolution scatter in arcsec.
            Default is None.
        size : float, optional
            Size of the PSF image. Default is 2.
        **kwargs
            Additional arguments passed to `get_psfprofile`.

        Returns
        -------
        array_like
            The encircled energy.
        """
        from .profiles import psfimage_to_encircledenergy
        
        radius = np.atleast_1d(radius)
        size = np.max([radius.max()*1.01, size]) # 1.01 just to make sure
        psf, pixelarea, arcsec_to_pixel, _ = \
          self.get_psfprofile(lbda, profile=profile, 
                                normal_scatter=normal_scatter,
                                position=(0, 0), # force the center
                                shape=(size, size),
                                norm_pixels=False, # we do it after
                                  **kwargs)
        # psf.sum() => 1
        psf *= pixelarea
        
        centroid = np.asarray(psf.shape)/2 - 0.5
        radius_eff = radius * arcsec_to_pixel
        return psfimage_to_encircledenergy(psf, radius_eff, position=centroid)

    def get_encircled_energy_radius(self, lbda, ee,
                                    profile="airy", normal_scatter=None,
                                    max_radius=2, nbins=1_000, **kwargs):
        """Get the radius for a given encircled energy.

        Parameters
        ----------
        lbda : float or array_like
            Wavelength in Angstrom.
        ee : float or array_like
            Encircled energy or energies to find the radius for.
        profile : str, optional
            PSF profile to use. Default is "airy".
        normal_scatter : float, optional
            Sigma of an additional normal convolution scatter in arcsec.
            Default is None.
        max_radius : float, optional
            Maximum radius to search for. Default is 2.
        nbins : int, optional
            Number of bins to use for the search. Default is 1000.
        **kwargs
            Additional arguments passed to `get_encircled_energy`.

        Returns
        -------
        array_like
            The radius or radii for the given encircled energy.
        """
        ee_radius = np.linspace(0, max_radius, nbins)
        encirle_energy = self.get_encircled_energy(lbda, ee_radius, profile=profile,
                                                   normal_scatter=normal_scatter,
                                                   **kwargs)
        ee = np.atleast_1d(ee)
        return ee_radius[np.argmin( (encirle_energy-ee[:,None])**2, axis=1)]
    
    # --------- #
    #  Plotting #
    # --------- #
    def show_psfprofile(self, lbda, ax=None, profile="airy",
                        oversampling=10,
                        show_radius=False, shape=None,
                        normal_scatter=None,
                        **kwargs): # pragma: no cover
        """Show the PSF profile.

        Parameters
        ----------
        lbda : float
            Wavelength in Angstrom.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        profile : str, optional
            PSF profile to use. Default is "airy".
        oversampling : int, optional
            Oversampling factor. Default is 10.
        show_radius : bool, optional
            If True, show the Airy radius. Default is False.
        shape : tuple, optional
            Shape of the PSF image. Default is None.
        normal_scatter : float, optional
            Sigma of an additional normal convolution scatter in arcsec.
            Default is None.
        **kwargs
            Additional arguments passed to `ax.imshow()`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        from matplotlib.colors import LogNorm
        

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        psf, pixelarea, arcsec_to_pixel, radius = self.get_psfprofile(lbda,
                                                                      profile=profile,
                                                                       shape=shape,
                                                                       normal_scatter=normal_scatter,
                                                                       oversampling=oversampling)
        norm_pixels = arcsec_to_pixel/oversampling
        shape = np.asarray(psf.shape)/oversampling
        hy, hx = np.asarray(shape) / 2  # Half total width [spx]
        extent = np.asarray([-hx-0.5, hx+0.5, -hy-0.5, hy+0.5]) / norm_pixels
            
        ax.imshow(psf, origin="lower", norm=LogNorm(), extent=extent, zorder=5, **kwargs)
        if show_radius:            
            prop = dict(color="k", alpha=0.5, lw=1, ls="-", zorder=6)
            ax.axvline(radius, **prop)
            ax.axvline(-radius, **prop)
            ax.axhline(radius, **prop)
            ax.axhline(-radius, **prop)

        return fig
    
    # ============= #
    #   Properties  #
    # ============= #
    @property
    def meta(self):
        """Metadata containing the mirror specifications."""
        return self._meta
    
    @property
    def surface(self):
        """Collecting area from outer and inner diameters in m^2."""
        if self._surface is None:
            return np.pi/4 * (self.diameter_ext ** 2 - self.diameter_int ** 2)
        
        return self._surface

    @property
    def diameter_ext(self):
        """External diameter (assumed circular) in m."""
        return self.meta.get("diameter_ext", None)

    @property
    def optics(self):
        """ThermalOptics object."""
        return self._optics
    @property
    def diameter_int(self):
        """Internal diameter (assumed circular) in m."""
        return self.meta.get("diameter_int", None)
    
    @property
    def temperature(self):
        """Mirror temperature in K."""
        return self.optics.temperature
    
    @property
    def emissivity(self):
        """Emissivity of the mirror."""
        return self.optics.emissivity

    @property
    def nelements(self):
        """Number of mirrors."""
        return self.optics.nelements
