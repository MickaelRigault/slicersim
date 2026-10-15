"""Telescope module.

This module simulates the telescope: primary mirror collecting area, thermal
emission of the mirrors and diffraction-limited (Airy) PSF.

.. autosummary::

   Telescope
"""

__author__ = "Mickael Rigault <m.rigault@ip2i.in2p3.fr>"

import warnings
from copy import deepcopy

import numpy as np
from astropy import units

from .thermal import ThermalOptics


class Telescope:
    """Telescope: collecting area, mirror thermal emission and Airy PSF.

    Parameters
    ----------
    surface : float or None, optional
        Collecting area in m^2. If None, it is computed from the
        ``"diameter_ext"`` and ``"diameter_int"`` entries of ``meta``.
        Default is None.
    optics : `~slicersim.thermal.ThermalOptics` or None, optional
        Thermal properties of the telescope mirrors. Default is None.
    meta : dict, optional
        Telescope specifications (e.g. ``"diameter_ext"`` and
        ``"diameter_int"`` in m). Default is an empty dict.

    Attributes
    ----------
    mutable_parameters : list of str
        Parameters that can be changed after initialization with `update`.

    Warns
    -----
    UserWarning
        If ``surface`` is None and the diameters are not given in ``meta``.
    """

    mutable_parameters = ['temperature', 'emissivity',
                          "diameter_ext", "diameter_int",
                          "surface"]

    def __init__(self, surface=None, optics=None, meta={}):
        """Initialize the Telescope (see the class docstring for parameters)."""
        
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
        """Build a telescope from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary (e.g. the ``[telescope]`` section of a
            configuration file). It may contain ``"surface"`` [m^2], otherwise
            ``"diameter_ext"`` and ``"diameter_int"`` [m] are used, and must
            contain the thermal parameters required by
            `~slicersim.thermal.ThermalOptics.from_config` (at least
            ``"temperature"`` and ``"emissivity"``). It is stored as `meta`.

        Returns
        -------
        Telescope
            The telescope instance.
        """
        surface = config.get("surface", None) # if None, diameter_ext and diameter_int used.
        optics = ThermalOptics.from_config( config, no_solidangle_ok=True)
        return cls(surface=surface, optics=optics, meta=config) # core information stored in meta
                       
    def __str__(self):
        """Return a human-readable summary of the telescope configuration.

        Returns
        -------
        str
            String describing the primary mirror collecting area, and, if
            thermal emission is modelled, the temperature and emissivity of
            each optical element.
        """
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

        See `mutable_parameters` for the list of accepted names. Setting
        ``diameter_ext`` or ``diameter_int`` resets a previously given
        ``surface``, which is then computed from the diameters.

        Parameters
        ----------
        reset_others : bool, optional
            If True, parameters not given in ``kwargs`` are reset to their
            initial values. Default is False.
        **kwargs
            Parameters to update.

        Warns
        -----
        UserWarning
            If a given parameter is not mutable (it is then ignored).
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
        """Get the mirror thermal signal received by a spaxel.

        Parameters
        ----------
        lbda_bin : array_like
            Wavelength bin(s) in Angstrom, either 1D ``[lbda_min, lbda_max]``
            or 2D ``[[lbda_min, lbda_max], [lbda_min, lbda_max], ...]``.
        solid_angle : float
            Spaxel solid angle in steradian.
        as_sum : bool, optional
            If True, return the sum of the contributions of all mirrors.
            Otherwise, return one entry per mirror. Default is True.

        Returns
        -------
        float or numpy.ndarray
            Thermal signal in ph/s/spaxel, integrated over each wavelength bin.
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
        """Get the Airy disk radius.

        The radius of the first minimum of the Airy disk is
        ``1.22 * lbda / diameter_ext`` (in radian).

        Parameters
        ----------
        lbda : float or array_like
            Wavelength(s) in Angstrom.
        norm_scale : float, optional
            Scale by which the radius is divided (e.g. a pixel size in
            arcsec to get the radius in pixels). Default is 1.

        Returns
        -------
        numpy.ndarray
            Radius of the first minimum in arcsec (divided by
            ``norm_scale``), one value per wavelength.
        """
        return 1.22 * np.atleast_1d(lbda) * units.angstrom.to("m") / self.diameter_ext * units.radian.to("arcsec") / norm_scale
    
    def get_nea_airy(self, lbda, norm_scale=1, padding=5, position=(0, 0), **kwargs): # pragma: no cover
        """Get the Noise Equivalent Area (NEA) of the Airy disk.

        Parameters
        ----------
        lbda : float or array_like
            Wavelength(s) in Angstrom.
        norm_scale : float, optional
            Scale by which the Airy radius is divided (see `get_airy_radius`).
            Default is 1.
        padding : float, optional
            Half-width of the evaluation grid, in units of the (mean) Airy
            radius. Default is 5.
        position : tuple of float, optional
            Position of the PSF centroid. Default is ``(0, 0)``.
        **kwargs
            Additional arguments passed to `~slicersim.nea.get_2dpsf_nea`.

        Returns
        -------
        float or numpy.ndarray
            Noise Equivalent Area of the Airy disk, in the squared units of
            the radius (arcsec^2 divided by ``norm_scale**2``).
        """
        from .nea import get_2dpsf_nea
        radius = self.get_airy_radius(lbda, norm_scale=norm_scale)[:,None, None] 
        padding_sigma = int(radius.mean()*padding)
        xx = f"{-padding_sigma}:{padding_sigma}:50j"
        yy = f"{-padding_sigma}:{padding_sigma}:50j"
        
        return get_2dpsf_nea("airy", radius=radius, xx=xx, yy=yy,
                                 position=position, **kwargs)

    def get_psfprofile(self, lbda, profile="airy", shape=(10, 10),
                           normal_scatter=None,
                           oversampling=10, **kwargs):
        """Get the diffraction-limited PSF image.

        Internally, the PSF is computed in units of the smallest Airy radius
        (a "pixel"), each pixel being divided into ``oversampling`` sub-pixels.

        Parameters
        ----------
        lbda : float or array_like
            Wavelength(s) in Angstrom used to compute the Airy radius.
        profile : str, optional
            PSF profile: ``"airy"``, or ``"gaussian"`` (aliases ``"gauss"``,
            ``"norm"``, ``"normal"``) with a sigma matching the Airy radius
            (see `~slicersim.profiles.airyradius_to_gaussiansigma`).
            Default is ``"airy"``.
        shape : tuple of float or None, optional
            Size ``(height, width)`` of the PSF image in arcsec. If None, ten
            times the largest Airy radius is used. Default is ``(10, 10)``.
        normal_scatter : float or None, optional
            Sigma of an additional Gaussian scatter convolved with the PSF, in
            arcsec. Default is None.
        oversampling : int, optional
            Number of sub-pixels per pixel. Default is 10.
        **kwargs
            Additional arguments passed to
            `~slicersim.profiles.get_profilepsf` (e.g. ``position``,
            ``norm_pixels``).

        Returns
        -------
        psf : numpy.ndarray
            PSF image (one image per wavelength if ``lbda`` is an array).
        pixelarea : float
            Area of a PSF image (sub-)pixel, in pixel^2 units.
        arcsec_to_pixels : float
            Conversion factor from arcsec to PSF image (sub-)pixels.
        radius : numpy.ndarray
            Airy radius in arcsec, one value per wavelength.
        """
        from .profiles import get_profilepsf
        radius = self.get_airy_radius(lbda)
        norm_pixels = 1/np.min(radius) # for corrected sampling
        if shape is None:
            shape = ( int(np.nanmax(radius)*10), int(np.nanmax(radius)*10) )
        
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
        """Get the fraction of the PSF energy encircled within given radii.

        Parameters
        ----------
        lbda : float
            Wavelength in Angstrom.
        radius : float or array_like
            Radius or radii in arcsec.
        profile : str, optional
            PSF profile to use (see `get_psfprofile`). Default is ``"airy"``.
        normal_scatter : float or None, optional
            Sigma of an additional Gaussian scatter convolved with the PSF, in
            arcsec. Default is None.
        size : float, optional
            Minimum size of the PSF image in arcsec; it is enlarged to
            include the largest radius if needed. Default is 2.
        **kwargs
            Additional arguments passed to `get_psfprofile`.

        Returns
        -------
        numpy.ndarray
            Encircled energy fraction, one value per radius.
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
        """Get the radius encircling a given fraction of the PSF energy.

        The encircled energy is computed on a grid of ``nbins`` radii between
        0 and ``max_radius`` and the closest grid radius is returned.

        Parameters
        ----------
        lbda : float
            Wavelength in Angstrom.
        ee : float or array_like
            Encircled energy fraction(s), between 0 and 1.
        profile : str, optional
            PSF profile to use (see `get_psfprofile`). Default is ``"airy"``.
        normal_scatter : float or None, optional
            Sigma of an additional Gaussian scatter convolved with the PSF, in
            arcsec. Default is None.
        max_radius : float, optional
            Maximum radius of the search grid in arcsec. Default is 2.
        nbins : int, optional
            Number of radii in the search grid. Default is 1000.
        **kwargs
            Additional arguments passed to `get_encircled_energy`.

        Returns
        -------
        numpy.ndarray
            Radius in arcsec, one value per ``ee``.
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
        """Show the PSF image (log scale, axes in arcsec).

        Parameters
        ----------
        lbda : float
            Wavelength in Angstrom.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        profile : str, optional
            PSF profile to use (see `get_psfprofile`). Default is ``"airy"``.
        oversampling : int, optional
            Number of sub-pixels per pixel. Default is 10.
        show_radius : bool, optional
            If True, show the Airy radius as horizontal and vertical lines.
            Default is False.
        shape : tuple of float or None, optional
            Size of the PSF image in arcsec. If None, ten times the Airy
            radius is used. Default is None.
        normal_scatter : float or None, optional
            Sigma of an additional Gaussian scatter convolved with the PSF, in
            arcsec. Default is None.
        **kwargs
            Additional arguments passed to `matplotlib.axes.Axes.imshow`.

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
        """Metadata dictionary containing the telescope specifications."""
        return self._meta
    
    @property
    def surface(self):
        """Collecting area in m^2.

        If not given explicitly, it is computed from `diameter_ext` and
        `diameter_int`.
        """
        if self._surface is None:
            return np.pi/4 * (self.diameter_ext ** 2 - self.diameter_int ** 2)
        
        return self._surface

    @property
    def diameter_ext(self):
        """External diameter (assumed circular) in m."""
        return self.meta.get("diameter_ext", None)

    @property
    def optics(self):
        """`~slicersim.thermal.ThermalOptics` describing the mirrors' thermal emission."""
        return self._optics
    @property
    def diameter_int(self):
        """Internal diameter (assumed circular) in m."""
        return self.meta.get("diameter_int", None)
    
    @property
    def temperature(self):
        """Mirror temperature(s) in K."""
        return self.optics.temperature
    
    @property
    def emissivity(self):
        """Mirror emissivity (one value per mirror)."""
        return self.optics.emissivity

    @property
    def nelements(self):
        """Number of mirrors sharing each temperature and emissivity."""
        return self.optics.nelements
