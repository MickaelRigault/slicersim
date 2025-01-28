"""
Spectrograph module, to simulate:

* mirror collecting surface and thermal emission
* spatial stage and MLA sampling
* spectral stage and dispersion law

Astrophysical scene is handled by :class:`mlaperf.scene.Scene` and detector by
:class:`mlaperf.detector.Detector`.

.. autosummary::

   Mirror
   Spectrograph
"""

__author__ = "Mickael Rigault <m.rigault@ip2i.in2p3.fr>, Yannick Copin <y.copin@ip2i.in2p3.fr>"

import warnings
from dataclasses import dataclass
import numpy as np

import astropy.units as u

from .utils import integ_gaussian2D_erf, recursive_get
from . import iotools
from .mirrors import Mirror



@dataclass
class Camera:
    """
    Camera data class.
    """

    acceptance: float = 0.      #: Camera acceptance angle [rad]
    speed: float = np.inf       #: Camera speed (f-number)

    def __str__(self):

        return f"Camera: {np.degree(self.acceptance)} deg, f/{self.speed:.0f}"

# ================ #
#                  #
#   Spectrograph   #
#                  #
# ================ #
def lbda_from_respow(spectral_range, res_power, npx=2):
    r""" Compute wavelength ramp for constant n-px resolving power.

    .. math::

       \frac{\Delta\lambda}{\lambda} &= \frac{1}{n\mathcal{R}} &\\
       &= \ln 10\,\Delta\log\lambda &\\
       \log\lambda_i^e &= \log\lambda_0 + i\times \Delta\log\lambda
       &\quad\text{(edges)} \\
       \log\lambda_i &= (\log\lambda_i + \log\lambda_{i+1})/2
       &\quad\text{(center)}

    :param 2-tuple spectral_range: spectral domain
    :param float res_power: resolving power R
    :param float npx: n-px resolution (i.e. n px per spectral elements)
    :return: mid and bin edge wavelengths [same units as input domain]
    """

    wmin, wmax = spectral_range
    dlog = 1/(2.302585092994046 * res_power * npx)  # ln(10)=2.303...
    npx = round((np.log10(wmax) - np.log10(wmin)) / dlog)
    loglbda_edges = np.linspace(np.log10(wmin), np.log10(wmax), npx + 1)
    loglbda_mid = (loglbda_edges[:-1] + loglbda_edges[1:])/2

    return 10**loglbda_mid, 10**loglbda_edges

def build_lbda(spectral_range, spectral_resolution=None, wsol=None, dsol=None):
    """ set wavelength coordinates.

    Set :attr:`lbda` (:attr:`nlbda` mid wavelengths) and :attr:`lbda_egdes`
    (:attr:`nlbda` + 1 edge wavelengths).
    """
    if wsol is None:   # Compute from constant resolving power
        lbda, lbda_edges = lbda_from_respow(spectral_range, spectral_resolution)
    else:              # Compute from wavelength solution
        wmin, wmax = spectral_range
        npx = round(dsol(wmax) - dsol(wmin))    # Total nb of px
        lbda = wsol(np.r_[:npx])                # λ at bin center
        lbda_edges = wsol(np.r_[:npx+1] - 0.5)  # λ at bin edge
        
    return lbda, lbda_edges


def build_lbda_from_config(config):
    """ """
    spectral_range = np.asarray(config["spectral_range"], dtype="float32")

    dispersion_law = config.get("dispersion_law", None)
    dispersion_scale = float(config.get("dispersion_scale", 1))
    spectral_resolution = config.get("spectral_resolution", None)
    # Set by dispersion law ?
    if dispersion_law is not None:
        # => ok, you have a dispersion law
        wname, dname = 'wavelength', 'offset'
        tab = iotools.read_ecsv( dispersion_law, colnames=[wname, dname])
        assert tab[dname].unit == 'pix'
        
        # dispersion solution (wavelengths in Å, offset in pix)
        dsol = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[dname] * dispersion_scale,
            ext='extrapolate')
        
        # wavelength solution (wavelengths in Å, offset in pix)
        wsol = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[dname] * dispersion_scale,
            ext='extrapolate', inverse=True)
        
    elif spectral_resolution is None:
        raise ValueError("'spectral_resolution' OR 'dispersion_law' should be set.")
    else:
        wsol = dsol = None
        dispersion_scale = float(dispersion_scale)

    lbda, lbda_edges = build_lbda(spectral_range, wsol=wsol, dsol=dsol,
                                    spectral_resolution=spectral_resolution)
    return lbda, lbda_edges


def build_spaxels_from_config(config, 
                              psf_sigma_spectral=None,
                              spx_spatial_scale=None):
    """ spaxel coordinates [spx] from MLA shape.

    Set `self.(x,y)[_edges]` from :attr:`spatial_shape`.
    """
    if (spatial_shape:= config.get("spatial_shape", None)) is None:
        spatial_shape_insigma = config.get("spatial_shape_insigma", None)
        if spatial_shape_insigma is None:
            raise ValueError("'spatial_shape' OR 'spatial_shape_insigma' should be set.")

        spatial_shape_insigma = np.asarray(spatial_shape_insigma, dtype="float")
        if psf_sigma_spectral is None:
            psf_sigma_spectral = recursive_get(config, "psf_sigma_spectral")
        if spx_spatial_scale is None:
            spx_spatial_scale = recursive_get(config, "spatial_scale")
        
        spatial_shape = np.round(spatial_shape_insigma * psf_sigma_spectral / spx_spatial_scale)
    
    ny, nx = spatial_shape = np.asarray(spatial_shape, dtype="int")
    hnx, hny = (nx - 1)/2, (ny - 1)/2
    y, x = np.ogrid[-hny:hny:ny*1j,
                              -hnx:hnx:nx*1j]  # Central coord. grids [spx]
    y_edges, x_edges = np.ogrid[
        -hny - 0.5:hny + 0.5:(ny + 1)*1j,
        -hnx - 0.5:hnx + 0.5:(nx + 1)*1j]      # Edge coord. grids [spx]
            
    return {"shape": (nx, ny), "centroids": (x, y), "edges": (x_edges, y_edges),
            "spatial_scale": spx_spatial_scale}

def build_throughput_from_config(config):
    """ """
    try:
        # throughput is a constant
        throughput = float(config["throughput"])
        throughput_name = self.throughput_name_interp = None
            
    except ValueError:
        # throughput is a filename
        throughput_name = config["throughput"]  #: Throughput filename
        wname, tname = 'wavelength', 'throughput'
        tab = iotools.read_ecsv(throughput_name,
                                colnames=[wname, tname])
        #: Throughput interpolator (wavelengths in Å)
        throughput = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[tname], ext='zeros')
        
    return throughput
            
class Spectrograph:
    """
    Spectrograph simulation.

    .. Warning:: the spectral PSF impacts the cross-dispersion profile,
                 but is not applied in the dispersion direction.
    """

    #: Mutable parameters (list)
    mutable_parameters = ['spectral_range', 'spectral_resolution', # lbda
                          'xdisp_sigma_spectral', 'xdisp_sigma', # xdisp_profile,
                          'psf_sigma_spectral', 'guiding_sigma', # psf_profile
                          'spatial_scale', 'spatial_scale_insigma', # spaxels
                          'spatial_shape', 'spatial_shape_insigma',
                          #'camera.acceptance', 'camera.speed',
                          ]

    
    def __init__(self, lbda, mirror,
                 spaxels={}, throughput=None,
                 xdispersion={}, spatial_psf={},
                 lbda_edges=None,
                 meta={}):
        """
        Initialize the spectrograph properties from `config` dictionary.

        :param dict config: spectrograph configuration dictionary
        :param bool verbose: verbose mode
        """
        # wavelength
        self.lbda = lbda
        self.lbda_edges = lbda_edges

        # mirror
        self.mirror = mirror

        # Spaxels
        self._spaxels = spaxels

        # Throughput
        if callable(throughput): # this is a function
            self._throughput_interp = throughput
            self.throughput = self._throughput_interp(self.lbda)
        else:
            self._throughput_interp = None
            self.throughput = throughput

        self.xdispersion = xdispersion
        self.spatial_psf = spatial_psf
        
        # affect the instance
        self.mutable_parameters = self.mutable_parameters + [f"mirror.{k}" for k in self.mirror.mutable_parameters]
        
        self._meta_in = meta.copy()      #: Meta-parameters as input
        self._meta = meta.copy()         #: Meta-parameters as used
        
    @classmethod
    def from_config(cls, config):
        """ """
        # wavelengths
        lbda, lbda_edges = build_lbda_from_config(config)

        # Mirror
        mirror = Mirror.from_config(config['mirror'])
        
        # Spaxels
        spaxels = build_spaxels_from_config(config)
        
        # throughput
        throughput = build_throughput_from_config(config)
        
        # PSF at the detector level
        xdispersion = {"sigma_spectral": float(config["psf"]["detector"]["xdisp_sigma_spectral"]),
                      "sigma": float(config["psf"]["detector"]["xdisp_sigma"]),
                      "profile": config["psf"]["detector"]["xdisp_profile"]}

        # in coming PSF
        psf = {"sigma_spectral": float(config["psf"]["spatial"]["psf_sigma_spectral"]),
               "guiding_sigma": float(config["psf"]["spatial"]["guiding_sigma"]),
               "profile": config["psf"]["spatial"]["psf_profile"]}

        
        init_prop = {"lbda":lbda, 
                     "lbda_edges":lbda_edges, 
                     "mirror": mirror,
                     "spaxels":spaxels,
                     "throughput": throughput, 
                     "xdispersion":xdispersion, 
                     "spatial_psf": psf,
                     "meta": config
                    }
        return cls(**init_prop)
        
    @staticmethod
    def build_lbda_from_config(config):
        return build_lbda_from_config(config)
        
    def __str__(self):

        wmin, wmax = self.spectral_range
        avwres0 = self.effective_resolution(npx=2, sigma=0, average=True)
        avwres = self.effective_resolution(npx=2, average=True)
        wres = self.effective_resolution(npx=2)
        imin = np.argmin(wres)

        s = f"Spectrograph {self.name!r}:"
        s += f"\n  Spectral range: {wmin:_.0f}-{wmax:_.0f} Å, {self.nlbda} px"
        if self.wsol is not None:
            s += "\n  Spectral dispersion: " \
                f"{self.dispersion_law!r} ×{self.dispersion_scale} (R0~{avwres0:.0f})"
        else:
            s += f"\n  Fixed resolving power (2 px): {avwres0:.0f}"
        s += f"\n  Spectral PSF: chromatic σ={self.xdisp_sigma_spectral:.2f} px at 1 µm, "
        s += f"x-disp. σ={self.xdisp_sigma:.2f} px"
        s += "\n  Resolving power (2-px + σ): " \
            f"R~{avwres:.0f} (λ-average), " \
            f"min={wres[imin]:.0f} at {self.lbda[imin]:_.0f} Å"
        shape = "×".join([ str(i) for i in self.spatial_shape ])
        s += f"\n  MLA: {shape} spx of {self.spx_spatial_scale*1e3:.0f} mas"
        s += f"\n  Spatial PSF: chromatic σ={self.psf_sigma_spectral*1e3:.0f} mas at 1 µm, "
        s += f"guiding σ={self.guiding_sigma*1e3:.0f} mas"

        if self.throughput_name:
            s += (f"\n  Total throughput: {self.throughput_name!r} "
                  f"(~{self.throughput.mean():.0%})")  # px-average
        else:
            s += f"\n  Total throughput: constant {self.throughput:.0%}"

        s += "\n  " + str(self.mirror)

        return s

    def update(self, reset_others=False, **kwargs):
        """ Update any mutable attribute of the spectrograph. """
    
        updates = {}
        mirror_updates = {}
        lbda_updates = {}
        xdisp_updates = {}
        psf_updates = {}
        spaxel_updates = {}      
        
        # == Filling the update == #
        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            if v is None:        # Skip
                continue

            # mirror
            if k.startswith("mirror."):
                mirror_updates[k.replace("mirror.", "")] = v
                continue

            # lbda
            elif k in ('spectral_range', 'spectral_resolution'):
                
                # Simulation.update is in charge of updating other chromatic
                # quantities if 'spectral_range' or 'spectral_resolution' are
                # modified.
                if k == "spectral_resolution" and self.wsol is not None:
                    warnings.warn("Switching to constant resolving power.")
                    lbda_updates['dispersion_law'] = None
                    
                lbda_updates[k] = v

            # change PSF
            elif k in self.meta["psf"]["spatial"].keys():
                psf_updates[k.replace("psf_","")] = v

            # change PSF xdisp
            elif k in self.meta["psf"]["detector"].keys():
                xdisp_updates[k.replace("xdisp_","")] = v
                
            # spaxels
            elif k in ('spatial_scale', 'spatial_scale_insigma', 
                     'spatial_shape', 'spatial_shape_insigma'):
                spaxel_updates[k] = v
            else:
                warning.warn(f"{k=} is unparsed")
            
        
        # update mirror
        self.mirror.update(**mirror_updates, reset_others=reset_others)
        
        # psf
        self.spatial_psf = self.spatial_psf | psf_updates
        self.xdispersion = self.xdispersion | xdisp_updates
        
        # lbda
        if spaxel_updates:
            self._spaxels = build_spaxels_from_config( self._meta | spaxel_updates )
            
        if lbda_updates:
            self.update_lbda(*build_lbda_from_config(lbda_updates),
                             update_throughput=True)
        
        # the rest if any
        if reset_others:
            self._meta = self._meta_in |updates
        else:
            self._meta = self._meta | spaxel_updates

    def update_lbda(self, lbda, lbda_edges, update_throughput=True):
        """ """
        self.lbda = lbda
        self.lbda_edges = lbda_edges
        if update_throughput:
            self._update_throughput()
            
    def _update_throughput(self):
        """ """
        if self._throughput_interp is not None:
            self.throughput = self.throughput_interp(self.lbda)
    
    #            
    # - getter
    #    
    def get_xdisp_sigma_spectral(self, xdims=0, xdisp_sigma=None):
        """ Get spectral PSF stddev [px].

        The total (Gaussian) spectral PSF is made of two components:

        - the optical (chromatic) component, with stddev proportional
          to wavelength, normalized at wref=1 µm,
        - the achromatic component, with constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        :param int xdims: extra dimensions to be appended
        :param float xdisp_sigma: constant sigma override [px]
                                  (None for default)
        :return: total sigma [px]
        """
        if xdisp_sigma is None:
            xdisp_sigma = self.xdispersion["sigma"]

        return self._get_chromatic_sigma(self.lbda,
                                         chromatic_sigma = self.xdispersion["sigma_spectral"],
                                         constant_sigma = xdisp_sigma,
                                         wref = self.lbda_ref,
                                         xdims = xdims)

    def get_psf_sigma_spectral(self, xdims=0, guiding_sigma=None,
                              in_spaxels=False):
        """ Get spatial PSF stddev [arcsec].

        The total (Gaussian) spatial PSF is made of two components:

        - the optical (chromatic) component, with stddev proportional
          to wavelength, normalized at wref=1 µm,
        - the guiding (achromatic) component, with constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        :param int xdims: extra dimensions to be appended
        :param float guiding_sigma: guiding sigma override [arcsec]
                                    (None for default)
        :return: total sigma [arcsec]
        """

        if guiding_sigma is None:
            guiding_sigma = self.spatial_psf["guiding_sigma"]

        sigma = self._get_chromatic_sigma(self.lbda,
                                        chromatic_sigma = self.spatial_psf["sigma_spectral"],
                                        constant_sigma = guiding_sigma,
                                        wref = self.lbda_ref,
                                        xdims=xdims)
        if in_spaxels:
            sigma /= self.spx_spatial_scale
            
        return sigma
    
    def get_spatial_psf(self, position=(0, 0)):
        """ Get normalized 2D spatial PSF on the MLA.

        This uses the exact 2D Gaussian PSF integration over the spx.

        :param 2-tuple position: point source position in MLA [spx]
        :return: normalized PSF (nlbda, ny, nx)
        """

        sigmas = self.get_psf_sigma_spectral(xdims=2) / self.spx_spatial_scale
        psf = integ_gaussian2D_erf(self.spx_edges,  # ((1, nx), (ny, 1)) [spx]
                                   sigmas,          # (nlbda, 1, 1) [spx]
                                   position,        # [spx]
                                   normed=True)     # sum(axis=(1, 2)) = 1

        return psf                         # (nlbda, ny, nx)

    def get_thermal_signal(self, domains=None, temperature=None, emissivity=None,
                               as_cube=False):
        """ Mirror thermal signal [ph/s/spx/Δλ].

        Parameters
        ----------
        domains:  
            (nlbda, 2) list of spectral domains [Å] or spectral px by default

        temperature: float
            mirror temperature [K], or default one
            
        emissivity: float
            mirror emissivity, or default one
            
        as_cube: bool
            output format (3d cube of float or float)

        Returns
        -------
        thermal signal in ph/s/spx/Δλ (3d cube or float, see as_cube)
        """
        if domains is None:
            domains = np.vstack([self.lbda_edges[:-1],
                                 self.lbda_edges[1:]]).T  # (nlbda, 2) [Å]

        solid_angle = self.omega                  # Spx solid angle [sr]
        signal = self.mirror.get_thermal_signal(domains,
                                                solid_angle=solid_angle,
                                                temperature=temperature,
                                                emissivity=emissivity)
        if as_cube:
            signal = np.full((self.nlbda, *self.spx_shape[::-1]),
                                 signal[:, np.newaxis, np.newaxis])  # (nlbda, ny, nx)
            
        return signal                              # [ph/s/spx/Δλ]

    
    def get_nea(self, position=(0,0), nea_spatial=None, nea_pixels=None):
        """ noise effective area (PSF => Spaxel => detector (through x-dispersion)

        Parameters
        ----------
        nea_spatial: float, array
            noise equivalent area of the spatial PSF (in unit of spaxel / slice).
            If None, self.get_nea_spatial() is used.
            (array size must broadcast with self.lbda)

        nea_spatial: float, array
            noise equivalent area of a spaxel / slice caused by x-dispersion (in pixels)
            If None, self.get_nea_pixels() is used.
            (array size must broadcast with self.lbda)
        
        position: (float, float)
            # ignored if nea_spatial is given #
            position of the PSF in unit of spaxel / slicer.
            
        Returns
        -------
        array
        """
        
        # NEA_Spatial PSF on the MLA (2D) / Slicer (1d) | in slice / spaxel
        if nea_spatial is None:
           nea_spatial = self.get_nea_spatial(position = position)
        
        # NEA_pixel of 1 spaxel / slice on the dectetor
        if nea_pixels is None:
            nea_pixels = self.get_nea_pixels()
    
        return nea_spatial * nea_pixels
    
    def get_nea_spatial(self, position=(0,0)):
        """ noise equivalent area in unit of slice/spaxels 

        i.e., how many "spaxel noise")
        
        Parameters
        ----------
        position: (float, float)
            position of the PSF in unit of spaxel / slicer

        Returns
        -------
        array
        """
        from .profiles import get_2dnorm_nea
        
        if self.spatial_psf["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian spatial PSF profile implemented, but: {self.spatial_psf['profile']=}")
            
        sigma_at_mla = self.get_psf_sigma_spectral(in_spaxels=True, xdims=2)
        return get_2dnorm_nea(sigma_at_mla, mean = position)

    def get_nea_pixels(self):
        """ noise equivalent area of a spaxel /slice in unit of pixels

        Returns
        -------
        array
        """
        from .profiles import get_1dnorm_nea
        if self.xdispersion["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian xdispersion PSF profile implemented, but: {self.xdispersion['profile']=}")
            
        sigma_xdisp_at_detector = self.get_xdisp_sigma_spectral()  # in pixels 
        return get_1dnorm_nea(sigma_xdisp_at_detector)

    # ------------- #
    #   Others      #
    # ------------- #
    
    def point_source_variance(self, varcube, position=(0, 0), radius=5,
                                  optimal=True, verbose=False):
        """
        Point-source extracted variance from variance cube.

        It is evaluated from a plain summation over the aperture, or from an
        optimal extraction, i.e. inverse-variance weighted least-square fit of
        the PSF profile (assumed Gaussian).  As the extraction is assumed to do
        a perfect job on the signal, only the variance depends on instrumental
        parameters and aperture definition.

        :param varcube: variance cube (nlbda, ny, nx) [ADU²]
        :param 2-tuple position: point source position in MLA [spx]
        :param float radius: MLA aperture [spx]
        :param bool optimal: optimal vs. plain extraction
        :return: spx spectrum [ADU] and variance [ADU²]
        """

        # Spatial PSF
        psf = self.get_spatial_psf(position=position)  # (nlbda, ny, nx)

        # Aperture
        x0, y0 = position
        x, y = self.spx_centroids
        r = np.hypot(x - x0, y - y0)  # (ny, nx) [spx]
        aper = (r <= radius)
        nspx = np.count_nonzero(aper)           # Selected spx
        if verbose:
            print(f"Aperture of {radius=} spx: {nspx} spx selected")

        # If cube is (nlbda, ny, nx), cube[:, aper] is (nlbda, nspx)
        if optimal:  # Optimal extraction variance = 1/sum(psf**2/var)
            variance = 1 / (psf[:, aper]**2 / varcube[:, aper]).sum(axis=-1)
        else:        # Plain summation: variance = sum(variance)
            variance = varcube[:, aper].sum(axis=-1)

        return variance                  # (nlbda,) [ADU²]

    def effective_resolution(self, npx=2, sigma=None, average=False):
        r""" Effective spectral resolution.

        .. math::

           R &= \frac{2}{n \delta\lambda} \\
           \delta\lambda &= \max(1, \sigma) \times \Delta\lambda

        where :math:`\Delta\lambda` is the spectral step [Å] and
        :math:`\sigma` is the spectral resolution [px].

        :param float npx: n-px resolution (i.e. n px per spectral elements)
        :param sigma: spectral PSF stddev override
        :param bool average: chromatic average (weighted by spectral step)
        :return: effective n-px wavelength resolution
                 (as a function of wavelength or averaged)
        """
        if sigma is None:
            sigma = self.get_xdisp_sigma_spectral()  # (nlbda,)
        sigma = np.maximum(sigma, 1)

        dlbda = np.diff(self.lbda_edges)
        wres = self.lbda / (npx * sigma * dlbda)  # (nlbda,)

        if average:             # Chromatic average
            wres = np.average(wres, weights=dlbda)

        return wres    

    # - generate        
    def generate_point_source(self, spectrum, position=(0, 0)):
        """ Generate a photon flux cube from a point source spectrum.

        :param spectrum: point source spectrum [erg/s/cm²/Å]
        :param 2-tuple position: point source position in MLA [spx]
        :return: (nlbda, ny, nx) photon flux cube [ph/s/spx]
        """
        # Spatial PSF
        psf = self.get_spatial_psf(position=position)      # (nlbda, ny, nx)

        # erg/s/cm²/Å / erg/ph * cm² * Å = ph/s
        flux = spectrum * self.flambda2photon      # (nlbda,) [ph/s]

        return np.reshape(flux, (-1, 1, 1)) * psf  # Point source (nlbda, ny, nx)

                       
    def generate_background(self, spectrum):
        """ Generate a photon flux cube from uniform scene background spectrum.

        :param spectrum: uniform scene background spectrum [erg/s/cm²/Å/arcsec²]
        :return: (nlbda, ny, nx) photon flux cube [ph/s/spx]
        """
        # erg/s/cm²/Å/arcsec² * cm² * Å / erg/ph * arcsec² = ph/s/spx
        flux = spectrum * self.flambda2photon * self.spx_spatial_scale**2

        return np.full((self.nlbda, *self.spx_shape[::-1]), # y, x
                       flux[:, np.newaxis, np.newaxis])  # (nlbda, ny, nx)

    #
    # - tools
    #                       
    def chromatic_average(self, quantity):
        """ Chromatic average of a quantity (averaged over wavelength rather than px)

        :param array quantity: chromatic quantity (nlbda,)
        :return: chromatic average
        """
        return np.average(quantity, weights=np.diff(self.lbda_edges))
                       
    #
    # - Internal
    #
    @staticmethod
    def _get_chromatic_sigma(lbda, chromatic_sigma,
                                 constant_sigma,
                                 wref,
                                 xdims=0):
        """
        Get total PSF, including chromatic and constant components.

        The total (Gaussian) stddev is the quadratic sum of two components:

        - the chromatic stddev, proportional to wavelength,
          normalized at `wref`,
        - the achromatic, constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        :param lbda: wavelength
        :param chromatic_sigma: chromatic (linear) stddev
        :param constant_sigma: achromatic (constant) stddev
        :param float wref: reference wavelength (same unit as `lbda`)
        :param int xdims: extra dimensions to be appended
        :return: total stddev as function of wavelength
        """

        lmin, lmax = np.array(lbda)[[0, -1]]  # 1st and last wavelengths
        assert lmin > wref/3 and lmax < wref*3, \
            "Input and reference wavelengths probably not in same units."

        sigma = np.hypot(constant_sigma,
                         chromatic_sigma * (lbda / wref))  # [px]
        if xdims:
            sigma = sigma.reshape(sigma.shape + (1,) * xdims)

        return sigma
    
    
    # ================= #
    #  Properties       #
    # ================= #
    @property
    def mla_extent(self):
        """ MLA extent [spx]."""
        hx, hy = self.spx_shape / 2  # Half total width [spx]
        return [-hx, hx, -hy, hy]
    
    @property
    def flambda2photon(self):
        """ Chromatic conversion factor from erg/s/cm²/Å to ph/s. """
        dlbda = np.diff(self.lbda_edges)  # Spectral step (nlbda,)
        hnu = 1.9864459e-08 / self.lbda   # Photon energy [erg] with lbda in [Å]

        # erg/s/cm²/Å * (cm² * throughput * Å / erg/ph) = ph/s
        return (self.mirror.surface * 1e4 *
                self.throughput * dlbda / hnu)  # (nlbda,) [ph/s]

    # Spaxels
    @property
    def spaxels(self):
        """ """
        return self._spaxels

    @property
    def spx_shape(self):
        """ """
        return self._spaxels["shape"]

    @property
    def spx_centroids(self):
        """ """
        return self._spaxels["centroids"]
        
    @property
    def spx_edges(self):
        """ """
        return self._spaxels["edges"]

    @property
    def spx_spatial_scale(self):
        """ """
        return self._spaxels["spatial_scale"]
        
    @property
    def nlbda(self):
        """ number of spectral pixels. """
        return len(self.lbda)
    
    @property
    def meta(self):
        """ metadata of the instance. """
        return self._meta

    @property
    def name(self):
        """ name of the spectrograph (if any) """
        return self.meta.get("name", "")

    @property
    def lbda_ref(self):
        """ reference wavelength """
        # 1micron by default
        return self.meta.get("lbda_ref", 10_000) 
    
    @property
    def omega(self):
        """ spaxel solid angle [sr]. """
        hspx = self.spx_spatial_scale / 2  # [arcsec]
        hspx *= 4.84813681109536e-06   # [rad]
        return np.pi * np.sin(hspx)**2

    @property
    def psf_sigma_spectral(self):
        """ """
        return self.spatial_psf["sigma_spectral"]

    @property
    def xdisp_sigma_spectral(self):
        """ """
        return self.xdispersion["sigma_spectral"]
        
    

def plot_spectral_resolution(spectro, ax=None):
    """
    Plot effective spectral resolution.
    """

    wres = spectro.effective_resolution(npx=2)
    mwres = spectro.effective_resolution(npx=2, average=True)
    dlbda = np.diff(spectro.lbda_edges)           # [Å]
    sigma = spectro.get_xdisp_sigma_spectral() * dlbda  # [Å]

    lbda_mu = spectro.lbda / 1e4  # [µm]

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(tight_layout=True)

    ax.plot(lbda_mu, wres, c='C00')
    ax.set(xlabel="Wavelength [µm]",
           title="Resolving power (2 δλ): " +
           f"λ-mean={mwres:.0f}, min={wres.min():.0f}")
    ax.axhline(mwres, ls='--', c='C00')
    ax.set_ylabel(r"Resolving power $\mathcal{R} = \lambda/(2\delta\lambda)$",
                  c='C00')
    ax.tick_params(axis='y', labelcolor='C00');
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(lbda_mu, np.maximum(dlbda, sigma), c='C01')
    ax2.plot(lbda_mu, dlbda, c='C01', ls=':',
             label=f"Δλ ({spectro.dispersion_law!r}×{spectro.dispersion_scale})")
    ax2.plot(lbda_mu, sigma, c='C01', ls='--',
             label=f"σ ({spectro.xdisp_sigma_spectral} λ/µm & {spectro.xdisp_sigma} px)")
    ax2.set_ylabel("δλ [Å]", color='C01')
    ax2.tick_params(axis='y', labelcolor='C01');
    ax2.legend()

    return ax


if __name__ == "__main__":

    from mlaperf.iotools import get_config
    config = get_config("instrument.toml")

    spectro = Spectrograph(config["spectrograph"])
    print(spectro)
