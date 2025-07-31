"""
Spectrograph module, to simulate:

* spatial stage and MLA sampling
* spectral stage and dispersion law

Astrophysical scene is handled by :class:`mlaperf.scene.Scene` and detector by
:class:`mlaperf.detector.Detector`.

.. autosummary::

   Spectrograph
"""

__author__ = "Mickael Rigault <m.rigault@ip2i.in2p3.fr>, Yannick Copin <y.copin@ip2i.in2p3.fr>"

import warnings
import numpy as np
from copy import deepcopy
import astropy.units as u

from .utils import recursive_get
from . import iotools
from .telescope import Telescope
from .profiles import build_pixels
from .thermal import ThermalOptics

# ================ #
#                  #
#   Spectrograph   #
#                  #
# ================ #
def lbda_from_respow(spectral_range, res_power, npx_resolution=2):
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
    :param float npx_resolution: n-px resolution (i.e. n px per spectral elements)
    :return: mid and bin edge wavelengths [same units as input domain]
    """

    wmin, wmax = spectral_range
    dlog = 1/(2.302585092994046 * res_power * npx_resolution)  # ln(10)=2.303...
    npx = round((np.log10(wmax) - np.log10(wmin)) / dlog)
    loglbda_edges = np.linspace(np.log10(wmin), np.log10(wmax), npx + 1)
    loglbda_mid = (loglbda_edges[:-1] + loglbda_edges[1:])/2 # mean

    return 10**loglbda_mid, 10**loglbda_edges

def build_lbda(spectral_range, spectral_resolution=None, wsol=None, dsol=None, npx_resolution=2):
    """ set wavelength coordinates.

    Set :attr:`lbda` (:attr:`nlbda` mid wavelengths) and :attr:`lbda_egdes`
    (:attr:`nlbda` + 1 edge wavelengths).
    """
    if wsol is None:   # Compute from constant resolving power
        lbda, lbda_edges = lbda_from_respow(np.asarray(spectral_range, dtype="float32"), spectral_resolution,
                                                npx_resolution=npx_resolution)
    else:              # Compute from wavelength solution
        wmin, wmax = spectral_range
        npx = round(dsol(wmax) - dsol(wmin))+1  # Total nb of px
        lbda = wsol(np.r_[:npx])                # λ at bin center
        lbda_edges = wsol(np.r_[:npx+1] - 0.5)  # λ at bin edge
        
    return lbda, lbda_edges

def build_lbda_from_config(config):
    """ """
    spectral_range = config.get("spectral_range", None)

    dispersion_law = config.get("dispersion_law", None)
    dispersion_scale = float(config.get("dispersion_scale", 1))
    spectral_resolution = config.get("spectral_resolution", None)
    npx_resolution = config.get("dispersion_resolution", 2) # number of pixel per resolution element.
    
    # Set by dispersion law ?
    if dispersion_law is not None:
        # => ok, you have a dispersion law
        wname, dname = 'wavelength', 'offset'
        tab = iotools.read_ecsv( dispersion_law, colnames=[wname, dname])
        assert tab[dname].unit == 'pix'
        
        # dispersion solution (wavelengths in Å, offset in pix)
        dsol = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[dname] * dispersion_scale * npx_resolution/2,
            ext='extrapolate')
        
        # wavelength solution (wavelengths in Å, offset in pix)
        wsol = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[dname] * dispersion_scale * npx_resolution/2,
            ext='extrapolate', inverse=True)
        
    elif spectral_resolution is None:
        raise ValueError("'spectral_resolution' OR 'dispersion_law' should be set.")
    else:
        wsol = dsol = None
        dispersion_scale = float(dispersion_scale)

    lbda, lbda_edges = build_lbda(spectral_range, wsol=wsol, dsol=dsol,
                                  spectral_resolution=spectral_resolution,
                                  npx_resolution=npx_resolution)
    return lbda, lbda_edges

def build_spaxels_from_config(config):
    """ spaxel coordinates [spx] from MLA shape.

    Set `self.(x,y)[_edges]` from :attr:`spatial_shape`.
    """
    # Spaxel spatial scale (size of a single spaxel in arcsec)
    spatial_shape = recursive_get(config, "spatial_shape")
    spx_spatial_scale = recursive_get(config, "spatial_scale")
    
    return {"shape": spatial_shape,
            "spx_scale": spx_spatial_scale}

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
    _SPECTROGRAPH_TYPE = "Unknown"

    _SAMPLING = {"fine": {'spatial_shape': [58, 116], 'spatial_scale': 0.04},
                 "medium": {'spatial_shape': [58, 116], 'spatial_scale': 0.08}
                 }
    
    #: Mutable parameters (list)
    mutable_parameters = [#'spectral_range', 'spectral_resolution', # lbda
                          "dispersion_resolution",
                          'xdisp_sigma_spectral', 'xdisp_sigma', # xdisp_profile,
                          'psf_sigma_spectral', 'guiding_sigma', # psf_profile
                          'spatial_scale','spatial_shape' , # spaxels
                          'spx_scale','spx_shape' , # spaxels
                          ]

    def __init__(self, lbda, telescope=None,
                 spaxels={}, throughput=None,
                 spatial_psf={},
                 lbda_edges=None,
                 optics={},
                 dispersion_resolution=2, 
                 meta={}):
        """ Initialize the spectrograph. 
        You likely want to create it using the .from_config() constructor

        Parameters
        ----------
        lbda: array
            wavelengths in Angstrom
            
        telescope: slicersim.Telescope
            telescope object

        spaxels: dict
            spaxel information: 
            {shape: (N,M), spx_scale: float [in arcsec]}
            
        throughput: array, func
            throughput of the spectrograph. 
            if func => throughput = func(lbda)            
        """
        # wavelength
        self.lbda = lbda
        self.lbda_edges = lbda_edges

        # mirror
        self.telescope = telescope

        # optics
        self.optics = optics
        
        # Spaxels
        self.set_spaxels(**spaxels)
        
        # Throughput: flaot or func, see get_throughput()
        self.throughput = throughput

        self.spatial_psf = spatial_psf
        
        # affect the instance
        self.mutable_parameters = self.mutable_parameters + \
          [f"telescope.{k}" for k in self.telescope.mutable_parameters] + \
          [f"optics.{k}" for k in self.optics.mutable_parameters]

        meta["dispersion_resolution"] = dispersion_resolution
        self._meta_in = meta.copy()      #: Meta-parameters as input
        self._meta = meta.copy()         #: Meta-parameters as used

    @classmethod
    def from_config(cls, config, telescope=None):
        """ """
        # spectrograph that inherit Spectrograph simply need to update _parse_config()
        init_prop, _ = cls._parse_config(config)
        return cls(telescope=telescope, **init_prop)
        
    @staticmethod
    def _parse_config(config):
                # make sure the input config is not changed
        input_config = deepcopy(config)
        
        # wavelengths
        lbda, lbda_edges = build_lbda_from_config(input_config)
        
        # Spaxels
        spaxels = build_spaxels_from_config(input_config)
        
        # throughput
        throughput = build_throughput_from_config(input_config)

        # thermal optics
        optics = ThermalOptics.from_config(input_config["optics"])
        
        # in coming PSF
        psf = {"sigma_spectral": input_config["psf"]["spatial"]["psf_sigma_spectral"],
               "guiding_sigma": float(input_config["psf"]["spatial"]["guiding_sigma"]),
               "profile": input_config["psf"]["spatial"]["psf_profile"]}

        init_prop = {"lbda":lbda, 
                     "lbda_edges":lbda_edges, 
                     "spaxels":spaxels,
                     "throughput": throughput, 
                     "spatial_psf": psf,
                     "optics": optics,
                     "meta": input_config}
            
        return init_prop, input_config

        
    @staticmethod
    def build_lbda_from_config(config):
        return build_lbda_from_config(config)
        
    def update(self, reset_others=False, **kwargs):
        """ Update any mutable attribute of the spectrograph. """
        NAME_ALT = {"spx_shape": "spatial_shape",
                    "spx_scale": "spatial_scale"}
            
        updates = {}
        telescope_updates = {}
        optics_updates = {}
        lbda_updates = {}
        psf_updates = {}
        spaxel_updates = {}      
        
        # == Filling the update == #
        for k, v in kwargs.items():
            k = NAME_ALT.get(k, k)
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            if v is None:        # Skip
                continue

            # telescope
            if k.startswith("telescope."):
                telescope_updates[k.replace("telescope.", "")] = v
                continue

            # optics
            if k.startswith("optics."):
                optics_updates[k.replace("optics.", "")] = v
                continue            

            # lbda
            if k == "dispersion_resolution":
                # update the meta (for lsf)
                updates["dispersion_resolution"] = v
                # and update the wavelengths
                lbda_updates["dispersion_resolution"] = v
                
            # not allowed to change.
            
            # change PSF
            elif k in self.meta["psf"]["spatial"].keys():
                psf_updates[k.replace("psf_","")] = v
                
            # spaxels
            elif k in ('spatial_scale', 'spatial_scale_insigma', 
                     'spatial_shape', 'spatial_shape_insigma'):
                spaxel_updates[k] = v
            else:
                warning.warn(f"{k=} is unparsed")
            
        what_changed = []
        # update telescope
        if telescope_updates:
            what_changed.append("telescope")
            self.telescope.update(**telescope_updates, reset_others=reset_others)

        # update internal optics
        if optics_updates:
            what_changed.append("optics")
            self.optics.update(**optics_updates)
        
        # psf
        if psf_updates:
            what_changed.append("psf")
            self.spatial_psf = self.spatial_psf | psf_updates

        # lbda
        if spaxel_updates:
            what_changed.append("spaxels")
            spaxels = build_spaxels_from_config( self._meta | spaxel_updates )
            self.set_spaxels(**spaxels)
            
        if lbda_updates:
            what_changed.append("lbda")            
            self.update_lbda(*build_lbda_from_config(self._meta | lbda_updates) )

        if updates:
            what_changed.append("meta")
            
        # the rest if any
        if reset_others:
            self._meta = self._meta_in | updates
        else:
            self._meta = self._meta | updates
            
        return what_changed
        

    def update_lbda(self, lbda, lbda_edges):
        """ """
        self.lbda = lbda
        self.lbda_edges = lbda_edges

    def set_spaxels(self, shape, spx_scale):
        """ """
        self._spaxels = {"shape": shape, "spx_scale": spx_scale}
        self._spaxel_coords = {}

    # --------- #
    #  GETTER   #
    # --------- #
    def get_throughput(self):
        """ """
        if callable(self.throughput):
            return self.throughput(self.lbda)
        # float
        return self.throughput
    
    def get_spectrograph_shape(self, oversampling=None):
        """ """
        if oversampling is None or oversampling==1:
            return self.spx_shape
        
        return (np.asarray(self.spx_shape)*oversampling).astype(int)
    
    def get_spaxel_centroids(self, in_arcsec=False, squeeze=False, oversampling=None):
        """ """
        # numpy and all() test enables to have list/array oversampling (or float/int)
        if oversampling is None or (np.asarray(oversampling) == self.spaxels["oversampling"]).all():
            spaxels_coords = self.spaxels
        else:
            spaxels_coords = build_pixels(self.spx_shape, oversampling=oversampling)
        
        x, y = spaxels_coords["centroids"]
        if in_arcsec:
            if np.ndim(self.spx_spatial_scale) == 0:
                spx_y = spx_x = self.spx_spatial_scale
            else:
                spx_y, spx_x = self.spx_spatial_scale
            # not /= not to change 
            x = x * spx_x / spaxels_coords["oversampling"]  # in arcsec
            y = y * spx_y / spaxels_coords["oversampling"]  # in arcsec

        if squeeze:
            x, y = np.squeeze(x), np.squeeze(y)
            
        return (x, y), spaxels_coords["oversampling"]
        
    def get_psf_sigma_spectral(self, xdims=0, guiding_sigma=None,
                              in_spaxels=False, lbda=None):
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
        if lbda is None:
            lbda = self.lbda
            
        if guiding_sigma is None:
            guiding_sigma = self.spatial_psf["guiding_sigma"]

        sigma = self._get_chromatic_sigma(lbda,
                                        chromatic_sigma = self.psf_sigma_spectral,
                                        constant_sigma = guiding_sigma,
                                        wref = self.lbda_ref,
                                        xdims=xdims)
        if in_spaxels:
            # not inplace to allow non-float spx_spatial_scale
            sigma = sigma/self.spx_spatial_scale
            
        return sigma
    
    def get_spatial_psf(self, profile="default", position=(0, 0),
                            guiding_sigma=None, oversampling=None,
                            as_oversampled=False,
                            **kwargs):
        """ Get normalized 2D spatial PSF on the MLA.
        
        Parameters
        ----------
        profile: str
            profile to be used:
            - gaussian: gaussian profile using exact erf intergration [*default*]
            - airy: airy dist profile generated by the telescope mirror
            
        position: (float, float)
            location of the point source within the slicer/mla in unit of slice/spaxel

        guiding_sigma: float, None
            gaussian noise convolution (in arcsec) caused by jitter.

        oversampling: int
            oversampling factor used for the PSF (except if profile is gaussian, where erf is used)
            If oversampling is None, 5 is used by default. Set oversampling=1 for no oversampling.

        Returns
        -------
        psf:
            # (nlbda, ny, nx)

        This uses the exact 2D Gaussian PSF integration over the spx.

        :param 2-tuple position: point source position in MLA [spx]
        :return: normalized PSF (nlbda, ny, nx)
        """
        from . import profiles
        if oversampling is None:
            oversampling = 1
            
        # Gaussian        
        if profile in ["gaussian", "default", "normal", "norm"]:
            sigmas = self.get_psf_sigma_spectral(guiding_sigma=guiding_sigma)
            # allows asymetric spx_spatial_scale
            sigmas = sigmas[:, None, None] / self.spx_spatial_scale
            prop = dict(in_arcsec=False, squeeze=False, oversampling=1)
            if as_oversampled: # no need to oversample as this uses exact erf functions.
                print(f"changin oversampling to {oversampling=}")
                prop["oversampling"] = oversampling

            (xx, yy), oversampling = self.get_spaxel_centroids(**prop)
                 
            psf = profiles.get_gaussian2d(xx, yy, sigma=sigmas, mean=position, **kwargs)
            
            if as_oversampled:
                if np.ndim(oversampling) == 1:
                    oversampling_y, oversampling_x = oversampling
                else:
                    oversampling_y = oversampling_x = oversampling
                    
                psf *= oversampling_x*oversampling_y # to conserve energy
                
            return psf

        #
        # Generic PSF
        #
        if guiding_sigma is None:
            guiding_sigma = self.spatial_psf["guiding_sigma"]


        # to accomodate with non-square spaxels (like slicer)
        # we work in arcsec, not in spaxels.
        if profile in ["airy", "mirror", "telescope", "airydisk"]:
            radius = self.telescope.get_airy_radius(self.lbda) # in arcsec

            position = np.asarray(position)  # in spaxels
            psf_func = profiles.get_profilemodel("airy", position=position,
                                                      radius=radius[:, None, None],
                                                      normalized=True)
        else:
            raise ValueError(f"psf profile {profile=} not implemented")

        # coordinates including oversampling for exact PSF profile and guiding convolution
        # x and y are in arcsec as radius is given in arcsec.
        (xx, yy), oversampling = self.get_spaxel_centroids(in_arcsec=True, squeeze=False,
                                                           oversampling=oversampling)
        psf = psf_func(xx, yy)
         
        if guiding_sigma is not None and guiding_sigma>0:
            psf = self._apply_guiding(psf, guiding_arcsec=guiding_sigma, oversampling=oversampling)
            
        if oversampling !=1 and not as_oversampled:
            psf = self._remove_oversampling(psf, oversampling=oversampling, func=np.nanmean)

        return psf                         # (nlbda, ny, nx)

    # - internal tricks
    def _apply_guiding(self, image, guiding_arcsec, oversampling=1):
        """ """
        from scipy.ndimage import gaussian_filter
        # gaussian convolution | guiding_sigma is given in arcsec
        # works by itself if spx_spatial_scale is a list or a float.
        sigma_guiding_pixels = guiding_arcsec / (self.spx_spatial_scale/oversampling) # in arcsec=>spaxels
        return gaussian_filter(image, sigma_guiding_pixels, axes=(-2,-1))

    @staticmethod
    def _remove_oversampling(image, oversampling, func=np.mean):
        """ """
        if np.ndim(image) == 2:
            block_size = (oversampling, oversampling)
        elif np.ndim(image) == 3:
            block_size = (1, oversampling, oversampling)
        else:
            raise ValueError(f"image ndim must be 2 or 3 {np.ndim(image)=} given.")
        
        from astropy import nddata
        return nddata.block_reduce(image, block_size=(1, oversampling, oversampling), func=func)

    #  internal tricks 

    def cube_to_slice(self, cube, lbda_range, func=np.nansum, squeeze=False):
        """ get slices of the given cube. 
    
        cube: 3d-array
            cube of data
    
        lbda_range: (float, float), list
            wave_min, wave_max [A] (or list of: e.g. [[4000, 5000], [6000, 9000]])
            in obs-frame.
    
        func: function
            function to apply to merge wavelengths into one slice (np.mean, np.max, np.sum etc.)
    
        squeeze: bool
            for conveniance, should np.squeeze be called to remove (1,) dimensions ?
    
        Returns
        -------
        array: 
            (n-lbda_range, ny, nx)        
        """
        slices = []
        for wmin, wmax in np.atleast_2d(lbda_range):
            flag_lbda = ((self.lbda >= wmin) & (self.lbda <= wmax))
            slices.append( func(cube[flag_lbda], axis=0) )
            
        slices = np.asarray(slices)
        if squeeze:
            slices = np.squeeze(slices)
    
        return slices    

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

    def get_nea_telescope_airy(self, position=(0,0), in_spaxels=True):
        """ """
        if in_spaxels:
            norm_scale = self.spx_spatial_scale
        
        return self.telescope.get_nea_airy(self.lbda, norm_scale=norm_scale, position=position)

    # ------------- #
    #   Others      #
    # ------------- #
    def point_source_variance(self, varcube, position=(0, 0), radius=5,
                                  psf_profile="default",
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
        if np.ndim(radius) == 0:
            radius_y = radius_x = radius
        elif np.ndim(radius) == 1 and len(radius) == 2:
            radius_y, radius_x = radius
        else:
            raise ValueError(f"cannot parse input {raduis=}")
        
        # Spatial PSF
        psf = self.get_spatial_psf(position=position, profile=psf_profile)  # (nlbda, ny, nx)

        # Aperture
        x0, y0 = position
        (x, y), oversampling = self.get_spaxel_centroids()
        r_radius = np.hypot((x - x0)/radius_x, (y - y0)/radius_y)  # (ny, nx) [spx]
        aper = (r_radius <= 1)
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
            sigma = self.get_spectral_dispersion()
            
        sigma = np.maximum(sigma, 1)

        dlbda = np.diff(self.lbda_edges)
        wres = self.lbda / (npx * sigma * dlbda)  # (nlbda,)

        if average:             # Chromatic average
            wres = np.average(wres, weights=dlbda)

        return wres

    def get_spectral_dispersion(self):
        """ """
        raise NotImplementedError()

    def get_lsf_dispersion(self):
        """ get the gaussian LSF dispersion sigma in units of wavelength bin """
        return self.meta.get("dispersion_resolution", 2) /2.

    def apply_line_spread_function(self, fluxes, **kwargs):
        """ """
        from scipy.ndimage import gaussian_filter1d
        
        # wavelength is the first direction of fluxes
        lsf_sigma = self.get_lsf_dispersion()
        
        # apply a gaussian filter to input fluxes ; wavelength axis is 0 by default
        return gaussian_filter1d(fluxes, lsf_sigma, **({"axis":0} | kwargs) )
    
    # ------------- #
    #    generate   #
    # ------------- #
    # Point Souce
    def generate_point_source(self, spectrum, position=(0, 0),
                                  psf_profile="default",
                                  oversampling=None,
                                  as_oversampled=False,
                                  apply_lsf=True):
        """ Generate a photon flux cube from a point source spectrum.

        :param spectrum: point source spectrum [erg/s/cm²/Å]
        :param 2-tuple position: point source position in MLA [spx]
        :return: (nlbda, ny, nx) photon flux cube [ph/s/spx]
        """
        # Spatial PSF
        psf = self.get_spatial_psf(profile=psf_profile, position=position,
                                    as_oversampled=as_oversampled,
                                    oversampling=oversampling,
                                  )       # (nlbda, ny, nx)

        # erg/s/cm²/Å / erg/ph * cm² * Å = ph/s
        flux = spectrum * self.flambda2photon      # (nlbda,) [ph/s]

        psf_cube = np.reshape(flux, (-1, 1, 1)) * psf  # Point source (nlbda, ny, nx)
        if apply_lsf:
            psf_cube = self.apply_line_spread_function(psf_cube)
            
        return psf_cube

    # Flat Background    
    def generate_background(self, spectrum, oversampling=None,
                                apply_lsf=True):
        """ Generate a photon flux cube from uniform scene background spectrum.

        :param spectrum: uniform scene background spectrum [erg/s/cm²/Å/arcsec²]
        :return: (nlbda, ny, nx) photon flux cube [ph/s/spx]
        """

        # erg/s/cm²/Å/arcsec² * cm² * Å / erg/ph * arcsec² = ph/s/spx
        flux = spectrum * self.flambda2photon * self.spx_area

        bkgd_cube = np.full((self.nlbda, *self.get_spectrograph_shape(oversampling=oversampling) ), # y, x
                                flux[:, np.newaxis, np.newaxis])  # (nlbda, ny, nx)
        if apply_lsf:
            bkgd_cube = self.apply_line_spread_function(bkgd_cube)
                                
        return bkgd_cube

    # Structured background
    def generate_structured_background(self, *args, apply_lsf=True, **kwargs):
        """ Generate a photon flux cube from uniform scene a structured background scene (e.g., host). """
        raise NotImplementedError("generate_structured_background() has not been implemented.")

    # Themal (pre-dispersor)
    def generate_thermal_signal(self, lbda_bin=None,
                                as_cube=False, oversampling=None,
                                apply_lsf=True, as_sum=True):
        """ Telescope mirror thermal signal [ph/s/spx/Δλ].

        Parameters
        ----------
        lbda_bin:  
            (nlbda, 2) list of spectral domains [Å] or spectral px by default

        temperature: float
            mirror temperature [K], or default one
            
        emissivity: float
            mirror emissivity, or default one
            
        as_cube: bool
            output format (3d cube of float or float)

        as_sum : bool
            if multiple mirrors, should this be the sum of all contribution (as_sum=True)
            or the list of (False).

        Returns
        -------
        thermal signal in ph/s/spx/Δλ (3d cube or float, see as_cube)
        """
        if lbda_bin is None:
            lbda_bin = self.lbda_bin

        signal = self.telescope.get_thermal_signal(lbda_bin,
                                                   solid_angle=self.omega,     # Spx solid angle [sr]
                                                   as_sum=as_sum)
        # output formating
        if as_cube:
            signal = np.full((self.nlbda, *self.get_spectrograph_shape(oversampling=oversampling)),
                                 signal[..., np.newaxis, np.newaxis])  # (nlbda, ny, nx)
                                 
        if apply_lsf:
            signal = self.apply_line_spread_function(signal)
            
        return signal                              # [ph/s/spx/Δλ]

    def get_thermal_dark(self, pixel_area, as_sum=True):
        """ 
        Parameters
        ----------
        pixel_area: float
            detector pixel surface in m2

        Returns
        -------
        float, array
            ph/s associated to thermal radiations.
        """
        # properties at the detector
        ## each pixels integrate the whole spectral range.
        lbda_bin = self.lbda[[0,-1]]

        
        # get the dark signal in ph/s
        signals = self.optics.get_signal(lbda_bin = lbda_bin,        # expectedin in [A]
                                         area = pixel_area)          # Collecting area [m²]
        
                                          
        # sum over 1 element if only 1 temperature
        if as_sum:
            signals = np.sum(signals, axis=0)

        return signals

        
    # Empty cube
    def get_empty_cube(self, filled=0, oversampling=None):
        """ """
        # no apply LSF as zeros...
        ny, nx = self.get_spectrograph_shape(oversampling=oversampling)
        return filled * np.ones( (self.nlbda, ny, nx) )

    # ------------ #
    #   GETTER     #
    # ------------ #
    def get_nea_spatial(self, position=(0,0), in_spaxels=True, guiding_sigma=None):
        """ noise equivalent area in unit of slice/spaxels 

        i.e., how many "spaxel noise")
        
        Parameters
        ----------
        position: (float, float)
            position of the PSF in unit of spaxel / slicer

        in_spaxels: bool
            shall the unit of the area be in spaxels**2 ? 
            If False => arcsec**2

        Returns
        -------
        array
        """
        from .nea import get_2dnorm_nea
        
        if self.spatial_psf["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian spatial PSF profile implemented, but: {self.spatial_psf['profile']=}")
            
        sigma_at_spectro = self.get_psf_sigma_spectral(in_spaxels=in_spaxels,
                                                   guiding_sigma=guiding_sigma,
                                                   xdims=2)
        return get_2dnorm_nea(sigma_at_spectro, mean=position)
    
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

    def show_nea(self, ax=None, position=(0,0), legend=True ):
        """ """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            
        # data
        nea_spatial = self.get_nea_spatial(in_spaxels=True, position=position)
        nea_no_guiding = self.get_nea_spatial(guiding_sigma=0, in_spaxels=True, position=position)
        nea_telescope = self.get_nea_telescope_airy(in_spaxels=True, position=position)


        
        ax.plot(self.lbda, self.get_nea(position = position, nea_spatial=nea_spatial),
                    color="#194D80", label="total")
        ax.plot(self.lbda, self.get_nea(position = position, nea_spatial=nea_no_guiding),
                    color="#194D80", ls="--", label="without guiding")
        ax.plot(self.lbda, self.get_nea(position = position, nea_spatial=nea_telescope),
                    color="#F8AD05", label="airy from telescope")
        ax.legend(fontsize="small", frameon=False)

        ax.set_xlabel(f"wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("NEA [in pixels]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)
        
        return fig
    
    def show_nea_spatial(self, ax=None, position=(0,0), legend=True ):
        """ """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            
        # data
        nea = self.get_nea_spatial(in_spaxels=True, position=position)
        nea_no_guiding = self.get_nea_spatial(guiding_sigma=0, in_spaxels=True, position=position)
        nea_telescope = self.get_nea_telescope_airy(in_spaxels=True, position=position)
        
        ax.plot(self.lbda, nea, color="#194D80", label="total")
        ax.plot(self.lbda, nea_no_guiding, color="#194D80", ls="--", label="without guiding")
        ax.plot(self.lbda, nea_telescope, color="#F8AD05", label="airy from telescope")

        ax.set_xlabel(f"wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("NEA [in spaxels]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)
        
        return fig

    def show_fwhm(self, ax=None, legend=True, guiding_arcsec=None, in_arcsec=False, show_band=True):
        """ """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        in_spaxels = not in_arcsec
        norm_scale = 1 if in_arcsec else self.spx_spatial_scale
        norm_sampling = self.spx_spatial_scale if in_arcsec else 1
        # sigma
        sigma_at_mla = self.get_psf_sigma_spectral(in_spaxels=in_spaxels, guiding_sigma=guiding_arcsec)
        sigma_at_mla_no_guiding = self.get_psf_sigma_spectral(in_spaxels=in_spaxels, guiding_sigma=0)
        radius = self.telescope.get_airy_radius(self.lbda, norm_scale=1 if in_arcsec else self.spx_spatial_scale)

        ax.plot(self.lbda, 2.35*sigma_at_mla, color="#194D80", label="total scatter")
        ax.plot(self.lbda, 2.35*sigma_at_mla_no_guiding, color="#194D80", ls="--", label="without guiding")
        ax.plot(self.lbda, 0.8*radius, color="#F8AD05", label="airy from telescope")
        
        if show_band:
            ax.axhspan(2*norm_sampling, 2.35*norm_sampling, color="tab:orange", alpha=0.05, lw=0)
            _ylow, _ = ax.get_ylim()
            ax.axhline(2*norm_sampling , color="tab:red", alpha=1, ls="--", lw=0.5)
            ax.axhspan(0, 2*norm_sampling, color="tab:red", alpha=0.05, lw=0)
            ax.set_ylim(_ylow)
            
        ax.set_xlabel(f"wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("FWHM [in spaxels]" if in_spaxels else "FWHM [in arcsec]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)
            
        return fig

    def show_psf(self, lbda_range, profile="default", 
                 guiding_arcsec=None, axes=None,
                 position=(0,0), oversampling=5,
                 in_arcsec=False,
                 norm="log", **kwargs):
        """ """
        from matplotlib import colors
        
        if norm is None or norm in ["linear"]:
            norm = colors.Normalize
        elif norm in ["log"]:
            norm = colors.LogNorm

        extent = np.asarray(self.mla_extent)
        if in_arcsec:
            extent *= self.spx_spatial_scale
            
            
        if guiding_arcsec is None:
            guiding_arcsec = self.spatial_psf["guiding_sigma"]
    
        if axes is None:
            import matplotlib.pyplot as plt
            fig, (ax, axg, axsl) = plt.subplots(nrows=3, 
                                                figsize=(5,7), 
                                                gridspec_kw={"hspace":0.04})
        else:
            (ax, axg, axsl) = axes
            fig = ax.figure
            
        # highly-resolved cube
        psf_cube = self.get_spatial_psf(profile=profile, guiding_sigma=0, # no guiding error yet
                                        oversampling=oversampling,
                                        as_oversampled=True, 
                                       position=position)
    
        # perfect model
        psf_slice = self.cube_to_slice(psf_cube, lbda_range, squeeze=True)
        ax.imshow(psf_slice, norm=norm(), extent=extent, **kwargs)
        ax.set_xticklabels([])
        
        # adding guiding
        psf_cube = self._apply_guiding(psf_cube, guiding_arcsec=guiding_arcsec, oversampling=oversampling)
        psf_slice = self.cube_to_slice(psf_cube, lbda_range, squeeze=True)
        axg.imshow(psf_slice, norm=norm(), extent=extent, **kwargs)
        axg.set_xticklabels([])
        
        # removing oversampling
        psf_cube = self._remove_oversampling(psf_cube, oversampling=oversampling)
        psf_slice = self.cube_to_slice(psf_cube, lbda_range, squeeze=True)
        axsl.imshow(psf_slice, norm=norm(), extent=extent, **kwargs)
    
        return fig
        
    # ================= #
    #  Properties       #
    # ================= #
    @property
    def mla_extent(self):
        """ MLA extent [spx]."""
        hy, hx = np.asarray(self.spx_shape) / 2  # Half total width [spx]
        return [-hx, hx, -hy, hy]
    
    @property
    def flambda2photon(self):
        """ Chromatic conversion factor from erg/s/cm²/Å to ph/s. """
        dlbda = np.diff(self.lbda_edges)  # Spectral step (nlbda,)
        hnu = 1.9864459e-08 / self.lbda   # Photon energy [erg] with lbda in [Å]

        # erg/s/cm²/Å * (cm² * throughput * Å / erg/ph) = ph/s
        return (self.telescope.surface * 1e4 *
                self.get_throughput() * dlbda / hnu)  # (nlbda,) [ph/s]

    # Spaxels
    @property
    def _hspaxels(self):
        """ spaxel properties {shape: (N,M), spx_scale: float [in arcsec]} """
        # 
        return self._spaxels

    @property
    def spaxels(self):
        """ """
        # build from self._spaxels
        if self._spaxel_coords is None or len(self._spaxel_coords) == 0:
            self._spaxel_coords = build_pixels(self._spaxels["shape"], oversampling=1)
            
        return self._spaxel_coords
    
    @property
    def spx_shape(self):
        """ """
        return self.spaxels["shape"]

    @property
    def spx_centroids(self):
        """ """
        return self.spaxels["centroids"]
        
    @property
    def spx_edges(self):
        """ """
        return self.spaxels["edges"]
        
    @property
    def spx_area(self):
        """ """
        if np.ndim(self.spx_spatial_scale) == 0:
            spx_spatial_scale_y = spx_spatial_scale_x = self.spx_spatial_scale
        else:
            spx_spatial_scale_y, spx_spatial_scale_x = self.spx_spatial_scale
        return spx_spatial_scale_y * spx_spatial_scale_x
        
    @property
    def spx_spatial_scale(self):
        """ """
        spx_spatial_scale = self._spaxels.get("spx_scale", None)
        if spx_spatial_scale is None:
            from .profiles import airyradius_to_gaussiansigma
            warnings.warn("Setting spaxels scale from airy")
            radius_airy = self.telescope.get_airy_radius(self.lbda_ref)
            spx_spatial_scale = airyradius_to_gaussiansigma(radius_airy, on="fwhm")
            
        return spx_spatial_scale 
        
    @property
    def nlbda(self):
        """ number of spectral pixels. """
        return len(self.lbda)

    @property
    def lbda_bin(self):
        """ wavelength edge bins. """
        return np.vstack([self.lbda_edges[:-1],
                          self.lbda_edges[1:]]).T  # (nlbda, 2) [Å]
                                  
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
        hspx = self.spx_spatial_scale * 4.84813681109536e-06   # [rad]
        if np.ndim(hspx)==0:
            hspx_y = hspx_x = hspx
        else:
            hspx_y, hspx_x = hspx

        # Case: square | could remove np.sin that has no impact
        return np.sin(hspx_y) * np.sin(hspx_x)
    
        # Case: circular            
        # return np.pi * np.sin(hspx_y/2) * np.sin(hspx_x/2)

    @property
    def skyarea(self):
        """ full sky area (nspaxel * spaxel area) """
        pass

    @property
    def psf_sigma_spectral(self):
        """ """
        if self.spatial_psf["sigma_spectral"] is None or \
          self.spatial_psf["sigma_spectral"] in ["default"]:
            # 2.9 is the airy equivalent.
            from .profiles import airyradius_to_gaussiansigma
            radius_airy = self.telescope.get_airy_radius(self.lbda_ref)
            self.spatial_psf["sigma_spectral"] = airyradius_to_gaussiansigma(radius_airy, on="fwhm")
            
        return float(self.spatial_psf["sigma_spectral"])

    @property
    def type(self):
        """ """
        return self._SPECTROGRAPH_TYPE

    
    
class MLASpectrograph( Spectrograph ):
    """ a spectrograph with no anamorphose and x-dispersion for traces """
    _ANAMORPHOSE = None
    _SPECTROGRAPH_TYPE = "mla"

    def __init__(self, xdispersion={}, *args, **kwargs):
        """ """
        _ = super().__init__(*args, **kwargs)
        
        # an MLA has a cross-dispersion
        self.xdispersion = xdispersion

    # this is what from_config needs
    @classmethod
    def _parse_config(cls, config):
        """ """
        
        init_prop, config = super()._parse_config(config)
        
        # PSF at the detector level
        xdispersion = {"sigma_spectral": float(config["psf"]["detector"]["xdisp_sigma_spectral"]),
                      "sigma": float(config["psf"]["detector"]["xdisp_sigma"]),
                      "profile": config["psf"]["detector"]["xdisp_profile"]}
        init_prop["xdispersion"] = xdispersion
        return init_prop, config
    
    def update(self, reset_others=False, **kwargs):
        """ """
        # do xdispersion stuffs
        xdisp_updates = {}
        for k, v in kwargs.items():
            if k in self.meta["psf"]["detector"].keys():
                xdisp_updates[k.replace("xdisp_", "")] = v
                _ = kwargs.pop(k) # remove them
                
        self.xdispersion = self.xdispersion | xdisp_updates
        # and the rest (normal spectrograph)
        return super().update(reset_others=False, **kwargs)

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
    def get_spectral_dispersion(self):
        """ this is given by the x-dispersion profile. """
        return self.get_xdisp_sigma_spectral()  # (nlbda,)


    def get_nea_spatial(self, position=(0,0), in_spaxels=True, guiding_sigma=None):
        """ noise equivalent area in unit of slice/spaxels 

        i.e., how many "spaxel noise")
        
        Parameters
        ----------
        position: (float, float)
            position of the PSF in unit of spaxel / slicer

        in_spaxels: bool
            shall the unit of the area be in spaxels**2 ? 
            If False => arcsec**2

        Returns
        -------
        array
        """
        from .nea import get_2dnorm_nea
        
        if self.spatial_psf["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian spatial PSF profile implemented, but: {self.spatial_psf['profile']=}")
            
        sigma_at_mla = self.get_psf_sigma_spectral(in_spaxels=in_spaxels,
                                                   guiding_sigma=guiding_sigma,
                                                   xdims=2)
        return get_2dnorm_nea(sigma_at_mla, mean = position)

    def get_nea_pixels(self):
        """ noise equivalent area of a spaxel /slice in unit of pixels

        Returns
        -------
        array
        """
        from .nea import get_1dnorm_nea
        if self.xdispersion["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian xdispersion PSF profile implemented, but: {self.xdispersion['profile']=}")
            
        sigma_xdisp_at_detector = self.get_xdisp_sigma_spectral()  # in pixels 
        return get_1dnorm_nea(sigma_xdisp_at_detector)
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def xdisp_sigma_spectral(self):
        """ """
        return self.xdispersion["sigma_spectral"]


class SlicerSpectrograph( Spectrograph ):
    """ """
    _ANAMORPHOSE = (2, 1)
    _SPECTROGRAPH_TYPE = "slicer"
    
    def set_spaxels(self, shape, spx_scale):
        """ """
        shape = np.asarray(shape) * self._ANAMORPHOSE
        spx_scale = np.asarray(spx_scale) / self._ANAMORPHOSE
        self._spaxels = {"shape": shape, "spx_scale": spx_scale}
        self._spaxel_coords = {}

    def get_nea_pixels(self):
        """ noise equivalent area of a slice (per wavelength) in unit of pixels

        Returns
        -------
        array
        """
        return 1 # just an image now. 
        
#    def get_spectrograph_shape(oversampling=None):
#        """ """
#        oversampling = self._parse_oversampling_(oversampling)
#        return super().get_spectrograph_shape(oversampling=oversampling)

#    @classmethod
#    def _parse_oversampling_(cls, oversampling):
#        """ oversampling accounting for the anamorphose """
#        if oversampling is None:
#            oversampling = cls._ANAMORPHOSE
            
#        elif np.ndim(oversampling) == 0: #int or float
#            oversampling = oversampling*np.asarray( cls._ANAMORPHOSE ).astype(int)
#
#        elif np.ndim(oversampling) != 1 or len(oversampling) != 2: #int or float
#            raise ValueError(f"could not parse {oversampling=}, should be int or (nslicers, npixels) ")
        
        
