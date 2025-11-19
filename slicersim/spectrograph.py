""" Spectrograph module. """

import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
import pandas

from . import iotools
from .profiles import build_pixels
from .thermal import ThermalOptics
from .utils import recursive_get


# ================ #
#                  #
#   Spectrograph   #
#                  #
# ================ #
def lbda_from_resolution_power(spectral_range, res_power, npx_resolution=2):
    """Compute wavelength ramp for constant n-px resolving power.

    Parameters
    ----------
    spectral_range : tuple
        Spectral domain (wmin, wmax).
    res_power : float
        Resolving power R.
    npx_resolution : float, optional
        Number of pixels per spectral element. Default is 2.

    Returns
    -------
    lbda : array_like
        Mid-bin wavelengths.
    lbda_edges : array_like
        Bin edge wavelengths.

    Notes
    -----
    .. math::

       \frac{\Delta\lambda}{\lambda} &= \frac{1}{n\mathcal{R}} \\
       &= \ln 10\,\Delta\log\lambda \\
       \log\lambda_i^e &= \log\lambda_0 + i\times \Delta\log\lambda
       &\quad\text{(edges)} \\
       \log\lambda_i &= (\log\lambda_i + \log\lambda_{i+1})/2
       &\quad\text{(center)}
    """

    wmin, wmax = spectral_range
    # ln(10)=2.303...
    dlog = 1 / (2.302585092994046 * res_power * npx_resolution)  
    npx = round((np.log10(wmax) - np.log10(wmin)) / dlog)
    loglbda_edges = np.linspace(np.log10(wmin), np.log10(wmax), npx + 1)
    loglbda_mid = (loglbda_edges[:-1] + loglbda_edges[1:]) / 2  # mean

    return 10**loglbda_mid, 10**loglbda_edges


def build_lbda(spectral_range, spectral_resolution=None,
                   wsol=None, dsol=None,
                   npx_resolution=2):
    """Set wavelength coordinates.

    This function sets the mid-bin wavelengths (`lbda`) and the bin edge
    wavelengths (`lbda_edges`).

    Parameters
    ----------
    spectral_range : tuple
        Spectral domain (wmin, wmax).
    spectral_resolution : float, optional
        Resolving power R. Default is None.
    wsol : callable, optional
        Wavelength solution (pixel -> wavelength). Default is None.
    dsol : callable, optional
        Dispersion solution (wavelength -> pixel). Default is None.
    npx_resolution : float, optional
        Number of pixels per spectral element. Default is 2.

    Returns
    -------
    lbda : array_like
        Mid-bin wavelengths.
    lbda_edges : array_like
        Bin edge wavelengths.
    """
    if wsol is None:  # Compute from constant resolving power
        lbda, lbda_edges = lbda_from_resolution_power(np.asarray(spectral_range, dtype="float32"),
                                                          spectral_resolution,
                                                          npx_resolution=npx_resolution)
    else:  # Compute from wavelength solution
        wmin, wmax = spectral_range
        npx = round(dsol(wmax) - dsol(wmin)) + 1  # Total nb of px
        lbda = wsol(np.r_[:npx])  # λ at bin center
        lbda_edges = wsol(np.r_[:npx + 1] - 0.5)  # λ at bin edge

    return lbda, lbda_edges


def build_lbda_from_config(config):
    """Build wavelength coordinates from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
        - spectral_range
        - dispersion_law # is a file
        - dispersion_scale # float
        - spectral_resolution
        - dispersion_resolution
        - 

    Returns
    -------
    lbda : array_like
        Mid-bin wavelengths.
    lbda_edges : array_like
        Bin edge wavelengths.
    """
    spectral_range = config.get("spectral_range", None)

    dispersion_law = config.get("dispersion_law", None)
    dispersion_scale = float(config.get("dispersion_scale", 1))
    
    spectral_resolution = config.get("spectral_resolution", None)
    npx_resolution = config.get("dispersion_resolution", 2)  # number of pixel per resolution element.

    # Set by dispersion law ?
    if dispersion_law is not None:
        # => ok, you have a dispersion law
        wname, dname = 'wavelength', 'offset'
        tab = iotools.read_ecsv(dispersion_law, colnames=[wname, dname])
        assert tab[dname].unit == 'pix'

        # dispersion solution (wavelengths in Å, offset in pix)
        dsol = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[dname] * dispersion_scale * npx_resolution / 2, # 2. is the expected nyquist limit
            ext='extrapolate')

        # wavelength solution (wavelengths in Å, offset in pix)
        wsol = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[dname] * dispersion_scale * npx_resolution / 2,# 2. is the expected nyquist limit
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
    """Build spaxel coordinates from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary with spaxel information ('shape' and 'spx_scale').
    """
    # Spaxel spatial scale (size of a single spaxel in arcsec)
    spatial_shape = recursive_get(config, "spatial_shape")
    spx_spatial_scale = recursive_get(config, "spatial_scale")

    return {"shape": spatial_shape,
            "spx_scale": spx_spatial_scale}


def build_throughput_from_config(config):
    """Build throughput from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    float or callable
        Throughput as a constant value or a function of wavelength.
    """
    try:
        # throughput is a constant
        throughput = float(config["throughput"])

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
    """Spectrograph simulation.

    .. warning::
        The spectral PSF impacts the cross-dispersion profile,
        but is not applied in the dispersion direction.
    """
    _SPECTROGRAPH_TYPE = "Unknown"

    _SAMPLING = {"fine": {'spatial_shape': [29, 58], 'spatial_scale': 0.04},
                 "medium": {'spatial_shape': [29, 58], 'spatial_scale': 0.08}
                 }

    #: Mutable parameters (list)
    mutable_parameters = [  # 'spectral_range', 'spectral_resolution', # lbda
        "dispersion_resolution", "spotsize", "dispersion_scale",
        'xdisp_sigma_spectral', 'xdisp_sigma',  # xdisp_profile,
        'psf_sigma_spectral', 'guiding_sigma',  # psf_profile
        'spatial_scale', 'spatial_shape',  # spaxels
        'spx_scale', 'spx_shape',  # spaxels
        ]

    def __init__(self, lbda, telescope=None,
                 spaxels={}, throughput=None,
                 spatial_psf={},
                 lbda_edges=None,
                 optics={},
                 dispersion_resolution=2,
                 meta={}):
        """Initialize the spectrograph.

        You likely want to create it using the .from_config() constructor.

        Parameters
        ----------
        lbda : array_like
            Wavelengths in Angstrom.
        telescope : slicersim.Telescope, optional
            Telescope object. Default is None.
        spaxels : dict, optional
            Spaxel information: {"shape": (N,M), "spx_scale": float [in arcsec]}.
            Default is {}.
        throughput : float or callable, optional
            Throughput of the spectrograph. If callable, throughput = func(lbda).
            Default is None.
        spatial_psf : dict, optional
            Spatial PSF information. Default is {}.
        lbda_edges : array_like, optional
            Wavelength bin edges in Angstrom. Default is None.
        optics : dict, optional
            Optics information. Default is {}.
        dispersion_resolution : int, optional
            Dispersion resolution. Default is 2.
        meta : dict, optional
            Metadata dictionary. Default is {}.
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
        self.set_throughput(throughput)

        self.spatial_psf = spatial_psf

        # affect the instance
        self.mutable_parameters = self.mutable_parameters + \
                                  [f"telescope.{k}" for k in self.telescope.mutable_parameters] + \
                                  [f"optics.{k}" for k in self.optics.mutable_parameters]

        meta["dispersion_resolution"] = dispersion_resolution
        self._meta_in = meta.copy()  #: Meta-parameters as input
        self._meta = meta.copy()  #: Meta-parameters as used

    @classmethod
    def from_config(cls, config, telescope=None):
        """Create a Spectrograph instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        telescope : slicersim.Telescope, optional
            Telescope object. Default is None.

        Returns
        -------
        Spectrograph
            A Spectrograph instance.
        """
        # spectrograph that inherit Spectrograph simply need to update _parse_config()
        init_prop, _ = cls._parse_config(config)
        
        return cls(telescope=telescope, **init_prop)

    @staticmethod
    def _parse_config(config):
        """Parse the configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        dict
            Initialization properties.
        dict
            Input configuration.
        """
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

        init_prop = {"lbda": lbda,
                     "lbda_edges": lbda_edges,
                     "spaxels": spaxels,
                     "throughput": throughput,
                     "spatial_psf": psf,
                     "optics": optics,
                     "meta": input_config}

        return init_prop, input_config

    @staticmethod
    def build_lbda_from_config(config):
        """Build wavelength coordinates from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        lbda : array_like
            Mid-bin wavelengths.
        lbda_edges : array_like
            Bin edge wavelengths.
        """
        return build_lbda_from_config(config)

    def update(self, reset_others=False, **kwargs):
        """Update any mutable attribute of the spectrograph.

        remark: the method accepts django like format such that
                a__b is understood as a.b. 
                For instance: optics__temperature => optics.temperature.
                so update(**{'optics.temperature':220}) is equivalent to 
                update(optics__temperature=220)

        Information:
        ------------
           # lbda:
           - changing `dispersion_resolution` redefines self.lbda such that 
             the resolving_power() is unchanged
           - changing `spotsize` updates self.dispersion_resolution 
             *without* updating lbda, effectively changing the resolving_power()


        Parameters
        ----------
        reset_others : bool, optional
            If True, reset other parameters to their initial values.
            Default is False.
        **kwargs
            Parameters to update.
        """
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
            k = NAME_ALT.get(k, k).replace("__", ".") # accept django like format
            
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            if v is None:  # Skip
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
            if k in ["dispersion_resolution", "spotsize"]:
                # update the meta (for lsf and instrumental psf)
                updates["dispersion_resolution"] = v
                # and update the wavelengths
                if k == "dispersion_resolution":
                    # you changed the actual dispersion resolution, so let's change lbda
                    lbda_updates["dispersion_resolution"] = v

            elif k == "dispersion_scale":
                lbda_updates["dispersion_scale"] = v

            # not allowed to change.

            # change PSF
            elif k in self.meta["psf"]["spatial"].keys():
                psf_updates[k.replace("psf_", "")] = v

            # spaxels
            elif k in ('spatial_scale', 'spatial_scale_insigma',
                       'spatial_shape', 'spatial_shape_insigma'):
                spaxel_updates[k] = v
            else:
                warnings.warn(f"{k=} is unparsed")

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
            spaxels = build_spaxels_from_config(self._meta | spaxel_updates)
            self.set_spaxels(**spaxels)

        if lbda_updates:
            what_changed.append("lbda")
            self.update_lbda(*build_lbda_from_config(self._meta | lbda_updates))

        if updates:
            what_changed.append("meta")

        # the rest if any
        if reset_others:
            self._meta = self._meta_in | updates
        else:
            self._meta = self._meta | updates

        return what_changed

    def update_lbda(self, lbda, lbda_edges):
        """Update wavelength coordinates.

        Parameters
        ----------
        lbda : array_like
            Mid-bin wavelengths.
        lbda_edges : array_like
            Bin edge wavelengths.
        """
        self.lbda = lbda
        self.lbda_edges = lbda_edges

    def set_spaxels(self, shape, spx_scale):
        """Set spaxel information.

        Parameters
        ----------
        shape : tuple
            Spaxel grid shape (ny, nx).
        spx_scale : float
            Spaxel scale in arcsec.
        """
        self._spaxels = {"shape": shape, "spx_scale": spx_scale}
        self._spaxel_coords = {}

    def set_throughput(self, throughput):
        """ set the throughput of the spectrograph

        Parameters
        ----------
        throughput: pandas.Series, float, array, func
            The throughput of the system as a function of wavelength (0->1)
            - float: constant throughput
            - array: must broadcast with self.lbda
            - func: function that input self.lbda such that throughput_array = throughput(self.lbda)
            - pandas.Series: throughput with lbda [AA] as index. This is converted input a function 
              using a cubic interpolation.
        
        Returns
        -------
        None
        """
        
        # 
        if type(throughput) is pandas.DataFrame:
            throughput = throughput.iloc[:,0] # convert as serie

        # Serie => func
        if type(throughput) is pandas.Series:
            from . import iotools            
            throughput = iotools.chromatic_interpolator(
                throughput.index, throughput.values, ext='zeros')

        self.throughput = throughput
        
    # --------- #
    #  GETTER   #
    # --------- #
    def get_lbda(self, units=None, oversample=None):
        """ get the wavelength array 

        Parameters
        ----------
        units: None,string, astropy.units
            the unit of the returned array. 
            None (default) corresponds to AA (Angstrom).

        oversample: None, int
            specify if you want to oversample the wavelength binning.

        Returns
        -------
        lbda: array
            wavelength array (see units).
        """
        lbda = self.lbda.copy() # in AA
        if units is not None:
            lbda *= getattr(units, "AA").to(units)

        if oversample is not None:
            nlbda = len(lbda)
            lbda = np.interp( np.arange(nlbda, step=1/oversample), np.arange(nlbda), lbda)

        return lbda
        
    def get_throughput(self):
        """Get the spectrograph throughput.

        Returns
        -------
        array_like or float
            Throughput as a function of wavelength or as a constant.
        """
        if callable(self.throughput):
            return self.throughput(self.lbda)
        
        # float
        return self.throughput

    def get_spectrograph_shape(self, oversampling=None):
        """Get the shape of the spectrograph detector.

        Parameters
        ----------
        oversampling : int, optional
            Oversampling factor. Default is None.

        Returns
        -------
        tuple
            Shape of the detector (ny, nx).
        """
        if oversampling is None or oversampling == 1:
            return self.spx_shape

        return (np.asarray(self.spx_shape) * oversampling).astype(int)

    def get_resolving_power(self):
        """ this is 'R' such as $\mathcal{R} = \lambda/\Delta\lambda$ 
        
        it is computed as lbda / np.difflbda_edges) / dispersion_resolution
        
        Returns
        -------
        'R': array
            the resolving power for shape (nlbda, )
        """
        return self.lbda / np.diff(self.lbda_edges) / self.dispersion_resolution


    def get_spaxel_centroids(self, in_arcsec=False, squeeze=False, oversampling=None):
        """Get the spaxel centroids.

        Parameters
        ----------
        in_arcsec : bool, optional
            If True, return centroids in arcsec. Default is False.
        squeeze : bool, optional
            If True, squeeze the output arrays. Default is False.
        oversampling : int, optional
            Oversampling factor. Default is None.

        Returns
        -------
        tuple
            Tuple containing (x, y) centroid coordinates and the oversampling factor.
        """
        # numpy and all() test enables to have list/array oversampling (or float/int)
        if oversampling is None or (np.asarray(oversampling) == self.spaxels["oversampling"]).all():
            spaxels_coords = self.spaxels
        else:
            spaxels_coords = build_pixels(self.spx_shape, oversampling=oversampling)

        x, y = spaxels_coords["centroids"] # dense if oversampling
        if in_arcsec:
            if np.ndim(self.spx_spatial_scale) == 0:
                spx_y = spx_x = self.spx_spatial_scale
            else:
                spx_y, spx_x = self.spx_spatial_scale
            # not /= not to change
            x = x * spx_x # / spaxels_coords["oversampling"]  # in arcsec
            y = y * spx_y # / spaxels_coords["oversampling"]  # in arcsec

        if squeeze:
            x, y = np.squeeze(x), np.squeeze(y)

        return (x, y), spaxels_coords["oversampling"]

    def get_psf_sigma_spectral(self, guiding_sigma=None,
                                   in_spaxels=False, lbda=None):
        """Get spatial PSF standard deviation in arcsec.

        The total (Gaussian) spatial PSF is made of two components:
        - the optical (chromatic) component, with stddev proportional
          to wavelength, normalized at wref=1 µm,
        - the guiding (achromatic) component, with constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        Parameters
        ----------
        guiding_sigma : float, optional
            Guiding sigma override in arcsec. Default is None.
        in_spaxels : bool, optional
            If True, return sigma in spaxels. Default is False.
        lbda : array_like, optional
            Wavelength array in Angstrom. If None (Default), 
            this uses self.lbda.
        Returns
        -------
        array_like
            Total sigma in arcsec or spaxels.
        """
        if lbda is None:
            lbda = self.lbda

        if guiding_sigma is None:
            guiding_sigma = self.spatial_psf["guiding_sigma"]

        sigma = self._get_chromatic_sigma(lbda,
                                          chromatic_sigma=self.psf_sigma_spectral,
                                          constant_sigma=guiding_sigma,
                                          wref=self.lbda_ref)
        # broadcast as (nlbda, 1, 1)
        sigma = sigma[..., None, None]
        
        if in_spaxels:
            # for broadcasting reasons /= does not work.
            sigma = sigma/ self.spx_spatial_scale

        return sigma

    def get_additonal_spatial_scatter(self, guiding_sigma=None, incl_instrument=True,
                                          in_spaxels=True):
        """ scale of the scatter induced by telescope jitter and instrumental psf
        
        Note: current implementation assumes scatter to the gaussian.
        
        Parameters
        ----------
        guiding_sigma : float, optional
            Gaussian noise convolution (in arcsec) caused by jitter.
            Default is None.
        incl_instrument: bool
            should this also include instrumental induced PSF
            = this only apply to slicer spectrograph =
        Returns
        -------
        scatter: 
            (x, y) gaussian sigma
        """
        if guiding_sigma is None:
            guiding_sigma = self.spatial_psf["guiding_sigma"]

        if guiding_sigma is not None:
            # make sure it is 2d (RA, Dec) 
            guiding_sigma = np.full((2,), guiding_sigma) / self.spx_spatial_scale # in spaxels

        # Note:
        # Slicer: instrumental PSF affects the LSF along the "x" direction
        #     and the PSF along the "y" direction.
        #     => The effective "sigma" this thus guiding_sigma & (0, inst_psf[1])
        # MLA: it affects the LSF and the x-dispersion profile.
        #     Not the spatial PSF
        #     => the effective "sigma" this thus just guiding_sigma
        if self._SPECTROGRAPH_TYPE == "slicer" and incl_instrument:
            inst_psf = self.get_instrumental_psf() # in spaxels
            if inst_psf is not None: # if None, nothing to do.
                # make sure the inst_psf is 2d (x and y)
                inst_psf = np.full((2,), inst_psf)
                # and remove the "x" contribution (see LSF)
                inst_psf[0] = 0
        
            if guiding_sigma is None and inst_psf is None:
                effective_sigma = None
            elif guiding_sigma is None:
                effective_sigma = inst_psf
            elif inst_psf is None:
                effective_sigma = guiding_sigma
            else:
                effective_sigma = np.hypot(guiding_sigma, inst_psf)
        else:
            effective_sigma = guiding_sigma

        if in_spaxels:
            return effective_sigma
        
        return effective_sigma * self.spx_spatial_scale

    def get_spatial_psf(self, profile="default", position=(0, 0),
                        guiding_sigma=None, oversampling=5,
                        as_oversampled=False,
                        **kwargs):
        """Get normalized 2D spatial PSF on the MLA.

        This uses the exact 2D Gaussian PSF integration over the spaxels.

        Parameters
        ----------
        profile : str, optional
            Profile to be used:
            - "gaussian": gaussian profile using exact erf integration (default).
            - "airy": airy disk profile generated by the telescope mirror.
            Default is "default".
        position : tuple, optional
            Location of the point source within the slicer/mla in unit of slice/spaxel.
            Default is (0, 0).
        guiding_sigma : float, optional
            Gaussian noise convolution (in arcsec) caused by jitter.
            Default is None.
        oversampling : int, optional
            Oversampling factor used for the PSF (except if profile is gaussian, where erf is used).
            If oversampling is None, 3 is used by default. Set oversampling=1 for no oversampling.
            Default is None.
        as_oversampled : bool, optional
            If True, return the oversampled PSF. Default is False.
        **kwargs
            Additional arguments passed to the profile function.

        Returns
        -------
        array_like
            Normalized PSF (nlbda, ny, nx).
        """
        from . import profiles
        if oversampling is None:
            oversampling = 3 # like ~10 (9) subpixels

        # to solve the numpy vs. plot confusion
        position_xy = position
            
            
        # effective_sigma is 2d (x-scatter, y-scatter) | x == dispersion y=slice  
        effective_sigma = self.get_additonal_spatial_scatter(guiding_sigma=guiding_sigma,
                                                              incl_instrument=True)
        
        # 
        # Gaussian, special as convolution with additional gaussan scatter is analytic
        #
        if profile in ["default", "gaussian", "normal", "norm"]:
            # adding gaussian_sigma in arcsec (in_spaxels=False)| e.g. jitter
            sigmas = self.get_psf_sigma_spectral(in_spaxels=True,
                                                 guiding_sigma=None, # explicitely null, see after.
                                                ) # in spaxels
            
            # adding gaussian scatter (e.g. jitter)
            if effective_sigma is not None:
                # effective_sigma is in spaxel 
                sigmas = np.hypot(sigmas, effective_sigma) # in spaxels

            prop = dict(in_arcsec=False, squeeze=False, oversampling=1)
            if as_oversampled:  # no need to oversample as this uses exact erf functions.
                warnings.warn(f"changin oversampling to {oversampling=}")
                prop["oversampling"] = oversampling

            (xx, yy), oversampling = self.get_spaxel_centroids(**prop)

            psf = profiles.get_gaussian2d(xx, yy, sigma=sigmas, mean=position_xy, **kwargs)
            # remark: if oversampled, the flux need to be *summed*, not *averaged*
            #         as the energy is conserved in the current structure.
            #         => smaller pixel (oversampled) -> less flux.
            
            return psf

        # ----------------------------------------------------- #
        #
        # Generic PSF,
        #    any profile convolved with additional
        #    gaussian scatter (like jitter)
        #
        # ----------------------------------------------------- #
        #

        # This part of the code works in arcsec.
        position_xy = np.asarray(position_xy) * self.spx_spatial_scale[::-1] # in spaxels => arcsec
#        if self.type == "slicer": # inject the anamorphone back in
#            print(position_xy)
#            position_xy /= self._ANAMORPHOSE
#            print(f"becomes {position_xy=}")

        # to accomodate with non-square spaxels (like slicer)
        # we work in arcsec, not in spaxels.
        if profile in ["airy", "mirror", "telescope", "airydisk"]:
            radius = self.telescope.get_airy_radius(self.lbda)  # in arcsec
            # assumed symetric on x and y
            psf_func = profiles.get_profilemodel("airy", position=position_xy,
                                                  radius=radius[:, None, None],
                                                  normalized=True)
            
        elif profile in ["Gaussian2D", "gaussian2d", "gaussian"]:
            sigmas = self.get_psf_sigma_spectral(in_spaxels=False,
                                                 guiding_sigma=None, # explicitely null, see after.
                                                ) # in arcsec 
            psf_func = profiles.get_profilemodel("Gaussian2D", position=position_xy,
                                                  sigma=sigmas,
                                                  normalized=True)

        else:
            raise ValueError(f"psf profile {profile=} not implemented")

        # coordinates including oversampling for exact PSF profile and guiding convolution
        # x and y are in arcsec as radius is given in arcsec.
        (xx, yy), oversampling = self.get_spaxel_centroids(in_arcsec=True, squeeze=False,
                                                           oversampling=oversampling)
        psf = psf_func(xx, yy)
        # Since we are working in arcsec, let's make sure this goes away.
        psf *= self.spx_area

        # Note: Normalisation have been tested. See:
        # - psf_scipy = self.get_spatial_psf("normal") # works in spaxels
        # - psf_profile = self.get_spatial_psf("Gaussian2D") # works in arcsec
        #   they must both agree within numerical errors.
        

        # oversampling: The PSF intensity need to be devided by oversampling area 
        #               to conserve energy. Hence the rebinning needs to sum (not average)
        if oversampling is not None:
            if np.ndim(oversampling) == 1:
                oversampling_y, oversampling_x = oversampling
            else:
                oversampling_y = oversampling_x = oversampling
                
            psf /= oversampling_y*oversampling_x

        # effective_sigma is either None or [sigma_x, sigma_y]
        if effective_sigma is not None and np.any(effective_sigma > 0):
            # effective_sigma is already in spaxels.
            psf = self._apply_guiding(psf, guiding=effective_sigma, oversampling=oversampling,
                                          in_units="spaxels")

        if oversampling != 1 and not as_oversampled:
            # in the current structure, you need to sum oversampled pixels as energy is conserved.
            psf = self._remove_oversampling(psf, oversampling=oversampling, func=np.nansum)

        return psf  # (nlbda, ny, nx)

    def get_instrumental_psf(self, spotsize=None, expected_size=2,
                                 in_spaxels=True):
        """ get the contribution of the spectrograph to the total spot size.
        This contribution is null if spot_size <= expected_size.

        This simply provides: 
        extra_pixels = np.sqrt(spotsize**2 - expected_size**2)

        Parameters
        ----------
        spotsize: float
            total size of a pointsource on the detector. 
            This is the total contribution, Telescope + Spectrograph.

        expected_size: float
            this is the expected geometric footprint of the psf on the 
            detector.

        in_spaxels: bool
            should the 'extra_pixels' be given in spaxel units 
            (spaxel=pixel for slicer along the x-slice direction)
            or should this be given in arcsec.
        
        Returns
        -------
        extra_pixels: array, float
            extra scattering. Format depends of in_spaxels and self.spx_spatial_scale
        """
        if spotsize is None:
            spotsize = self.dispersion_resolution

        if spotsize <= expected_size:
            return None # nothing to do | should a warning be added ? This should not happen
    
        # additional contribution (assumed gaussian) for the effective spotsize
        extra_pixels = np.sqrt(spotsize**2 - expected_size**2)
    
        # get the scatter scale in pixels/spaxels 
        # pixels/spaxels: same for a slicer in that direction
        if in_spaxels: 
            return extra_pixels
    
        # or get this scale in arcsec
        return extra_pixels * self.spx_spatial_scale # in arsec

    # - internal tricks
    def _apply_guiding(self, image, guiding, oversampling=1, in_units="arcsec"):
        """Apply guiding error convolution.

        Parameters
        ----------
        image : array_like
            Image to convolve.
        guiding : float
            Guiding error [arcsec or spaxel, see in_units].
        oversampling : int, optional
            Oversampling factor. Default is 1.
        in_units: bool
            the guiding parameter units: ["arcsec" or "spaxels"] 

        Returns
        -------
        array_like
            Convolved image.
        """
        from scipy.ndimage import gaussian_filter
        # gaussian convolution | guiding_sigma is given in arcsec
        # works by itself if spx_spatial_scale is a list or a float.
        if in_units=="arcsec":
            guiding = guiding/self.spx_spatial_scale
        elif in_units not in ["spaxels", "spx", "pixels"]:
            raise ValueError(f"in_units should be arcsec or spaxels, {in_units=} given.")

        # guiding is in spaxels/pixels at this stage.
        sigma_guiding_pixels = guiding * oversampling  # in spaxels
        return gaussian_filter(image, sigma_guiding_pixels, axes=(-2, -1))

    @staticmethod
    def _remove_oversampling(image, oversampling, func=np.mean):
        """Remove oversampling from an image.

        Parameters
        ----------
        image : array_like
            Oversampled image.
        oversampling : int
            Oversampling factor.
        func : callable, optional
            Function to use for downsampling. Default is `np.mean`.

        Returns
        -------
        array_like
            Downsampled image.
        """
        from astropy import nddata
        
        if np.ndim(image) == 2:
            block_size = (oversampling, oversampling)
        elif np.ndim(image) == 3:
            block_size = (1, oversampling, oversampling)
        else:
            raise ValueError(f"image ndim must be 2 or 3 {np.ndim(image)=} given.")

        return nddata.block_reduce(image, block_size=block_size, func=func)

    #  internal tricks

    def cube_to_slice(self, cube, lbda_range, func=np.nansum, squeeze=False):
        """Get slices of the given cube.

        Parameters
        ----------
        cube : array_like
            3D data cube.
        lbda_range : tuple or list of tuples
            Wavelength range(s) in Angstrom (wmin, wmax).
        func : callable, optional
            Function to apply to merge wavelengths into one slice.
            Default is `np.nansum`.
        squeeze : bool, optional
            If True, squeeze the output array. Default is False.

        Returns
        -------
        array_like
            (n_ranges, ny, nx) array of slices.
        """
        slices = []
        for wmin, wmax in np.atleast_2d(lbda_range):
            flag_lbda = ((self.lbda >= wmin) & (self.lbda <= wmax))
            slices.append(func(cube[flag_lbda], axis=0))

        slices = np.asarray(slices)
        if squeeze:
            slices = np.squeeze(slices)

        return slices

    def get_nea(self, position=(0, 0), nea_spatial=None, nea_pixels=None):
        """Noise Equivalent Area (PSF -> Spaxel -> detector).

        Parameters
        ----------
        position : tuple, optional
            Position of the PSF in unit of spaxel/slicer.
            Ignored if `nea_spatial` is given. Default is (0, 0).
        nea_spatial : float or array_like, optional
            Noise equivalent area of the spatial PSF (in spaxels).
            If None, `self.get_nea_spatial()` is used. Default is None.
        nea_pixels : float or array_like, optional
            Noise equivalent area of a spaxel/slice caused by x-dispersion (in pixels).
            If None, `self.get_nea_pixels()` is used. Default is None.

        Returns
        -------
        array_like
            Total Noise Equivalent Area.
        """

        # NEA_Spatial PSF on the MLA (2D) / Slicer (1d) | in slice / spaxel
        if nea_spatial is None:
            nea_spatial = self.get_nea_spatial(position=position)

        # NEA_pixel of 1 spaxel / slice on the dectetor
        if nea_pixels is None:
            nea_pixels = self.get_nea_pixels()

        return nea_spatial * nea_pixels

    def get_nea_telescope_airy(self, position=(0, 0), in_spaxels=True):
        """Get the NEA of the telescope's Airy disk.

        Parameters
        ----------
        position : tuple, optional
            Position of the PSF in spaxels. Default is (0, 0).
        in_spaxels : bool, optional
            If True, return NEA in spaxels^2. Default is True.

        Returns
        -------
        array_like
            NEA of the Airy disk.
        """
        if in_spaxels:
            norm_scale = self.spx_spatial_scale

        return self.telescope.get_nea_airy(self.lbda, norm_scale=norm_scale, position=position)

    # ------------- #
    #   Others      #
    # ------------- #
    def pointsource_variance(self, varcube, position=(0, 0), radius=5,
                              psf_profile="default",
                              optimal=True, verbose=False):
        """Point-source extracted variance from variance cube.

        It is evaluated from a plain summation over the aperture, or from an
        optimal extraction, i.e. inverse-variance weighted least-square fit of
        the PSF profile (assumed Gaussian). As the extraction is assumed to do
        a perfect job on the signal, only the variance depends on instrumental
        parameters and aperture definition.

        Parameters
        ----------
        varcube : array_like
            Variance cube (nlbda, ny, nx) in ADU^2.
        position : tuple, optional
            Point source position in MLA in spaxels. Default is (0, 0).
        radius : float, optional
            MLA aperture radius in spaxels. Default is 5.
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        optimal : bool, optional
            If True, use optimal extraction. Default is True.
        verbose : bool, optional
            If True, print verbose output. Default is False.

        Returns
        -------
        array_like
            Extracted variance spectrum (nlbda,) in ADU^2.
        """
        if np.ndim(radius) == 0:
            radius_y = radius_x = radius
        elif np.ndim(radius) == 1 and len(radius) == 2:
            radius_y, radius_x = radius
        else:
            raise ValueError(f"cannot parse input {radius=}")

        # Spatial PSF
        psf = self.get_spatial_psf(position=position, profile=psf_profile)  # (nlbda, ny, nx)

        # Aperture
        x0, y0 = position
        (x, y), oversampling = self.get_spaxel_centroids()
        r_radius = np.hypot((x - x0) / radius_x, (y - y0) / radius_y)  # (ny, nx) [spx]
        aper = (r_radius <= 1)
        nspx = np.count_nonzero(aper)  # Selected spx
        if verbose:
            print(f"Aperture of {radius=} spx: {nspx} spx selected")

        # If cube is (nlbda, ny, nx), cube[:, aper] is (nlbda, nspx)
        if optimal:  # Optimal extraction variance = 1/sum(psf**2/var)
            variance = 1 / (psf[:, aper] ** 2 / varcube[:, aper]).sum(axis=-1)
        else:  # Plain summation: variance = sum(variance)
            variance = varcube[:, aper].sum(axis=-1)

        return variance  # (nlbda,) [ADU²]

    def effective_resolution(self, average=False):
        """ Effective spectral resolution.

        Parameters
        ----------
        average : bool, optional
            If True, return the chromatic average. Default is False.

        Returns
        -------
        float or array_like
            Effective n-px wavelength resolution.

        Notes
        -----
        .. math::

           R &= \frac{2}{n \delta\lambda} \\
           \delta\lambda &= \max(1, \sigma) \times \Delta\lambda

        where :math:`\Delta\lambda` is the spectral step [Å] and
        :math:`\sigma` is the spectral resolution [px].
        """
        dispersion_resolution = self.get_lsf_dispersion(as_ = "resolution") #

        dlbda = np.diff(self.lbda_edges)
        wres = self.lbda / (dispersion_resolution * dlbda)  # (nlbda,)

        if average:  # Chromatic average
            wres = np.average(wres, weights=dlbda)

        return wres

    def get_lsf_dispersion(self, as_="scale"):
        """Get the gaussian LSF dispersion sigma in units of wavelength bin.

        Parameters
        ----------
        as: str
            Specify how you when the dispersion defined.
            - sigma/scale: equivalent of a normal distribution sigma (i.e. 'scale' in scipy, numpy definition)
        """
        if as_ is None or as_ in ["resolution"]:
            disp_ = self.dispersion_resolution
        
        elif as_ in ["scale", "sigma"]:
            disp_ = self.dispersion_resolution / 2.
        else:
            raise NotImplementedError(f"{as_=} is not implemented, use: scale or resolution")
        
        return disp_

    def apply_line_spread_function(self, fluxes, **kwargs):
        """Apply the line spread function to a flux cube.

        Parameters
        ----------
        fluxes : array_like
            Flux cube (nlbda, ny, nx).
        **kwargs
            Additional arguments passed to `scipy.ndimage.gaussian_filter1d`.

        Returns
        -------
        array_like
            Flux cube convolved with the LSF.
        """
        from scipy.ndimage import gaussian_filter1d

        # wavelength is the first direction of fluxes
        lsf_sigma = self.get_lsf_dispersion(as_="scale")

        # apply a gaussian filter to input fluxes ; wavelength axis is 0 by default
        return gaussian_filter1d(fluxes, lsf_sigma, **({"axis": 0} | kwargs))

    # ------------- #
    #    generate   #
    # ------------- #
    # Point Souce
    def generate_pointsource(self, spectrum, position=(0, 0),
                              psf_profile="default",
                              oversampling=None,
                              as_oversampled=False,
                              apply_lsf=True):
        """Generate a photon flux cube from a point source spectrum.

        Parameters
        ----------
        spectrum : array_like
            Point source spectrum in erg/s/cm²/Å.
        position : tuple, optional
            Point source position in MLA in spaxels. Default is (0, 0).
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        oversampling : int, optional
            Oversampling factor. Default is None.
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.

        Returns
        -------
        array_like
            Photon flux cube (nlbda, ny, nx) in ph/s/spx.
        """
        # Spatial PSF
        psf = self.get_spatial_psf(profile=psf_profile, position=position,
                                    as_oversampled=as_oversampled,
                                    oversampling=oversampling,
                                    )  # (nlbda, ny, nx)

        # erg/s/cm²/Å / erg/ph * cm² * Å = ph/s
        flux = spectrum * self.flambda2photon  # (nlbda,) [ph/s]

        psf_cube = np.reshape(flux, (-1, 1, 1)) * psf  # Point source (nlbda, ny, nx)
        if apply_lsf:
            psf_cube = self.apply_line_spread_function(psf_cube)

        return psf_cube

    # Flat Background
    def generate_background(self, spectrum, oversampling=None,
                            apply_lsf=True):
        """Generate a photon flux cube from uniform scene background spectrum.

        Parameters
        ----------
        spectrum : array_like
            Uniform scene background spectrum in erg/s/cm²/Å/arcsec².
        oversampling : int, optional
            Oversampling factor. Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.

        Returns
        -------
        array_like
            Photon flux cube (nlbda, ny, nx) in ph/s/spx.
        """

        # erg/s/cm²/Å/arcsec² * cm² * Å / erg/ph * arcsec² = ph/s/spx
        flux = spectrum * self.flambda2photon * self.spx_area

        bkgd_cube = np.full((self.nlbda, *self.get_spectrograph_shape(oversampling=oversampling)),  # y, x
                             flux[:, np.newaxis, np.newaxis])  # (nlbda, ny, nx)
        if apply_lsf:
            bkgd_cube = self.apply_line_spread_function(bkgd_cube)

        return bkgd_cube

    # Structured background
    def generate_structured_background(self, *args, apply_lsf=True, **kwargs):
        """Generate a photon flux cube from a structured background scene.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("generate_structured_background() has not been implemented.")

    # Themal (pre-dispersor)
    def generate_thermal_signal(self, lbda_bin=None,
                                as_cube=False, oversampling=None,
                                apply_lsf=True, as_sum=True):
        """Telescope mirror thermal signal in ph/s/spx/Δλ.

        Parameters
        ----------
        lbda_bin : array_like, optional
            (nlbda, 2) list of spectral domains in Angstrom.
            Default is None (use `self.lbda_bin`).
        as_cube : bool, optional
            If True, return a 3D cube. Default is False.
        oversampling : int, optional
            Oversampling factor. Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        as_sum : bool, optional
            If True, sum the contribution from all mirrors. Default is True.

        Returns
        -------
        array_like or float
            Thermal signal in ph/s/spx/Δλ.
        """
        if lbda_bin is None:
            lbda_bin = self.lbda_bin

        signal = self.telescope.get_thermal_signal(lbda_bin,
                                                   solid_angle=self.omega,  # Spx solid angle [sr]
                                                   as_sum=as_sum)
        # output formating
        if as_cube:
            signal = np.full((self.nlbda, *self.get_spectrograph_shape(oversampling=oversampling)),
                             signal[..., np.newaxis, np.newaxis])  # (nlbda, ny, nx)

        if apply_lsf:
            signal = self.apply_line_spread_function(signal)

        return signal  # [ph/s/spx/Δλ]

    def get_thermal_dark(self, pixel_area, as_sum=True):
        """Get the thermal dark current.

        Parameters
        ----------
        pixel_area : float
            Detector pixel surface in m^2.
        as_sum : bool, optional
            If True, sum the contribution from all optical elements.
            Default is True.

        Returns
        -------
        float or array_like
            Thermal dark current in ph/s.
        """
        # properties at the detector
        ## each pixels integrate the whole spectral range.
        lbda_bin = self.lbda[[0, -1]]

        # get the dark signal in ph/s
        signals = self.optics.get_signal(lbda_bin=lbda_bin,  # expectedin in [A]
                                         area=pixel_area)  # Collecting area [m²]

        # note: this sums over 1 element if only 1 temperature
        if as_sum:
            signals = np.sum(signals, axis=0)

        return signals

    # Empty cube
    def get_empty_cube(self, filled=0, oversampling=None):
        """Get an empty data cube.

        Parameters
        ----------
        filled : float, optional
            Value to fill the cube with. Default is 0.
        oversampling : int, optional
            Oversampling factor. Default is None.

        Returns
        -------
        array_like
            An empty data cube.
        """
        # no apply LSF as zeros...
        ny, nx = self.get_spectrograph_shape(oversampling=oversampling)
        
        return filled * np.ones((self.nlbda, ny, nx))

    # ------------ #
    #   GETTER     #
    # ------------ #
    def get_nea_spatial(self, position=(0, 0), in_spaxels=True, guiding_sigma=None):
        """Noise equivalent area in unit of slice/spaxels.

        i.e., how many "spaxel noise".

        Parameters
        ----------
        position : tuple, optional
            Position of the PSF in unit of spaxel/slicer. Default is (0, 0).
        in_spaxels : bool, optional
            If True, return the area in spaxels^2. If False, in arcsec^2.
            Default is True.
        guiding_sigma : float, optional
            Guiding sigma override in arcsec. Default is None.

        Returns
        -------
        array_like
            Noise equivalent area.
        """
        from .nea import get_2dnorm_nea

        if self.spatial_psf["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian spatial PSF profile implemented, but: {self.spatial_psf['profile']=}")

        sigma_at_spectro = self.get_psf_sigma_spectral(in_spaxels=in_spaxels,
                                                       guiding_sigma=guiding_sigma)
        
        return get_2dnorm_nea(sigma_at_spectro, mean=position)

    #
    # - tools
    #
    def chromatic_average(self, quantity):
        """Chromatic average of a quantity (averaged over wavelength rather than px).

        Parameters
        ----------
        quantity : array_like
            Chromatic quantity (nlbda,).

        Returns
        -------
        float
            Chromatic average.
        """
        return np.average(quantity, weights=np.diff(self.lbda_edges))

    #
    # - Internal
    #
    @staticmethod
    def _get_chromatic_sigma(lbda, chromatic_sigma,
                             constant_sigma,
                             wref):
        """Get total PSF, including chromatic and constant components.

        The total (Gaussian) stddev is the quadratic sum of two components:
        - the chromatic stddev, proportional to wavelength,
          normalized at `wref`,
        - the achromatic, constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        Parameters
        ----------
        lbda : array_like
            Wavelength.
        chromatic_sigma : float
            Chromatic (linear) stddev.
        constant_sigma : float
            Achromatic (constant) stddev.
        wref : float
            Reference wavelength (same unit as `lbda`).
        xdims : int, optional
            Extra dimensions to be appended. Default is 0.

        Returns
        -------
        array_like
            Total stddev as function of wavelength.
        """

        lmin, lmax = np.array(lbda)[[0, -1]]  # 1st and last wavelengths
        assert lmin > wref / 3 and lmax < wref * 3, \
            "Input and reference wavelengths probably not in same units."

        return np.hypot(constant_sigma,
                        chromatic_sigma * (lbda / wref))  # [px]

    # -------- #
    #  SHOW    #
    # -------- #
    def show_nea(self, ax=None, position=(0, 0), legend=True):
        """Show the Noise Equivalent Area.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        position : tuple, optional
            Position of the PSF in spaxels. Default is (0, 0).
        legend : bool, optional
            If True, show the legend. Default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # data
        nea_spatial = self.get_nea_spatial(in_spaxels=True, position=position)
        nea_no_guiding = self.get_nea_spatial(guiding_sigma=0, in_spaxels=True, position=position)
        nea_telescope = self.get_nea_telescope_airy(in_spaxels=True, position=position)

        ax.plot(self.lbda, self.get_nea(position=position, nea_spatial=nea_spatial),
                color="#194D80", label="total")
        ax.plot(self.lbda, self.get_nea(position=position, nea_spatial=nea_no_guiding),
                color="#194D80", ls="--", label="without guiding")
        ax.plot(self.lbda, self.get_nea(position=position, nea_spatial=nea_telescope),
                color="#F8AD05", label="airy from telescope")
        ax.legend(fontsize="small", frameon=False)

        ax.set_xlabel("wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("NEA [in pixels]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)

        return fig

    def show_nea_spatial(self, ax=None, position=(0, 0), legend=True):
        """Show the spatial Noise Equivalent Area.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        position : tuple, optional
            Position of the PSF in spaxels. Default is (0, 0).
        legend : bool, optional
            If True, show the legend. Default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
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

        ax.set_xlabel("wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("NEA [in spaxels]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)

        return fig

    def show_fwhm(self, ax=None, legend=True, guiding_arcsec=None, in_arcsec=False, show_band=True):
        """Show the Full Width at Half Maximum (FWHM) of the PSF.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        legend : bool, optional
            If True, show the legend. Default is True.
        guiding_arcsec : float, optional
            Guiding error in arcsec. Default is None.
        in_arcsec : bool, optional
            If True, show the FWHM in arcsec. Default is False (in spaxels).
        show_band : bool, optional
            If True, show the sampling bands. Default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
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
        radius = self.telescope.get_airy_radius(self.lbda, norm_scale=norm_scale)

        ax.plot(self.lbda, 2.35 * sigma_at_mla, color="#194D80", label="total scatter")
        ax.plot(self.lbda, 2.35 * sigma_at_mla_no_guiding, color="#194D80", ls="--", label="without guiding")
        ax.plot(self.lbda, 0.8 * radius, color="#F8AD05", label="airy from telescope")

        if show_band:
            ax.axhspan(2 * norm_sampling, 2.35 * norm_sampling, color="tab:orange", alpha=0.05, lw=0)
            _ylow, _ = ax.get_ylim()
            ax.axhline(2 * norm_sampling, color="tab:red", alpha=1, ls="--", lw=0.5)
            ax.axhspan(0, 2 * norm_sampling, color="tab:red", alpha=0.05, lw=0)
            ax.set_ylim(_ylow)

        ax.set_xlabel("wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("FWHM [in spaxels]" if in_spaxels else "FWHM [in arcsec]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)

        return fig

    def show_psf(self, lbda_range, profile="default",
                 guiding_arcsec=None, axes=None,
                 position=(0, 0), oversampling=5,
                 in_arcsec=False,
                 norm="log", **kwargs):
        """Show the spatial PSF.

        Parameters
        ----------
        lbda_range : tuple
            Wavelength range to integrate over.
        profile : str, optional
            PSF profile to use. Default is "default".
        guiding_arcsec : float, optional
            Guiding error in arcsec. Default is None.
        axes : list of matplotlib.axes.Axes, optional
            List of 3 axes to plot on. If None, a new figure and axes are created.
            Default is None.
        position : tuple, optional
            Position of the PSF in spaxels. Default is (0, 0).
        oversampling : int, optional
            Oversampling factor. Default is 5.
        in_arcsec : bool, optional
            If True, show the extent in arcsec. Default is False (in spaxels).
        norm : str, optional
            Normalization for the color scale ("linear" or "log"). Default is "log".
        **kwargs
            Additional arguments passed to `ax.imshow()`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
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
                                                figsize=(5, 7),
                                                gridspec_kw={"hspace": 0.04})
        else:
            (ax, axg, axsl) = axes
            fig = ax.figure

        # highly-resolved cube
        psf_cube = self.get_spatial_psf(profile=profile, guiding_sigma=0,  # no guiding error yet
                                        oversampling=oversampling,
                                        as_oversampled=True,
                                        position=position)

        # perfect model
        psf_slice = self.cube_to_slice(psf_cube, lbda_range, squeeze=True)
        ax.imshow(psf_slice, norm=norm(), extent=extent, **kwargs)
        ax.set_xticklabels([])

        # adding guiding
        psf_cube = self._apply_guiding(psf_cube, guiding=guiding_arcsec, oversampling=oversampling,
                                           in_units="arcsec")
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
        """MLA extent in spaxels."""
        hy, hx = np.asarray(self.spx_shape) / 2  # Half total width [spx]
        return [-hx, hx, -hy, hy]

    @property
    def flambda2photon(self):
        """Chromatic conversion factor from erg/s/cm²/Å to ph/s."""
        dlbda = np.diff(self.lbda_edges)  # Spectral step (nlbda,)
        hnu = 1.9864459e-08 / self.lbda  # Photon energy [erg] with lbda in [Å]

        # erg/s/cm²/Å * (cm² * throughput * Å / erg/ph) = ph/s
        return (self.telescope.surface * 1e4 *
                self.get_throughput() * dlbda / hnu)  # (nlbda,) [ph/s]

    # Spaxels
    @property
    def _hspaxels(self):
        """Spaxel properties {shape: (N,M), spx_scale: float [in arcsec]}."""
        #
        return self._spaxels

    @property
    def spaxels(self):
        """Spaxel coordinates and properties."""
        # build from self._spaxels
        if self._spaxel_coords is None or len(self._spaxel_coords) == 0:
            self._spaxel_coords = build_pixels(self._spaxels["shape"], oversampling=1)

        return self._spaxel_coords

    @property
    def spx_shape(self):
        """Shape of the spaxel grid (ny, nx)."""
        return self.spaxels["shape"]

    @property
    def spx_centroids(self):
        """Spaxel centroids."""
        return self.spaxels["centroids"]

    @property
    def spx_edges(self):
        """Spaxel edges."""
        return self.spaxels["edges"]

    @property
    def spx_area(self):
        """Area of a spaxel in arcsec^2."""
        if np.ndim(self.spx_spatial_scale) == 0:
            spx_spatial_scale_y = spx_spatial_scale_x = self.spx_spatial_scale
        else:
            spx_spatial_scale_y, spx_spatial_scale_x = self.spx_spatial_scale
        return spx_spatial_scale_y * spx_spatial_scale_x

    @property
    def spx_spatial_scale(self):
        """Spatial scale of a spaxel in arcsec."""
        spx_spatial_scale = self._spaxels.get("spx_scale", None)
        if spx_spatial_scale is None:
            from .profiles import airyradius_to_gaussiansigma
            warnings.warn("Setting spaxels scale from airy")
            radius_airy = self.telescope.get_airy_radius(self.lbda_ref)
            spx_spatial_scale = airyradius_to_gaussiansigma(radius_airy, on="fwhm")

        return spx_spatial_scale

    @property
    def dispersion_resolution(self):
        """ provide the effective dispersion resolution in pixels. 
        note: 2 is optimal. Lower, the PSF is unresolved, 
        higher the PSF is oversampled hence producing unecessary pixel noise. """
        return self.meta.get("dispersion_resolution", 2)

    
    @property
    def nlbda(self):
        """Number of spectral pixels."""
        return len(self.lbda)

    @property
    def lbda_bin(self):
        """Wavelength edge bins."""
        return np.vstack([self.lbda_edges[:-1],
                          self.lbda_edges[1:]]).T  # (nlbda, 2) [Å]

    @property
    def meta(self):
        """Metadata of the instance."""
        return self._meta

    @property
    def name(self):
        """Name of the spectrograph (if any)."""
        return self.meta.get("name", "")

    @property
    def lbda_ref(self):
        """Reference wavelength in Angstrom."""
        # 1micron by default
        return self.meta.get("lbda_ref", 10_000)

    @property
    def omega(self):
        """Spaxel solid angle in steradians."""
        hspx = self.spx_spatial_scale * 4.84813681109536e-06  # [rad]
        if np.ndim(hspx) == 0:
            hspx_y = hspx_x = hspx
        else:
            hspx_y, hspx_x = hspx

        # Case: square | could remove np.sin that has no impact
        return np.sin(hspx_y) * np.sin(hspx_x)

        # Case: circular
        # return np.pi * np.sin(hspx_y/2) * np.sin(hspx_x/2)

    @property
    def skyarea(self):
        """Full sky area (nspaxel * spaxel area)."""
        pass

    @property
    def psf_sigma_spectral(self):
        """Chromatic PSF sigma in arcsec."""
        if self.spatial_psf["sigma_spectral"] is None or \
                self.spatial_psf["sigma_spectral"] in ["default"]:
            # 2.9 is the airy equivalent.
            from .profiles import airyradius_to_gaussiansigma
            radius_airy = self.telescope.get_airy_radius(self.lbda_ref)
            self.spatial_psf["sigma_spectral"] = airyradius_to_gaussiansigma(radius_airy, on="fwhm")

        return float(self.spatial_psf["sigma_spectral"])

    @property
    def type(self):
        """Type of the spectrograph."""
        return self._SPECTROGRAPH_TYPE


class MLASpectrograph(Spectrograph):
    """A spectrograph with no anamorphose and x-dispersion for traces."""
    _ANAMORPHOSE = None
    _SPECTROGRAPH_TYPE = "mla"

    def __init__(self, xdispersion={}, *args, **kwargs):
        """Initialize the MLASpectrograph.

        Parameters
        ----------
        xdispersion : dict, optional
            Cross-dispersion information. Default is {}.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        _ = super().__init__(*args, **kwargs)

        # an MLA has a cross-dispersion
        self.xdispersion = xdispersion

    # this is what from_config needs
    @classmethod
    def _parse_config(cls, config):
        """Parse the configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        dict
            Initialization properties.
        dict
            Input configuration.
        """

        init_prop, config = super()._parse_config(config)

        # PSF at the detector level
        xdispersion = {"sigma_spectral": float(config["psf"]["detector"]["xdisp_sigma_spectral"]),
                       "sigma": float(config["psf"]["detector"]["xdisp_sigma"]),
                       "profile": config["psf"]["detector"]["xdisp_profile"]}
        init_prop["xdispersion"] = xdispersion
        return init_prop, config

    def update(self, reset_others=False, **kwargs):
        """Update any mutable attribute of the spectrograph.

        Parameters
        ----------
        reset_others : bool, optional
            If True, reset other parameters to their initial values.
            Default is False.
        **kwargs
            Parameters to update.
        """
        # do xdispersion stuffs
        xdisp_updates = {}
        for k, v in kwargs.items():
            if k in self.meta["psf"]["detector"].keys():
                xdisp_updates[k.replace("xdisp_", "")] = v
                _ = kwargs.pop(k)  # remove them

        self.xdispersion = self.xdispersion | xdisp_updates
        # and the rest (normal spectrograph)
        return super().update(reset_others=False, **kwargs)

    def get_xdisp_sigma_spectral(self, xdims=0, xdisp_sigma=None):
        """Get spectral PSF stddev in pixels.

        The total (Gaussian) spectral PSF is made of two components:
        - the optical (chromatic) component, with stddev proportional
          to wavelength, normalized at wref=1 µm,
        - the achromatic component, with constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        Parameters
        ----------
        xdims : int, optional
            Extra dimensions to be appended. Default is 0.
        xdisp_sigma : float, optional
            Constant sigma override in pixels. Default is None.

        Returns
        -------
        array_like
            Total sigma in pixels.
        """
        if xdisp_sigma is None:
            xdisp_sigma = self.xdispersion["sigma"]

        # adding the internal PSF 
        inst_psf = self.get_instrumental_psf(in_spaxels=True)
        if inst_psf is not None and inst_psf>0:
            xdisp_sigma = np.hypot(xdisp_sigma, inst_psf)
            
        return self._get_chromatic_sigma(self.lbda,
                                         chromatic_sigma=self.xdispersion["sigma_spectral"],
                                         constant_sigma=xdisp_sigma,
                                         wref=self.lbda_ref,
                                         xdims=xdims)
    
    def get_nea_spatial(self, position=(0, 0), in_spaxels=True, guiding_sigma=None):
        """Noise equivalent area in unit of slice/spaxels.

        i.e., how many "spaxel noise".

        Parameters
        ----------
        position : tuple, optional
            Position of the PSF in unit of spaxel/slicer. Default is (0, 0).
        in_spaxels : bool, optional
            If True, return the area in spaxels^2. If False, in arcsec^2.
            Default is True.
        guiding_sigma : float, optional
            Guiding sigma override in arcsec. Default is None.

        Returns
        -------
        array_like
            Noise equivalent area.
        """
        from .nea import get_2dnorm_nea

        if self.spatial_psf["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian spatial PSF profile implemented, but: {self.spatial_psf['profile']=}")

        sigma_at_mla = self.get_psf_sigma_spectral(in_spaxels=in_spaxels,
                                                   guiding_sigma=guiding_sigma)
        return get_2dnorm_nea(sigma_at_mla, mean=position)

    def get_nea_pixels(self):
        """Noise equivalent area of a spaxel/slice in unit of pixels.

        Returns
        -------
        array_like
            Noise equivalent area in pixels.
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
        """Chromatic cross-dispersion sigma in pixels."""
        return self.xdispersion["sigma_spectral"]


class SlicerSpectrograph(Spectrograph):
    """A spectrograph with an anamorphosis and a slicer."""
    _ANAMORPHOSE = (2, 1)
    _SPECTROGRAPH_TYPE = "slicer"

    def set_spaxels(self, shape, spx_scale):
        """Set spaxel information.

        Parameters
        ----------
        shape : tuple
            Spaxel grid shape (ny, nx).
        spx_scale : float
            Spaxel scale in arcsec.
        """
        shape = np.asarray(shape) * self._ANAMORPHOSE
        spx_scale = np.asarray(spx_scale) / self._ANAMORPHOSE
        self._spaxels = {"shape": shape, "spx_scale": spx_scale}
        self._spaxel_coords = {}

    def get_nea_pixels(self):
        """Noise equivalent area of a slice (per wavelength) in unit of pixels.

        Returns
        -------
        int
            Noise equivalent area in pixels (1 for a slicer).
        """
        return 1
