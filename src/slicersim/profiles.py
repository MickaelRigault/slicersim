

import numpy as np
from astropy.modeling import functional_models
from .utils import integ_gaussian2D_erf
from scipy import stats
# =========== #
#  Internal   #
# =========== #
def _center_to_edge_(xx):
    """ """
    step_ = xx[1]-xx[0]
    return np.append(xx, xx[-1]+step_) - step_/2

def airyradius_to_gaussiansigma(radius, on="fwhm"):
    """ """
    if on.lower() == "fwhm":
        ratio = 2.35/0.8
    else:
        raise NotImplementedError(f"only on='fwhm' implemented, {on=} given")
        
    return radius / ratio

def gaussiansigma_to_airyradius(sigma, on="fwhm"):
    """ """
    if on.lower() == "fwhm":
        ratio = 0.8/2.35
    else:
        raise NotImplementedError(f"only on='fwhm' implemented, {on=} given")
        
    return sigma / ratio

def build_pixels(shape, oversampling=10):
    """ """
    if oversampling is None:
        oversampling = 1
    else:
        # generic for int or list of.
        oversampling = np.asarray(oversampling, dtype="int")

    # assymetric oversampling
    if np.ndim(oversampling) == 1:
        oversampling_y, oversampling_x = oversampling
    else:
        oversampling_y = oversampling_x = oversampling
    
        
    ny, nx = np.asarray(shape, dtype=int)
    x = np.linspace(1, nx, (nx)*oversampling_x) - nx/2. -0.5 # centered
    y = np.linspace(1, ny, (ny)*oversampling_y) - ny/2. -0.5 # centered
    pixel_sizex = x[1]-x[0]
    pixel_sizey = y[1]-y[0]
    x_edges = np.append(x, x[-1]+pixel_sizex) - pixel_sizex/2
    y_edges = np.append(y, y[-1]+pixel_sizex) - pixel_sizey/2
    return {"centroids":(x[None,:], y[:,None]), 
            "edges": (x_edges[None,:], y_edges[:,None]),
            "shape": shape,
            "pixelarea": pixel_sizex*pixel_sizey,
            "fullshape": (ny*oversampling_y, nx*oversampling_x),
            "oversampling": oversampling
           }

# =========== #
#  Generic    #
# =========== #
def psfimage_to_encircledenergy(psfimage, radius, position=(0,0)):
    """ """
    import sep
    r = np.atleast_1d(radius)
    return sep.sum_circle(psfimage, [position[1]], [position[0]] , r=r)[0]

def get_profilepsf(profile, shape=(10, 10), oversampling=10, 
                   position=(0, 0), normal_scatter=None,
                   norm_pixels=False, **kwargs):
    """ compute the 2d psf image associated for the given profile.

    Parameters
    ----------
    profile: str
        name of the profile:
        - airy
        - gaussian

    shape: (int, int)
        shape of the output image

    oversampling: int
        number of sub-pixels per shape-unit

    position: (float, float)
        coordinate of the PSF with respect to the center of the images
        in shape-units

    normal_scatter: None, float
        additional normal scattering in shape-units

    **kwargs goes to profile:
        - radius mandatory for profile='airy'
        - sigma mandatory for profile='gaussian'
        
    Returns
    -------
    psf, centroid
        psf: image
        centroid: position in the unit of psf image pixels.
    """
    
    # build pixels
    pxl = build_pixels(shape, oversampling=oversampling)
    
    # Airy
    if profile == "airy":
        radius = kwargs.pop("radius") # fails if not given
        radius = np.atleast_1d(radius)[:, None, None]
        psf_func = get_profilemodel("airy", position=position,
                                             radius=radius,
                                             **kwargs)
        psf = psf_func(*pxl["centroids"]) # normed
        
    # Gaussian
    elif profile in ["normal", "gaussian", "gauss"]:
        sigma = kwargs.pop("sigma") # fails if not given
        sigma = np.atleast_1d(sigma)[:, None, None]
        psf = integ_gaussian2D_erf(pxl["edges"], sigma=sigma, mu=position,
                                        **kwargs)
        psf /= pxl["pixelarea"]
    # Fails otherwise
    else:
        raise NotImplementedError(f"unknown profile {profile=}")

    if norm_pixels:
        psf *= pxl["pixelarea"]
    
    # add normal scatter
    if normal_scatter is not None and normal_scatter>0:
        # gaussian convolution
        from scipy.ndimage import gaussian_filter
        scatter_pixels = normal_scatter * oversampling # 
        psf = gaussian_filter(psf, scatter_pixels, axes=(-2,-1))

    # centroid in psf image coordinates
    img_center = np.asarray( psf.shape[-2:] )/2 - 0.5
    centroid = img_center + np.asarray(position) * oversampling

    return np.squeeze(psf), centroid, pxl["pixelarea"]

def get_profilemodel(name, position=(0,0), normalized=True, **kwargs):
    """ get profile model function.
    
    PSF models comes from astropy.modeling.functional_models and can be used as follow
    ```python
    sigmas = 2 # or array of shape (M, 1, 1)
    airy = get_psfmodel("airy", position=(0.4, 1.2), radius=sigmas)
    xx = np.r_[-7:7:50j] 
    yy = np.r_[-7:7:49j] # different dim for example
    psf_stamp = airy(xx[None,:], yy[:,None]) # resulting shape (M, 49, 50)
    ```

    Parameters
    ----------
    name: str
        name of the 2D PSF model from astropy.
        Shortcut:
        - "airy", "airydisk": "AiryDisk2D",
        - "gaussian","gauss", "normal": "Gaussian2D",
        - "moffat":  "Moffat2D",
        - "sersic": "Sersic2D",
        - "ricker", "mexican", "mexicanhat": "RickerWavelet2D"
        - "lorentz": "Lorentz2D"
    
    position: (float, float)
        location of the centroid.

    **kwargs goes to corresponding astropy model init.
       => see https://docs.astropy.org/en/latest/modeling/predef_models2D.html
       All have amplitude=1 and:
       - AiryDisk2D: radius=1
       - Gaussian2D: x_stddev=None, y_stddev=None, theta=None, cov_matrix=None
       - Moffat2D: gamma=1, alpha=1
       - Sersic2D: r_eff=1, n=4, ellip=0, theta=0.0, ...
       - RickerWavelet2D: , sigma=1
       - Lorentz2D: fwhm=1

    Returns
    -------
    func
    """
    SHORT_CUTS = {"airy": "AiryDisk2D",
                  "gaussian": "Gaussian2D",
                  "moffat": "Moffat2D",
                  "sersic": "Sersic2D",
                  "ricker": "RickerWavelet2D",
                  "lorentz": "Lorentz2D"}
    SHORT_CUTS |= {k: SHORT_CUTS["airy"] for k in ["airydisk"]}
    SHORT_CUTS |= {k: SHORT_CUTS["gaussian"] for k in ["gauss", "normal"]}
    SHORT_CUTS |= {k: SHORT_CUTS["ricker"] for k in ["mexican", "mexicanhat"]}
    
    name = SHORT_CUTS.get(name, name)
    
    if name == "Gaussian2D":
        kwargs["x_mean"], kwargs["y_mean"] = position
        if "sigma" in kwargs:
            std = kwargs.pop("sigma")
            kwargs["x_stddev"] = kwargs["y_stddev"] = std
        elif "radius" in kwargs:
            std = kwargs.pop("radius")
            kwargs["x_stddev"] = kwargs["y_stddev"] = std
                
    else:
        kwargs["x_0"], kwargs["y_0"] = position

    if normalized:
        if name == "AiryDisk2D":
            # See e.g. photutils => #AiryDiskPSF.evaluate()
            from scipy.special import jn_zeros
            radius = kwargs.get("radius", 1)
            radius_x = radius_y = radius
            
            _rz = jn_zeros(1, 1)[0] / np.pi
            norm = (4.0 / np.pi) * (radius_x*radius_y / _rz**2)
            
        elif name == "Gaussian2D":
            # norm of a 2D symetric gaussian (no ellipticity)
            norm = 2*np.pi * kwargs["x_stddev"] * kwargs["y_stddev"]
            
        else:
            raise ValueError(f"Only AiryDisk2D & Gaussian2D norms havs been implemented, not {name=}")
            
        kwargs["amplitude"] = 1/norm
        
    return getattr(functional_models, name)(**kwargs)

# =========== #
#  Airy       #
# =========== #


# =========== #
#  Gaussian   #
# =========== #
def get_gaussian1d(xx, sigma, mean=0):
    """ """
    sigma = np.atleast_1d(sigma)
    mean = np.atleast_1d(mean)
    return stats.norm.pdf(xx, loc=mean, scale=sigma[:,None])
    
def get_gaussian2d(xx, yy, sigma, mean=(0,0), **kwargs):
    """ exact PSF from erf function using pixel edges. """
    from .utils import integ_gaussian2D_erf
    
    sigma = np.atleast_2d(sigma)
    
    # center to edge
    xedge = _center_to_edge_(np.squeeze(xx)) # (1, N)=>(N,) 
    yedge = _center_to_edge_(np.squeeze(yy)) # (N, 1)=>(N,)
    
    #return xedge, yedge
    pixels = integ_gaussian2D_erf(
             (xedge[None,:], yedge[:,None]),  # ((1, nx), (ny, 1)) [spx]
              sigma,                          # (nlbda, 1, 1) [spx]
              mean,                           # [spx]
              normed=True,                    # sum(axis=(1, 2)) = 1z
              **kwargs)                    
              
    return pixels
