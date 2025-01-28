""" function that manage PSF profile at the lenses/slicer and at the detector """


import numpy as np
from scipy import stats


# ========================= #
#                           #
#   Generic Astropy PSF     #
#                           #
# ========================= #
def get_psfmodel(name, position=(0,0), **kwargs):
    """ get PSF model function.
    
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
    from astropy.modeling import functional_models
    SHORT_CUTS = {"airy": "AiryDisk2D",
                  "gaussian": "Gaussian2D",
                  "moffat":  "Moffat2D",
                  "sersic": "Sersic2D",
                  "ricker": "RickerWavelet2D",
                  "lorentz": "Lorentz2D"}
    SHORT_CUTS |= {k:SHORT_CUTS["airy"] for k in ["airydisk"]}
    SHORT_CUTS |= {k:SHORT_CUTS["gaussian"] for k in ["gauss", "normal"]}
    SHORT_CUTS |= {k:SHORT_CUTS["ricker"] for k in ["mexican", "mexicanhat"]}
    name = SHORT_CUTS.get(name, name)
    
    if name == "Gaussian2D":
        kwargs["x_mean"], kwargs["y_mean"] = position
        if "sigma" in kwargs:
            kwargs["x_stddev"] = kwargs["y_stddev"] = kwargs.pop("sigma")
    else:
        kwargs["x_0"], kwargs["y_0"] = position
        
    return getattr(functional_models, name)(**kwargs)


def get_2dpsf_nea(name, xx="-7:7:15j", yy="-7:7:15j", norm_by_step=True,
                      position=(0, 0), **kwargs):
    """ get the Noise Equivalent area of a PSF model 

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

    xx, yy: str, array
        coordinate where the PSF is evaluated

    norm_by_step: bool
        should the xx binning be accounted for ? 
        if so, xx binning should be linearly increasing such that 
        bin_step = x[1]-x[0].
        if true, then get_1d_nea(func, xx="-9:9:0.5") and 
        get_1d_nea(func, xx="-9:9:0.05") are equal.
        
    position: (float, float)
        location of the PSF centroid.

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
    float, array
        noise equivalent area given the PSF model (name) and parameter (kwargs)
    """
    psfmodel = get_psfmodel(name, position=position, **kwargs)
    return get_2d_nea(psfmodel, xx=xx, yy=yy, norm_by_step=norm_by_step)

# ========================= #
#                           #
#   Noise Equivalent Area   #
#                           #
# ========================= #
def get_1d_nea(func, xx="-7:7:15j", norm_by_step=True, **kwargs):
    """ get the noise equivalent area of a 1d function

    Parameters
    ----------
    func: func
        1D PSF function such that: pixels = func(xx, **kwargs)

    xx: str, array
        coordinate where the PSF is evaluated

    norm_by_step: bool
        should the xx binning be accounted for ? 
        if so, xx binning should be linearly increasing such that 
        bin_step = x[1]-x[0].
        if true, then get_1d_nea(func, xx="-9:9:0.5") and 
        get_1d_nea(func, xx="-9:9:0.05") are equal.
        
    Returns
    -------
    nea (dim: integrated over xx)
    """
    if type(xx) is str:
        xx = eval(f"np.r_[{xx}]")

    if norm_by_step:
        norm = xx[1]-xx[0]
    else:
        norm = 1
        
    pixels = func(xx, **kwargs)
    return np.nansum(pixels, axis=1)**2 / np.nansum(pixels**2, axis=1) * norm

def get_2d_nea(func, xx="-7:7:15j", yy="-7:7:15j",
                  norm_by_step=True, **kwargs):
    """ get the noise equivalent area of a 2d function

    Parameters
    ----------
    func: func
        2D PSF function such that: pixels = func(xx, yy, **kwargs)
        where xx and yy are (N,) and (M,) centroid 1d arrays. 
        returning in pixels as a (..., M, N,) array. 
        
    xx, yy: str, array
        coordinate where the PSF is evaluated

    norm_by_step: bool
        should the xx binning be accounted for ? 
        if so, xx binning should be linearly increasing such that 
        bin_step = x[1]-x[0].
        if true, then get_1d_nea(func, xx="-9:9:0.5") and 
        get_1d_nea(func, xx="-9:9:0.05") are equal.

    Returns
    -------
    nea (dim: integrated over xx and yy)
    """
    if type(xx) is str:
        xx = eval(f"np.r_[{xx}]")
    
    if type(yy) is str:
        yy = eval(f"np.r_[{yy}]")

    if norm_by_step:
        norm = (xx[1]-xx[0]) * (yy[1]-yy[0])
    else:
        norm = 1

    pixels = func(xx[None,:], yy[:,None], **kwargs)
    return np.nansum(pixels, axis=(-2,-1))**2 / np.nansum(pixels**2, axis=(-2,-1)) * norm
        
# ------------------ #
#  Gaussian Shortcut #
# ------------------ #
def _center_to_edge_(xx):
    """ """
    step_ = xx[1]-xx[0]
    return np.append(xx, xx[-1]+step_) - step_/2

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
    xedge = _center_to_edge_(np.squeeze(xx)) # (1,N)=>(N,) 
    yedge = _center_to_edge_(np.squeeze(yy)) # (N,1)=>(N,) 
    #return xedge, yedge
    pixels = integ_gaussian2D_erf(
             (xedge[None,:], yedge[:,None]),  # ((1, nx), (ny, 1)) [spx]
              sigma,                          # (nlbda, 1, 1) [spx]
              mean,                           # [spx]
              normed=True,                    # sum(axis=(1, 2)) = 1z
              **kwargs)                    
              
    return pixels

def get_1dnorm_nea(sigma, xx="-7:7:15j", mean=0):
    """ """
    return get_1d_nea(get_gaussian1d, xx=xx, sigma=sigma, mean=mean)

def get_2dnorm_nea(sigma, xx="-7:7:15j", yy="-7:7:15j", mean=(0,0),
                  norm_by_step = True):
    """ """
    return get_2d_nea( get_gaussian2d, xx=xx, yy=yy, norm_by_step=True,
                       sigma=sigma, mean=mean)
    



