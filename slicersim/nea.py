""" function that manage PSF profile at the lenses/slicer and at the detector """

import warnings
import numpy as np
from scipy import stats


# ========================= #
#                           #
#   Generic Astropy PSF     #
#                           #
# ========================= #
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
    from .profiles import get_profilemodel
    psfmodel = get_profilemodel(name, position=position, **kwargs)
    return get_2d_nea(psfmodel, xx=xx, yy=yy, norm_by_step=norm_by_step)

# ========================= #
#                           #
#   Noise Equivalent Area   #
#                           #
# ========================= #
def pixels_to_nea(pixels, norm=1):
    """ apply the NEA calculation on input pixels
    nea = sum(pixel)**2 / sum( pixel**2).
    
    here the sum is done on the last dimension (ndim=2) or last 2 dimensions (ndim>=3)

    """
    if np.ndim(pixels) == 2:
        nea = np.nansum(pixels, axis=1)**2 / np.nansum(pixels**2, axis=1) * norm
    else:
        if np.ndim(pixels) > 3:
            warnings.warn("input pixels dimension greater than 3... integration over last 2 used.")
        
        nea = np.nansum(pixels, axis=(-2,-1))**2 / np.nansum(pixels**2, axis=(-2,-1)) * norm
        
    return nea

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
    return pixels_to_nea(pixels, norm=norm)

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
    return pixels_to_nea(pixels, norm=norm)
        
# ------------------ #
#  Gaussian Shortcut #
# ------------------ #
def get_1dnorm_nea(sigma, xx="-7:7:15j", mean=0):
    """ """
    return get_1d_nea(get_gaussian1d, xx=xx, sigma=sigma, mean=mean)

def get_2dnorm_nea(sigma, xx="-7:7:15j", yy="-7:7:15j", mean=(0,0),
                  norm_by_step = True):
    """ """
    return get_2d_nea( get_gaussian2d, xx=xx, yy=yy, norm_by_step=True,
                       sigma=sigma, mean=mean)
    



