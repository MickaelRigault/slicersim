""" function that manage PSF profile at the lenses/slicer and at the detector """


import numpy as np
from scipy import stats
from .utils import integ_gaussian2D_erf

# ========================= #
#                           #
#   Noise Equivalent Area   #
#                           #
# ========================= #
def _center_to_edge_(xx):
    """ """
    step_ = xx[1]-xx[0]
    return np.append(xx, xx[-1]+step_) - step_/2

def get_1d_nea(func, xx="-7:7:15j", norm_by_step = True, **kwargs):
    """ get the noise equivalent area of a 1d function

    Parameters
    ----------
    func: func
        1D PSF function such that: pixels = func(xx, **kwargs)

    xx: str, array
        numpy array to estimate the PSF. 

    norm_by_step: bool
        should the xx binning be accounted for ? 
        if so, xx binning should be linearly increasing such that 
        bin_step = x[1]-x[0].
        if true, then get_1d_nea(func, xx="-9:9:0.5") and get_1d_nea(func, xx="-9:9:0.05") are equal.
        
    Returns
    -------
    nea (in dim less than pixels)
    """
    if type(xx) is str:
        xx = eval(f"np.r_[{xx}]")

    if norm_by_step:
        norm = xx[1]-xx[0]
    else:
        norm = 1
        
    pixels = func(xx, **kwargs)
    return np.nansum(pixels, axis=1)**2 / np.nansum(pixels**2, axis=1) * norm

def get_1dnorm_nea(sigma, xx="-7:7:15j", mean=0):
    """ """
    sigma = np.atleast_1d(sigma)
    mean = np.atleast_1d(mean)
    cross_disp_func = stats.norm(loc=mean, scale=sigma[:,None]).pdf
    
    return get_1d_nea(cross_disp_func, xx=xx)
    
def get_2dnorm_nea(sigma, xx="-7:7:15j", yy="-7:7:15j", mean=(0,0),
                  norm_by_step = True):
    """ """
    if type(xx) is str:
        xx = eval(f"np.r_[{xx}]")
    
    if type(yy) is str:
        yy = eval(f"np.r_[{yy}]")

    if norm_by_step:
        norm = (xx[1]-xx[0]) * (yy[1]-yy[0])
    else:
        norm = 1

    sigma = np.atleast_2d(sigma)
    
    # center to edge
    xedge = _center_to_edge_(xx)
    yedge = _center_to_edge_(yy)
    #return xedge, yedge
    pixels = integ_gaussian2D_erf(
             (xedge[None,:], yedge[:,None]),  # ((1, nx), (ny, 1)) [spx]
              sigma,                          # (nlbda, 1, 1) [spx]
              mean,                           # [spx]
              normed=True)                    # sum(axis=(1, 2)) = 1z

    return np.nansum(pixels, axis=(-2,-1))**2 / np.nansum(pixels**2, axis=(-2,-1)) * norm



