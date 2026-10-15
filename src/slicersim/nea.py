r"""Noise Equivalent Area (NEA) of PSF profiles.

The NEA of a PSF sampled on pixels :math:`p_i` is

.. math::

    \mathrm{NEA} = \frac{\left(\sum_i p_i\right)^2}{\sum_i p_i^2}
    \times \Delta,

where :math:`\Delta` is the pixel area (or size in 1D) when the result is
normalized by the sampling step, such that it does not depend on the
sampling of the PSF.
"""

import warnings

import numpy as np


# ========================= #
#                           #
#   Generic Astropy PSF     #
#                           #
# ========================= #
def get_2dpsf_nea(name, xx="-7:7:15j", yy="-7:7:15j", norm_by_step=True,
                      position=(0, 0), **kwargs): # pragma: no cover 
    """Get the Noise Equivalent Area (NEA) of an astropy 2D PSF model.

    Parameters
    ----------
    name : str
        Name of the 2D PSF model from astropy, see
        `~slicersim.profiles.get_profilemodel`. Shortcuts:

        - ``"airy"``, ``"airydisk"``: ``"AiryDisk2D"``
        - ``"gaussian"``, ``"gauss"``, ``"normal"``: ``"Gaussian2D"``
        - ``"moffat"``: ``"Moffat2D"``
        - ``"sersic"``: ``"Sersic2D"``
        - ``"ricker"``, ``"mexican"``, ``"mexicanhat"``: ``"RickerWavelet2D"``
        - ``"lorentz"``: ``"Lorentz2D"``
    xx, yy : str or array_like, optional
        Coordinates where the PSF is evaluated. A string is interpreted as a
        `numpy.r_` slice expression. Default is ``"-7:7:15j"``.
    norm_by_step : bool, optional
        If True, multiply the NEA by the pixel area
        ``(xx[1] - xx[0]) * (yy[1] - yy[0])`` (``xx`` and ``yy`` must then be
        regularly spaced), such that the result does not depend on the
        sampling. Default is True.
    position : tuple of float, optional
        ``(x, y)`` location of the PSF centroid. Default is (0, 0).
    **kwargs
        Goes to `~slicersim.profiles.get_profilemodel`, and from there to the
        astropy model initializer, see
        https://docs.astropy.org/en/latest/modeling/predef_models2D.html.
        Models other than ``AiryDisk2D`` and ``Gaussian2D`` require
        ``normalized=False`` (the NEA does not depend on the amplitude).

    Returns
    -------
    numpy.ndarray
        Noise equivalent area, in units of ``xx`` squared if
        ``norm_by_step`` (in pixels of the sampling grid otherwise).

    Notes
    -----
    The model must be evaluated on a stack of PSFs (see Notes of
    `get_2d_nea`), e.g. by giving its size parameter as an array of shape
    ``(n, 1, 1)``.
    """
    from .profiles import get_profilemodel
    psfmodel = get_profilemodel(name, position=position, **kwargs)
    return get_2d_nea(psfmodel, xx=xx, yy=yy, norm_by_step=norm_by_step)

# ========================= #
#                           #
#   Noise Equivalent Area   #
#                           #
# ========================= #
def pixels_to_nea(pixels, norm=1): # pragma: no cover 
    r"""Compute the Noise Equivalent Area (NEA) from sampled pixels.

    .. math::

        \mathrm{NEA} = \mathrm{norm} \times
        \frac{\left(\sum p\right)^2}{\sum p^2}

    The sums ignore NaNs and are done over the last axis if
    ``pixels.ndim == 2`` (1D profiles), or over the last 2 axes otherwise
    (2D profiles).

    Parameters
    ----------
    pixels : numpy.ndarray
        Input pixels, of shape ``(n, nx)`` (1D profiles) or
        ``(..., ny, nx)`` (2D profiles).
    norm : float, optional
        Normalization factor, typically the pixel size (1D) or area (2D).
        Default is 1.

    Returns
    -------
    numpy.ndarray
        Noise Equivalent Area.

    Warns
    -----
    UserWarning
        If ``pixels`` has more than 3 dimensions.
    """
    if np.ndim(pixels) == 2:
        nea = np.nansum(pixels, axis=1)**2 / np.nansum(pixels**2, axis=1) * norm
    else:
        if np.ndim(pixels) > 3:
            warnings.warn("input pixels dimension greater than 3... integration over last 2 used.")
        
        nea = np.nansum(pixels, axis=(-2,-1))**2 / np.nansum(pixels**2, axis=(-2,-1)) * norm
        
    return nea

def get_1d_nea(func, xx="-7:7:15j", norm_by_step=True, **kwargs): # pragma: no cover 
    """Get the Noise Equivalent Area (NEA) of a 1D function.

    Parameters
    ----------
    func : callable
        1D PSF function such that ``pixels = func(xx, **kwargs)`` is an
        array of shape ``(n, len(xx))``.
    xx : str or array_like, optional
        Coordinates where the PSF is evaluated. A string is interpreted as a
        `numpy.r_` slice expression. Default is ``"-7:7:15j"``.
    norm_by_step : bool, optional
        If True, multiply the NEA by the step ``xx[1] - xx[0]`` (``xx`` must
        then be regularly spaced), such that the result does not depend on
        the sampling: ``get_1d_nea(func, xx="-9:9:0.5")`` and
        ``get_1d_nea(func, xx="-9:9:0.05")`` are then equal.
        Default is True.
    **kwargs
        Goes to ``func``.

    Returns
    -------
    numpy.ndarray
        NEA (integrated over ``xx``), of shape ``(n,)``.
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
                  norm_by_step=True, **kwargs): # pragma: no cover 
    """Get the Noise Equivalent Area (NEA) of a 2D function.

    Parameters
    ----------
    func : callable
        2D PSF function such that ``pixels = func(x, y, **kwargs)``, where
        ``x`` and ``y`` are the open mesh-grid ``xx[None, :]`` (shape
        ``(1, N)``) and ``yy[:, None]`` (shape ``(M, 1)``), returns an array
        of shape ``(..., M, N)``.
    xx, yy : str or array_like, optional
        Coordinates where the PSF is evaluated. A string is interpreted as a
        `numpy.r_` slice expression. Default is ``"-7:7:15j"``.
    norm_by_step : bool, optional
        If True, multiply the NEA by the pixel area
        ``(xx[1] - xx[0]) * (yy[1] - yy[0])`` (``xx`` and ``yy`` must then be
        regularly spaced), such that the result does not depend on the
        sampling. Default is True.
    **kwargs
        Goes to ``func``.

    Returns
    -------
    numpy.ndarray
        NEA (integrated over ``xx`` and ``yy``), of shape ``(...)``.

    Notes
    -----
    `pixels_to_nea` treats 2D inputs as a set of 1D profiles: ``func`` must
    return an array with at least one leading axis, i.e. of shape
    ``(n, M, N)``, for the NEA to be computed over the 2D images.
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
def get_1dnorm_nea(sigma, xx="-7:7:15j", mean=0): # pragma: no cover 
    """Get the Noise Equivalent Area (NEA) of a 1D Gaussian.

    This is a shortcut for `get_1d_nea` with
    ``func=get_gaussian1d`` (see `~slicersim.profiles.get_gaussian1d`) and
    ``norm_by_step=True``.

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Standard deviation(s) of the Gaussian, in units of ``xx``.
    xx : str or array_like, optional
        Coordinates where the PSF is evaluated. A string is interpreted as a
        `numpy.r_` slice expression. Default is ``"-7:7:15j"``.
    mean : float, optional
        Mean of the Gaussian. Default is 0.

    Returns
    -------
    numpy.ndarray
        NEA (integrated over ``xx``), in units of ``xx``, one per ``sigma``.
    """
    from .profiles import get_gaussian1d
    return get_1d_nea(get_gaussian1d, xx=xx, sigma=sigma, mean=mean)

def get_2dnorm_nea(sigma, xx="-7:7:15j", yy="-7:7:15j", mean=(0,0),
                  norm_by_step = True): # pragma: no cover 
    """Get the Noise Equivalent Area (NEA) of a 2D Gaussian.

    This is a shortcut for `get_2d_nea` with ``func=get_gaussian2d`` (see
    `~slicersim.profiles.get_gaussian2d`).

    Parameters
    ----------
    sigma : array_like
        Standard deviations of the Gaussian, in units of ``xx``, of shape
        ``(n, 1, 2)`` with the last axis being ``(sigma_y, sigma_x)`` (see
        Notes of `get_2d_nea`).
    xx, yy : str or array_like, optional
        Coordinates where the PSF is evaluated. A string is interpreted as a
        `numpy.r_` slice expression. Default is ``"-7:7:15j"``.
    mean : tuple of float, optional
        ``(x, y)`` center of the Gaussian. Default is (0, 0).
    norm_by_step : bool, optional
        Currently ignored: the NEA is always normalized by the pixel area
        (as with ``norm_by_step=True`` in `get_2d_nea`). Default is True.

    Returns
    -------
    numpy.ndarray
        NEA (integrated over ``xx`` and ``yy``), in units of ``xx`` squared,
        of shape ``(n,)``.
    """
    from .profiles import get_gaussian2d
    return get_2d_nea( get_gaussian2d, xx=xx, yy=yy, norm_by_step=True,
                       sigma=sigma, mean=mean)
    



