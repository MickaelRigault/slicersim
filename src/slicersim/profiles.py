"""2D PSF profiles (Airy disk, Gaussian) and pixel grids to sample them.

Coordinates and sizes are given in "shape units", i.e. in units of the
(non-oversampled) pixels of the image the profiles are evaluated on, with
(0, 0) at the image center.
"""

import numpy as np
from astropy.modeling import functional_models
from scipy import stats

from .utils import integ_gaussian2D_erf


# =========== #
#  Internal   #
# =========== #
def _center_to_edge_(xx):
    """Convert center coordinates to edge coordinates.

    Shifts and extends an array of center coordinates to represent pixel edges
    by adding half-steps before and after.

    Parameters
    ----------
    xx : numpy.ndarray
        Regularly spaced center coordinates.

    Returns
    -------
    numpy.ndarray
        Edge coordinates, of length ``len(xx) + 1``.
    """
    step_ = xx[1]-xx[0]
    return np.append(xx, xx[-1]+step_) - step_/2

def airyradius_to_gaussiansigma(radius, on="fwhm"):
    r"""Convert an Airy disk radius into an equivalent Gaussian sigma.

    The conversion matches the FWHM of both profiles:

    .. math::

        \sigma = r \times 0.8 / 2.35

    Parameters
    ----------
    radius : float or numpy.ndarray
        Radius of the first dark ring of the Airy disk (any unit).
    on : str, optional
        Quantity matched between the two profiles. Only ``"fwhm"`` is
        implemented. Default is ``"fwhm"``.

    Returns
    -------
    float or numpy.ndarray
        Equivalent Gaussian sigma, in the same unit as ``radius``.

    Raises
    ------
    NotImplementedError
        If ``on`` is not ``"fwhm"``.

    See Also
    --------
    gaussiansigma_to_airyradius : The inverse conversion.
    """
    if on.lower() == "fwhm":
        ratio = 2.35/0.8
    else:
        raise NotImplementedError(f"only on='fwhm' implemented, {on=} given")

    return radius / ratio

def gaussiansigma_to_airyradius(sigma, on="fwhm"):
    r"""Convert a Gaussian sigma into an equivalent Airy disk radius.

    The conversion matches the FWHM of both profiles:

    .. math::

        r = \sigma \times 2.35 / 0.8

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Standard deviation of the Gaussian (any unit).
    on : str, optional
        Quantity matched between the two profiles. Only ``"fwhm"`` is
        implemented. Default is ``"fwhm"``.

    Returns
    -------
    float or numpy.ndarray
        Equivalent Airy disk radius, in the same unit as ``sigma``.

    Raises
    ------
    NotImplementedError
        If ``on`` is not ``"fwhm"``.

    See Also
    --------
    airyradius_to_gaussiansigma : The inverse conversion.
    """
    if on.lower() == "fwhm":
        ratio = 0.8/2.35
    else:
        raise NotImplementedError(f"only on='fwhm' implemented, {on=} given")

    return sigma / ratio

def build_pixels(shape, oversampling=10):
    """Build a pixel coordinate system with optional oversampling.

    Coordinates are in units of the (non-oversampled) pixels, with (0, 0) at
    the center of the image.

    Parameters
    ----------
    shape : tuple of int
        Shape ``(ny, nx)`` of the image.
    oversampling : int or list of int or None, optional
        Oversampling factor: a single integer for symmetric oversampling, or
        ``[oversampling_y, oversampling_x]`` for asymmetric oversampling.
        None means 1. Default is 10.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"centroids"``: ``(x, y)`` pixel centers, as an open mesh-grid of
          shapes ``(1, nx * oversampling_x)`` and ``(ny * oversampling_y, 1)``;
        - ``"edges"``: ``(x_edges, y_edges)`` pixel edges, as an open
          mesh-grid (one more element than ``"centroids"`` along each axis);
        - ``"shape"``: the input ``shape``;
        - ``"pixelarea"``: area of a single oversampled pixel;
        - ``"fullshape"``: shape of the oversampled image;
        - ``"oversampling"``: the oversampling factor(s) used.
    """
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
    x = (np.arange(nx * oversampling_x) + 0.5) / oversampling_x - nx / 2
    y = (np.arange(ny * oversampling_y) + 0.5) / oversampling_y - ny / 2
    pixel_sizex = 1 / oversampling_x
    pixel_sizey = 1 / oversampling_y
    x_edges = np.arange(nx * oversampling_x + 1) / oversampling_x - nx / 2
    y_edges = np.arange(ny * oversampling_y + 1) / oversampling_y - ny / 2

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
    """Calculate the encircled energy within given radii.

    This uses ``sep.sum_circle`` (requires the ``sep`` package).

    Parameters
    ----------
    psfimage : numpy.ndarray
        2D PSF image.
    radius : float or numpy.ndarray
        Aperture radius or radii, in pixels of ``psfimage``.
    position : tuple of float, optional
        ``(x, y)`` position of the aperture center, in pixels of
        ``psfimage`` (``(0, 0)`` being the center of the first pixel).
        Default is (0, 0).

    Returns
    -------
    numpy.ndarray
        Sum of ``psfimage`` within each radius.
    """
    import sep
    r = np.atleast_1d(radius)
    return sep.sum_circle(psfimage, [position[1]], [position[0]] , r=r)[0]

def get_profilepsf(profile, shape=(10, 10), oversampling=10,
                   position=(0, 0), normal_scatter=None,
                   norm_pixels=False, **kwargs):
    """Compute the oversampled 2D PSF image for a given profile.

    Parameters
    ----------
    profile : str
        Name of the profile:

        - ``"airy"``: Airy disk (see `get_profilemodel`);
        - ``"gaussian"``, ``"gauss"``, ``"normal"``: Gaussian profile
          integrated over pixels (see `~slicersim.utils.integ_gaussian2D_erf`).
    shape : tuple of int, optional
        Shape ``(ny, nx)`` of the (non-oversampled) image. Default is
        (10, 10).
    oversampling : int, optional
        Number of sub-pixels per shape-unit along each axis. Default is 10.
    position : tuple of float, optional
        ``(x, y)`` coordinates of the PSF with respect to the center of the
        image, in shape-units. Default is (0, 0).
    normal_scatter : float or None, optional
        Additional Gaussian scattering (standard deviation) in shape-units,
        applied by convolution. Default is None.
    norm_pixels : bool, optional
        If True, multiply the PSF by the pixel area, such that pixel values
        are the fraction of the flux in the pixel. Otherwise, pixel values
        are a surface density (per shape-unit squared). Default is False.
    **kwargs
        Additional arguments passed to the profile:

        - ``radius``: mandatory for ``profile="airy"`` (shape-units, float or
          array_like for several PSFs);
        - ``sigma``: mandatory for ``profile="gaussian"`` (shape-units).

    Returns
    -------
    psf : numpy.ndarray
        PSF image(s), of shape ``(ny * oversampling, nx * oversampling)``,
        or ``(n, ny * oversampling, nx * oversampling)`` for ``n`` radii or
        sigmas.
    centroid : numpy.ndarray
        Position of the PSF centroid, in (oversampled) pixels of ``psf``,
        computed as ``psf.shape[-2:] / 2 - 0.5 + position * oversampling``.
    pixelarea : float
        Area of a single oversampled pixel, in shape-units squared.

    Raises
    ------
    KeyError
        If the mandatory ``radius`` or ``sigma`` is not given.
    NotImplementedError
        If ``profile`` is not supported.
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
        scatter_pixels = normal_scatter * oversampling
        psf = gaussian_filter(psf, scatter_pixels, axes=(-2,-1))

    # centroid in psf image coordinates
    img_center = np.asarray( psf.shape[-2:] )/2 - 0.5
    centroid = img_center + np.asarray(position) * oversampling

    return np.squeeze(psf), centroid, pxl["pixelarea"]

def get_profilemodel(name, position=(0,0), normalized=True, **kwargs):
    """Get a 2D profile model from `astropy.modeling.functional_models`.

    Parameters
    ----------
    name : str
        Name of the 2D astropy model. Shortcuts:

        - ``"airy"``, ``"airydisk"``: ``"AiryDisk2D"``
        - ``"gaussian"``, ``"gauss"``, ``"normal"``: ``"Gaussian2D"``
        - ``"moffat"``: ``"Moffat2D"``
        - ``"sersic"``: ``"Sersic2D"``
        - ``"ricker"``, ``"mexican"``, ``"mexicanhat"``: ``"RickerWavelet2D"``
        - ``"lorentz"``: ``"Lorentz2D"``
    position : tuple of float, optional
        Location ``(x, y)`` of the centroid. Default is (0, 0).
    normalized : bool, optional
        If True, set the model amplitude such that the profile integrates to
        1. Only implemented for ``AiryDisk2D`` and ``Gaussian2D``.
        Default is True.
    **kwargs
        Additional arguments passed to the astropy model initializer, see
        https://docs.astropy.org/en/latest/modeling/predef_models2D.html.
        Model defaults are amplitude=1 and:

        - AiryDisk2D: radius=1
        - Gaussian2D: x_stddev=None, y_stddev=None, theta=None,
          cov_matrix=None. For convenience, ``sigma`` (or ``radius``) can be
          given to set both ``x_stddev`` and ``y_stddev``.
        - Moffat2D: gamma=1, alpha=1
        - Sersic2D: r_eff=1, n=4, ellip=0, theta=0.0
        - RickerWavelet2D: sigma=1
        - Lorentz2D: fwhm=1

    Returns
    -------
    astropy.modeling.Fittable2DModel
        The requested astropy model with its parameters set; call it as
        ``model(x, y)``.

    Raises
    ------
    ValueError
        If ``normalized`` is True for a model other than ``AiryDisk2D`` or
        ``Gaussian2D``.

    Examples
    --------
    >>> sigmas = 2  # or array of shape (M, 1, 1)
    >>> airy = get_profilemodel("airy", position=(0.4, 1.2), radius=sigmas)
    >>> xx = np.r_[-7:7:50j]
    >>> yy = np.r_[-7:7:49j]  # different dim for example
    >>> airy(xx[None, :], yy[:, None]).shape  # (M, 49, 50) for M radii
    (49, 50)
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
    """Evaluate a normalized 1D Gaussian at given points.

    Parameters
    ----------
    xx : numpy.ndarray
        Points at which to evaluate the Gaussian, of shape ``(nx,)``.
    sigma : float or numpy.ndarray
        Standard deviation(s) of the Gaussian: a scalar, or an array of
        shape ``(n,)`` for multiple Gaussians.
    mean : float, optional
        Mean of the Gaussian. Default is 0.

    Returns
    -------
    numpy.ndarray
        Probability density at ``xx``, of shape ``(n, nx)`` (``n = 1`` for a
        scalar ``sigma``).
    """
    sigma = np.atleast_1d(sigma)
    mean = np.atleast_1d(mean)
    return stats.norm.pdf(xx, loc=mean, scale=sigma[:,None])

def get_gaussian2d(xx, yy, sigma, mean=(0,0), **kwargs):
    """Compute an exact pixel-integrated 2D Gaussian PSF.

    The Gaussian profile is integrated over the area of each pixel using the
    error function (see `~slicersim.utils.integ_gaussian2D_erf`).

    Parameters
    ----------
    xx : numpy.ndarray
        Regularly spaced x-coordinates of the pixel centers, of shape
        ``(nx,)`` or ``(1, nx)``.
    yy : numpy.ndarray
        Regularly spaced y-coordinates of the pixel centers, of shape
        ``(ny,)`` or ``(ny, 1)``.
    sigma : array_like
        Standard deviations of the Gaussian, in the same units as ``xx``, of
        shape ``(..., 2)`` with the last axis being ``(sigma_y, sigma_x)``,
        e.g. ``(m, 1, 2)`` for ``m`` PSFs.
    mean : tuple of float, optional
        ``(x, y)`` center of the Gaussian. Default is (0, 0).
    **kwargs
        Additional arguments passed to
        `~slicersim.utils.integ_gaussian2D_erf`.

    Returns
    -------
    numpy.ndarray
        Pixel values, normalized such that the sum over an infinite grid
        equals 1, of shape ``(ny, nx)`` or ``(m, ny, nx)``.
    """
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
