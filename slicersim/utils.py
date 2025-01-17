"""
Miscellaneous functions.
"""

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>", "Mickael Rigault <m.rigault@ipnl.in2p3.fr>",

import numpy as np


def mesh_kwargs(**kwargs):
    """
    Create a dataframe with all parameter combinations.

    >>> mesh_kwargs(scale=[0.02, 0.05], sigma=[0.03, 0.06])
       scale  sigma
    0   0.02   0.03
    1   0.05   0.03
    2   0.02   0.06
    3   0.05   0.06
    """
    import pandas
    scan_params = list(kwargs.keys())     # (npar,)
    # Number of scanned parameters (i.e. columns in final df)
    # npar = len(scan_params)

    mesh = np.meshgrid(*kwargs.values())  # npar × (nval1, nval2, ...)
    mshape = np.shape(mesh)               # (npar, nval1, nval2, ...)
    shape = (mshape[0], np.prod(mshape[1:]))  # (npar, nval1*nval2*...)
    df_params = pandas.DataFrame(
        np.reshape(mesh, shape).T,        # (nval1*nval2*..., npar)
        columns=scan_params)              # (npar,)

    return df_params

def inspect_func( func ):
    """ inspect the given function parameters
    
    Parameters
    ----------
    func: function
        function to inspect

    Returns
    -------
    list, dict
       - names of the parameters (args and kwargs)
       - dict of the kwargs only.
    """
    import inspect
    inspect_full = inspect.getfullargspec( func )
    all_params = inspect_full.args
    # kwargs_ are the default function values
    if inspect_full.defaults is not None:
        kwargs_ = dict(zip(all_params[::-1], inspect_full.defaults[::-1]))
    else:
        kwargs_ = {}
    
    return all_params, kwargs_


def restride(arr, binfactor, squeezed=True, flattened=False):
    """ Rebin ND-array `arr` by `binfactor`.

    Let `arr.shape = (s1, s2, ...)` and `binfactor = (b1, b2, ...)` (same
    length), new shape will be `(s1/b1, s2/b2, ... b1, b2, ...)` (squeezed).

    If `binfactor` is an iterable of length < `arr.ndim`, it is prepended
    with 1's.

    If `binfactor` is an integer, it is considered as the bin factor for all
    axes.

    If `flattened`, the bin axes are explicitely flattened into a single
    axis. Note that this will probably induce a copy of the array.

    Bin 2D-array by a factor 2:

    >>> restride(np.ones((6, 8)), 2).shape
    (3, 4, 2, 2)

    Bin 2D-array by a factor 2, with flattening of the last 2 bin axes:

    >>> restride(np.ones((6, 8)), 2, flattened=True).shape
    (3, 4, 4)

    Bin 2D-array by uneven factor (3, 2):

    >>> restride(np.ones((6, 8)), (3, 2)).shape
    (2, 4, 3, 2)

    Bin 3D-array by factor 2 over the last 2 axes, and take bin average:

    >>> q = np.arange(2*4*6).reshape(2, 4, 6)
    >>> restride(q, (2, 2)).mean(axis=(-1, -2))
    array([[[ 3.5,  5.5,  7.5],
            [15.5, 17.5, 19.5]],
           [[27.5, 29.5, 31.5],
            [39.5, 41.5, 43.5]]])

    Bin 3D-array by factor 2, and take bin average:

    >>> restride(q, 2).mean(axis=(-1, -2, -3))
    array([[15.5, 17.5, 19.5],
           [27.5, 29.5, 31.5]])
    """

    try:                        # binfactor is list-like
        # Convert binfactor to [1, ...] + binfactor
        binshape = [1] * (arr.ndim - len(binfactor)) + list(binfactor)
    except TypeError:           # binfactor is not list-like
        binshape = [binfactor] * arr.ndim

    assert len(binshape) == arr.ndim, "Invalid bin factor (shape)."
    assert (~np.mod(arr.shape, binshape).astype('bool')).all(), \
        "Invalid bin factor (modulo)."

    # New shape
    rshape = [ d // b for d, b in zip(arr.shape, binshape) ] + binshape
    # New stride
    rstride = [ d * b for d, b in zip(arr.strides, binshape) ] + list(arr.strides)

    rarr = np.lib.stride_tricks.as_strided(arr, rshape, rstride)

    if flattened:               # Flatten bin axes, which may induce a costful copy!
        rarr = rarr.reshape(rarr.shape[:-(rarr.ndim - arr.ndim)] + (-1,))

    return rarr.squeeze() if squeezed else rarr  # Remove length-1 axes


def bin_array(arr, binfactor, function):
    """
    Bin (through `function`) ND-array `arr` by `binfactor`.

    Let `arr.shape = (s1, s2, ...)` and `binfactor = (b1, b2, ...)` (same
    length), new shape will be `(s1/b1, s2/b2, ...)`.

    If `binfactor` is an iterable of length < `arr.ndim`, it is prepended
    with 1's.

    If `binfactor` is an integer, it is considered as the bin factor for all
    axes.

    >>> bin_array(np.arange(4*6).reshape(4, 6), 2, np.mean).shape
    (2, 3)
    >>> bin_array(np.arange(4*6).reshape(4, 6), (2,), np.mean).shape
    (4, 3)
    >>> bin_array(np.arange(4*6).reshape(4, 6), (2, 1), np.mean).shape
    (2, 6)

    .. Note:: this is the opposite of :func:`unbin_array()`.
    """

    try:                        # binfactor is list-like
        # Convert binfactor to [1, ...] + binfactor
        binshape = [1] * (arr.ndim - len(binfactor)) + list(binfactor)
    except TypeError:           # binfactor is not list-like
        binshape = [binfactor] * arr.ndim

    return function(
        restride(arr, binshape, squeezed=False),  # (s1/b1, s2/b2, ..., b1, b2, ...)
        axis=tuple(range(-arr.ndim, 0))           # (..., -2, -1)
    )                                             # (s1 / b1, s2 / b2, ...)


def unbin_array(arr, binfactor):
    """
    Unbin (by repetition) ND-array `arr` by `binfactor`.

    Let `arr.shape = (s1, s2, ...)` and `binfactor = (b1, b2, ...)` (same
    length), new shape will be `(s1*b1, s2*b2, ...)`.

    If `binfactor` is an iterable of length < `arr.ndim`, it is prepended
    with 1's.

    If `binfactor` is an integer, it is considered as the bin factor for all
    axes.

    >>> unbin_array(np.arange(2*3).reshape(2, 3), 2).shape
    (4, 6)
    >>> unbin_array(np.arange(2*3).reshape(2, 3), (2,)).shape
    (2, 6)
    >>> unbin_array(np.arange(2*3).reshape(2, 3), (2, 1)).shape
    (4, 3)

    .. Note:: this is the opposite of :func:`bin_array()`.
    """

    try:                        # binfactor is list-like
        # Convert binfactor to [1, ...] + binfactor
        binshape = [1] * (arr.ndim - len(binfactor)) + list(binfactor)
    except TypeError:           # binfactor is not list-like
        binshape = [binfactor] * arr.ndim

    return (
        np.broadcast_to(
            # Add a new dimension for each axis (i.e. at axis 1, 3, 5, ...)
            np.expand_dims(arr, list(range(1, 2 * arr.ndim + 1, 2))),  # (s1, 1, s2, 1, ...)
            [ j for i in zip(arr.shape, binshape) for j in i ])       # (s1, b1, s2, b2, ...)
        .reshape([ i * j for i, j in zip(arr.shape, binshape) ]))     # (s1 * b1, s2 * b2, ...)


def romberg(z, steps=1):
    """
    Romberg ND-integration using samples of a ND function.

    Axis -1 is x, -2 is y, etc. See
    :func:`scipy.integrate.romb` for details.

    >>> nz, ny, nx = 2**2 + 1, 2**4 + 1, 2**3 + 1
    >>> zlims, ylims, xlims = (0, 1/2), (0, 2), (0, 1)
    >>> z, y, x = np.ogrid[zlims[0]:zlims[1]:nz*1j,
                           ylims[0]:ylims[1]:ny*1j,
                           xlims[0]:xlims[1]:nx*1j]
    >>> dz, dy, dx = (z[1, 0, 0] - z[0, 0, 0],
                      y[0, 1, 0] - y[0, 0, 0],
                      x[0, 0, 1] - x[0, 0, 0])
    >>> integrand = (2 * x + y + z / 2)**2
    # int_{x=0}^{1} int_{y=0}^{2} int_{z=0}^{1/2} = 83/16
    >>> np.isclose(romberg(integrand, (dz, dy, dx)), 83/16)
    True
    """

    from scipy.integrate import romb

    steps = np.resize(steps, (z.ndim,))  # Make it a 1D vector

    if z.ndim == 1:
        return romb(z, steps[0])

    for axis in range(z.ndim):
        step = steps[-(axis+1)]  # Start on axis=-1, i.e. the last step
        if axis == 0:
            integral = [romb(zz, step) for zz in z]
        else:
            integral = romb(integral, step)

    return integral


def integ_gaussian1D_midpoint(x_edges, sigma, mu=0, normed=True):
    """
    Return 1D-Gaussian integrated over pixels (midpoint approximation).

    :param x_edges: x-coordinates of pixel edges (nx+1,)
    :param sigma: Gaussian standard deviation (scalar or shape (nsig, 1))
    :param mu: center of the 1D Gaussian
    :param normed: use a total flux-normalized 1D Gaussian
    :return: sigma.shape + (nx,) array

    >>> x_edges = np.linspace(-10, 10, 41); sigma = 2
    >>> np.isclose(
    ...     integ_gaussian1D_midpoint(x_edges, sigma=sigma, normed=True).sum(),
    ...     1)
    True
    >>> np.isclose(
    ...     integ_gaussian1D_midpoint(x_edges, sigma=sigma, normed=False).sum(),
    ...     sigma * (2*np.pi)**0.5)
    True
    """

    x = (x_edges[1:] + x_edges[:-1]) / 2  # Px center coordinate (nx,)
    dx = np.diff(x_edges)                 # Px width (nx,)

    # Unnormalized Gaussian, sig.shape + (nx,)
    f = np.exp(-0.5 * ((x - mu) / sigma)**2) * dx
    if normed:
        f /= (2 * np.pi)**0.5 * sigma

    return f  # sigma.shape + (nx,)


def integ_gaussian1D_erf(x_edges, sigma, mu=0, normed=True):
    """
    Return 1D-Gaussian integrated over pixels (exact).

    :param x_edges: x-coordinates of pixel edges (nx+1,)
    :param sigma: Gaussian standard deviation (scalar or shape (nsig, 1))
    :param mu: center of the 1D Gaussian
    :param normed: use a total flux-normalized 1D Gaussian
    :return: sigma.shape + (nx,) array

    >>> x_edges = np.linspace(-10, 10, 41); sigma = 2
    >>> np.isclose(
    ...     integ_gaussian1D_erf(x_edges, sigma=sigma, normed=True).sum(),
    ...     1)
    True
    >>> np.isclose(
    ...     integ_gaussian1D_erf(x_edges, sigma=sigma, normed=False).sum(),
    ...     sigma * (2*np.pi)**0.5)
    True
    """

    from scipy.special import erf

    tmp = erf((x_edges - mu) / (1.4142135623730951 * sigma))  # sig.shape + (nx+1,)
    # Normalized Gaussian, sig.shape + (nx,)
    f = (tmp[..., 1:] - tmp[..., :-1]) / 2
    if not normed:
        f *= (2 * np.pi)**0.5 * sigma

    return f


def integ_gaussian2D_midpoint(xy_edges, sigma, mu=(0, 0), normed=True):
    """
    Return 2D-Gaussian integrated over pixels (midpoint approximation).

    :param xy_edges: x- and y-coordinates of pixel edges
                     (1, nx+1) and (ny+1, 1)
    :param sigma: Gaussian standard deviation (scalar or shape (nsig, 1, 1))
    :param mu: center of the 2D Gaussian
    :param normed: use a total flux-normalized 2D Gaussian
    :return: sigma.shape + (ny, nx) array

    >>> y, x = np.ogrid[-11:11:45j, -10:10:41j]; sigma = 2
    >>> np.isclose(
    ...     integ_gaussian2D_midpoint((x, y), sigma=sigma, normed=True).sum(),
    ...     1)
    True
    >>> np.isclose(
    ...     integ_gaussian2D_midpoint((x, y), sigma=sigma, normed=False).sum(),
    ...     sigma**2 * 2*np.pi)
    True
    """

    x_edges, y_edges = xy_edges
    mux, muy = mu

    x = (x_edges[0:1, 1:] + x_edges[0:1, :-1]) / 2  # Px center coord. (1, nx)
    y = (y_edges[1:, 0:1] + y_edges[:-1, 0:1]) / 2  # (ny, 1)
    dx = np.diff(x_edges[0:1, :], axis=1)           # Px widths (1, nx)
    dy = np.diff(y_edges[:, 0:1], axis=0)           # (ny, 1)

    # Unnormalized Gaussian, sig.shape + (ny, nx)
    f = np.exp(-0.5 * ((x - mux)**2 + (y - muy)**2) / sigma**2) * dx * dy
    if normed:
        f /= 2 * np.pi * sigma**2

    return f  # sigma.shape + (ny, nx)


def integ_gaussian2D_erf(xy_edges, sigma, mu=(0, 0), normed=True):
    """
    Return (axisymmetric) 2D-Gaussian integrated over pixels (exact).

    :param xy_edges: x- and y-coordinates of pixel edges
                     (1, nx+1) and (ny+1, 1)
    :param sigma: Gaussian standard deviation (scalar or shape (nsig, 1, 1))
    :param mu: center of the 2D Gaussian
    :param normed: use a total flux-normalized 2D Gaussian
    :return: sigma.shape + (ny, nx) array

    >>> y, x = np.ogrid[-5:5.1, -6:6.1]; sigma = 1
    >>> np.isclose(
    ...     integ_gaussian2D_erf((x, y), sigma=sigma, normed=True).sum(),
    ...     1)
    True
    >>> np.isclose(
    ...     integ_gaussian2D_erf((x, y), sigma=sigma, normed=False).sum(),
    ...     sigma**2 * 2*np.pi)
    True
    """

    from scipy.special import erf

    x_edges, y_edges = xy_edges  # (1, nx + 1) and (ny + 1, 1)
    if x_edges.shape[0] > 1 or y_edges.shape[1] > 1:
        raise NotImplementedError("Only 'open' mesh-grid is supported.")

    mux, muy = mu

    sqrt2sig = 1.4142135623730951 * sigma
    tmpx = erf((x_edges - mux) / sqrt2sig)  # sig.shape + (1, nx+1)
    tmpy = erf((y_edges - muy) / sqrt2sig)  # sig.shape + (ny+1, 1)
    # Normalized Gaussian, sig.shape + (ny, nx)
    f = np.diff(tmpx, axis=-1) * np.diff(tmpy, axis=-2) / 4
    if not normed:
        f *= 2 * np.pi * sigma**2

    return f  # sigma.shape + (ny, nx)


def complete_dims(arr, xdims, squeeze=True):
    """
    Complete dimensions of an array.

    * If `xdims > 0`, add `xdims` dimensions, final shape is
      `arr.shape + (1,) * xdims`;
    * If `xdims < 0`, add as many dimensions as necessary to reach
      `|xdims|` dimensions at least, final shape is
      `arr.shape + (1,) * (|xdims| - arr.ndim)`.

    :param np.array arr: input array
    :param int xdims: extra dimensions to be appended
    :param bool squeeze: remove *trivial* dimensions
    :return: expanded array, of shape `arr.shape + (1,) * xdims`

    >>> complete_dims(1, 2, False).shape  # () + (1,)*2
    (1, 1)
    >>> complete_dims(1, 2).shape
    ()
    >>> complete_dims([[0, 1, 2]], 4, False).shape  # (1, 3) + (1,)*4
    (1, 3, 1, 1, 1, 1)
    >>> complete_dims([[0, 1, 2]], 4).shape
    (3, 1, 1, 1, 1)
    >>> complete_dims([[0, 1, 2]], -4, False).shape  # (1, 3) + (1,)*(4 - 2)
    (1, 3, 1, 1)
    >>> complete_dims([[0, 1, 2]], -4).shape
    (3, 1, 1)
    """

    from itertools import takewhile

    # faster than np.expand_dims(arr, axis=tuple(range(-xdims, 0))
    if xdims >= 0:
        ns = np.shape(arr) + (1,) * xdims
    else:
        ns = np.shape(arr) + (1,) * (-xdims - np.ndim(arr))

    nsarr = np.reshape(arr, ns)
    if squeeze:   # remove all trivial dimensions, up to first non-trivial one
        naxes = len(tuple(takewhile(lambda a: a == 1, np.shape(nsarr))))
        nsarr = np.squeeze(nsarr, axis=tuple(range(naxes)))

    return nsarr


def embed_array(arr, shape, mode='constant', **kwargs):
    """
    Embed and center array *arr* in a larger *shape* array.

    :param arr: input array to be embedded
    :param tuple shape: target shape
    :param str mode: padding mode
    :param kwargs: padding parameters
    :return: embedded array

    >>> arr = np.arange(2 * 3).reshape(2, 3)
    >>> embed_array(arr, (2, 5))
    array([[0, 0, 1, 2, 0],
           [0, 3, 4, 5, 0]])

    .. Note:: this is a wrapper to :func:`numpy.pad`.
    """

    inshape = np.shape(arr)
    if len(inshape) != len(shape) or not (inshape <= shape):
        raise ValueError(
            f"Incompatible shapes: {inshape=} cannot be embeded in {shape=}.")

    pads = [(n := (n_out - n_in) // 2, (n_out - n_in - n))
            for n_in, n_out in zip(inshape, shape)]

    return np.pad(arr, pads, mode=mode, **kwargs)


def crop_array(arr, shape):
    """
    Crop central part of array *arr* in a smaller *shape* array.

    :param arr: input array to be cropped
    :param tuple shape: target shape
    :return: cropped array

    >>> arr = np.arange(2 * 3).reshape(2, 3)  # (2, 3)
    >>> crop_array(embed_array(arr, (2, 5)), (-1, 3))
    array([[0, 1, 2],
           [3, 4, 5]])

    .. Note:: this is the opposite of :func:`embed_array()`.
    """

    inshape = np.shape(arr)
    if len(inshape) != len(shape) or not (inshape >= shape):
        raise ValueError(
            f"Incompatible shapes: {inshape=} cannot be cropped into {shape=}.")

    pads = [(n := (n_in - n_out) // 2, n + n_out) if n_out > 0 else (0, n_in)
            for n_in, n_out in zip(inshape, shape)]

    return arr[np.ix_(*[np.arange(b, a) for b, a in pads])]


def recover_bin_edges(wave, order=2):
    """
    Recover the edges of a set of wavelength bins given the bin centers.

    See `sncosmo.Spectrum._recover_bin_edges` for details.
    """

    wave = np.asarray(wave)

    # First order estimate
    o1 = (wave[:-1] + wave[1:]) / 2.

    if order == 1:
        # Estimate front and back edges
        f1 = 2 * wave[0] - o1[0]
        b1 = 2 * wave[-1] - o1[-1]
        bin_edges = np.hstack([f1, o1, b1])
    elif order == 2:
        # Second order correction
        o2 = 1.5*o1[1:-1] - (o1[2:] + o1[:-2]) / 4.
        # Estimate front and back edges
        f2 = 2 * wave[1] - o2[0]
        f1 = 2 * wave[0] - f2
        b2 = 2 * wave[-2] - o2[-1]
        b1 = 2 * wave[-1] - b2
        # Stack everything together
        bin_edges = np.hstack([f1, f2, o2, b2, b1])

    return bin_edges


def cart2pol(x, y):
    """
    Convert cartesian coordinates into polar coordinates.
    """

    return np.hypot(x, y), np.arctan2(y, x)


def pol2cart(rho, phi):
    """
    Convert polar coordinates into cartesian coordinates.
    """

    return rho * np.cos(phi), rho * np.sin(phi)


def ogrid_domain(domains, npts):
    """
    Open ND-coord. grid from N domains and nb of points (wrapper to np.ogrid).

    >>> z, y, x = ogrid_domain(((0, 1), (1, 2), (2, 3)), (2, 3, 5))
    >>> zn, yn, xn = np.ogrid[0:1:2j, 1:2:3j, 2:3:5j]
    >>> np.allclose(x, xn) and np.allclose(y, yn) and np.allclose(z, zn)
    """

    npts = np.resize(npts, len(domains))  # Make it a 1D vector if needed

    return np.ogrid[[slice(vmin, vmax, npt * 1j)
                     for (vmin, vmax), npt in zip(domains, npts)]]


def ogrid_steps(ogrid):
    """
    Compute ..., y, x steps from open coordinate grid.

    >>> ogrid_steps(np.ogrid[0:1:2j, 1:2:3j, 3:4:5j])
    [1.0, 0.5, 0.25]
    """

    steps = []
    for arr in ogrid:
        sq = np.squeeze(arr)
        steps.append((sq[-1] - sq[0]) / (len(sq) - 1))

    return steps


def ogrid_mid(ogrid):
    """
    Compute ..., y, x mid-coordinates from open coordinate grid.

    >>> ogrid_mid(np.ogrid[0:1:2j, 1:2:3j, 3:4:5j])
    [array([[[0.5]]]),
     array([[[1.25],
             [1.75]]]),
     array([[[3.125, 3.375, 3.625, 3.875]]])]
    """

    mids = []
    for i, arr in enumerate(ogrid):
        sq = np.squeeze(arr)
        mid = (sq[1:] + sq[:-1]) / 2
        shape = list(arr.shape)
        shape[i] -= 1
        mids.append(mid.reshape(shape))

    return mids


def ogrid_coords(ogrid):
    """
    Convert open ND-coordinate grid to list of ND-coordinates.

    >>> ogrid_coords(np.ogrid[0:1:2j, 1:2:3j])
    array([[0. , 1. ],
           [0. , 1.5],
           [0. , 2. ],
           [1. , 1. ],
           [1. , 1.5],
           [1. , 2. ]])
    """

    return (np.moveaxis(np.broadcast_arrays(*ogrid), 0, -1)
            .reshape(-1, len(ogrid)))


def ogrid_extent(ogrid, reverse=True, full=False):
    """
    Get extent [xmin, xmax, ymin, ymax, ...] (reversed).

    >>> ogrid_extent(np.ogrid[0:1:2j, 1:2:3j], reverse=False)
    [0.0, 1.0, 1.0, 2.0]
    >>> ogrid_extent(np.ogrid[0:1:2j, 1:2:3j], reverse=True)
    [1.0, 2.0, 0.0, 1.0]
    >>> ogrid_extent(np.ogrid[0:1:2j, 1:2:3j], reverse=False, full=True)
    [-0.5, 1.5, 0.75, 2.25]
    >>> ogrid_extent(np.ogrid[0:1:2j, 1:2:3j], reverse=True, full=True)
    [0.75, 2.25, -0.5, 1.5]
    """

    ids = [-1, 0] if reverse else [0, -1]

    extent = []
    for arr in ogrid:
        extent.extend(np.squeeze(arr)[ids])

    if full:                    # Add ± 1/2 step on each side
        signs = [+1, -1] if reverse else [-1, +1]
        offsets = [sign * step / 2
                   for step in ogrid_steps(ogrid)
                   for sign in signs]  # [-dx/2, +dx/2, -dy/2...]
        extent = [lim + offset for lim, offset in zip(extent, offsets)]

    return extent[::-1] if reverse else extent


def dict_product(adic):
    """
    Same as :func:`itertools.product` but for a dictionary.

    >>> list(dict_product(dict(a=[1,2], b=[3])))
    [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
    """

    from itertools import product

    keys = adic.keys()
    values = product(*adic.values())

    return (dict(zip(keys, value)) for value in values)
