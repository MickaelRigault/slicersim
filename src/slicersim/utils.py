"""
Miscellaneous functions.
"""

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>", "Mickael Rigault <m.rigault@ipnl.in2p3.fr>",

import numpy as np


def recursive_get(dict_, key, default=None):
    """ Recursively get a key from a nested dictionary.

    Parameters
    ----------
    dict_ : dict
        The dictionary to search.
    key : str
        The key to search for.
    default : any, optional
        The value to return if the key is not found. Defaults to None.

    Returns
    -------
    any
        The value of the key if found, otherwise the default value.
    """
    if key in dict_.keys():
        return dict_.get(key)
        
    for key_, items in dict_.items():
        if type(items) is dict:
            out = recursive_get(items, key)
            if out is not None:
                return out
                
    return default

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
    """ Inspect the parameters of a function.

    This function returns the names of all parameters (args and kwargs) and a
    dictionary of the keyword arguments with their default values.

    Parameters
    ----------
    func : function
        The function to inspect.

    Returns
    -------
    list
        A list of the names of all parameters.
    dict
        A dictionary of the keyword arguments and their default values.
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
    np.True_
    >>> np.isclose(
    ...     integ_gaussian1D_erf(x_edges, sigma=sigma, normed=False).sum(),
    ...     sigma * (2*np.pi)**0.5)
    np.True_

    """

    from scipy.special import erf

    tmp = erf((x_edges - mu) / (1.4142135623730951 * sigma))  # sig.shape + (nx+1,)
    # Normalized Gaussian, sig.shape + (nx,)
    f = (tmp[..., 1:] - tmp[..., :-1]) / 2
    if not normed:
        f *= (2 * np.pi)**0.5 * sigma

    return f

def integ_gaussian2D_erf(xy_edges, sigma, mu=(0, 0), normed=True):
    """ Return (axisymmetric) 2D-Gaussian integrated over pixels (exact).

    :param xy_edges: x- and y-coordinates of pixel edges
                     (1, nx+1) and (ny+1, 1)
    :param sigma: Gaussian standard deviation (scalar or shape (nsig, 1, 1))
    :param mu: center of the 2D Gaussian
    :param normed: use a total flux-normalized 2D Gaussian
    :return: sigma.shape + (ny, nx) array

    >>> y, x = np.ogrid[-5:5.1, -6:6.1]
    >>> sigma = np.array([1])
    >>> np.isclose(
    ...     integ_gaussian2D_erf((x, y), sigma=sigma, normed=True).sum(),
    ...     1)
    np.True_
    >>> np.isclose(
    ...     integ_gaussian2D_erf((x, y), sigma=sigma, normed=False).sum(),
    ...     sigma**2 * 2*np.pi)
    array([ True])

    """

    from scipy.special import erf

    x_edges, y_edges = xy_edges  # (1, nx + 1) and (ny + 1, 1)
    if x_edges.shape[0] > 1 or y_edges.shape[1] > 1:
        raise NotImplementedError("Only 'open' mesh-grid is supported.")

    mux, muy = mu

    # allows for asymetric sigma
    sqrt2sig = 1.4142135623730951 * sigma
    if sqrt2sig.shape[-1] == 1:
        sqrt2sig_y = sqrt2sig_x = sqrt2sig
        
    elif sqrt2sig.shape[-1] == 2:
        sqrt2sig_y = sqrt2sig[..., 0][...,None] # no dim reduction
        sqrt2sig_x = sqrt2sig[..., 1][...,None] # no dim reduction
        
    tmpx = erf((x_edges - mux) / sqrt2sig_x)  # sig.shape + (1, nx+1)
    tmpy = erf((y_edges - muy) / sqrt2sig_y)  # sig.shape + (ny+1, 1)
    
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

