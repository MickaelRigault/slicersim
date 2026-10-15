"""
Miscellaneous functions.
"""

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>", "Mickael Rigault <m.rigault@ipnl.in2p3.fr>",

import numpy as np


def recursive_get(dict_, key, default=None):
    """Recursively get a key from a nested dictionary.

    The key is first looked for at the top level of ``dict_``, then in its
    nested dictionaries (depth first, in insertion order). The first non-None
    value found is returned.

    Parameters
    ----------
    dict_ : dict
        The dictionary to search.
    key : str
        The key to search for.
    default : object, optional
        The value to return if the key is not found. Default is None.

    Returns
    -------
    object
        The value of the key if found, otherwise ``default``.

    Examples
    --------
    >>> recursive_get({"a": 1, "b": {"c": 2}}, "c")
    2
    """
    if key in dict_:
        return dict_.get(key)

    for items in dict_.values():
        if type(items) is dict:
            out = recursive_get(items, key)
            if out is not None:
                return out

    return default

def mesh_kwargs(**kwargs):
    """Create a dataframe with all parameter combinations.

    Parameters
    ----------
    **kwargs
        Parameter names and the list of values each of them takes.

    Returns
    -------
    pandas.DataFrame
        One row per combination of the input values, one column per
        parameter.

    Notes
    -----
    The combinations follow the `numpy.meshgrid` default (``indexing="xy"``)
    ordering: for three parameters ``a, b, c``, ``b`` varies the slowest, then
    ``a``, and ``c`` the fastest.

    Examples
    --------
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
    """Inspect the parameters of a function.

    This function returns the names of all positional-or-keyword parameters
    and a dictionary of those having a default value.

    Parameters
    ----------
    func : callable
        The function to inspect.

    Returns
    -------
    all_params : list of str
        Names of all the parameters.
    defaults : dict
        Parameters that have a default value, mapped to that value.
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
    """Restride an N-D array into bins, without copying.

    Let ``arr.shape = (s1, s2, ...)`` and ``binfactor = (b1, b2, ...)`` (same
    length), the new shape will be ``(s1/b1, s2/b2, ..., b1, b2, ...)``
    (squeezed).

    Parameters
    ----------
    arr : numpy.ndarray
        Input array. Each ``s_i`` must be a multiple of the corresponding
        ``b_i``.
    binfactor : int or tuple of int
        Bin factor. If an integer, it is used for all axes. If its length is
        lower than ``arr.ndim``, it is prepended with 1's.
    squeezed : bool, optional
        Remove the length-1 axes from the output. Default is True.
    flattened : bool, optional
        Flatten the bin axes into a single axis. Note that this will probably
        induce a copy of the array. Default is False.

    Returns
    -------
    numpy.ndarray
        Restrided view of ``arr``, of shape
        ``(s1/b1, s2/b2, ..., b1, b2, ...)``.

    Raises
    ------
    AssertionError
        If ``binfactor`` is incompatible with the shape of ``arr``.

    Examples
    --------
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
    <BLANKLINE>
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
    """Bin an N-D array by a given factor, using a reduction function.

    Let ``arr.shape = (s1, s2, ...)`` and ``binfactor = (b1, b2, ...)`` (same
    length), the new shape will be ``(s1/b1, s2/b2, ...)``.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array. Each ``s_i`` must be a multiple of the corresponding
        ``b_i``.
    binfactor : int or tuple of int
        Bin factor. If an integer, it is used for all axes. If its length is
        lower than ``arr.ndim``, it is prepended with 1's.
    function : callable
        Reduction function accepting an ``axis`` argument, e.g. `numpy.mean`
        or `numpy.sum`.

    Returns
    -------
    numpy.ndarray
        Binned array of shape ``(s1/b1, s2/b2, ...)``.

    See Also
    --------
    unbin_array : The opposite operation.
    restride : The underlying binning view.

    Examples
    --------
    >>> bin_array(np.arange(4*6).reshape(4, 6), 2, np.mean).shape
    (2, 3)
    >>> bin_array(np.arange(4*6).reshape(4, 6), (2,), np.mean).shape
    (4, 3)
    >>> bin_array(np.arange(4*6).reshape(4, 6), (2, 1), np.mean).shape
    (2, 6)
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
    """Unbin (by repetition) an N-D array by a given factor.

    Let ``arr.shape = (s1, s2, ...)`` and ``binfactor = (b1, b2, ...)`` (same
    length), the new shape will be ``(s1*b1, s2*b2, ...)``.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    binfactor : int or tuple of int
        Unbin factor. If an integer, it is used for all axes. If its length
        is lower than ``arr.ndim``, it is prepended with 1's.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(s1*b1, s2*b2, ...)``, each input element being
        repeated ``b1 x b2 x ...`` times.

    See Also
    --------
    bin_array : The opposite operation.

    Examples
    --------
    >>> unbin_array(np.arange(2*3).reshape(2, 3), 2).shape
    (4, 6)
    >>> unbin_array(np.arange(2*3).reshape(2, 3), (2,)).shape
    (2, 6)
    >>> unbin_array(np.arange(2*3).reshape(2, 3), (2, 1)).shape
    (4, 3)
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
    """Return a 1D Gaussian integrated over pixels (exact).

    The integral over each pixel is computed analytically with the error
    function.

    Parameters
    ----------
    x_edges : numpy.ndarray
        Coordinates of the pixel edges, of shape ``(nx+1,)``.
    sigma : float or numpy.ndarray
        Gaussian standard deviation, scalar or of shape ``(nsig, 1)``, in the
        same units as ``x_edges``.
    mu : float, optional
        Center of the 1D Gaussian. Default is 0.
    normed : bool, optional
        If True, use a unit-integral Gaussian; otherwise use a Gaussian of
        unit peak amplitude. Default is True.

    Returns
    -------
    numpy.ndarray
        Integrated Gaussian of shape ``sigma.shape[:-1] + (nx,)`` (``(nx,)``
        for a scalar ``sigma``).

    Examples
    --------
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
    """Return a 2D Gaussian integrated over pixels (exact).

    The Gaussian is aligned with the x and y axes (no rotation), and the
    integral over each pixel is computed analytically with the error
    function.

    Parameters
    ----------
    xy_edges : tuple of numpy.ndarray
        ``(x_edges, y_edges)`` coordinates of the pixel edges, as an "open"
        mesh-grid of shapes ``(1, nx+1)`` and ``(ny+1, 1)`` (see
        `numpy.ogrid`).
    sigma : array_like
        Gaussian standard deviations, in the same units as ``xy_edges``, of
        shape ``(..., 2)`` with the last axis being ``(sigma_y, sigma_x)``.
        A 2D input of shape ``(nsig, 2)`` is reshaped as ``(nsig, 1, 2)``.
    mu : tuple of float or array_like, optional
        ``(x, y)`` center of the 2D Gaussian; it must broadcast to the shape
        of ``sigma``. Default is (0, 0).
    normed : bool, optional
        If True, use a unit-integral Gaussian; otherwise use a Gaussian of
        unit peak amplitude. Default is True.

    Returns
    -------
    numpy.ndarray
        Integrated Gaussian of shape ``(ny, nx)`` for a single ``sigma``, or
        ``(nsig, ny, nx)`` for ``sigma`` of shape ``(nsig, 2)`` or
        ``(nsig, 1, 2)``.

    Raises
    ------
    NotImplementedError
        If ``xy_edges`` is not an open mesh-grid.

    Examples
    --------
    >>> y, x = np.ogrid[-5:5.1, -6:6.1]
    >>> pixels = integ_gaussian2D_erf((x, y), sigma=[1, 1])
    >>> pixels.shape
    (10, 12)
    >>> np.isclose(pixels.sum(), 1)
    np.True_
    >>> integ_gaussian2D_erf((x, y), sigma=[[1, 1], [2, 2], [3, 3]]).shape
    (3, 10, 12)
    """
    from scipy.special import erf

    x_edges, y_edges = xy_edges  # (1, nx + 1) and (ny + 1, 1)
    if x_edges.shape[0] > 1 or y_edges.shape[1] > 1:
        raise NotImplementedError("Only 'open' mesh-grid is supported.")

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    if sigma.ndim == 2 and sigma.shape[0]>1:
        sigma = sigma[:, None, :]

    mu = np.broadcast_to(mu, sigma.shape)
    mux = mu[..., 0][..., None] # mu is (x, y)
    muy = mu[..., 1][..., None]

    # allows for asymetric sigma
    sqrt2sig = 1.4142135623730951 * sigma
    if sqrt2sig.shape[-1] == 1:
        sqrt2sig_y = sqrt2sig_x = sqrt2sig

    elif sqrt2sig.shape[-1] == 2:
        sqrt2sig_y = sqrt2sig[..., 0][..., None] # no dim reduction
        sqrt2sig_x = sqrt2sig[..., 1][..., None] # no dim reduction



    tmpx = erf((x_edges - mux) / sqrt2sig_x)  # sig.shape + (1, nx+1)
    tmpy = erf((y_edges - muy) / sqrt2sig_y)  # sig.shape + (ny+1, 1)

    f = (np.diff(tmpx, axis=-1) * np.diff(tmpy, axis=-2) / 4)

    if not normed:
        f *= 2 * np.pi * sigma**2

    return f  # sigma.shape + (ny, nx)


def complete_dims(arr, xdims, squeeze=True):
    """Complete the dimensions of an array with trailing length-1 axes.

    - If ``xdims >= 0``, add ``xdims`` dimensions, the final shape is
      ``arr.shape + (1,) * xdims``;
    - If ``xdims < 0``, add as many dimensions as necessary to reach
      ``|xdims|`` dimensions at least, the final shape is
      ``arr.shape + (1,) * (|xdims| - arr.ndim)``.

    Parameters
    ----------
    arr : array_like
        Input array.
    xdims : int
        Number of extra dimensions to append (if positive), or minimal
        number of dimensions of the output (if negative).
    squeeze : bool, optional
        Remove the *leading* length-1 dimensions (up to the first non-trivial
        one). Default is True.

    Returns
    -------
    numpy.ndarray
        Expanded array.

    Examples
    --------
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
