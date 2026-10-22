"""Read configuration and data files.

Files given without a directory are looked for in the package configuration
directory ``PACKAGE_PATH.joinpath("config")`` (``slicersim/config``).
"""

import os
import warnings
from importlib.resources import files

import astropy.units as u
import numpy as np
from astropy.table import Table
from scipy.interpolate import UnivariateSpline

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # External

PACKAGE_PATH = files("slicersim")     #: Path to data & config files.

def expand_path(filename):
    """Get the full file path, including the config path if necessary.

    If the input filename does not specifically include a directory, it is
    looked for in the default ``PACKAGE_PATH.joinpath("config")`` directory.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    str or pathlib.Path
        ``filename`` itself if it includes a directory, otherwise the path to
        ``filename`` in ``PACKAGE_PATH.joinpath("config")``.
    """

    if os.path.dirname(filename):  # filename includes a path
        fname = filename
    else:                          # use PACKAGE_PATH.joinpath("config") as default
        fname = PACKAGE_PATH.joinpath("config").joinpath(filename)

    return fname

def merge_dicts(d1, d2):
    """Recursively merge ``d2`` into a copy of ``d1``.

    Values in ``d2`` override values in ``d1`` in case of conflicts; nested
    dictionaries are merged recursively. ``d1`` is not modified.

    Parameters
    ----------
    d1 : dict
        First dictionary.
    d2 : dict
        Second dictionary.

    Returns
    -------
    dict
        Merged dictionary.
    """
    from copy import deepcopy
    d1 = deepcopy(d1) # do not change input d1.

    for key in d2:
        if key in d1:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                d1[key] = merge_dicts(d1[key], d2[key])
            else:
                d1[key] = d2[key]  # Override value
        else:
            d1[key] = d2[key]

    return d1


def get_config(scene="supernova.toml", instrument="lazuli_cbe.toml"):
    """Read the scene and instrument configuration files.

    The final configuration has the following structure::

        {'scene': {'pointsource': {...},
                   'background': {...}
                  },
         'telescope': {...},
         'spectrograph': {...},
         'detector': {...},
         'extraction': {...},
        }

    Parameters
    ----------
    scene : str, list or dict, optional
        Scene configuration, as accepted by `read_config`: a filename (e.g.
        ``"supernova"``), a list of filenames and/or dictionaries merged from
        left to right, or a configuration dictionary.
        Default is ``"supernova.toml"``.
    instrument : str, list or dict, optional
        Instrument configuration, as accepted by `read_config`.
        Default is ``"lazuli_cbe.toml"``.

    Returns
    -------
    dict
        Configuration dictionary.

    Notes
    -----
    The scene and instrument configurations are combined with a shallow
    merge: a top-level key present in both is taken from ``instrument``.
    """
    return read_config(scene) | read_config(instrument)


def read_config(filename, verbose=False):
    """Read a single configuration file.

    - If the input filename does not specifically include a directory, it is
      looked for in the default ``PACKAGE_PATH.joinpath("config")`` directory.
    - Currently, only ``.toml`` configuration files are supported.

    Parameters
    ----------
    filename : str, list or dict
        Filename of the configuration file. If no extension is provided,
        ``.toml`` is assumed: ``filename="supernova"`` is equivalent to
        ``filename="supernova.toml"``. A dict is returned as is.

        ``filename`` can also be a list of names, dicts, or a mix of both.
        These nested dictionaries are merged from left to right, such that
        for ``filename = ["supernova", "zodi"]``, the configuration from
        ``"zodi"`` overwrites the corresponding (nested) entries from
        ``"supernova"``. A dict element is assumed to be a configuration.
        For example,
        ``["supernova", {"scene": {"pointsource": {"source": [lbda_ref, flux_ref]}}}]``
        overwrites the ``scene.pointsource.source`` entry of the
        ``"supernova"`` configuration.
    verbose : bool, optional
        If True, print the name of the file being read (not propagated to the
        elements of a list). Default is False.

    Returns
    -------
    dict
        Configuration as a nested dictionary.

    Raises
    ------
    NotImplementedError
        If the configuration file extension is not supported.
    """
    # dict structure
    if type(filename) is dict:
        return filename

    # list / array structure
    if hasattr(filename, "__iter__") and type(filename) not in (str, np.bytes_):
        d = {}
        for filename_ in filename:
            d = merge_dicts(d, read_config(filename_))

        return d

    # core
    fname = expand_path(filename)
    if verbose:
        print(f"Reading configuration from {fname!r}...")

    _, extension = os.path.splitext(fname)
    if extension is None or len(extension) ==0:
        extension = ".toml"
        fname = f"{fname}{extension}"

    if fname == "lazuli.toml":
        warnings.warn("lazuli.toml is deprecated, lazuli_cbe.toml to use.")
        fname = "lazuli_cbe.toml"

    if extension.lower() == ".toml":
        with open(fname, "rb") as f:
            config = tomllib.load(f)
    else:
        raise NotImplementedError(
            f"Unknown configuration extension {extension!r}.")

    return config


def override_config(cfg, copy=False, **kwargs):
    """Override elements of a (nested dict) configuration using dot notation.

    Parameters
    ----------
    cfg : dict
        Nested dictionary configuration.
    copy : bool, optional
        If True, return a modified deep copy (original left untouched);
        otherwise ``cfg`` is modified in place. Default is False.
    **kwargs
        Override values, with keys in dot notation, e.g.
        ``**{'key1.key2.key3': newval}``.

    Returns
    -------
    dict
        Modified configuration (``cfg`` itself, or its copy if ``copy``).

    Raises
    ------
    KeyError
        If a key does not already exist in ``cfg``.

    Examples
    --------
    >>> cfg = dict(a=1, b=dict(c=2, d=3))
    >>> override_config(cfg, **{'b.c': 4})
    {'a': 1, 'b': {'c': 4, 'd': 3}}
    >>> override_config(cfg, **{'b.e': 5}) # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    KeyError: "Unknown key 'b.e'."

    A whole sub-dictionary can be overridden at once:

    >>> override_config(cfg, **{'b': dict(c=3, d=4)})
    {'a': 1, 'b': {'c': 3, 'd': 4}}
    """

    if copy:
        from copy import deepcopy
        cfg = deepcopy(cfg)

    for key, val in kwargs.items():
        subcfg = cfg
        tokens = key.split('.')
        for token in tokens[:-1]:     # Dig down the configuration dict
            subcfg = subcfg[token]
            if not isinstance(subcfg, dict):
                raise KeyError(f"Unknown key {key!r}.")
        if tokens[-1] not in subcfg:  # Overriden key has to exists
            raise KeyError(f"Unknown key {key!r}.")
        else:
            subcfg[tokens[-1]] = val  # Override

    return cfg


def read_ecsv(filename, colnames=None, description=''):
    """Read an ECSV file.

    Parameters
    ----------
    filename : str
        ECSV file name.
    colnames : list of str or None, optional
        Requested column names. If None, all columns are returned.
        Default is None.
    description : str, optional
        Description of the file; if not empty, information about the file is
        printed. Default is ``''``.

    Returns
    -------
    astropy.table.Table
        File content as a table.

    Raises
    ------
    NotImplementedError
        If the file extension is not ``.ecsv``.
    KeyError
        If a requested column does not exist.
    """

    fname = expand_path(filename)
    if description:
        print(f"Reading {description} file {fname!r}...")

    _, extension = os.path.splitext(fname)
    if extension.lower() == ".ecsv":
        tab = Table.read(fname, format='ascii.ecsv')
    else:
        raise NotImplementedError(f"Unknown file extension {extension!r}.")

    if description:
        print(f"  {len(tab)} rows, {len(tab.colnames)} columns: {tab.colnames}")
        if (comments := tab.meta.get('comments')):
            print(f"  Comments: {comments}")

    if colnames:
        try:
            return tab[colnames]  # Return requested columns
        except KeyError as exc:
            colname = exc.args[0]
            raise KeyError(f"No column {colname} in {filename}.") from exc
    else:
        return tab                # Return full table


# ============= #
#  Galaxies     #
# ============= #
def get_galaxy_filepath(name, which="spectrum"):
    """Get the path(s) to the spectrum and/or thumbnail files of a galaxy.

    Files are looked for in the package data directory
    ``PACKAGE_PATH.joinpath("data")``.

    Parameters
    ----------
    name : str
        Galaxy name (case insensitive). Currently, only NGC galaxies are
        supported, given as ``"ngcXXXX"`` or ``"ngc_XXXX"``.
    which : str, optional
        Which file(s) to return: ``"spectrum"`` (or ``"spec"``) for the
        spectrum file, ``"image"`` (or ``"thumbnail"``) for the thumbnail
        image file, or ``"both"`` for both. Default is ``"spectrum"``.

    Returns
    -------
    pathlib.Path or list of pathlib.Path
        Path to the requested file if a single file is requested, otherwise
        the list ``[spectrum_path, image_path]``.

    Examples
    --------
    >>> spec, image = get_galaxy_filepath("NGC4321", which="both")  # doctest: +SKIP
    """
    try:
        from lazuli_astrosim_data.io import get_galaxy_filepath
    except ImportError:
        raise ImportError("lazuli_astrosim_data is required to access galaxy data. Please install it via pip.") from None

    return get_galaxy_filepath(name, which=which)

# ============= #
#  CalSpec      #
# ============= #
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"

def read_calspec(filename, url=CALSPEC_URL, wrange=[3_990, 17_010],
                 description=''):
    """Read a CalSpec reference flux FITS file.

    Parameters
    ----------
    filename : str
        Name of the reference flux file. If it is not found locally (or in
        ``PACKAGE_PATH.joinpath("config")``), it is read from ``url``.
    url : str, optional
        Base URL of the remote CalSpec repository. Default is
        ``CALSPEC_URL``.
    wrange : list of float, optional
        ``[min, max]`` wavelength domain to keep (inclusive), in Angstrom.
        Default is ``[3_990, 17_010]``.
    description : str, optional
        Description of the file; if not empty, the file name is printed.
        Default is ``''``.

    Returns
    -------
    astropy.table.Table
        File content as a table with columns ``"wavelength"`` (Angstrom)
        and ``"flux"`` (erg/s/cm^2/A).

    Raises
    ------
    AssertionError
        If the file units are not ``ANGSTROMS`` and ``FLAM``.

    Examples
    --------
    >>> tab = read_calspec('gd71_mod_012.fits',
    ...                    description='GD71') # doctest: +ELLIPSIS
    Reading GD71 CalSpec file 'https://archive.stsci.edu/...
    >>> tab.info()  # doctest: +NORMALIZE_WHITESPACE
    <Table length=43735>
       name     dtype           unit           format
    ---------- ------- ---------------------- --------
    wavelength float64               Angstrom {:10.4g}
          flux float32 erg / (Angstrom s cm2) {:12.4e}
    """

    fname = expand_path(filename)  # use explicit or config/ file
    if not os.path.exists(fname):  # use distant repository
        fname = os.path.join(url, filename)
    if description:
        print(f"Reading {description} CalSpec file {fname!r}...")

    tab = Table.read(fname, unit_parse_strict="silent")
    assert (tab['WAVELENGTH'].unit == "ANGSTROMS" and
            tab['FLUX'].unit == "FLAM"), \
        f"Incompatible CalSpec table {fname!r}"
    # Rename and assign proper units
    tab['WAVELENGTH'].name = "wavelength"
    tab['wavelength'].unit = "angstrom"
    tab['FLUX'].name = 'flux'
    tab['flux'].unit = u.erg / u.s / u.cm**2 / u.angstrom  # flambda

    wmin, wmax = wrange
    wcol = tab['wavelength']
    wsel = ~((wcol < wmin) | (wcol > wmax))  # Wavelength select

    return tab[wsel]['wavelength', 'flux']


def chromatic_interpolator(wavelength, quantity,
                           k=3, ext='raise', inverse=False):
    """Build a chromatic interpolator.

    The interpolator is an interpolating spline (no smoothing, ``s=0``).

    Parameters
    ----------
    wavelength : array_like
        Wavelengths, in the unit expected by the interpolator.
    quantity : array_like
        Quantity to be interpolated.
    k : int, optional
        Degree of the spline. Default is 3.
    ext : str or int, optional
        Extrapolation mode (``'extrapolate'``, ``'zeros'``, ``'raise'``,
        ``'const'``), see `scipy.interpolate.UnivariateSpline`.
        Default is ``'raise'``.
    inverse : bool, optional
        If True, interpolate wavelength as a function of ``quantity`` (e.g.
        wavelength solution); otherwise ``quantity`` as a function of
        wavelength. In the inverse case, ``quantity`` must be increasing.
        Default is False.

    Returns
    -------
    scipy.interpolate.UnivariateSpline
        Chromatic interpolator.
    """

    options = dict(s=0, k=k, ext=ext)
    if inverse:                 # wavelength(quantity)
        interp = UnivariateSpline(quantity, wavelength, **options)
    else:                       # quantity(wavelength)
        interp = UnivariateSpline(wavelength, quantity, **options)

    return interp
