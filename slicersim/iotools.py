"""Access package data files."""

import os
import sys
import numpy as np

from scipy.interpolate import UnivariateSpline
import astropy.units as u
from astropy.table import Table

if sys.version_info[:2] >= (3, 10):
    from importlib.resources import files  # Python 3.10+
else:
    from importlib_resources import files  # External backport
try:
    import tomllib                         # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib                # External


MLAPERF_PATH = files("slicersim.config")     #: Path to data & config files.


def expand_path(filename):
    """Get the full file path, including the config path if necessary.

    If the input filename does not specifically include a path, it will be
    looked for in the default :data:`MLAPERF_PATH` directory.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    str
        Filename including the default path if needed.
    """

    if os.path.dirname(filename):  # filename includes a path
        fname = filename
    else:                          # use MLAPERF_PATH as default
        fname = MLAPERF_PATH.joinpath(filename)

    return fname

def merge_dicts(d1, d2):
    """Recursively merge d2 into a copy of d1 and return the result.

    Values in d2 will override values in d1 in case of conflicts.

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


def get_config(scene="supernova.toml", instrument="lazuli.toml"):
    """Read configuration files.

    The final configuration has the following structure::

        {'scene': {'pointsource': {...},
                   'zodi': {...}
                  },
         'spectrograph': {...},
         'detector': {...},
         'extraction': {...},
        }

    Parameters
    ----------
    scene : str or dict, optional
        Filename of the scene, or a dictionary of filenames.
        For example, `scene="supernovae"` or
        `{"pointsource": "supernovae", "background": "zodi"}`.
        Default is "supernova.toml".
    instrument : str or dict, optional
        Filename defining the instrument, or a dictionary giving details.
        Default is "lazuli.toml".

    Returns
    -------
    dict
        Configuration dictionary.
    """
    return read_config(scene) | read_config(instrument)


def read_config(filename, verbose=False):
    """Read a single configuration file.

    - If the input filename does not specifically include a path, it will be
      looked for in the default :data:`MLAPERF_PATH` directory.
    - Currently, only `.toml` configuration files are supported.

    Parameters
    ----------
    filename : str or list
        Filename of the configuration file. If no extension is provided,
        `.toml` is assumed. `filename="supernova"` is equivalent to
        `filename="supernova.toml".

        `filename` can also be a list of names or a mix of names and dicts.
        These nested dictionaries are merged from left to right, such that for
        `filename = ["supernova", "zodi"]`, the configuration from "zodi" will
        overwrite the corresponding (nested) entry from "supernova".
        If an element of the list is a dict, it will be assumed to be a config.
        For example, `["supernova", {"scene": {"pointsource": {"source":
        [lbda_ref, flux_ref]}}}]` will overwrite the `scene.pointsource.source`
        from the "supernova" configuration.
    verbose : bool, optional
        Verbose mode. Default is False.

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
        
    if extension.lower() == ".toml":
        config = tomllib.load(open(fname, "rb"))
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
        Return a modified deep copy (original left untouched). Default is False.
    **kwargs : dict
        Override values, e.g., `{'key1.key2.key3': newval}`.

    Returns
    -------
    dict
        Modified (copy of) configuration.

    Examples
    --------
    >>> cfg = dict(a=1, b=dict(c=2, d=3))
    >>> override_config(cfg, **{'b.c': 4})
    {'a': 1, 'b': {'c': 4, 'd': 3}}
    >>> override_config(cfg, **{'b.e': 5}) # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    KeyError: "Unknown key 'b.e'."

    .. note::
        Multi-key override is supported:

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
    colnames : list, optional
        List of requested column names. If None, all columns are returned.
        Default is None.
    description : str, optional
        Description string (used as a verbose flag). Default is ''.

    Returns
    -------
    astropy.table.Table
        File content as a table.
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


#: CalSpec repository URL
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"


def read_calspec(filename, url=CALSPEC_URL, wrange=[3_990, 17_010],
                 description=''):
    """Read a CalSpec reference flux FITS file.

    Parameters
    ----------
    filename : str
        Name of the reference flux file.
    url : str, optional
        URL to the reference flux file. Default is `CALSPEC_URL`.
    wrange : list, optional
        Wavelength domain in Angstroms. Default is `[3_990, 17_010]`.
    description : str, optional
        Description string (used as a verbose flag). Default is ''.

    Returns
    -------
    astropy.table.Table
        File content as a table with columns `["wavelength", "flux"]`.

    Examples
    --------
    >>> tab = read_calspec('gd71_mod_011.fits',
    ...                    description='GD71') # doctest: +ELLIPSIS
    Reading GD71 CalSpec file 'https://archive.stsci.edu/...
    >>> tab.info()
    <Table length=43736>
     name       dtype                   unit                 format
    ---------- ------- ------------------------------------ --------
    wavelength float64                             Angstrom {:10.4g}
          flux float32 erg / (Angstrom s cm2) {:12.4e}
    """

    fname = expand_path(filename)  # use explicit or config/ file
    if not os.path.exists(fname):  # use distant repository
        fname = os.path.join(url, filename)
    if description:
        print(f"Reading {description} CalSpec file {fname!r}...")

    tab = Table.read(fname, cache=True)  # cache the table
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


def read_xshooter(filename, wrange=[3_990, 17_010], description=''):
    """Read an XShooter PN spectrum FITS file.

    Parameters
    ----------
    filename : str
        Name of the spectrum file.
    wrange : list, optional
        Wavelength domain in Angstroms. Default is `[3_990, 17_010]`.
    description : str, optional
        Description string (used as a verbose flag). Default is ''.

    Returns
    -------
    astropy.table.Table
        File content as a table with columns `["wavelength", "flux"]`.
    """

    fname = expand_path(filename)  # use explicit or config/ file
    if description:
        print(f"Reading {description} XShooter file {fname!r}...")

    tab = Table.read(fname, cache=True)  # cache the table
    # Rename and assign proper units
    tab['WAVE'].name = "wavelength"
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

    Parameters
    ----------
    wavelength : array_like
        Wavelength array in interpolated units.
    quantity : array_like
        Quantity to be interpolated.
    k : int, optional
        Degree of the smoothing spline. Default is 3.
    ext : str, optional
        Extrapolation mode ('extrapolate', 'zeros', 'raise', 'const').
        Default is 'raise'.
    inverse : bool, optional
        Inverse interpolation (e.g., wavelength solution). Default is False.

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
