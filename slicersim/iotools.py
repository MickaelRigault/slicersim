"""
Access package data files.

.. autosummary::

   expand_path
   get_config
   read_ecsv
   chromatic_interpolator
"""

import os
import sys

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
    """
    Get full file path, including config path if necessary.

    If the input filename does not specifically include a path, it
    will be looked for in the default :data:`MLAPERF_PATH` directory.

    :param str filename: file name
    :return: filename including default path if needed
    """

    if os.path.dirname(filename):  # filename includes a path
        fname = filename
    else:                          # use MLAPERF_PATH as default
        fname = MLAPERF_PATH.joinpath(filename)

    return fname


def get_config(scene="supernova.toml", instrument="lazuli.toml"):
    """
    Read configuration files.

    Final configuration has the following structure::

      {'scene': {'point_source': {...}
                 'zodi': {...}
                },
       'spectrograph': {...},
       'detector': {...},
       'extraction': {...},
      }

    :param str instrument: intrument configuration file name
    :param str scene: scene configuration file name
    :return dict: configuration (nested dictionary)
    """

    return {**read_config(scene),
            **read_config(instrument)}


def read_config(filename, verbose=False):
    """
    Read single configuration file.

    * If the input filename does not specifically include a path, it will be
      looked for in the default :data:`MLAPERF_PATH` directory.
    * As for now, only `toml` configuration files are supported.

    :param str filename: configuration file name
    :param bool verbose: verbose mode
    :return dict: configuration (nested dictionary)
    :raise NotImplementedError: unknown configuration extension
    """

    fname = expand_path(filename)
    if verbose:
        print(f"Reading configuration from {fname!r}...")

    _, extension = os.path.splitext(fname)
    if extension.lower() == ".toml":
        config = tomllib.load(open(fname, "rb"))
    else:
        raise NotImplementedError(
            f"Unknown configuration extension {extension!r}.")

    return config


def override_config(cfg, copy=False, **kwargs):
    """
    Override elements of a (nested dict) configuration using dot notation.

    :param dict cfg: nested dictionary configuration
    :param bool copy: return a modified deep copy (original left untouched)
    :param kwargs: override values `{'key1.key2.key3': newval}`
    :return: modified (copy of) configuration

    >>> cfg = dict(a=1, b=dict(c=2, d=3))
    >>> override_config(cfg, **{'b.c': 4})
    {'a': 1, 'b': {'c': 4, 'd': 3}}
    >>> override_config(cfg, **{'b.e': 5})
    KeyError: "Unknwown key 'b.e'."

    .. Note:: Multi-key override is supported:

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
                raise KeyError(f"Unknwown key {key!r}.")
        if tokens[-1] not in subcfg:  # Overriden key has to exists
            raise KeyError(f"Unknwown key {key!r}.")
        else:
            subcfg[tokens[-1]] = val  # Override

    return cfg


def read_ecsv(filename, colnames=None, description=''):
    """
    Read ECSV file.

    :param str filename: ECSV file name
    :param list colnames: list of requested column names (or None for all)
    :param str description: description string (used as a verbose flag)
    :return: file content as a table
    :rtype: astropy.table.Table
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
    """
    Read CalSpec reference flux FITS file.

    :param str filename: name of the reference flux file
    :param str url: URL to the reference flux file
    :param list wrange: wavelength domain [Å]
    :param str description: description string (used as a verbose flag)
    :return: file content as a table with columns `["wavelength", "flux"]`
    :rtype: astropy.table.Table

    >>> tab = read_calspec('gd71_mod_011.fits',
    ...                    description='GD71) # doctest: +ELLIPSIS
    Reading GD71 CalSpec file 'https://archive.stsci.edu/...
    >>> tab.info()
    <Table length=43736>
    name     dtype           unit           format
    ---------- ------- ---------------------- --------
    wavelength float64               Angstrom {:10.4g}
          flux float32 erg / (Angstrom cm2 s) {:12.4e}
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
    """
    Read XShooter PN spectrum FITS file.

    :param str filename: name of the spectrum file
    :param list wrange: wavelength domain [Å]
    :param str description: description string (used as a verbose flag)
    :return: file content as a table with columns `["wavelength", "flux"]`
    :rtype: astropy.table.Table
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
    """
    Build a chromatic interpolator.

    :param list wavelength: wavelength array (in interpolated units)
    :param list quantity: quantity to be interpolated
    :param int k: degree of the smoothing spline
    :param str ext: extrapolation mode
                    ('extrapolate', 'zeros', 'raise', 'const')
    :param bool inverse: inverse interpolation (e.g. wavelength solution)
    :return: chromatic interpolator
    :rtype: scipy.interpolate.UnivariateSpline
    """

    options = dict(s=0, k=k, ext=ext)
    if inverse:                 # wavelength(quantity)
        interp = UnivariateSpline(quantity, wavelength, **options)
    else:                       # quantity(wavelength)
        interp = UnivariateSpline(wavelength, quantity, **options)

    return interp
