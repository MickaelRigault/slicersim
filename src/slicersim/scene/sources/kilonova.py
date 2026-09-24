""" This module handles Kilonova sources.

Kilonova spectra are computed from POSSIS radiative transfer models
(`Bulla 2019 <https://arxiv.org/abs/1906.04205>`_, Bulla 2023). These models provide a
spectral time series for a grid of viewing angles; they are wrapped here as
an `AngularTimeSeriesSource` (a `sncosmo.Source` with an extra ``theta``
parameter) to be used through `sncosmo.Model`.

Two models are directly available through shortcut names:

- "bulla19": a single POSSIS model from Bulla (2019)
  (``mejdyn=0.02``, ``mejwind=0.13``, ``phi=30``).
- "bulla23": a POSSIS model from Bulla (2023) [default].

Any other POSSIS file (.txt or .fits) can be used by giving its path. Model
grids can be found at https://github.com/mbulla/kilonova_models.

The entry points are `get_kilonova_pointsource`, used by
`slicersim.scene.get_scene` to build a point source configuration, and
`get_kilonova_flux`, the ``model_func`` of the corresponding
`~slicersim.scene.pointsource.PointSource`.
"""
import os
import warnings
from functools import lru_cache

import numpy as np
import sncosmo
from scipy.interpolate import RectBivariateSpline as Spline2d
from astropy.cosmology import Planck18 as cosmology

# stored on S3 for slicersim, but also available on github:
# => https://github.com/mbulla/kilonova_models
# This is just one case
bulla19_url = "https://slicersim-data-104767970225-eu-west-3-an.s3.eu-west-3.amazonaws.com/nsns_nph1.0e%2B06_mejdyn0.020_mejwind0.130_phi30.txt"
bulla23_url = "https://slicersim-data-104767970225-eu-west-3-an.s3.eu-west-3.amazonaws.com/bulla23_knmodel.fits"

# Top level method

def get_kilonova_pointsource(source=None, **kwargs):
    """Get a generic configuration for a kilonova point source.

    Parameters
    ----------
    source : str, optional
        Name of the kilonova model (see `get_kilonova_flux`).
        If None the default is used.

        - "bulla23" [default]
        - "bulla19"
    **kwargs
        Additional parameters to update the configuration, i.e. any
        parameter of `get_kilonova_flux` (e.g. redshift, phase, theta,
        magabs, magobs, band) or ``position``.

    Returns
    -------
    dict
        Configuration dictionary for a kilonova point source, ready to be
        passed to `~slicersim.scene.pointsource.PointSource.from_config`.

    See Also
    --------
    slicersim.scene.get_scene : Build a full scene around a kilonova.
    get_kilonova_flux : The corresponding model function.

    Examples
    --------
    >>> get_kilonova_pointsource(redshift=0.05, theta=45)
    {'name': 'kilonova', 'redshift': 0.05, 'phase': 1.4, 'position': [1, 0.5], 'source': 'bulla23', 'theta': 45}
    """
    # This is basically a place holder for more complexity in the future.
    if source is None:
        source = "bulla23" # default
    generic = { 'name': 'kilonova',
                'redshift': 0.2,
                'phase': 1.4, # days since merger
                'position': [1, 0.5],
                "source": source
            }

    return generic | kwargs


class AngularTimeSeriesSource(sncosmo.Source):
    r"""A single-component spectral time series model with a viewing angle.

    The spectral flux density of this model is given by

    .. math::

        F(t, \lambda) = A \times M(t, \lambda, \cos\theta)

    where :math:`M` is the flux defined on a grid in phase, wavelength and
    cosine of the viewing angle, and :math:`A` (amplitude) and
    :math:`\theta` (viewing angle, in degrees) are the free parameters of
    the model. The flux at a given :math:`\theta` is linearly interpolated
    in :math:`\log M` between the ``cos_theta`` grid points.

    The amplitude :math:`A` is a simple unitless scaling factor applied to
    the model flux, normalized at initialization to a maximum of 1.
    Therefore, it has no intrinsic meaning and can only be interpreted in
    conjunction with the model values.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.

    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.

    cos_theta : `~numpy.ndarray`
        Cosine of the viewing angles of the grid.

    flux : `~numpy.ndarray`
        Model spectral flux density in arbitrary units.
        Must have shape ``(num_phases, num_waves, num_cos_theta)``.
        Negative values are set to 0 and null values to a small positive
        value (1e-10 after normalization) to allow log-interpolation.

    zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).

    zero_after : bool, optional
        If True, flux at phases after minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the maximum phase (``flux[-1, :]`` in the input array).

    name : str, optional
        Name of the model. Default is `None`.

    version : str, optional
        Version of the model. Default is `None`.

    See Also
    --------
    get_kilonova_model : Build a `sncosmo.Model` from a POSSIS file.
    """

    _param_names = ['amplitude', 'theta']
    param_names_latex = ['A', r'\theta']

    def __init__(self, phase, wave, cos_theta, flux,
                     zero_before=False, zero_after=False, name=None,
                     version=None):
        """Initialize the AngularTimeSeriesSource.

        See the class docstring for the parameters.
        """
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._cos_theta = cos_theta
        # cleaning bad flux definition.
        flux = flux.copy()
        flux[flux < 0] = 0
        flux /= flux.max()
        flux[flux == 0] = 1e-10

        self._flux_array = flux
        self._parameters = np.array([1., 0.])
        self._current_theta = 0.
        self._zero_before = zero_before
        self._zero_after = zero_after
        self._set_theta()

    def _set_theta(self):
        """Build the (phase, wave) flux interpolator at the current theta.

        The log-flux is interpolated along ``cos_theta`` at the current
        viewing angle for every phase, and stored as a 2D spline in
        (phase, wave). This is called whenever ``theta`` changes.
        """
        logflux_ = np.zeros(self._flux_array.shape[:2])

        for k in range(len(self._phase)):
            f_tmp = Spline2d(self._wave, self._cos_theta, np.log(self._flux_array[k]),
                             kx=1, ky=1)
            logflux_[k] = f_tmp(self._wave, np.cos(self._parameters[1]*np.pi/180)).T

        self._model_flux = Spline2d(self._phase, self._wave, logflux_, kx=1, ky=1)
        self._current_theta = self._parameters[1]

    def _flux(self, phase, wave):
        """Compute the model flux.

        Parameters
        ----------
        phase : array_like
            Phases in days.
        wave : array_like
            Wavelengths in Angstroms.

        Returns
        -------
        numpy.ndarray
            Flux of shape ``(len(phase), len(wave))``.
        """
        if self._current_theta != self._parameters[1]:
            self._set_theta()

        f = self._parameters[0] * (np.exp(self._model_flux(phase, wave)))

        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.

        if self._zero_after:
            mask = np.atleast_1d(phase) > self.maxphase()
            f[mask, :] = 0.

        return f

# ================ #
#   POSSIS MODEL   #
# ================ #
def read_possis_fits(fitsfile):
    """Read in a POSSIS spectral model stored as a fits file.

    The fits file must contain the "MODEL", "PHASE", "LBDA" and "COSTHETA"
    extensions, with MODEL of shape ``(num_cos_theta, num_phases, num_waves)``.

    Parameters
    ----------
    fitsfile : str
        Path or url of the fits file. Remote files are stored in the
        astropy download cache.

    Returns
    -------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    cos_theta : `~numpy.ndarray`
        Cosine of viewing angle.
    flux : `~numpy.ndarray`
        Model spectral flux density in arbitrary units, of shape
        ``(num_phases, num_waves, num_cos_theta)``.

    See Also
    --------
    read_possis_file : Generic POSSIS reader (txt or fits).
    """
    from astropy.io import fits

    with fits.open(fitsfile) as modelfits:
        # np.array: load in memory before the file is closed.
        flux = np.array(modelfits["MODEL"].data)
        flux = np.moveaxis(flux, (0), (-1)) # consistency with original formal
        phase = np.array(modelfits["PHASE"].data)
        wave = np.array(modelfits["LBDA"].data)
        cos_theta = np.array(modelfits["COSTHETA"].data)

    return phase, wave, cos_theta, flux

@lru_cache
def read_possis_file(filename):
    """Read in a spectral model created by POSSIS (1906.04205).

    This is as appropriate for injestion as an `AngularTimeSeriesSource`.
    Model grids can be found here: https://github.com/mbulla/kilonova_models.

    Results are cached (per filename) so a model is only read (or
    downloaded) once per session; the returned arrays must hence not be
    modified in place.

    Parameters
    ----------
    filename : str
        Path or url (https) to the POSSIS file (.txt or .fits)

    Returns
    -------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    cos_theta : `~numpy.ndarray`
        Cosine of viewing angle.
    flux : `~numpy.ndarray`
        Model spectral flux density in arbitrary units, of shape
        ``(num_phases, num_waves, num_cos_theta)``.

    See Also
    --------
    read_possis_fits : The reader used for fits files.
    """
    # go to custumed loader if fits file.
    if np.any([filename.endswith(fits_ext) for fits_ext in  [".fits", ".fits.gz"]]):
        # fits handles distant or local sources.
        return read_possis_fits(filename)

    # use hand made reader.
    if filename.startswith("https"):
        import requests
        kn_possis = requests.get(filename)
        kn_possis.raise_for_status()
        lines = kn_possis.text.splitlines()
    else:
        with open(filename) as f:
            lines = f.read().splitlines()

    nobs = int(lines[0])
    nwave = float(lines[1])
    line3 = (lines[2]).split(' ')
    ntime = int(line3[0])
    t_i = float(line3[1])
    t_f = float(line3[2])

    cos_theta = np.linspace(0, 1, nobs)  # 11 viewing angles
    phase = np.linspace(t_i, t_f, ntime)  # epochs

    file_ = np.genfromtxt(lines, skip_header=3)

    wave = file_[0:int(nwave),0]
    flux = []
    for i in range(int(nobs)):
        flux.append(file_[i*int(nwave):i*int(nwave)+int(nwave),1:])
    flux = np.array(flux).T

    phase = np.linspace(t_i, t_f, len(flux.T[0][0]))  # epochs

    return phase, wave, cos_theta, flux

def get_kilonova_model(filename=None,
                       effects=None, effect_names=None, effect_frames=None):
    """Get a kilonova model from a POSSIS file.

    Parameters
    ----------
    filename : str
        Path or url of the POSSIS file (.txt or .fits),
        see `read_possis_file`.
    effects : list of `sncosmo.PropagationEffect`, optional
        Propagation effects (e.g. dust), see `sncosmo.Model`.
        Default is None.
    effect_names : list of str, optional
        Names of the effects, see `sncosmo.Model`. Default is None.
    effect_frames : list of str, optional
        Frames of the effects ("rest", "obs" or "free"), see `sncosmo.Model`.
        Default is None.

    Returns
    -------
    sncosmo.Model
        Model with an `AngularTimeSeriesSource` source, i.e. with parameters
        ``z``, ``t0``, ``amplitude`` and ``theta`` (in degrees).
    """
    phase, wave, cos_theta, flux = read_possis_file(filename)
    source = AngularTimeSeriesSource(phase=phase, wave=wave, flux=flux, cos_theta=cos_theta,
                                         name="kilonova")
    model = sncosmo.Model(source, effects=effects, effect_names=effect_names,
                          effect_frames=effect_frames)
    return model

def get_kilonova_flux(lbda, phase=0,
                       redshift=0.1, theta=0,
                       magobs=None, magabs=-15.8,
                       band="sdssr", magsys="ab",
                       source="bulla19", cosmo=cosmology):
    """Get the flux of a kilonova source.

    The POSSIS model is evaluated at the requested viewing angle and
    redshift, normalized to the given peak magnitude, and computed at the
    requested phase and (observer-frame) wavelengths.

    Parameters
    ----------
    lbda : array_like
        Observer-frame wavelength array in Angstrom.
    phase : float, optional
        Time since the merger in days (observer-frame). Default is 0.
    redshift : float, optional
        Redshift of the kilonova. Default is 0.1.
    theta : float, optional
        Viewing angle in degrees (0 is pole-on, 90 is edge-on).
        Default is 0.
    magobs : float, optional
        Peak observed magnitude in `band`. If given, `magabs` is ignored.
        Default is None.
    magabs : float, optional
        Peak absolute magnitude (rest-frame) in `band`, converted to an
        observed magnitude using `cosmo`. Ignored if `magobs` is given.
        If both are None the model is not normalized.
        Default is -15.8.
    band : str, optional
        Name of the bandpass (must be known by `sncosmo`) used for the
        normalization. Default is "sdssr".
    magsys : str, optional
        Name of the magnitude system (see `sncosmo`). Default is "ab".
    source : str, optional
        Kilonova model to use. Either a path to a POSSIS file or a shortcut
        name:

        - "bulla19": Bulla (2019) model.
        - "bulla23": Bulla (2023) model.

        Default is "bulla19".
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology used to convert `magabs` into an observed magnitude.
        Default is Planck18.

    Returns
    -------
    numpy.ndarray
        The kilonova flux in erg/s/cm^2/A, matching the shape of ``lbda``.

    Raises
    ------
    ValueError
        If ``source`` is neither a file nor a known shortcut name.

    Warns
    -----
    UserWarning
        If both `magobs` and `magabs` are given.

    See Also
    --------
    get_kilonova_model : The underlying `sncosmo.Model`.

    Examples
    --------
    >>> import numpy as np
    >>> lbda = np.linspace(4000, 20000, 1000)
    >>> flux = get_kilonova_flux(lbda, phase=1.4, redshift=0.1, theta=30,
    ...                          source="bulla23")
    """
    # source could be a filename or a shortcut.
    if type(source) is str and os.path.isfile(source):
        filename = source
    elif type(source) is str and "bulla19" in source:
        filename = bulla19_url
    elif type(source) is str and "bulla23" in source:
        filename = bulla23_url
    else:
        raise ValueError(f"Unknown source {source!r}. Must be a filename or 'bulla19' or 'bulla23'.")

    # sncosmo model
    model = get_kilonova_model(filename)
    model.set(theta=theta, z=redshift, t0=0)

    if magobs is not None:
        if magabs is not None:
            warnings.warn("magobs and magabs have been specified. magabs ignored.")
        model.set_source_peakmag(m=magobs, band=band, magsys=magsys)
    elif magabs is not None:
        model.set_source_peakabsmag(absmag=magabs,
                                    band=band, magsys=magsys,
                                    cosmo=cosmo)
    return model.flux(phase, lbda)
