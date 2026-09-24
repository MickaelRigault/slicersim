import os
import warnings
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
    """ """
    # This is basically a place holder for more complexity in the future.
    if source is None:
        source = "bulla23" # default
    generic = { 'name': 'kilonova',
                'redshift': 0.2,
                'phase': 1.4, # peak mag
                'position': [1, 0.5],
                "source": source
            }

    return generic | kwargs


class AngularTimeSeriesSource(sncosmo.Source):
    r""" A single-component spectral time series model.
        The spectral flux density of this model is given by
        .. math::
        F(t, \lambda) = A \\times M(t, \lambda, \cos_\theta)
        where _M_ is the flux defined on a grid in phase and wavelength
        and _A_ (amplitude) is the single free parameter of the model. The
        amplitude _A_ is a simple unitless scaling factor applied to
        whatever flux values are used to initialize the
        ``TimeSeriesSource``. Therefore, the _A_ parameter has no
        intrinsic meaning. It can only be interpreted in conjunction with
        the model values. Thus, it is meaningless to compare the _A_
        parameter between two different ``TimeSeriesSource`` instances with
        different model data.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.

    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.

    flux : `~numpy.ndarray`
        Model spectral flux density in arbitrary units.
        Must have shape `(num_phases)`.

    zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).

    zero_after : bool, optional
        If True, flux at phases after minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the maximum phase (``flux[-1, :]`` in the input array).

    cos_theta : `~numpy.ndarray`
        cosine of viewing angle

    name : str, optional
        Name of the model. Default is `None`.

    version : str, optional
        Version of the model. Default is `None`.

    """

    _param_names = ['amplitude', 'theta']
    param_names_latex = ['A', r'\theta']

    def __init__(self, phase, wave, cos_theta, flux,
                     zero_before=False, zero_after=False, name=None,
                     version=None):
        """ """
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
        logflux_ = np.zeros(self._flux_array.shape[:2])

        for k in range(len(self._phase)):
            f_tmp = Spline2d(self._wave, self._cos_theta, np.log(self._flux_array[k]),
                             kx=1, ky=1)
            logflux_[k] = f_tmp(self._wave, np.cos(self._parameters[1]*np.pi/180)).T

        self._model_flux = Spline2d(self._phase, self._wave, logflux_, kx=1, ky=1)
        self._current_theta = self._parameters[1]

    def _flux(self, phase, wave):
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
    """ """
    from astropy.io import fits

    modelfits = fits.open(fitsfile)
    flux = modelfits["MODEL"].data
    flux = np.moveaxis(flux, (0), (-1)) # consistency with original formal
    phase = modelfits["PHASE"].data
    wave = modelfits["LBDA"].data
    cos_theta = modelfits["COSTHETA"].data

    return phase, wave, cos_theta, flux

def read_possis_file(filename):
    """Read in a spectral model created by POSSIS (1906.04205).

    This is as appropriate for injestion as a
    `skysurvey.source.angular.AngularTimeSeriesSource`. Model grids can be
    found here: https://github.com/mbulla/kilonova_models.

    Parameters
    ----------
    filename : str
        Path to the POSSIS file (.txt or .fits)

    Returns
    -------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    cos_theta : `~numpy.ndarray`
        Cosine of viewing angle.
    flux : `~numpy.ndarray`
        Model spectral flux density in arbitrary units. Must have shape
        `(num_phases)`.
    """
    # go to custumed loader if fits file.
    if np.any([filename.endswith(fits_ext) for fits_ext in  [".fits", ".fits.gz"]]):
        # fits handles distant or local sources.
        return read_possis_fits(filename)

    # use hand made reader.
    if filename.startswith("https"):
        import requests
        kn_possis = requests.get(filename)
        lines = kn_possis.text.splitlines()
    else:
        f = open(filename)
        lines = f.readlines()

    nobs = int(lines[0])
    nwave = float(lines[1])
    line3 = (lines[2]).split(' ')
    ntime = int(line3[0])
    t_i = float(line3[1])
    t_f = float(line3[2])

    cos_theta = np.linspace(0, 1, nobs)  # 11 viewing angles
    phase = np.linspace(t_i, t_f, ntime)  # epochs

    file_ = np.genfromtxt(filename, skip_header=3)

    wave = file_[0:int(nwave),0]
    flux = []
    for i in range(int(nobs)):
        flux.append(file_[i*int(nwave):i*int(nwave)+int(nwave),1:])
    flux = np.array(flux).T

    phase = np.linspace(t_i, t_f, len(flux.T[0][0]))  # epochs

    return phase, wave, cos_theta, flux

def get_kilonova_model(filename=None,
                       effects=None, effect_names=None, effect_frames=None):
    """Get a kilonova model from a POSSIS file."""
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
    """ """
    # source could be a filename or a shortcut.
    if os.path.isfile(source):
        filename = source
    elif type(source) == str and "bulla19" in source:
        filename = bulla19_url
    elif type(source) == str and "bulla23" in source:
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
                                    cosmo=cosmology)
    #
    return model.flux(phase, lbda)
