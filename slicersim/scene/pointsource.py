""" Scene point source """
import inspect
import numpy as np

from astropy.cosmology import Planck18 as cosmology
from .base import SceneElement

from twins_embedding import TwinsEmbeddingModel
twins_embedding_model = TwinsEmbeddingModel()


"""
    @staticmethod
    def stdstar_from_config(config, lbda, k=3, verbose=True):
        '''
        Generate stdstar spectrum from point-source config.

        The interpolation does not account for spectral resolution,
        but account for chromatic average over (approximate) pixel.

        :param dict config: configuration dictionary
        :param array lbda: bin central wavelengths [Å] (nlbda,)
        :param int k: interpolation order
        :param bool verbose: verbose flag
        :return: flux density array [flambda] (nlbda,)
        '''

        from astropy.units import UnitsWarning

        # Disable astropy.units.UnitsWarning (CalSpec spectra are not compliant)
        warnings.simplefilter('ignore', category=UnitsWarning)

        name = config["source"]         # Std star name
        abmag = float(config["ABmag"])  # target AB-magn.
        band = config["bandpass"]       # Bandpass (sncosmo)

        # Estimate bin edge wavelengths
        lbda_edges = recover_bin_edges(lbda, order=2)  # (nlbda+1,)

        # Read reference flux over bin extent
        wrange = lbda_edges[[0, -1]]
        tab = read_calspec(config["reference"],  # Reference filename
                           wrange=wrange,        # Wavelength domain
                           description=name if verbose else '')  # Std star name

        # Chromatic interpolator
        model = chromatic_interpolator(tab['wavelength'],    # [Å]
                                       tab['flux'],          # [erg/s/cm²/Å]
                                       ext='extrapolate', k=k)

        # Average flux over spectral pixels
        int_model = model.antiderivative()     # Flux primitive
        int_flux = int_model(lbda_edges)       # Flux integral at bin edges
        target_spec = np.diff(int_flux) / np.diff(lbda_edges)  # (nlbda,)

        # Compute reference magnitude and scale factor
        spec = sncosmo.Spectrum(lbda, target_spec)  # [Å, flambda]
        abmag0 = spec.bandmag(band, "ab")   # reference AB-magn.
        if not np.isnan(abmag):
            scale = 10**(0.4*(abmag - abmag0))  # > 1 for fainter
        else:
            config["ABmag"] = abmag0
            scale = 1

        if verbose:
            print(f"{name}: {band}={abmag0:.2f} AB, scale factor: {scale:.1f}")

        return target_spec / scale  # point source spectrum [erg/s/cm²/Å] (nlbda,)




    @staticmethod
    def pn_from_config(config, lbda,
                       k=3, sigma=None, oversampling=5, verbose=True):
        '''
        Generate planetary nebula spectrum from point-source config.

        The interpolation accounts for integration over pixel and
        spectral resolution.

        :param dict config: configuration dictionary
        :param array lbda: bin central wavelengths [Å] (nlbda,)
        :param float sigma: constant LSF sigma [Å]
        :param float oversampling: oversampling factor
        :param int k: interpolation order
        :param bool verbose: verbose flag
        :return: flux density array [flambda] (nlbda,)

        .. Warning:: variable LSF is not supported.
        '''

        from astropy.units import UnitsWarning

        # Disable astropy.units.UnitsWarning (CalSpec spectra are not compliant)
        warnings.simplefilter('ignore', category=UnitsWarning)

        name = config["source"]         # PN name

        # Estimate bin edge wavelengths
        lbda_edges = recover_bin_edges(lbda, order=2)  # (nlbda+1,)

        # Read reference flux over bin extent
        wrange = lbda_edges[[0, -1]]
        tab = read_xshooter(config["reference"],  # Reference filename
                            wrange=wrange,        # Wavelength domain
                            description=name if verbose else '')  # PN name

        # Chromatic interpolator
        model = chromatic_interpolator(tab['wavelength'],    # [Å]
                                       tab['flux'],          # [erg/s/cm²/Å]
                                       ext='extrapolate', k=k)
        int_model = model.antiderivative()  # Model primitive

        if np.ndim(sigma) > 0:
            raise NotImplementedError("Variable LSF sigma not implemented.")

        if sigma is not None:               # Gaussian kernel convolution
            raise NotImplementedError("LSF not implemented.")
            from scipy.ndimage import gaussian_filter1d

            # Rebin on a fine linear grid
            dl = min(np.diff(lbda_edges).min(), sigma) / oversampling
            n = round((lbda_edges[-1] - lbda_edges[0]) / dl)
            l = np.linspace(lbda_edges[0], lbda_edges[-1], n)
            dl = (l[-1] - l[0]) / (n - 1)

            print("DEBUG", l[[0, -1]], dl, n, sigma / dl)

            f = np.diff(int_model(l)) / dl
            # Gaussian smoothing
            gf = gaussian_filter1d(f, sigma / dl)
            # Interpolate
            model = chromatic_interpolator(l, gf, ext='extrapolate', k=k)
            int_model = model.antiderivative()  # Model primitive

        # Average flux over spectral pixels
        int_flux = int_model(lbda_edges)       # Flux integral at bin edges
        target_spec = np.diff(int_flux) / np.diff(lbda_edges)  # (nlbda,)

        # No flux scaling

        return target_spec  # point source spectrum [erg/s/cm²/Å] (nlbda,)
"""
# ==================== #
#  Top level shortcut  #
# ==================== #
def source_to_modelfunc(source):
    """ """
    if "salt" in source:
        model_func = get_saltmodel_flux
    elif source == 'twins-embedding':
        model_func = get_twins_embedding_flux
    else:
        raise NotImplementedError(f"no model_func defined for source: {source}")

    return model_func


def obsmag_to_redshift(mag, magabs,
                        redshift_scan="0.001:3:0.01",
                        cosmo=cosmology):
    """
    Linearly interpolate app. magn. into redshift given an abs. magn.

    :param mag: apparent magnitude(s)
    :param float magabs: absolute magnitude
    :param redshift_scan: redshift ramp (argument to `np.r_`)
    :param cosmo: :class:`astropy.cosmology.Cosmology`
    :return: linearly interpolated redshift(s)
    """

    zz = eval(f"np.r_[{redshift_scan}]")
    obsmag = cosmo.distmod(zz).value + magabs

    # Linear interpolation
    return np.interp(mag, obsmag, zz)


def get_saltmodel(redshift=0.1,
                  MBmax=-19.3, source="salt2-extended", cosmo=cosmology,
                  x1=0, c=0, alpha=-0.14, beta=3.15):
    """
    :pypi:`sncosmo` SALT2 SN Ia model at redshift z and with peak `MBmax`.

    The returned model is monkey-patched with additional method `flux_extended`
    which returns null fluxes outside the valid spectral domain of the model.

    :param float z: redshift
    :param float MBmax: peak absolute Bessell-B AB-mag.
    :param str source: model source (see `sncosmo`)
    :param cosmo: :class:`astropy.cosmology.Cosmology`
    :param float x1: SALT2 stretch
    :param float c: SALT2 color
    :param float alpha: stretch standardization factor
    :param float beta: color standardization factor
    :return: monkey-patched :class:`sncosmo.SALT2Source`
    """
    import sncosmo

    model = sncosmo.Model(source=source)
    model.set(z=redshift, c=c, x1=x1)              # SALT2 parameters
    eff_mbmax = MBmax + (x1*alpha + c*beta)  # Tripp relation (no env bias)
    
    # set effective peak magnitude
    model.set_source_peakabsmag(eff_mbmax,
                                "bessellb", "AB", cosmo=cosmo)

    def get_flux(wave, time): # make sure get_flux exists.

        wmin, wmax = model.minwave(), model.maxwave()
        wave = np.atleast_1d(wave)
        sel = (wave > wmin) & (wave < wmax)
        flux = np.zeros_like(wave)
        flux[sel] = model.flux(time, wave[sel])

        return flux

    model.get_flux = get_flux  # Monkey patching
    return model

# ============== #
#                #
#   Models       #
#                #
# ============== #
# explicit here the parameters to enable mutable_parameters parsing
def get_saltmodel_flux(lbda, phase,
                        abmag=None, # extra 
                        redshift=0.1,
                        MBmax=-19.3, source="salt2-extended", cosmo=cosmology,
                        x1=0, c=0, alpha=-0.14, beta=3.15):
    """ """
    if abmag is not None:
        if redshift is not None:
            warnings.warn(f"abmag and redshift are set, redshift is ignored and derived from abmag.")
            
        redshift = obsmag_to_redshift(abmag, MBmax, cosmo=cosmo)
        
    model = get_saltmodel(redshift=redshift, MBmax=MBmax,
                          source=source, cosmo=cosmo,
                          x1=x1, c=c, alpha=alpha, beta=beta)

    return model.get_flux(lbda, phase)


def get_twins_embedding_flux(lbda, phase, redshift=0.05,
                             magnitude=0., color=0., coordinates=(0., 0., 0.), cosmo=cosmology):
    flux, flux_error = twins_embedding_model.evaluate(phase, magnitude, color, list(coordinates))
    wl_obs = twins_embedding_model.wave * (1. + redshift)
    dist_ratio = cosmo.luminosity_distance(0.05).value / cosmo.luminosity_distance(redshift).value
    cosmo_k_corr = 1.05 / (1. + redshift)
    flux_obs = flux * 1e-15 * dist_ratio ** 2. * cosmo_k_corr
    return np.interp(lbda, wl_obs, flux_obs, left=np.nan, right=np.nan)


class PointSource (SceneElement):
    
    """ A SceneElement with a position """
    def __init__(self, model_func, position, lbda=None, meta={}):
        """ """
        self._position = position
        meta = meta.copy()
        if "position" not in meta:
            meta["position"] = position

        super().__init__(model_func=model_func, lbda=lbda, meta=meta)
    
    @classmethod
    def from_config(cls, config):
        """ Generate supernova from point-source config.

        Parameters
        ----------
        config: dict
            configuration dictionary

        """
        position = config.get("position", (0, 0))
        model_func = config.get("model_func", None)
        # look for one
        if model_func is None: 
            if "source" in config:
                model_func = source_to_modelfunc(config["source"])
            else:
                raise ValueError("no model_func not source in the config. one is needed.")

        
        return cls(model_func=model_func, position=position, meta=config.copy())
        
    # ========== #
    #  Getter    #
    # ========== #
    def get_spectrum(self, lbda=None, phase=None, restframe=False):
        """ get the spectrum at the given  phase 

        Parameters
        ----------
        phase: float
            phase with respect to maximum light

        lbda: array
            wavelength of definition in Angstrom

        restframe: bool
            is the input phase restframe or obs-frame ?

        Returns
        -------
        lbda, flux
            lbda (see input units)
            flux is in erg/s/cm^2/A | see self.model
        
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        if phase is None:
            phase = self.phase
            
        elif restframe and phase !=0:
            if (z:=self.redshift) is None:
                raise ValueError("no known redshift for the target. Cannot use restrame")
            
            phase = phase/(1+z)
        
        # Actual flux
        model_kwargs = self._parse_model_kwargs_()
        flux = self.model_func(lbda, phase, **model_kwargs) # compute spectrum
        return lbda, flux
        
    # ================ #
    #   Properties     #
    # ================ #
    @property
    def redshift(self):
        """ """
        return self.meta.get("redshift", None)
                
    @property
    def position(self):
        """ """
        return self._position

    @property
    def phase(self):
        """ """
        return self.meta.get("phase", None)

    @property
    def mutable_parameters(self):
        """ """
        return self._model_mutables + ["position"]
