""" Scene point source """
import inspect
import numpy as np

from astropy.cosmology import Planck18 as cosmology
from astropy import units
from .base import SceneElement

try:
    from twins_embedding import TwinsEmbeddingModel
    twins_embedding_model = TwinsEmbeddingModel()
except ImportError:
    twins_embedding_model = None



def get_snia_pointsource(model="salt", **kwargs):
    """ generic top level configuration setting 

    Parameters
    ----------
    model: str
        name of the modelling to use
        - salt (or any sncosmo salt source name).
        - twin

    **kwargs update the pointsource
    """
    generic = {'name': 'SN Ia', 
               'redshift': 1.5, 
               'phase': 0, 'position': [1, 0.5]}
    
    if "salt" in model.lower():
        if model == "salt":
            model = "salt2-extended"
        
        pointsource = {'source': model, 'MBmax': -19.3, 'c': 0., 'x1': 0.}
        
    elif "twin" in model.lower():
        pointsource = {'magnitude':0., 'color':0., 'coordinates':(0., 0., 0.)}

    return generic | pointsource | kwargs


    
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

    elif source == "blackbody":
        model_func = get_blackbody_flux
        
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
_FLAMBDA_units = units.erg / (units.cm**2 * units.s * units.AA)
def get_blackbody_flux(lbda, temperature, mag, 
                       band="sdssr", magsys="ab"):
    """ 
    Parameters
    ----------
    temperature: float
        temperature of the black body in Kelvin
    
    mag: float
        target magnitude integrated over the 'band'

    band: string
        name of the band (from sncosmo)

    magsys: string
        name of the magnitude system (see sncosmo.)
    """
    from sncosmo import Spectrum
    from astropy.modeling.models import BlackBody
    
    if not hasattr(lbda, 'unit'): # assumed Angstrom
        lbda = units.Quantity(lbda, units.AA)
        
    blackbody = BlackBody(temperature=temperature*units.K)
    flux_nu = blackbody(lbda) * units.sr # rm sr in unit

    # flux with whatever magnitude.
    flux = flux_nu.to(_FLAMBDA_units, units.spectral_density(lbda))

    # let's get it to the target mag using sncosmo
    spec_in = Spectrum(wave=lbda.value, flux=flux.value)
    # the "whatever mag"
    native_mag = spec_in.bandmag(band, magsys)
    # conver the flux to the good amplitude
    fluxcoef_for_target_mag = 10**(-0.4*(mag-native_mag))
    
    return flux.value * fluxcoef_for_target_mag # numpy array
    
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
                             magnitude=0., color=0., coordinates=(0., 0., 0.),
                             cosmo=cosmology, ref_redshift=0.05,
                             norm=1e-15):
    """ get a snia spectrum assuming the Twin Embedding model

    Parameters
    ----------
    lbda: array
        wavelength [A]

    phase: float
        rest-frame phase [days]

    redshift: float
        reshift of the simulated flux

    magnitude: float
        magnitude offset for the twin model (dmag)
        
    color: float
        color term of the twin model (Av)

    coordinates: list
        embedding parameters for the twin model (xi)

    cosmo: astropy.Cosmology
        cosmology to be used to redshift the simulated target

    ref_redshift: float
        reference redshift

    norm: float
        normalisation of the output flux.
    
    Returns
    -------
    Array
    """
    if twins_embedding_model is None:
        raise ImportError("The 'twins_embedding' module is not available.")
    
    flux, flux_error = twins_embedding_model.evaluate(phase, magnitude, color, list(coordinates))
    wl_obs = twins_embedding_model.wave * (1. + redshift)
    dist_ratio = cosmo.luminosity_distance(ref_redshift).value / cosmo.luminosity_distance(redshift).value
    cosmo_k_corr = (1. + ref_redshift) / (1. + redshift) 
    flux_obs = flux * norm * dist_ratio ** 2. * cosmo_k_corr
    
    return np.interp(lbda, wl_obs, flux_obs, left=np.nan, right=np.nan)


# ============= #
#               #
#  PointSource  #
#               #
# ============= #
class PointSource( SceneElement ):
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
            configuration dictionary.
            Config must contain:
            - position: (0, 0)
            - model_func [func] or source [string]
        """
        position = config.get("position", (0, 0))
        model_func = config.get("model_func", None)
        # look for one
        if model_func is None: 
            if "source" in config:
                source_ = config["source"]
                if type(source_) in [str, np.str_]:
                    model_func = source_to_modelfunc(config["source"])
                else: # assume it's a spectrum
                    mag = config.get("mag", None)
                    band = config.get("band", None)
                    lbda_, flux_ = source_
                    return cls.from_spectrum(lbda_, flux_,
                                              mag=mag, band="bessellb",
                                             position=position, meta=config.copy() ) 
            else:
                raise ValueError("neither 'model_func' nor 'source' in the config. One is needed.")

        return cls(model_func=model_func, position=position,
                       meta=config.copy())

    @classmethod
    def from_spectrum(cls, lbda_, flux_, mag=20, band="bessellb",
                     position=(0,0), lbda=None, meta={}):
        """ Generate a pointsource from any spectrum
        
        Parameters
        ----------
        lbda_: Array
            wavelength [A] of the reference spectrum

        flux_: Array
            flux of the reference spectrum.
            (the unit will be given by the magnitude)

        mag: float
            default magnitude of the target (see band)

        band: str
            name of the band (must be known by sncosmo)

        position: list
            x and y location of the point source within the IFU

        lbda: None or Array
            default wavelenght at which the spectrum may be gerenated
            (ignore if unsure)

        meta: dict
            meta information carried by the object.

        Returns
        -------
        PointSource
        """
        from sncosmo import Spectrum
        meta["mag"] = mag
        meta["band"] = band
        meta["flux_ref"] = flux_
        meta["lbda_ref"] = lbda_        
        this = cls(None, position=position, lbda=lbda, meta=meta)

        # internal function 
        def _internal_get_flux(lbda, mag, band):
            """ """
            if mag is None:
                flux_ratio = 1
            else:
                in_mag = Spectrum(meta["lbda_ref"], meta["flux_ref"]
                                      ).bandmag(band, "ab")
                flux_ratio = 10**( -0.4*(mag - in_mag) )
                
            flux_ = np.interp(lbda, meta["lbda_ref"], meta["flux_ref"],
                                  left=np.nan, right=np.nan)
            return flux_ * flux_ratio
        
        this._model_func = _internal_get_flux
        return this
    
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
        if phase is not None:
            model_kwargs |= {"phase":phase} # allow
            
        flux = self.model_func(lbda, **model_kwargs) # compute spectrum
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
