""" Scene point source """
import inspect
import numpy as np

from astropy.cosmology import Planck18 as cosmology


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

def get_pointsource_from_config(config):
    """ parse the pointsource name from the config and 
    returns the corresponding object
    
    """
    name = config.get("name", None)
    if name is None:
        raise ValueError("no 'name' entry in the config. On needed to automatically parse the pointsource.")
    
    if name in ["SN", "SNIa", "SN Ia"]:
        target = Supernovae.from_config(config)
    else:
        raise NotImplementedError(f"pointsource {name} not implemented")

    return target


class PointSource:
    """
    """
    def __init__(self, position=None, lbda=None, meta={}):
        """ """
        self._position = position
        meta = meta.copy()
        if "position" not in meta:
            meta["position"] = position

        self._meta = meta.copy()
        self._meta_in = meta.copy() # stored
        self._lbda = lbda

    def __str__(self):
        import pprint
        meta = pprint.pformat(self.meta, sort_dicts=False)
        return meta

    def __repr__(self):
        return self.__str__()
    
    def get_spectrum(self, lbda=None):
        """ """
        raise NotImplementedError("PointSource object must define get_spectrum")


    def _update_pointsource(self, **kwargs):
        """ """
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)
            
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def position(self):
        """ """
        return self._position

    @property
    def lbda(self):
        """ wavelegnth of definition [A] """
        return self._lbda

    @property
    def meta(self):
        """ meta parameters of the object, if any """
        return self._meta

    @property
    def _pointsource_mutable(self):
        """ """
        return ["position", "lbda", "abmag"]

# ================= #
#                   #
# Type Ia Superovae #
#                   #
# ================= #
def get_snia_model(redshift, source="salt2-extended", cosmo=cosmology, **kwargs):
    """ """
    if "salt" in source:
        model = get_saltmodel(redshift=redshift, source=source, cosmo=cosmo, **kwargs)
        
    else:
        raise NotImplementedError(f"only salt model sources have been implemented. {source} given")

    return model

def _get_mutable_parameters_(source):
    """ """
    from ..utils import inspect_func
    if "salt" in source:
        func = get_saltmodel
    else:
        raise NotImplementedError(f"only salt model sources have been implemented. {source} given")

    names, kwargs = inspect_func(func)
    return names, kwargs

# = Actual sources
    
def get_saltmodel(redshift, MBmax=-19.3, source="salt2-extended", cosmo=cosmology,
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
    model.set(c=c, x1=x1)              # SALT2 parameters

    eff_mbmax = MBmax + (x1*alpha + c*beta)  # Tripp relation (no env bias)
    model.set(z=redshift)
    # set effective peak magnitude
    model.set_source_peakabsmag(eff_mbmax,
                                "bessellb", "AB", cosmo=cosmo)

    def get_flux(time, wave): # make sure get_flux exists.

        wmin, wmax = model.minwave(), model.maxwave()
        wave = np.atleast_1d(wave)
        sel = (wave > wmin) & (wave < wmax)
        flux = np.zeros_like(wave)
        flux[sel] = model.flux(time, wave[sel])

        return flux

    model.get_flux = get_flux  # Monkey patching
    return model

def obsmag_to_redshift(mag, magabs=-19.3,
                       redshift_scan="0.001:3:0.01", cosmo=cosmology):
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


# = PointSource with phase dependencies = #
class Transient( PointSource ):
    """ """
    @property
    def _pointsource_mutable(self):
        """ """
        return list(super()._pointsource_mutable) + ["phase"]

    @property
    def phase(self):
        """ """
        if not hasattr(self, "_phase"):
            self._phase = 0
            
        return self._phase
    
class Supernovae( Transient ):
    """ """
    
    def __init__(self, model, position=None, lbda=None, flux=None, meta={}):
        """ """
        self._model = model
        super().__init__(position=position, lbda=lbda, meta=meta)
    
    @classmethod
    def from_config(cls, config):
        """ Generate supernova from point-source config.

        Parameters
        ----------
        config: dict
            configuration dictionary

        """
        position = config.get("position", (0, 0))
        model, fullconfig = cls._config_to_model(config)
        return cls(model=model, position=position, meta=fullconfig.copy())

    @classmethod
    def _config_to_model(cls, config):
        """ """
        config = config.copy()
        source = config.get("source", "salt2-extended")
        mutable_params, defaults = _get_mutable_parameters_(source)
        param = {k: v for k in mutable_params if (v := config.get(k, None)) is not None}
        
        return get_snia_model(**param), defaults | config
        
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
            if (z:=self.meta.get("redshift", None)) is None:
                raise ValueError("no known redshift for the target. Cannot use restrame")
            
            phase = phase/(1+z)
        
        # Actual flux
        flux = self.model.get_flux(phase, lbda)  # compute spectrum
        return lbda, flux

    def update(self, **kwargs):
        """ """
        model_update = {}
        pointsource_update = {}
        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue
            
            if v is None:        # Skip
                continue

            # extra cases
            if k == "abmag":
                k = "redshift"
                v = obsmag_to_redshift(v, self.meta["MBmax"])
                
            # updates
            if k in self._model_mutables:
                model_update[k] = v
                
            if k in self._pointsource_mutable:
                pointsource_update[k] = v

        if len(model_update)>0:
            target_model = self._meta_in | model_update
            self._model, model_update = self._config_to_model(target_model)
            
        self._update_pointsource(**pointsource_update)
        self._meta = self._meta_in | model_update | pointsource_update
        
    # ================ #
    #   Properties     #
    # ================ #
    @property
    def model(self):
        """ model that contains the get_flux() method"""
        return self._model

    @property
    def mutable_parameters(self):
        """ """
        extra = ["abmag"] # see update()
        model_mutables = self._model_mutables
        pointsource_mutables = self._pointsource_mutable
        return list(pointsource_mutables) + list(model_mutables) + extra
    
    @property
    def _model_mutables(self):
        """ """
        source = self.meta.get("source", self.model.source.name)
        mutable, defaults = _get_mutable_parameters_(source)
        return mutable

    @property
    def redshift(self):
        """ """
        return self.meta.get("redshift", self.model.get("z"))
