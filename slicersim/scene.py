"""
Scene module.

.. autosummary::

   cosmology
   zodiacal_spectrum
   sn_model_salt2
   Scene

.. Warning:: the astrophysical scene does not include the thermal
   contribution from mirror (if any).
"""

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"

import warnings
import pprint

import numpy as np

import astropy.cosmology
import sncosmo

from .iotools import read_calspec, read_xshooter, chromatic_interpolator
from .utils import recover_bin_edges

cosmology = astropy.cosmology.Planck18  #: Default cosmology


# ================= #
#                   #
# Type Ia Superovae #
#                   #
# ================= #
def sn_spectrum(z, lbda, phase=0, source="salt2-extended",
                      MBmax=-19.3,  cosmo=cosmology,
                      **kwargs):
    """ """
    if "salt2" in source:
        target = sn_model_salt2(z, MBmax=MBmax, source=source, cosmo=cosmo,
                                    **kwargs)
        target_flux = target.flux_extended(time=phase, wave=lbda)
    else:
        raise NotImplementedError("only 'salt2' SN Ia source implemented. {source} given")
    
    return target_flux

def sn_model_salt2(z, MBmax=-19.3, source="salt2-extended", cosmo=cosmology,
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

    .. plot::

       import matplotlib.pyplot as plt
       from mlaperf.spectrograph import Spectrograph
       from mlaperf.scene import sn_model_salt2, cosmology
       lbda, _ = Spectrograph.lbda_from_respow([4000, 17000], 75)
       z, x1, c = 0.5, 0.1, 0.1
       alpha, beta = -0.14, 3.15
       MBmax = -19.3
       sn_model = sn_model_salt2(z, MBmax=MBmax, cosmo=cosmology, x1=0, c=0)
       fig, ax = plt.subplots()
       for z in (0.1, 0.5, 1, 1.5, 2):
           sn_model.set(z=z)
           sn_model.set_source_peakabsmag(MBmax,
                                          "bessellb", "AB", cosmo=cosmology)
           l, = ax.plot(lbda / 1e4, sn_model.flux_extended(0, lbda),
                        ds='steps-mid', label=f"{z=:.1f}")
           sn_model.set(x1=x1, c=c)
           sn_model.set_source_peakabsmag(MBmax + alpha*x1 + beta*c,
                                          "bessellb", "AB", cosmo=cosmology)
           l, = ax.plot(lbda / 1e4, sn_model.flux_extended(0, lbda),
                        ds='steps-mid', ls='--', c=l.get_color())
       ax.set(xlabel="Wavelength [µm]",
              ylabel="Flux [fλ]", yscale='log',
              title=f"{sn_model._source.name} v{sn_model._source.version}: " \
                    f"{MBmax=:.1f} (--: x1=c=0.1)")
       ax.legend()
    """

    model = sncosmo.Model(source=source)
    model.set(z=z, c=c, x1=x1)              # SALT2 parameters

    effMBmax = MBmax + (x1*alpha + c*beta)  # Tripp relation (no env bias)
    model.set_source_peakabsmag(effMBmax,
                                "bessellb", "AB", cosmo=cosmo)

    def flux_extended(time, wave):

        wmin, wmax = model.minwave(), model.maxwave()
        wave = np.atleast_1d(wave)
        sel = (wave > wmin) & (wave < wmax)
        flux = np.zeros_like(wave)
        flux[sel] = model.flux(time, wave[sel])

        return flux

    model.flux_extended = flux_extended  # Monkey patching

    return model

# ================= #
#                   #
#    Background     #
#                   #
# ================= #
def zodiacal_spectrum(lbda, scale=1, model="Aldering01.BB5800"):
    """
    Sky background (zodi) spectrum at the North Ecliptic Pole.

    Two models are implemented from Aldering 2001 (LBNL-51157):

    * Black-Body approximation (`Aldering01.BB5800`)
    * Truncated Power Law (`Aldering01.TPL`)

    :param lbda: wavelength [Å]
    :param float scale: scaling factor
    :param str model: model name
    :return: flux in erg/s/cm²/Å/arcsec²

    See also :ads:`Scaramella+22 <2022A&A...662A.112E>`.

    .. plot::

       import matplotlib.pyplot as plt
       from mlaperf.spectrograph import Spectrograph
       from mlaperf.scene import zodiacal_spectrum
       lbda, _ = Spectrograph.lbda_from_respow([4000, 17000], 75)
       fig, ax = plt.subplots()
       ax.plot(lbda / 1e4,
               np.log10(zodiacal_spectrum(lbda, model="Aldering01.BB5800")),
               ds='steps-mid', label="BB5800")
       ax.plot(lbda / 1e4,
               np.log10(zodiacal_spectrum(lbda, model="Aldering01.TPL")),
               ds='steps-mid', label="TPL")
       ax.legend()
       ax.set(xlabel="Wavelength [µm]",
              ylabel="log(Flux [fλ/arcsec²])",
              title="Zodiacal spectrum (Aldering01)")
    """

    if model == "Aldering01.BB5800":
        # h*c/(k*5800 K) = 24806.498 Å
        zodi = 1e-17 * (lbda * 1e-4)**-4.64 / (np.exp(24806.498 / lbda) - 1)
    elif model == "Aldering01.TPL":
        zodi = 10**(-17.755 - 0.740*(lbda * 1e-4 - 0.61))
        zodi[lbda < 6100] = 10**(-17.755)
    else:
        raise NotImplementedError(f"Unknow model {model!r}.")

    return scale * zodi         # Account for scale factor
    
# ================= #
#                   #
#    Internal       #
#                   #
# ================= #

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


# ================= #
#                   #
#     Class         #
#                   #
# ================= #
class Scene:
    """
    Scene description.

    A scene will potentially contain three components:

    * a target point source (an SN, possibly a standard star),
    * a uniform background (e.g. the zodiacal background),
    * a structured background (e.g. the host galaxy).

    .. Warning:: the structured `host` is not implemented, and the scene
       does not include the thermal background from telescope.
    """

    #: Mutable parameters (dict by origin)
    scene_mutable_parameters = {
        "target": ("position",
                   "redshift", "phase", "MBmax", "c", "x1",  # supernova
                   "ABmag"),                                 # std star
        "background": ("scale", "model"),
        "host": (),
    }
    #: Mutable parameters (concatenated list)
    mutable_parameters = (scene_mutable_parameters["target"] +
                          scene_mutable_parameters["background"] +
                          scene_mutable_parameters["host"])

    def __init__(self, target=None, background=None, host=None,
                 lbda=None, meta={}, **kwargs):
        """
        A scene is initialized from spectro-spatial components.

        :param target: target point source spectrum [erg/s/cm²/Å]
        :param background: uniform background spectrum [erg/s/cm²/Å/arcsec²]
        :param host: structured background cube [erg/s/cm²/Å/arcsec²]
        :param lbda: wavelength [Å]
        :param dict meta: component metadata, `{target:{}, ...}`
        :param dict kwargs: more component metadada
        :return: the simulated scene

        :seealso from_config: load the scene from a configuration file.
        """

        self.target = target          #: Target point-source spectrum [erg/s/cm²/Å]
        self.background = background  #: Background spectrum [erg/s/cm²/Å/arcsec²]
        self.host = host              #: Host (**not yet implemented**)
        self.lbda = lbda              #: Wavelength [Å]
        self.meta = {**meta, **kwargs}  #: Meta-parameters

    def __str__(self):

        return pprint.pformat(self.meta, sort_dicts=False)

    @classmethod
    def from_config(cls, config, lbda):
        """
        Initiate scene from config dictionary.

        :param dict config: configuration dictionary containing (or not)
             `{"point_source": {}, "background": {}, "host":{}}`.  A missing
             component has no contribution to the scene.
        :param lbda: wavelength [Å]
        :return: scene instance
        """

        meta = {}
        target_config = config.get("point_source", {})
        background_config = config.get("background", {})
        host_config = config.get("host", {})

        if target_config:      # initialize point-source spectrum from config
            target_spec = cls.target_from_config(target_config, lbda)
            meta["target"] = target_config
        else:
            target_spec = None

        if background_config:  # initialize background spectrum from config
            background_spec = cls.background_from_config(background_config, lbda)
            meta["background"] = background_config
        else:
            background_spec = None

        if host_config:        # initialize host from config
            # raise NotImplementedError("Host component not implemented.")
            warnings.warn("Host component not implemented, ignored.")
            host_spec = None
        else:
            host_spec = None

        return cls(target=target_spec,
                   background=background_spec,
                   host=host_spec,
                   lbda=lbda, meta=meta)

    @staticmethod
    def target_from_config(config, lbda, verbose=True):
        """
        Generate point-source spectrum from point-source config.

        :param dict config: configuration dictionary
        :param array lbda: wavelength [Å]
        :return: spectrum
        """

        name = config.get("name")

        if name == "SN Ia":                # SN Ia spectrum from sncosmo
            target_spec = Scene.supernova_from_config(config, lbda)
        elif name == "StdStar":            # Std star spectrum from CalSpec
            target_spec = Scene.stdstar_from_config(config, lbda, verbose=verbose)
        elif name == "PN":                 # Planetary Nebula spectrum
            target_spec = Scene.pn_from_config(config, lbda, verbose=verbose)
        else:
            raise NotImplementedError(f"Unknown point-source {name!r}.")

        return target_spec                 # point source spectrum [erg/s/cm²/Å]

    @staticmethod
    def supernova_from_config(config, lbda):
        """
        Generate supernova spectrum from point-source config.

        :param dict config: configuration dictionary
        :param array lbda: wavelength [Å]
        :return: spectrum
        """

        # obs-mag given | pop will consume the parameter
        abmag = config.pop("ABmag", None)
        if abmag is None:
            redshift = float(config["redshift"])   # Redshift, mandatory if no ABmag
        else:
            redshift = obsmag_to_redshift(abmag, config["MBmax"])
            config["redshift"] = redshift

        phase = float(config.get("phase", 0))  # Phase [day]
        param  = { k: v
                   for k in ["MBmax", "source", "cosmo", "x1", "c"]
                   if (v := config.get(k, None)) is not None }
        model = sn_model_salt2(redshift, **param)       # sncosmo model
        target_spec = model.flux_extended(phase, lbda)  # compute spectrum

        return target_spec      # point source spectrum [erg/s/cm²/Å]

    @staticmethod
    def stdstar_from_config(config, lbda, k=3, verbose=True):
        """
        Generate stdstar spectrum from point-source config.

        The interpolation does not account for spectral resolution,
        but account for chromatic average over (approximate) pixel.

        :param dict config: configuration dictionary
        :param array lbda: bin central wavelengths [Å] (nlbda,)
        :param int k: interpolation order
        :param bool verbose: verbose flag
        :return: flux density array [flambda] (nlbda,)
        """

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
        """
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
        """

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

    @staticmethod
    def background_from_config(config, lbda):
        """
        Compute background spectrum from `scene.background` config.
        """

        name = config.get("name")
        if name == "zodi":      # zodiacal light
            param = { k: v
                      for k in ["model", "scale"]
                      if (v := config.get(k, None)) is not None }
            background_spec = zodiacal_spectrum(lbda, **param)
        else:
            raise NotImplementedError(f"Unknown background type {name!r}.")

        return background_spec  # background spectrum [erg/s/cm²/Å/arcsec²]

    @property
    def redshift(self):
        """
        Supernova redshift (from `self.meta`) or NaN.
        """

        return self.meta.get("target", {}).get("redshift", np.NaN)

    @property
    def phase(self):
        """
        Supernova phase (from `self.meta`) or NaN.
        """

        return self.meta.get("target", {}).get("phase", np.NaN)

    @property
    def ABmag(self):
        """
        Source AB-magn (from `self.meta`) or NaN.
        """

        return self.meta.get("target", {}).get("ABmag", np.NaN)

    @property
    def target_position(self):
        """
        Target position (from `self.meta`),
        """

        return self.meta.get("target", {}).get("position", [np.NaN, np.NaN])

    def has_component(self, component):
        """
        Tests if the component is present in scene.

        :param str component: component name (target, background, host)
        """

        return getattr(self, component, None) is not None

    def update_lbda(self, lbda, **kwargs):
        """
        Update chromatic components.

        :param lbda: wavelength [Å]
        """

        self.lbda = lbda
        # Update scene components
        self.target = self.target_from_config(
            self.meta.get("target"), self.lbda, **kwargs)
        self.background = self.background_from_config(
            self.meta.get("background"), self.lbda)
        #self.host = self.host_from_config(self.meta.get("host"), self.lbda)

    def update_target(self, ABmag=None, position=None,
                      redshift=None, phase=None, MBmax=None, x1=None, c=None,
                      verbose=False):
        """
        Update the target properties.

        :param float redshift: SN redshift
        :param float phase: SN phase [days wrt max]
        :param float MBmax: SN peak absolute Bessell-B AB-mag
        :param float x1: SN stretch
        :param float c: SN color
        :param float ABmag: stdstar AB-magn (in predefined band)
        :param tuple position: point source position in MLA [spx]
        :param bool verbose: verbose flag
        """

        if not self.has_component("target"):
            raise AttributeError("No target component in the scene.")

        new_prop = { k: v
                     for k, v in locals().items()  # new properties
                     if k != "self" and v is not None }

        if not new_prop:        # Nothing to do
            return

        newconfig = {**self.meta.get("target"), **new_prop}
        self.target = self.target_from_config(newconfig, self.lbda, verbose=verbose)
        self.meta["target"] = newconfig

    def update_background(self, scale=None, model=None):
        """
        Update the background properties.

        :param float scale: scale of the background spectrum
        :param str model: model name
        """

        if not self.has_component("background"):
            raise AttributeError("No background component in the scene.")

        new_prop = { k: v
                     for k, v in locals().items()  # new properties
                     if k != "self" and v is not None }

        if not new_prop:        # Nothing to do
            return

        newconfig = {**self.meta.get("background"), **new_prop}
        self.background = self.background_from_config(newconfig, self.lbda)
        self.meta["background"] = newconfig

    def update(self, **kwargs):
        """
        Update any mutable attribute of the scene.
        """

        updates_target = {}
        updates_host = {}
        updates_background = {}
        lbda = kwargs.pop("lbda", None)  # removes it from the kwargs

        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            if v is None:        # Skip
                continue
            if k in self.scene_mutable_parameters["target"]:
                updates_target[k] = v
            elif k in self.scene_mutable_parameters["background"]:
                updates_background[k] = v
            elif k in self.scene_mutable_parameters["host"]:
                updates_host[k] = v
            else:
                raise ValueError(f"Unknown scene parameter {k!r}.")

        self.update_target(**updates_target)
        self.update_background(**updates_background)
        # self.update_host(**updates_host)  # Not implemented

        # Update the lbda *after* all meta parameters have been updated.
        if lbda is not None:
            self.update_lbda(lbda)  # updates all chromatic components

    def get_component_spectra(self, fillna=0):
        """
        Get the stack of all scene component spectra (target, host, background).

        :param float fillna: if a spectrum is not defined, an array of fillna
                             will be set.
        """

        spectra = [ s_ if s_ is not None else np.full_like(self.lbda, fillna)
                    for s_ in [self.target, self.host, self.background]]

        return np.stack(spectra)

    def plot_background(self, ax=None, in_log=True, **kwargs):
        """
        Background spectrum plot.

        :param pyplot.Axes ax: plot axes
        :param bool in_log: plot the spectrum in log-scale
        :param kwargs: propagated to to `ax.plot()`
        :return: plot axes
        """

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

        flux = self.background  # background spectrum [erg/s/cm²/Å/arcsec²]
        if in_log:
            flux = np.log10(flux)
            ylabel = "log(Flux [fλ/arcsec²])"
        else:
            ylabel = "Flux [fλ/arcsec²]"

        default = dict()#ds='steps-mid')
        model_name = self.meta.get('background').get('model')
        ax.plot(self.lbda / 1e4, flux, **{**default, **kwargs})
        ax.set(xlabel="Wavelength [µm]",
               ylabel=ylabel,
               title=f"Background spectrum ({model_name})")

        return ax

    def plot_target(self, ax=None, in_log=True, **kwargs):
        """
        Point-source spectrum plot.

        :param pyplot.Axes ax: plot axes
        :param bool in_log: plot the spectrum in log-scale
        :param kwargs: propagated to to `ax.plot()`
        :return: plot axes
        """

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

        flux = self.target  # point-source spectrum [erg/s/cm²/Å]
        if in_log:
            flux = np.log10(flux)
            ylabel = "log(Flux [fλ])"
        else:
            ylabel = "Flux [fλ]"

        if self.meta.get("target", None) is None:
            title = None
        else:
            title = (f"{self.meta['target']['name']} "
                     f"({self.meta['target']['source']})")
            
            
        default = dict()#ds='steps-mid')
        ax.plot(self.lbda / 1e4, flux, **{**default, **kwargs})
        ax.set(xlabel="Wavelength [µm]",
               ylabel=ylabel,
               title=title)

        return ax


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mlaperf.spectrograph import Spectrograph

    print("Default cosmology:", cosmology)

    lbda, _ = Spectrograph.lbda_from_respow([4000, 17000], 75)

    fig, ax = plt.subplots()
    ax.plot(lbda / 1e4,
            np.log10(zodiacal_spectrum(lbda, model="Aldering01.BB5800")),
            ds='steps-mid', label="BB5800")
    ax.plot(lbda / 1e4,
            np.log10(zodiacal_spectrum(lbda, model="Aldering01.TPL")),
            ds='steps-mid', label="TPL")
    ax.legend()
    ax.set(xlabel="Wavelength [µm]",
           ylabel="log(Flux [fλ/arcsec²])",
           title="Zodiacal spectrum (Aldering01)")

    z = 0.5
    MBmax = -19.3
    sn_model = sn_model_salt2(z, MBmax=MBmax)
    print(sn_model)
    print(f"At {z=:.2f}: {sn_model.minwave():.0f}, {sn_model.maxwave():.0f} Å")

    fig, ax = plt.subplots()
    for z in (0.1, 0.5, 1, 1.5, 2):
        sn_model.set(z=z)
        sn_model.set_source_peakabsmag(MBmax, "bessellb", "AB", cosmo=cosmology)
        ax.plot(lbda / 1e4, sn_model.flux_extended(0, lbda),
                ds='steps-mid', label=f"{z=:.1f}")
    ax.set(xlabel="Wavelength [µm]",
           ylabel="Flux [fλ]", yscale='log',
           title=f"{sn_model._source.name} v{sn_model._source.version}: {MBmax=:.1f}")
    ax.legend()

    plt.show()
