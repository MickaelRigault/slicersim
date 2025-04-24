import warnings
import pprint

import numpy as np

from ..iotools import read_calspec, read_xshooter, chromatic_interpolator
from ..utils import recover_bin_edges


# ================= #
#                   #
#     Class         #
#                   #
# ================= #
class Scene:
    """
    Scene description.

    A scene will potentially contain three elements:

    * a target point source (an SN, possibly a standard star),
    * a uniform background (e.g. the zodiacal background),
    * a structured background (e.g. the host galaxy).

    .. Warning:: the structured `host` is not implemented, and the scene
       does not include the thermal background from telescope.
    """

    def __init__(self, target=None, background=None, host=None,
                 lbda=None, meta={}):
        """
        A scene is initialized from spectro-spatial elements.

        :param target: target point source spectrum [erg/s/cm²/Å]
        :param background: uniform background spectrum [erg/s/cm²/Å/arcsec²]
        :param host: structured background cube [erg/s/cm²/Å/arcsec²]
        :param lbda: wavelength [Å]
        :param dict meta: element metadata, `{target:{}, ...}`
        :param dict kwargs: more element metadada
        :return: the simulated scene

        :seealso from_config: load the scene from a configuration file.
        """

        self._target = target          #: Target point-source spectrum [erg/s/cm²/Å]
        self._background = background  #: Background spectrum [erg/s/cm²/Å/arcsec²]
        self._host = host              #: Host (**not yet implemented**)
        self._lbda = lbda              #: Wavelength [Å]
        self._scene_meta = meta.copy()#: Meta-parameters

    def __str__(self):

        return pprint.pformat(self.meta, sort_dicts=False)

    @classmethod
    def from_config(cls, config, lbda=None):
        """
        Initiate scene from config dictionary.

        :param dict config: configuration dictionary containing (or not)
             `{"point_source": {}, "background": {}, "host":{}}`.  A missing
             element has no contribution to the scene.
        :param lbda: wavelength [Å]
        :return: scene instance
        """
        from . import pointsource, background
        
        config = config.copy()
        target_config = config.pop("point_source", {}) # rename target ?
        background_config = config.pop("background", {})
        host_config = config.pop("host", {})

        if target_config:      # initialize point-source spectrum from config
            target = pointsource.PointSource.from_config(target_config)
        else:
            target = None

        if background_config:  # initialize background spectrum from config
            background = background.Background.from_config(background_config)
        else:
            background = None

        if host_config:        # initialize host from config
            # raise NotImplementedError("Host element not implemented.")
            warnings.warn("Host element not implemented, ignored.")
            host = None
        else:
            host = None

        return cls(target=target,
                   background=background,
                   host=host,
                   lbda=lbda,
                   meta=config) # config don't contains elements anymore.

  
    def has_element(self, element):
        """
        Tests if the element is present in scene.

        :param str element: element name (target, background, host)
        """

        return getattr(self, element, None) is not None


    def update(self, reset_others=False, **kwargs):
        """ change any mutable_parameters (see self.mutable_parameters) 
        
        for convinience, the update method respects the django '__' format, 
        such that, e.g. 'target__phase' is understood as 'target.phase'.
        This way, one can do:
        >>> self.update(target__phase = -1)
        """
        updates_target = {}
        updates_host = {}
        updates_background = {}
        
        for k, v in kwargs.items():
            k = k.replace("__", ".") # django like 
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            if v is None:        # Skip
                continue

            # special case
            if k == "lbda":
                self._lbda = v


            if k.startswith("target."):
                updates_target[k.replace("target.","")] = v
                
            elif k.startswith("background."):
                updates_background[k.replace("background.","")] = v
                
            elif k.startswith("host."):
                updates_host[k.replace("host.","")] = v

            else:
                raise ValueError(f"Unknown scene parameter {k!r}.")

        if self.target is not None:
            self.target.update(reset_others=reset_others, **updates_target)
            
        if self.background is not None:
            self.background.update(reset_others=reset_others, **updates_background)
            
        if self.host is not None:
            self.host.update(reset_others=reset_others, **updates_host)

    def get_element_spectrum(self, which, lbda=None, fillna=0, **kwargs):
        """ """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")
        
        element = getattr(self, which)
        if element is None:
            spec = np.full_like(lbda, fillna)
        else:
            _, spec = element.get_spectrum(lbda, **kwargs)
            
        return lbda, spec

        
    def get_stacked_spectra(self, lbda=None, fillna=0):
        """
        Get the stack of all scene element spectra (target, host, background).

        :param float fillna: if a spectrum is not defined, an array of fillna
                             will be set.
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        
        spectra = [s_.get_spectrum(lbda)[1] if s_ is not None else np.full_like(lbda, fillna)
                    for s_ in [self.target, self.host, self.background]]

        return lbda, np.stack(spectra)        

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

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def target(self):
        """ the scene point source """
        return self._target

    @property
    def background(self):
        """ the scene spatially flat background """
        return self._background

    @property
    def host(self):
        """ the scene structured background """
        return self._host
    
    @property
    def mutable_parameters(self):
        """ list of mutable parameters """
        # lbda is special. it is a shared parameters
        if self.target is not None:
            target_ = [f"target.{pname_}" for pname_ in self.target.mutable_parameters if pname_ != "lbda"]
        else:
            target_ = []
        
        if self.background is not None:
            background_ = [f"background.{pname_}" for pname_ in self.background.mutable_parameters if pname_ != "lbda"]
        else:
            background_ = []

        if self.host is not None:
            host_ = [f"host.{pname_}" for pname_ in self.host.mutable_parameters if pname_ != "lbda"]
        else:
            host_ = []

        return ["lbda"]+target_+background_+host_

    @property
    def meta(self):
        """ """
        return {"target": self.target.meta if self.target is not None else None,
                "background": self.background.meta if self.background is not None else None,
                "host": self.host.meta if self.host is not None else None
                } | self._scene_meta
                    

    
    @property
    def target_position(self):
        """
        Target position (from `self.meta`),
        """
        return self.target.position
    
