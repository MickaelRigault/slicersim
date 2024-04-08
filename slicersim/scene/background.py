import numpy as np

from ..utils import inspect_func
        

def get_background_model_func(name):
    """ top level method to parse name into model_func """
    if "zodi" in name:
        model_func = zodiacal_spectrum
    else:
        raise NotImplementedError(f"background: {name} is not implemented")
    
    return model_func

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


# ================ #
#                  #
#     Class        #
#                  #
# ================ #    
class Background:
    
    def __init__(self, model_func, lbda=None, meta={}):
        """ """
        self._model_func = model_func
        self._meta = meta.copy()
        self._meta_in = meta.copy()
        self._lbda = lbda
        
    def __str__(self):
        import pprint
        meta = pprint.pformat(self.meta, sort_dicts=False)
        return meta

    def __repr__(self):
        return self.__str__

    def _update_background(self, **kwargs):
        """ """
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)

    @classmethod
    def from_config(cls, config):
        """ """
        name = config.get("name", None)
        if name is None:
            raise ValueError("no name in config. One needed")

        model_func = get_background_model_func(name)
        _, default = inspect_func(model_func)
        return cls(model_func, meta= default | config)

    # ================ #
    #   Methods        #
    # ================ #                
    def update(self, **kwargs):
        """ change any mutable_parameters (see self.mutable_parameters) """
        model_update = {}
        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue
            
            if v is None:        # Skip
                continue

            # special case
            if k == "lbda":
                self._lbda = v
            
            # updates
            else:
                model_update[k] = v

        self._meta = self._meta_in | model_update
        # in background, the meta is called on the get_spectrum()

    def get_spectrum(self, lbda=None):
        """ get the spectrum 

        Parameters
        ----------
        lbda: array
            wavelength of definition in Angstrom

        Returns
        -------
        lbda, flux
            lbda (input unit) 
            flux (see self.model_func)
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")
        
        # get the kwargs used for the model.
        _, model_kwargs = inspect_func(self.model_func) # default
        func_kwarg_names = model_kwargs.keys()
        model_parameters = {k: self.meta[k] for k in func_kwarg_names if k in self.meta} # to be updated
        flux = self.model_func(lbda, **model_parameters)  # compute spectrum
        
        return lbda, flux
    
    # ================ #
    #   Properties     #
    # ================ #        
    @property
    def model_func(self):
        """ """
        return self._model_func
    
    @property
    def meta(self):
        """ meta parameters of the object, if any """
        return self._meta

    @property
    def lbda(self):
        """ wavelegnth of definition [A] """
        return self._lbda

    @property
    def mutable_parameters(self):
        """ list of mutable parameters """
        # no extra so far
        return self._model_mutables

    @property
    def _model_mutables(self):
        """ list of model mutable parameters"""
        return list(inspect_func(self.model_func)[0])
