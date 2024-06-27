import warnings
import pprint

import numpy as np

from ..utils import inspect_func

class SceneElement:
    
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

    @classmethod
    def from_config(cls, config):
        """ """
        raise NotImplementedError("No from_config implemented")
    
    # ================ #
    #   Methods        #
    # ================ #                
    def update(self, reset_others=False, **kwargs):
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

        if reset_others:
            self._meta = self._meta_in | model_update
        else:
            self._meta = self._meta | model_update

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
        model_kwargs = self._parse_model_kwargs_() # default updated by current meta
        flux = self.model_func(lbda, **model_kwargs)  # compute spectrum
        return lbda, flux

    def _parse_model_kwargs_(self):
        """ get default model_func parameters updated by current meta """
        # get default
        list_parameters, default_param = inspect_func(self.model_func)
        list_parameters.remove("lbda") # this should not be there.
        # names of parameters
        kwargs_names = default_param.keys()
        # get meta input if any        
        model_parameters = {k: self.meta[k] for k in list_parameters
                                if k in self.meta}
        # default updated by current meta 
        return default_param | model_parameters
    
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
