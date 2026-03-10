import warnings

from ..utils import inspect_func

class SceneElement:
    """Base class for scene elements."""
    
    def __init__(self, model_func, lbda=None, meta={}):
        """Initialize the SceneElement.

        Parameters
        ----------
        model_func : callable
            Function that returns the spectrum of the element.
            It must take `lbda` as first argument.
        lbda : array_like, optional
            Wavelength array in Angstrom. Default is None.
        meta : dict, optional
            Dictionary of parameters for the `model_func`.
            Default is {}.
        """
        self._model_func = model_func
        self._meta = meta.copy()
        self._meta_in = meta.copy()
        self._lbda = lbda
        
    @classmethod
    def from_config(cls, config):
        """Build the class from a configuration file.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.

        Returns
        -------
        SceneElement
            An instance of the class.
        """
        raise NotImplementedError("No from_config implemented")
    
    # ================ #
    #   Methods        #
    # ================ #                
    def update(self, reset_others=False, **kwargs):
        """Update mutable parameters.

        Parameters
        ----------
        reset_others : bool, optional
            If True, parameters not given in `kwargs` are reset to their
            initial values. Default is False.
        **kwargs
            Parameters to update. See `self.mutable_parameters`.
        """
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
        """Get the spectrum of the scene element.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, `self.lbda` is used.
            Default is None.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        flux : array_like
            Flux of the scene element. Unit depends on the model_func.
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
        """Get default `model_func` parameters updated by current meta.

        Returns
        -------
        dict
            Dictionary of parameters for `model_func`.
        """
        # get default
        list_parameters, default_param = inspect_func(self.model_func)
        # remove the first param which should be lbda/wave
        # see how get_spectrum() uses _parse_model_kwargs_()
        list_parameters = list_parameters[1:] 

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
        """Function that returns the spectrum of the element."""
        return self._model_func
    
    @property
    def meta(self):
        """Meta parameters of the object."""
        return self._meta

    @property
    def lbda(self):
        """Wavelength of definition in Angstrom."""
        return self._lbda

    @property
    def mutable_parameters(self):
        """List of mutable parameters."""
        # no extra so far
        return self._model_mutables

    @property
    def _model_mutables(self):
        """List of model mutable parameters."""
        return list(inspect_func(self.model_func)[0])
