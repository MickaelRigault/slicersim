"""Scene: base module.

`SceneElement` is the base class of every scene component (point source,
background, host). It wraps a ``model_func(lbda, ...)`` returning a spectrum
(a datacube for the host), and exposes the parameters of that function as
mutable parameters, stored in the element ``meta``.
"""
import warnings

from ..utils import inspect_func
from copy import deepcopy

class SceneElement:
    """Base class for scene elements.

    A scene element wraps a model function ``model_func(lbda, ...)`` that
    returns the element spectrum. The parameters of ``model_func`` (but
    ``lbda``) are read from the element `meta` when the spectrum is computed,
    falling back to the function defaults, and can be changed with `update`.

    Subclasses may override `mutable_parameters` and ``_model_func`` (e.g.
    `~slicersim.scene.host.Host`, whose parameters are those of a spatial and
    a spectral function).

    Parameters
    ----------
    model_func : callable or None
        Function that returns the spectrum of the element.
        It must take the wavelength (``lbda``) as first argument.
        If None, ``_model_func`` is not set here: it must be set afterwards
        or defined by the subclass.
    lbda : array_like, optional
        Default wavelength array in Angstrom. Default is None.
    meta : dict, optional
        Parameters of ``model_func`` (and any other metadata). It is
        deep-copied. Default is an empty dict.
    """

    def __init__(self, model_func, lbda=None, meta={}):
        """Initialize the SceneElement."""
        if model_func is not None and not isinstance(model_func, str): # e.g. "None"
            self._model_func = model_func

        self._meta = deepcopy(meta)
        self._meta_in = deepcopy(meta)
        self._lbda = lbda

    @classmethod
    def from_config(cls, config):
        """Build the class from a configuration dictionary.

        Must be implemented by subclasses.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.

        Returns
        -------
        SceneElement
            An instance of the class.

        Raises
        ------
        NotImplementedError
            Always, for the base class.
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
            If True, parameters not given in ``kwargs`` are reset to their
            initial values. Default is False.
        **kwargs
            Parameters to update, see `mutable_parameters`. ``lbda`` updates
            the default wavelength array. None values are skipped.

        Returns
        -------
        list of str
            Names of the updated model parameters (``lbda`` excluded).

        Warns
        -----
        UserWarning
            If a given parameter is not mutable (it is then ignored).
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

        return list( model_update.keys() )

    def get_spectrum(self, lbda=None):
        """Get the spectrum of the scene element.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, `lbda` (the property) is
            used. Default is None.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        flux : numpy.ndarray
            Flux of the scene element. Unit depends on ``model_func``
            (typically erg/s/cm^2/A, or erg/s/cm^2/A/arcsec^2 for
            backgrounds).

        Raises
        ------
        ValueError
            If ``lbda`` is None and no default wavelength is set.
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
        """Get default ``model_func`` parameters updated by current meta.

        Returns
        -------
        dict
            Dictionary of parameters for ``model_func``.
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
        """Meta parameters of the object (parameters of `model_func` and metadata)."""
        return self._meta

    @property
    def lbda(self):
        """Default wavelength array in Angstrom."""
        return self._lbda

    @property
    def mutable_parameters(self):
        """List of mutable parameters (the arguments of `model_func`, including ``lbda``)."""
        # no extra so far
        return self._model_mutables

    @property
    def _model_mutables(self):
        """List of model mutable parameters."""
        return list(inspect_func(self.model_func)[0])
