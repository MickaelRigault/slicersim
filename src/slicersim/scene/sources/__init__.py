""" Sources: spectral models feeding the scene elements.

This sub-package gathers the spectral models (the ``model_func`` of a
`~slicersim.scene.base.SceneElement`) that can be used to feed a scene.
Each module implements one family of sources:

- `~slicersim.scene.sources.supernovae`: type Ia supernova models
  (SALT and Twins-Embedding).
- `~slicersim.scene.sources.blackbody`: blackbody continuum sources.
- `~slicersim.scene.sources.calspec`: HST CalSpec spectrophotometric standards.
- `~slicersim.scene.sources.utils`: helper tools shared by the source models.

A model function always takes the wavelength array ``lbda`` (in Angstrom) as
first argument and returns a flux in erg/s/cm^2/A. The remaining parameters are
introspected by `~slicersim.scene.base.SceneElement` to build the list of
mutable parameters, so they must be explicit (no ``**kwargs``).
"""


def source_to_modelfunc(source):
    """Get the model function from a source name.

    Parameters
    ----------
    source : str
        Name of the source model.
        - "salt" (or any name containing "salt", e.g. "salt2-extended")
        - "twins-embedding"
        - "blackbody"

    Returns
    -------
    callable
        The model function for the given source. Its signature is
        ``model_func(lbda, ...)`` and it returns a flux in erg/s/cm^2/A.

    Raises
    ------
    NotImplementedError
        If no model function is associated to the given ``source``.

    See Also
    --------
    slicersim.scene.sources.supernovae.get_saltmodel_flux : SALT model flux.
    slicersim.scene.sources.supernovae.get_twins_embedding_flux : Twins-Embedding model flux.
    slicersim.scene.sources.blackbody.get_blackbody_flux : Blackbody flux.

    Examples
    --------
    >>> model_func = source_to_modelfunc("salt2-extended")
    >>> flux = model_func(np.linspace(4000, 9000, 100), phase=0)
    """
    if "salt" in source:
        from .supernovae import get_saltmodel_flux
        model_func = get_saltmodel_flux

    elif source == 'twins-embedding':
        from .supernovae import get_twins_embedding_flux
        model_func = get_twins_embedding_flux

    elif source == "blackbody":
        from .blackbody import get_blackbody_flux
        model_func = get_blackbody_flux

    elif source == "kilonova":
        from .kilonova import get_kilonova_flux
        model_func = get_kilonova_flux

    else:
        raise NotImplementedError(f"no model_func defined for source: {source}")

    return model_func
