""" module containing the scene elements. """
import warnings
from .pointsource import PointSource  # noqa: F401
from .scene import Scene  # noqa: F401

def get_scene(source="snia", background="zodi", host={}, **kwargs):
    """ Get a scene configuration.

    This function generates a dictionary that defines a scene containing a
    source, a background, and a host galaxy. The scene can then be used
    to generate a simulation.

    Parameters
    ----------
    source : str, optional
        The name of the source model to use. Can be "snia" or any other
        source name. Defaults to "snia".
    background : str, optional
        The name of the background model to use. Defaults to "zodi".
    host : dict, optional
        A dictionary defining the host galaxy. Defaults to an empty dict.
    **kwargs
        Additional keyword arguments to pass to the source model.

    Returns
    -------
    dict
        A dictionary defining the scene, with keys for the point source,
        background, and host.
    """
    if "snia" in source:
        from .sources.supernovae import get_snia_pointsource
        # pointsource
        pointsource = get_snia_pointsource(model=source.replace("snia-",""), **kwargs)

    elif source == "kilonova":
        from .sources.kilonova import get_kilonova_pointsource
        # pointsource
        pointsource = get_kilonova_pointsource(model=source.replace("kilonova-",""), **kwargs)

    else:
        raise NotImplementedError(f"Unknown source {source!r}. Only 'snia' and 'kilonova' implemented.")

    # background
    if background == "zodi":
        background = {'name': 'zodi', 'model': 'Aldering01.BB5800', 'scale': 2.0}

    return {"scene":{"pointsource": pointsource,
                        "background": background,
                        "host": host
                    }}

def get_sn_scene(model="salt", background="zodi", host={}, **kwargs):
    """ Get a scene configuration for a supernova model.

    This function generates a dictionary that defines a scene containing a
    supernova, a background, and a host galaxy. The scene can then be used
    to generate a simulation.

    Parameters
    ----------
    model : str, optional
        The name of the supernova model to use. Can be "salt" or any other
        sncosmo salt source name. Defaults to "salt".
    background : str, optional
        The name of the background model to use. Defaults to "zodi".
    host : dict, optional
        A dictionary defining the host galaxy. Defaults to an empty dict.
    **kwargs
        Additional keyword arguments to pass to the supernova model.

    Returns
    -------
    dict
        A dictionary defining the scene, with keys for the point source,
        background, and host.
    """
    warnings.warn(f"get_sn_scene is deprecated. Use get_scene(source='snia-{model}', ...) instead.", DeprecationWarning)
    return get_scene(source=f"snia-{model}", background=background, host=host, **kwargs)
