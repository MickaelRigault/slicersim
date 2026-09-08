""" module containing the scene elements. """

from .scene import Scene # noqa: F401
from .pointsource import PointSource # noqa: F401
from .extendedsource import ExtendedSource # noqa: F401


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
    host : dict or str, optional
        Definition of the host galaxy (structured background).
        - {} or None: no host (default).
        - str: name of the host spectrum model (e.g. "elliptical"),
          using the default Sersic profile parameters.
        - dict: parameters passed to `get_host_extendedsource`
          (e.g. {"spectrum": "elliptical", "index": 1, "mag": 21}).
    **kwargs
        Additional keyword arguments to pass to the supernova model.

    Returns
    -------
    dict
        A dictionary defining the scene, with keys for the point source,
        background, and host.
    """
    from .pointsource import get_snia_pointsource
    from .extendedsource import get_host_extendedsource

    # pointsource
    snia_pointsource = get_snia_pointsource(model=model, **kwargs)

    # background
    background = {'name': 'zodi', 'model': 'Aldering01.BB5800', 'scale': 2.0}

    # host
    if host is None or host == {}:
        host = {}
    elif isinstance(host, str):
        host = get_host_extendedsource(spectrum=host)
    else:
        host = get_host_extendedsource(**host)

    return {"scene":{"pointsource": snia_pointsource,
                     "background": background,
                     "host": host
                    }}
