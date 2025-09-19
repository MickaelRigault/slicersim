""" module containing the scene elements. """

from .scene import Scene
from .pointsource import PointSource


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
    from .pointsource import get_snia_pointsource
    
    # pointsource
    snia_pointsource = get_snia_pointsource(model=model, **kwargs)
    
    # background
    background = {'name': 'zodi', 'model': 'Aldering01.BB5800', 'scale': 2.0}

    # host
    host = {}

    return {"scene":{"pointsource": snia_pointsource, 
                     "background": background, 
                     "host": host
                    }}
