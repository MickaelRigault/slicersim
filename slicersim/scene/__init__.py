""" module containing the scene elements. """

from .scene import Scene
from .pointsource import PointSource


def get_sn_scene(model="salt", background="zodi", host={}, **kwargs):
    """ get a scene configuration for a sn model. 
    {scene: pointsource:{}, host:{}, backgound:{} """
    from .pointsource import get_snia_pointsource
    
    # pointsource
    snia_pointsource = get_snia_pointsource(model=model, **kwargs)
    
    # background
    background = {'name': 'zodi', 'model': 'Aldering01.BB5800', 'scale': 2.0}

    # host
    host = {}

    return {"scene":{"point_source": snia_pointsource, 
                     "background": background, 
                     "host": host
                    }}
