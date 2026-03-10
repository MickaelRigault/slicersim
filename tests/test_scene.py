import numpy as np

import slicersim
from slicersim.thermal import get_source_radiation
from slicersim import iotools

# ================= #
#   Background      #
# ================= #

def test_background_func_setup():
    """ """

    test_transmission = 1e-3
    
    def qth_flux(lbda, temperature, emissivity=0.35, ph_transmission=1):
        """ """
        bb_photonflux = get_source_radiation(lbda, temperature, emissivity=emissivity)
        return bb_photonflux * ph_transmission

    # normal bb case
    scene = {"scene": {"pointsource": None, 
                   "background": {"name": "blackbody", 
                                  "temperature": 2800,
                                  "emissivity": 0.35
                                 },
                   "host":None}}
    config = iotools.get_config(scene=scene)
    # => associated simulation
    simu = slicersim.Simulation.from_config(config)

    # customed bb case
    scene_func = {"scene": {"pointsource": None, 
                    "background": {"func": qth_flux, 
                                  "temperature": 2800,
                                  "emissivity": 0.35,
                                  "ph_transmission": test_transmission,
                                 },
                   "host":None}}
        
    config_func = iotools.get_config(scene=scene_func)
    # => associated simulation
    simu_func = slicersim.Simulation.from_config(config_func)

    lbda = simu.spectrograph.lbda
    _, spec = simu.scene.background.get_spectrum(lbda)
    _, spec_func = simu_func.scene.background.get_spectrum(lbda)
    
    assert np.isclose(spec_func/spec, test_transmission).all()
