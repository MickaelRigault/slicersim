
from slicersim.detector import Detector 

def get_default_detector():
    """ """
    
    from slicersim import iotools

    config = iotools.get_config()["detector"]
    return Detector.from_config(config)

# ============= #
#   Tests        #
# ============= #
def test_detector_from_config():
    """ """
    detector = get_default_detector()
    assert isinstance(detector, Detector), "Detector.from_config() failed to return a detector"



