import numpy as np

from slicersim.detector import Detector
from slicersim.thermal import ThermalOptics

import pytest
from slicersim.iotools import TEST_CONFIG as test_config

lbda_test = np.linspace(4_000, 18_000, 400)
thermaloptics_test = ThermalOptics([200, 300], emissivity=[0.1, 0.01], fratio=[[15, 30], [7.5, 5.2]])


@pytest.fixture
def detector():
    """ """
    return Detector.from_config( test_config["detector"] )

# ============= #
#   Tests        #
# ============= #
def test_instanciation(detector):
    """ """
    assert isinstance(detector, Detector), "Detector.from_config() failed to return a detector"

def test_qe(detector):
    """ """
    qe = detector.get_qe(lbda_test)
    
    assert qe.shape == lbda_test.shape, "test lbda shape and returned qe shape do not match"
    assert np.all((qe>=0) & (qe<=1)), "not all qe values are between 0 and 1"

def test_thermal_dark(detector):
    """ """
    # This does not test the actual expected values for the thermal dark.
    null_termal = detector.get_thermal_dark()
    if detector.thermaloptics is None: # expect nothing, do I get nothing ?
        assert null_termal == 0, "empty detector.get_thermal_dark() returns non-zero dark value"


    # testing functionalities
    thermaldark = detector.get_thermal_dark(thermaloptics_test,
                                                lbda_range=[10_000, 20_000],
                                                as_sum=True)
    assert thermaldark.shape == (1,), "format error on detector.get_thermal_dark(as_sum=True) "
    assert thermaldark > 0, "thermal dark is negative."

    
    thermaldark = detector.get_thermal_dark(thermaloptics_test,
                                                lbda_range=[10_000, 20_000],
                                                as_sum=False)
    assert thermaldark.shape == thermaloptics_test.temperature.shape, "format error on detector.get_thermal_dark(as_sum=False) "
    
    
    
