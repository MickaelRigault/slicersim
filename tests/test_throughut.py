
import numpy as np
import pytest
from slicersim.spectrograph import OpticsThroughput

from slicersim.iotools import TEST_CONFIG as test_config

def get_test_case(throughput):
    """ """
    test_name = list(throughput.names)[0]
    knots = throughput.curves[test_name].get_knots()
    lbda_test = np.linspace(knots[0], knots[1], 100)
    return test_name, lbda_test

@pytest.fixture
def throughput():
    """ """
    return OpticsThroughput.from_config( test_config["spectrograph"]["throughput"] )

def test_instanciation(throughput):
    """ """
    assert isinstance(throughput, OpticsThroughput), "throughput is not an OpticsThroughput"
    
def test_get_throughput(throughput):
    """ """
    test_name, test_lbda = get_test_case(throughput)

    
    current_curve_per_optics = throughput.get_element_throughput(test_name, test_lbda, incl_noptics=False)
    current_curve = throughput.get_element_throughput(test_name, test_lbda, incl_noptics=True)

    assert np.all( (current_curve>=0) * (current_curve<=1) ), "throughput should be defined between [0, 1]"
    assert np.all(current_curve_per_optics >=current_curve), "throughput per optics not greater or equal than all included."

def test_update(throughput):
    """ """
    test_name, test_lbda = get_test_case(throughput)
    
    throughput.update(**{f"noptics.{test_name}": 1})
    current_curve_ref = throughput.get_element_throughput(test_name, test_lbda, incl_noptics=True)

    throughput.update(**{f"noptics.{test_name}": 2})
    current_curve_comp = throughput.get_element_throughput(test_name, test_lbda, incl_noptics=True)

    
    assert np.all(current_curve_comp <= current_curve_ref)
    if np.any(current_curve_ref < 1): # not just perfect
        assert np.any(current_curve_comp < current_curve_ref) # strictly lower
        
    if np.all(current_curve_ref < 1): # not just perfect
        assert np.all(current_curve_comp < current_curve_ref) # strictly lower
