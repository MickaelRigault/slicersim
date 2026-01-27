import numpy as np
import pytest

from slicersim.telescope import Telescope
    
from slicersim.iotools import TEST_CONFIG as test_config

lbda_test = np.linspace(5000, 10000, 50)

@pytest.fixture
def telescope():
    """ """
    return Telescope.from_config(test_config["telescope"])

def test_instanciation(telescope):
    """ """
    assert isinstance(telescope, Telescope)

def test_update_and_airy(telescope):
    """ """
    telescope.update(diameter_ext=5)
    airy_5 = telescope.get_airy_radius(10_000)
    
    telescope.update(diameter_ext=3)
    airy_3 = telescope.get_airy_radius(10_000)
    assert np.all(airy_3 > airy_5)

def test_airy(telescope):
    """ """
    radius = telescope.get_airy_radius(lbda_test)
    assert radius.shape == ( len(lbda_test),)

def test_psfprofile(telescope):
    """ """
    psf, pixelarea, arcsec_to_pixels, radius = telescope.get_psfprofile(lbda_test, 
                                                                    shape=(0.5, 0.5))
    assert psf.shape[0] == len(lbda_test)
    assert np.ndim(psf) == 3

def test_ee(telescope):
    """ """
    radius = telescope.get_encircled_energy_radius(5000, 1)
    assert radius>0
