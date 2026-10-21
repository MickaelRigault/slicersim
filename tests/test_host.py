
import pytest
import numpy as np
from slicersim.scene import host

# default_config is defined as a fixture in conftest
@pytest.fixture
def default_host():
    config = host.get_host_config("sersic:bb", redshift=1)
    return host.Host.from_config( config )


def test_get_host_config():
    config = host.get_host_config("sersic:bb", redshift=1)
    assert type(config) is dict
    assert "temperature" in config # as for bb
    assert "n" in config # as in sersic



def test_instanciation(default_host):
    """ """
    assert isinstance(default_host, host.Host), "failed to instanciate the Host"

def test_get_cube(default_host):
    """ """
    lbda = np.linspace(1_000, 30_000, 1_000)
    _, cube = default_host.get_cube(lbda)
    assert cube.ndim == 3, "cube is not 3D"
    assert cube.shape[0] == len(lbda), "cube wavelength axis does not match"
