
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
