import pytest

from slicersim import iotools
from slicersim.target import VirtualTarget
from slicersim.lazuli import VirtualLazuliTarget


def pytest_configure(config):
    print("running pytest_configure")
    VirtualTarget._DEFAULT_CONFIG = {"instrument": "lazuli_test.toml"}
    VirtualLazuliTarget._INSTRUMENT = "lazuli_test.toml"


def pytest_unconfigure(config):
    print("running pytest_unconfigure")
    VirtualTarget._DEFAULT_CONFIG = {"instrument": "lazuli_cbe.toml"}
    VirtualLazuliTarget._INSTRUMENT = "lazuli_cbe.toml"


@pytest.fixture
def default_config():
    return iotools.get_config(instrument="lazuli_test.toml")
