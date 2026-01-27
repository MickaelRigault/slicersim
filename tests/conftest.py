import pytest

from slicersim import iotools
from slicersim.target import VirtualTarget


def pytest_configure(config):
    VirtualTarget._DEFAULT_CONFIG = {"instrument": "lazuli_test.toml"}


def pytest_unconfigure(config):
    VirtualTarget._DEFAULT_CONFIG = {"instrument": "lazuli.toml"}


@pytest.fixture
def default_config():
    return iotools.get_config(instrument="lazuli_test.toml")
