import pytest

from slicersim import iotools


@pytest.fixture
def default_config():
    return iotools.get_config(instrument="lazuli.toml")
