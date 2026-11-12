import numpy as np
import pytest

from slicersim.detector import Detector
from slicersim.thermal import ThermalOptics

@pytest.fixture
def detector(default_config):
    return Detector.from_config(default_config["detector"])

@pytest.fixture
def lbda_test():
    return np.linspace(4_000, 18_000, 400)

# ============= #
#   Tests        #
# ============= #
def test_instanciation(detector):
    """ """
    assert isinstance(detector, Detector), "Detector.from_config() failed to return a detector"

def test_qe(detector, lbda_test):
    """ """
    qe = detector.get_qe(lbda_test)

    assert qe.shape == lbda_test.shape, "test lbda shape and returned qe shape do not match"
    assert np.all((qe>=0) & (qe<=1)), "not all qe values are between 0 and 1"


def test_thermal_dark(detector):
    # This does not test the actual expected values for the thermal dark.

    null_termal = detector.get_thermal_dark()
    if detector.thermaloptics is None:  # expect nothing, do I get nothing ?
        assert null_termal == 0

    # testing functionalities
    thermaloptics_test = ThermalOptics(
        [200, 300], emissivity=[0.1, 0.01], fratio=[[15, 30], [7.5, 5.2]],
        meta={"dispersed": [False, False]} # directly as detector dark
        )
    thermaldark = detector.get_thermal_dark(
        thermaloptics_test, lbda_range=[10_000, 20_000], as_sum=True
    )
    assert thermaldark.shape == (1,)
    assert thermaldark > 0

    thermaldark = detector.get_thermal_dark(
        thermaloptics_test, lbda_range=[10_000, 20_000], as_sum=False
    )
    assert thermaldark.shape == thermaloptics_test.temperature.shape


def test_effective_ron(detector):
    """ """
    effective_ron = detector.get_effective_ron(nmd=(64, 1, 0), ron=25)
    assert effective_ron>7 and effective_ron<15 # expected ~10.5


def test_estimate_pixel_signal_npix_per_voxel(detector, lbda_test):
    """ spreading a voxel over npix pixels keeps the signal, adds npix times the per-pixel noise floor """
    flux = np.linspace(0, 50, len(lbda_test)) # ph/s/voxel
    detector_floor = detector.estimate_variance(np.zeros_like(flux)) # ADU²/pixel, no source

    signal_1, variance_1 = detector.estimate_pixel_signal(flux, lbda=lbda_test, saturation=None)
    signal_2, variance_2 = detector.estimate_pixel_signal(flux, lbda=lbda_test, saturation=None,
                                                              npix_per_voxel=2)

    assert np.allclose(signal_1, signal_2), "the signal should not depend on npix_per_voxel"
    assert np.allclose(variance_2 - variance_1, detector_floor), "each extra pixel should add one noise floor"

    # default is 1 pixel per voxel.
    signal_default, variance_default = detector.estimate_pixel_signal(flux, lbda=lbda_test, saturation=None)
    assert np.allclose(variance_default, variance_1)
