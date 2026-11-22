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


def test_estimate_pixel_signal_cached_and_incl_signal(detector, lbda_test):
    """ cached electron rate and incl_signal=False do not change the results, saturation included """
    flux = np.linspace(0, 50, len(lbda_test)) # ph/s/voxel
    signal, variance = detector.estimate_pixel_signal(flux, lbda=lbda_test)

    # signal not requested => None, same variance
    no_signal, variance_nosignal = detector.estimate_pixel_signal(flux, lbda=lbda_test, incl_signal=False)
    assert no_signal is None
    assert np.array_equal(variance, variance_nosignal)

    # cached electron rate, reused for the very same flux array only
    _, variance_cached = detector.estimate_pixel_signal(flux, lbda=lbda_test, cached_thermal=True)
    assert np.array_equal(variance, variance_cached)
    assert detector._cached_flux_e["flux"] is flux
    _, variance_other = detector.estimate_pixel_signal(flux*2, lbda=lbda_test, cached_thermal=True)
    assert not np.array_equal(variance, variance_other), "cache used for another flux"

    # saturation: same masking and count with or without signal
    saturation = np.nanmax(signal) / 2
    signal_sat, variance_sat = detector.estimate_pixel_signal(flux, lbda=lbda_test, saturation=saturation)
    nsaturated = detector.nsaturated_detpx
    _, variance_sat_nosignal = detector.estimate_pixel_signal(flux, lbda=lbda_test, saturation=saturation,
                                                                  incl_signal=False)
    assert nsaturated > 0 and detector.nsaturated_detpx == nsaturated
    assert np.array_equal(variance_sat, variance_sat_nosignal)
    assert np.all(np.isinf(variance_sat[np.isnan(signal_sat)]))

    # no saturation => nsaturated_detpx is 0
    _ = detector.estimate_pixel_signal(flux, lbda=lbda_test, saturation=np.nanmax(signal) * 10, incl_signal=False)
    assert detector.nsaturated_detpx == 0
