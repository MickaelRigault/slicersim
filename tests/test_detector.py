import numpy as np
import pytest

from slicersim.detector import Detector
from slicersim.thermal import ThermalOptics


@pytest.fixture
def detector(default_config):
    return Detector.from_config(default_config["detector"])


def test_detector_from_config(detector):
    assert isinstance(detector, Detector)


def test_qe(detector):
    lbda_test = np.linspace(4_000, 18_000, 400)
    qe = detector.get_qe(lbda_test)
    assert qe.shape == lbda_test.shape
    assert np.all((qe >= 0) & (qe <= 1))


def test_thermal_dark(detector):
    # This does not test the actual expected values for the thermal dark.

    null_termal = detector.get_thermal_dark()
    if detector.thermaloptics is None:  # expect nothing, do I get nothing ?
        assert null_termal == 0

    # testing functionalities
    thermaloptics_test = ThermalOptics(
        [200, 300], emissivity=[0.1, 0.01], fratio=[[15, 30], [7.5, 5.2]]
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
