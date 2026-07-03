import numpy as np
import slicersim
from slicersim.target import Target
from slicersim.lazuli import lazuli_etc, LazuliCalSpec
import pytest

@pytest.fixture
def instrument_config(default_config):
    """ """
    from copy import deepcopy
    instrument_config = deepcopy(default_config)
    _ = instrument_config.pop("scene")
    return instrument_config

@pytest.fixture
def target(instrument_config):
    """ """
    lbda = np.linspace(3_000, 20_000, 500)
    flux = np.ones( lbda.shape )*1e-18
    return Target(lbda, flux, instrument=instrument_config)

@pytest.fixture
def lazulitarget():
    """ """
    lbda = np.linspace(3_000, 20_000, 500)
    flux = np.ones( lbda.shape )*1e-18
    return slicersim.LazuliTarget(lbda, flux)

@pytest.fixture
def lazulisnia():
    """ """
    return slicersim.LazuliSupernova(redshift=1)

def test_supernovae(lazulisnia):
    """ """
    # there is no noise included, so spec must be greater or equal to zero
    lbda, spec, var = lazulisnia.get_spectrum(unit="flambda", incl_error=False)
    assert lbda.shape == spec.shape == var.shape
    assert (spec >= 0).all(), "all spectrum should be >=0 as no error)"

    # z=0.5 target
    lazulisnia.simulation.update(redshift=0.5)
    _ = lazulisnia.setup_to_snr(20)
    exptime = lazulisnia.get_exposure_time()
    assert exptime >= 1, "exposure shorter than 1s doesn't seem correct"

    lazulisnia.simulation.update(redshift=1)
    _ = lazulisnia.setup_to_snr(20)
    exptime_distance = lazulisnia.get_exposure_time()
    assert exptime_distance > exptime, "distance exposure time should be longer than nearby"

def test_target_and_lazuli(lazulitarget):
    """ """
    # config
    config = lazulitarget.get_readout_config()
    assert ("nmd" in config) & ("nramps" in config), "nmd and nramps must be in config"

    # sampling
    name, _ = lazulitarget.get_spectrograph_field()
    assert name in ["narrow", "wide"], "field names should be fine or medium"

    lazulitarget.change_spectrograph(field="narrow")
    name, _ = lazulitarget.get_spectrograph_field()
    assert name == "narrow", "field expected to be narrow"

    lazulitarget.change_spectrograph(field="wide")
    name, _ = lazulitarget.get_spectrograph_field()
    assert name == "wide", "field expected to be wide"


def test_variance_contribution(target):
    """ """
    # variance
    variance_df = target.get_variance_contribution()
    assert variance_df.shape[0] == len(target.simulation.spectrograph.lbda), "variance shape not correct."
    assert np.all([k in variance_df.columns for k in target.simulation.variance_sources]), "missing variance source"

def test_data_volume(target):
    """ """
    # data volume
    data_vol = target.get_data_volume("MB")
    assert data_vol > 1, "data volume lower than 1MB, strange"


def test_get_cube(target):
    """ """
    cube, cubevar = target.get_cube()
    assert cube.shape[0] == len(target.simulation.spectrograph.lbda), "cube shape not correct"

def test_get_cube_lazuli(lazulitarget):
    """ """
    cube_narrow, var_narrow = lazulitarget.get_cube(which="narrow")
    cube_wide, var_wide = lazulitarget.get_cube(which="wide")
    (cube_narrow2, var_narrow2), (cube_wide2, var_wide2) = lazulitarget.get_cube(which="both")

    assert np.all(cube_narrow2 == cube_narrow) & np.all(cube_wide == cube_wide2), "which on get_cube seems broken."

    # get samplig
    cube_current, var_current = lazulitarget.get_cube(which="current")
    name, _ = lazulitarget.get_spectrograph_field()
    if name == "narrow":
        assert np.all(cube_current == cube_narrow), "which on get_cube seems broken (fine not matching)."

    elif name == "wide":
        assert np.all(cube_current == cube_wide), "which on get_cube seems broken (medium not matching)."

def test_field_position(lazulitarget):
    """ """
    position_narrow, position_wide = lazulitarget._get_field_positions()
    # medium field is located below and zero is on top.
    assert position_narrow[1]<position_wide[1]

    # test case where we set the location of manually
    position_narrow, position_wide = lazulitarget._get_field_positions(np.asarray([0,0]), "wide")
    assert position_narrow[1]<position_wide[1], "verticale position seems wrong in field locations"

    # test case where we set the location of manually
    position_narrow, position_wide = lazulitarget._get_field_positions(np.asarray([-5,0]), "narrow")
    assert position_narrow[1]<position_wide[1], "verticale position seems wrong in field locations"


def test_lazuli_etc():
    lbda = np.linspace(3_000, 20_000, 500)
    flux = np.ones( lbda.shape )*1e-18

    exptime, target = lazuli_etc(lbda, flux, 20, mag=22, band="bessellb")
    assert exptime > 0, "negative exptime doesn't make sense"
    assert exptime > 10, "less than 10s for a mag=22 target is odd"

    exptime_mid, target_mid = lazuli_etc(lbda, flux, 20, mag=24, band="bessellb")
    assert exptime_mid > exptime, "mag 24 should take longer than mag 22"

    exptime_faint, target_fait = lazuli_etc(lbda, flux, 20, mag=25, band="bessellb")
    assert exptime_faint > exptime_mid, "mag 25 should take longer than mag 24"

    exptime_detailed, _ = lazuli_etc(lbda, flux, 50, mag=22, band="bessellb")
    assert exptime_detailed > exptime, "SNR of 50 should take longer than SNR of 20"

def test_calspec():
    """ """
    bd17 = LazuliCalSpec.from_name("bd_17", mag=22, band="sdssr")
    assert bd17 is not None, "failed to instanciate a LazuliCalSpec"

    _ = bd17.setup_to_snr(20)
    exptime = bd17.get_exposure_time()
    assert exptime > 10, "observing a mag=22 star should take more than 10s"
