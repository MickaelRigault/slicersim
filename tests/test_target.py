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
    name, _ = lazulitarget.get_spectrograph_sampling()
    assert name in ["fine", "medium"], "sampling names should be fine or medium"

    lazulitarget.change_spectrograph(sampling="fine")
    name, _ = lazulitarget.get_spectrograph_sampling()
    assert name == "fine", "sampling expected to be fine"

    lazulitarget.change_spectrograph(sampling="medium")
    name, _ = lazulitarget.get_spectrograph_sampling()
    assert name == "medium", "sampling expected to be medium"


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
    cube_fine, var_fine = lazulitarget.get_cube(which="fine")
    cube_medium, var_medium = lazulitarget.get_cube(which="medium")
    (cube_fine2, var_fine2), (cube_medium2, var_medium2) = lazulitarget.get_cube(which="both")
     
    assert np.all(cube_fine2 == cube_fine) & np.all(cube_medium == cube_medium2), "which on get_cube seems broken."

    # get samplig
    cube_current, var_current = lazulitarget.get_cube(which="current")
    name, _ = lazulitarget.get_spectrograph_sampling()
    if name == "fine":
        assert np.all(cube_current == cube_fine), "which on get_cube seems broken (fine not matching)."
        
    elif name == "medium":
        assert np.all(cube_current == cube_medium), "which on get_cube seems broken (medium not matching)."

def test_field_position(lazulitarget):
    """ """
    position_fine, position_medium = lazulitarget._get_field_positions()
    # medium field is located below and zero is on top.
    assert position_fine[1]<position_medium[1]

    # test case where we set the location of manually
    position_fine, position_medium = lazulitarget._get_field_positions(np.asarray([0,0]), "medium")
    assert position_fine[1]<position_medium[1], "verticale position seems wrong in field locations"

    # test case where we set the location of manually
    position_fine, position_medium = lazulitarget._get_field_positions(np.asarray([-5,0]), "fine")
    assert position_fine[1]<position_medium[1], "verticale position seems wrong in field locations"


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

def test_vega():
    """ Vega magnitude system, tapped from sncosmo (no synphot). """
    from sncosmo import Spectrum
    from slicersim.scene.pointsource import PointSource

    lbda = np.linspace(3_000, 11_000, 2_000)
    flux = np.ones(lbda.shape)
    band, mag = "bessellv", 15.0

    # normalization path: same spectrum/mag, different magnitude system
    ps_ab = PointSource.from_spectrum(lbda, flux, mag=mag, band=band, magsys="ab")
    ps_vega = PointSource.from_spectrum(lbda, flux, mag=mag, band=band, magsys="vega")
    _, f_ab = ps_ab.get_spectrum(lbda)
    _, f_vega = ps_vega.get_spectrum(lbda)
    realized = np.nanmedian(f_vega / f_ab)

    # expected amplitude ratio from the reference spectrum's mag in each system
    m_ab = Spectrum(lbda, flux).bandmag(band, "ab")
    m_vega = Spectrum(lbda, flux).bandmag(band, "vega")
    expected = 10 ** (-0.4 * (m_ab - m_vega))

    assert not np.isclose(realized, 1.0), "ab and vega normalization should differ"
    np.testing.assert_allclose(realized, expected, rtol=1e-3)

    # end-to-end plumbing: magsys reaches the scene pointsource through LazuliTarget
    t = slicersim.LazuliTarget(lbda, flux, mag=mag, band=band, magsys="vega")
    assert t.simulation.scene.pointsource.meta["magsys"] == "vega", "magsys not plumbed through"

def test_supernova_vs_lazulisupernova(instrument_config):
    """ """
    from slicersim.target import Supernova
    from slicersim.lazuli import LazuliSupernova
    
    snia = Supernova(instrument_config, redshift=1)
    lazulisnia = LazuliSupernova(redshift=1)
    cube_lazuli, _ = lazulisnia.get_cube(which="current")
    cube, _ = snia.get_cube()
    assert np.all(cube == cube_lazuli), "cube from lazulisupernova differ from supernova('lazuli.toml')"
