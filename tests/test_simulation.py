import numpy as np
import pytest

from slicersim.simulation import Simulation
from slicersim import iotools



@pytest.fixture
def simu():
    config = iotools.get_config()
    return Simulation.from_config(config)

# ======= #
#  TESTS  #
# ======= # 
def test_instanciation(simu):
    """ """
    assert isinstance(simu, Simulation), "failed to instanciate the Simulation"

    
def test_from_classmethods():
    """ """
    # from_source
    lbda = np.linspace(3000, 20_000, 150)
    flux = np.ones(lbda.shape) * 1e-17
    
    simu = Simulation.from_source(lbda, flux)
    lbda, spec, var = simu.get_spectrum()
    assert isinstance(simu, Simulation), "failed to instanciate the Simulation (from_source)"
    assert len(spec) > 1

    # from_scene
    simu = Simulation.from_scene(redshift=1, snr=20)
    lbda, spec, var = simu.get_spectrum()
    exptime = simu.get_times()["total_exptime"]
    assert isinstance(simu, Simulation), "failed to instanciate the Simulation (from_scene)"
    assert exptime > 1
    assert len(spec) > 1
    
# top functionalities
def test_cube(simu):
    """ """
    cube, cubevar = simu.get_cube()
    assert cube.shape  == (len(simu.spectrograph.lbda), *simu.spectrograph.spx_shape), "cube doesn't have the good shape"
    assert np.all(cubevar>=0), "variance in cubevar are not all positive"
    
def test_get_spectrum(simu):
    """ """
    lbda, spec, variance = simu.get_spectrum()

    assert lbda.shape == spec.shape == variance.shape, "mismatch of shapes returned by get_spectrum"
    assert np.all(lbda == simu.spectrograph.lbda), "mismatch of lbda"
    assert np.all(variance>=0), "there are negative variances"

def test_variance_sources(simu):
    """ """
    # default variance 
    lbda, spec, variance = simu.get_spectrum()

    # variance per sources
    variance_df = simu.get_variance_contribution()
    variance_sources = variance_df[simu.variance_sources]

    # test all positively defined
    assert np.all(variance_sources>=0), "not all variances are positively defined"

    # test that sum of variances is close to total variance
    variance_ratio = variance_df[simu.variance_sources].sum(axis=1) / variance
    assert np.isclose(variance_ratio, 1, rtol=0.1).all(), "sum of variances is not close to total variance"

    
def test_update(simu):
    """ """
    
    # update detector property
    simu.update(nramps=3)
    exptime_3 = simu.get_times()["total_exptime"]
    simu.update(nramps=1)
    exptime_1 = simu.get_times()["total_exptime"]
    assert np.isclose(exptime_3/exptime_1, 3.), "3 ramps is not 3x longer than 1 ramp."

    # update spectroscopic property
    request_shape  = [20,40]
    experted_output_shape = np.asarray(request_shape) * simu.spectrograph._ANAMORPHOSE
    simu.update(spatial_shape=request_shape)
    cube, cubevar = simu.get_cube()
    assert np.all(cube.shape[1:] == experted_output_shape), "wrong cube shape after update of spectrp schape"
    

    # update spectro top-level
    requested_rmin = 150
    requested_spotsize = 2.5
    simu.change_spectrograph_resolution(requested_rmin, spotsize=requested_spotsize)
    rpower = simu.spectrograph.get_resolving_power()
    assert np.isclose(rpower.min(), requested_rmin, 1), "failed to reach the requested rmin"
    assert simu.spectrograph.dispersion_resolution == requested_spotsize, "dispersion resolution is not seft consistant."    

def test_update_and_reset(simu):

    lbda_ref, spec_ref, var_ref = simu.get_spectrum()
    
    # Changing int
    simu.update(spx_shape=[10, 20], redshift=0.3, dark=0.01)
    simu.change_spectrograph_resolution(220, spotsize=2.2)
    lbda_changed, spec_changed, var_changed = simu.get_spectrum()
    
    # reset it
    simu.reset()
    lbda_test, spec_test, var_test = simu.get_spectrum()
    
    assert np.all(spec_test == spec_ref) & np.all(lbda_test == lbda_ref) & np.all(var_test == var_ref)
    if lbda_changed.shape == lbda_ref.shape:
            assert np.all(spec_changed != spec_ref) & np.all(lbda_changed != lbda_ref) & np.all(var_changed == var_ref)
    # else: # no need to test as shape differs so...
        
def test_etc(simu):
    """ """
    requested_snr = 25
    
    # redshift 0.5
    simu.update(redshift=0.5)
    _, snr_05, exptime_05 = simu.fetch_snr(requested_snr)
    
    # redshift 1
    simu.update(redshift=1.)
    _, snr_1, exptime_1 = simu.fetch_snr(requested_snr)

    # redshift 1.5
    simu.update(redshift=1.5)
    _, snr_15, exptime_15 = simu.fetch_snr(requested_snr)

    # snr
    assert np.isclose(snr_05, requested_snr, 2), "snr not reached at z=0.5"
    assert np.isclose(snr_1, requested_snr, 2), "snr not reached at z=1"
    assert np.isclose(snr_15, requested_snr, 2), "snr not reached at z=1.5"

    # exptime
    assert exptime_05 < exptime_1 < exptime_15


def test_data_volume(simu):
    """ """
    simu.update(nramp=1, nmd=(64, 8, 0)) # 1 ramp by default
    datavolume_1ramp = simu.get_data_volume("GB")
    simu.update(nramp=2) # 2 ramps as test
    datavolume_2ramp = simu.get_data_volume("GB")
    
    assert (datavolume_1ramp>0.01) # 1 ramp should be at least 10 MB (500 likely)
    assert (datavolume_2ramp == 2*datavolume_1ramp)
