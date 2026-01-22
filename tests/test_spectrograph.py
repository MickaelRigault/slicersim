import numpy as np
from slicersim.spectrograph import Spectrograph

@pytest.fixture
def spectro():
    """ """
    from slicersim import iotools
    config = iotools.get_config()
    telescope = Telescope.from_config( config["telescope"] )
    return Spectrograph.from_config( config["spectrograph"], telescope=telescope)

# ======= #
#  TESTS  #
# ======= #
def test_instanciation(spectro):
    """ """
    assert isinstance(spectro, Spectrograph), "failed to instanciate the Spectrograph"

def test_lbda(spectro):
    """ """
    assert len(spectro.lbda) > 0

def test_generate(spectro):
    """ """
    
    # pointsource
    lbda = spectro.lbda
    spec = np.ones( lbda.shape ) * 1e-18
    flux_ph = spectro.generate_pointsource(spec, psf_profile="gaussian")
    
    assert flux_ph.shape == (len(spectro.lbda), *spectro.spx_shape)
    assert np.all(flux_ph>=0), "not all data are positively defined."

    # backgroun
    lbda = spectro.lbda
    spec = np.ones( lbda.shape ) * 1e-19
    flux_ph = spectro.generate_background(spec)
    assert flux_ph.shape == (len(spectro.lbda), *spectro.spx_shape)
    assert np.all(flux_ph>=0), "not all data are positively defined."
    assert len(np.unique(flux_ph.sum(axis=0))) == 1, "non uniform"

def test_lsf(spectro):
    """ """
    lbda = spectro.lbda
    spec = np.zeros( lbda.shape )
    index_applied = int(len(lbda)/2)
    spec[index_applied] = 1
    spec_lsf = spectro.apply_line_spread_function(spec)
    assert len(spec_lsf[spec_lsf>0]) > 1, "lsf did applied"
    assert len(spec_lsf[spec_lsf>0]) > 2*spectro.dispersion_resolution, "lsf did applied"

def test_thermalchromatic(spectro):
    """ """
    thermal = spectro.generate_thermal_signal(as_cube=False, as_sum=True)
    assert (thermal.shape == spectro.lbda.shape)
    assert np.all(thermal>0), "lsf did applied"

def test_spaxels_and_update(spectro):
    """ """
    expected_shape = spectro.spx_shape
    (spaxel_x, spaxel_y), area = spectro.get_spaxel_centroids()
    assert len(spaxel_y.squeeze()) == expected_shape[0]
    assert len(spaxel_x.squeeze()) == expected_shape[1]

    # testing update
    requested_shape = [20, 10]
    spectro.update(spx_shape = requested_shape)
    (spaxel_x, spaxel_y), area = spectro.get_spaxel_centroids()
    assert len(spaxel_y.squeeze()) == requested_shape[0]
    assert len(spaxel_x.squeeze()) == requested_shape[1]

def test_throughput(spectro):
    """ """
    throughput = spectro.get_throughput()
    assert throughput.shape == spectro.lbda.shape
    assert np.all((throughput>=0) & (throughput<=1))

    requested_throughput = np.ones(spectro.lbda.shape)*0.5
    spectro.set_throughput(requested_throughput)
    throughput = spectro.get_throughput()
    assert np.all(throughput == requested_throughput)
