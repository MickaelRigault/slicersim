import numpy as np
import pytest

from slicersim.spectrograph import Spectrograph, SlicerSpectrograph


@pytest.fixture
def spectro():
    """ """
    from slicersim import iotools
    from slicersim.telescope import Telescope
    
    config = iotools.get_config()
    telescope = Telescope.from_config( config["telescope"] )
    return Spectrograph.from_config( config["spectrograph"], telescope=telescope)


@pytest.fixture
def slicer():
    """ """
    from slicersim import iotools
    from slicersim.telescope import Telescope
    config = iotools.get_config()
    telescope = Telescope.from_config( config["telescope"] )
    return SlicerSpectrograph.from_config( config["spectrograph"], telescope=telescope)

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

def test_resolving_power(spectro):
    """ """
    resolving_power = spectro.get_resolving_power()
    # this is another way to build it.
    # should be self consistant.

    # dispersion resolution. Is it self-consistant ? 
    dispersion_resolution = spectro.get_lsf_dispersion(as_ = "resolution") #
    assert (dispersion_resolution == spectro.dispersion_resolution)

    # definition of R. Is it self-consistant ? 
    dlbda = np.diff(spectro.lbda_edges)
    wres = spectro.lbda / (dispersion_resolution * dlbda)  # (nlbda,)
    assert (resolving_power == wres).all()

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


def test_oversampling(slicer):
    
    spectrum = np.ones( slicer.lbda.shape ) * 1e-18
    cube = slicer.generate_pointsource(spectrum)
    cube_oversampling = slicer.generate_pointsource(spectrum, oversampling=10)
    cube_oversampled = slicer.generate_pointsource(spectrum, oversampling=10, as_oversampled=True)

    assert cube_oversampling.shape == cube.shape
    assert np.all(cube_oversampled.shape[1:] == np.asarray(cube.shape[1:])*10) and cube_oversampled.shape[0] == cube.shape[0]

def test_cube_profile(slicer):
    """ """
    
    flux = np.ones(slicer.lbda.shape)
    
    # generate cubes using 3 different profiles.
    cube_gaussian = slicer.generate_pointsource( flux,  psf_profile="gaussian", oversampling=10)
    cube_gaussianastro = slicer.generate_pointsource( flux,  psf_profile="Gaussian2D", oversampling=10) # astropy Gaussian
    cube_airy = slicer.generate_pointsource( flux,  psf_profile="airy", oversampling=10) # Airy disk

    # spectrum from cube as sum over spatial dimensions.
    specsum_gaussian = cube_gaussian.sum((-2,-1))
    specsum_gaussianastro = cube_gaussianastro.sum((-2,-1))
    specsum_airy = cube_airy.sum((-2,-1))
    
    assert np.isclose(specsum_gaussian/specsum_airy, 1, rtol=0.1).all()
    assert np.isclose(specsum_gaussian/specsum_gaussianastro, 1, rtol=0.1).all()    


def test_thermal_dark(spectro):
    """ """
    pixel_size = 10 * 1e-6 # 10 micrometter
    pixel_area = pixel_size**2

    thermal_dark_sum = spectro.get_thermal_dark(pixel_area, as_sum=True)
    thermal_dark_details = spectro.get_thermal_dark(pixel_area, as_sum=False)
    
    assert thermal_dark_sum >= 0
    assert np.all(thermal_dark_details>=0)
    assert thermal_dark_details.shape == spectro.optics.temperature.shape    


def test_cube_to_slice(spectro):
    """ """
    # generate a cube
    flux = np.ones( spectro.lbda.shape ) * 1e-18
    cube = spectro.generate_pointsource( flux )

    # build 2 slice definitions
    lbda_range_blue = np.percentile(spectro.lbda, [10, 50])
    lbda_range_red = np.percentile(spectro.lbda, [80, 90])

    # test various cases.
    slice_blue = spectro.cube_to_slice( cube, lbda_range_blue)
    slice_bluesqueeze = spectro.cube_to_slice( cube, lbda_range_blue, func=np.nanmean, squeeze=True) # testing with mean
    slice_red = spectro.cube_to_slice( cube, lbda_range_blue)
    slice_blue_and_red = spectro.cube_to_slice( cube, [lbda_range_blue, lbda_range_red])

    slice_full = spectro.cube_to_slice( cube, [spectro.lbda[0], spectro.lbda[-1]], func=np.nansum, squeeze=True)
    
    assert slice_blue.shape == (1, *cube.shape[1:])
    assert slice_red.shape == (1, *cube.shape[1:])
    
    assert slice_bluesqueeze.shape == cube.shape[1:]
    assert slice_blue_and_red.shape == (2, *cube.shape[1:])
    assert (slice_full == np.nansum(cube, axis=0)).all()
