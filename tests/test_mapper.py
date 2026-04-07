import numpy as np
import pandas

from slicersim.utils import mesh_kwargs
from slicersim.mapper import SlicerMapper
from slicersim.iotools import expand_path

import pytest

DETECTOR_MAX_SIZE = 5_000
data = pandas.read_csv(expand_path("mapping_spotdata.csv"), sep=" ")

@pytest.fixture
def mapper():
    """ """
    return SlicerMapper.from_spotdata(data)


def test_instanciation(mapper):
    """ """
    assert isinstance(mapper, SlicerMapper)

def test_interpmap():
    """ """
    interpmap = SlicerMapper._build_interp_map_(data)
    xy =  interpmap(20, 0.5, 10_000 ) # slice #1 in near 0 the center of the slice (-1->1) at 1micron
    assert xy.shape == (2,), "expect shape doesn't match"
    
def test_get_pixel_positions(mapper):
    """ """
    slicepos = np.linspace(-1, 1, 5)
    wavelength = np.linspace(5_000, 10_000, 10)
    
    df_in = mesh_kwargs(sliceid=20, slicepos=slicepos, 
                                wavelength=wavelength)
    pixels = mapper.get_pixel_positions(df_in)
    assert np.all(pixels>=0), "pixels should be positive"
    assert np.all(pixels<=DETECTOR_MAX_SIZE), "pixels with values larger than 10_000. This is unlikely."
    assert np.all((pixels[0]-pixels[-1])>1), "dynamic range of tested pixels is lower and 1"

def test_project_slice(mapper):
    """ """

    # target to be projected
    from slicersim import LazuliTarget
    lbda_model = np.linspace(3000, 19_000, 1000)
    flux_model = np.ones( lbda_model.shape )
    target = LazuliTarget(lbda_model, flux_model, mag=20)
    _ = target.setup_to_snr(25)
    
    # get slicer data of the target. 
    (cube_fine, _), (cube_medium, _) = target.get_cube(which="both", 
                                                       psf_profile="airy", as_oversampled=False, 
                                                       switch_off=["background", "thermal"])

    # projection of the cube onto the detector.
    lbda = target.simulation.spectrograph.lbda
    img_fine = mapper.project_slice(np.arange(1, 59)[::-1], cube_fine, lbda)
    img_med = mapper.project_slice(np.arange(59, 117)[::-1], cube_medium, lbda)
    image = np.sum([img_med, img_fine], axis=0)
    assert np.mean(image>0)>0.10, "less than 10% of pixels have been illuminated"

def test_get_slice_contours(mapper):
    """ """
    contours = mapper.get_slice_contours(20, out_format='numpy')
    assert np.ndim(contours) == 2
    assert contours.shape[-1] == 2
    
    contours = mapper.get_slice_contours([20, 10], out_format='numpy')
    contours = np.stack(contours)
    assert np.ndim(contours) == 3
    assert contours.shape[-1] == 2
    assert np.all(contours>=0)
    assert np.all(contours<=DETECTOR_MAX_SIZE)
    
    try:
        import shapely
        contours = mapper.get_slice_contours(20, out_format='shapely')
        assert isinstance(contours, shapely.Polygon)
    except ImportError as e: # no need to test
        pass


def test_xy_to_or_from_slice_pos_wave(mapper):
    """ """
    # 3d -> 2d
    sliceid, fieldpos, lbda = [1, 10, 80], -1, 8000
    xy = mapper.slice_pos_wave_to_xy(sliceid, fieldpos, lbda)
    assert xy.shape == (2, 3)

    # 2d -> 3d
    sliceid_recovered, fieldpos_recovered, lbda_recovered = mapper.xy_to_slice_pos_wave(*xy)
    assert sliceid_recovered.shape == np.shape(sliceid)
    assert np.isclose(sliceid_recovered, sliceid).all()
    assert np.isclose(fieldpos_recovered, fieldpos).all()
    assert np.isclose(lbda_recovered, lbda).all()
