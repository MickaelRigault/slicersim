import numpy as np
import pytest

from slicersim.sample import Sample

@pytest.fixture
def sample():
    import numpy as np
    redshift = np.linspace(0.2, 1.5, 10)
    return Sample.from_sneia(redshift)


def test_instanciation(sample):
    """ """
    assert isinstance(sample, Sample)

def test_get_target(sample):
    """ """
    from slicersim.target import VirtualTarget

    target = sample.get_target(0)
    assert isinstance(target, VirtualTarget), "returned object is not a Target"

    redshift = target.get_properties("redshift")
    # see value on sample()
    assert (redshift==0.2), "returned redshift not what was expected"

def test_change_property(sample):
    """ """
    # change the dark value for all targets
    _ = sample.change_property(dark=0.3)
    # test it on a target
    target = sample.get_target(0)
    dark_of_target = target.get_properties("dark")
    assert dark_of_target == 0.3, "sample change of dark not propagated to individual target"

def test_get_volume(sample):
    """ """
    volume = sample.get_data_volume()
    assert len(volume) == sample.ntargets, "dimension issue on get_data_volume()"
    assert (np.asarray(volume)>0).all(), "at least some returned volume are lower or equal to zero."

def test_get_exposure_and_surveyduration(sample):
    """ """
    exptime = sample.get_exposure_time()
    assert len(exptime) == sample.ntargets, "dimension issue on get_exposure_time()"
    assert (np.asarray(exptime)>0).all(), "at least some returned exptime are lower or equal to zero."

    survey_duration = sample.get_total_surveyduration()
    assert survey_duration>0, "survey duration is lower or equal to 0 yr. Doesn't make sense."
    
def test_sample_setup_to_snr(sample):
    """ """
    config, snr = sample.setup_to_snr(20, per_resolution=False)
    snr = np.asarray(snr)
    assert len(config) == sample.ntargets, "unmatched dimensions"
    assert np.all( (snr>15) & (snr<25) ), "returned snr is not what was expected" 
