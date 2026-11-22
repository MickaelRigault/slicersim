import numpy as np
import pytest

from slicersim.sample import Sample

@pytest.fixture
def sample():
    import numpy as np
    redshift = np.linspace(0.2, 1.5, 10)
    return Sample.from_sneia(redshift, )


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


def test_dask_or_not_in_setup(sample):
    """ """
    # cannot test
    try:
        from dask.distributed import Client
    except ImportError as e:
        return

    configs, snrs = sample.setup_to_snr(25, per_resolution=True, show_progress=False)
    client = Client()
    configs_dask, snrs_dask = sample.setup_to_snr(25, per_resolution=True, client=client)
    assert np.isclose(snrs_dask, snrs).all()
    assert (configs_dask == configs).all().all()
    


# tests for the func set_thermal_dark
def test_set_thermal_dark(sample):
    """ the shared thermal dark is exactly each target's own, and kept by reset_cache """
    import warnings
    own = Sample.from_sneia([0.2, 1.5], cache_thermal=False) # narrow and wide fields
    reference = own.targets[0].simulation.detector.get_thermal_dark(units="e-/s")

    # shared at init (cache_thermal=True by default): used by cached calls, bitwise equal.
    for target in [sample.targets[0], sample.targets[-1]]: # narrow and wide fields
        detector = target.simulation.detector
        assert np.array_equal(detector.get_thermal_dark(units="e-/s", cached=True), reference)
        assert np.array_equal(detector.get_thermal_dark(units="e-/s"), reference), "not the target's own value"

    # same setup_to_snr results with and without shared thermal dark
    config, snr = sample.targets[-1].setup_to_snr(20, inplace=False)
    config_own, snr_own = own.targets[-1].setup_to_snr(20, inplace=False)
    assert config == config_own and snr == snr_own

    # kept by reset_cache (fetch_snr), unless requested
    simulation = sample.targets[0].simulation
    simulation.reset_cache()
    assert simulation.detector._cached_thermal is not None
    simulation.reset_cache(keep_thermal=False)
    assert simulation.detector._cached_thermal is None

    # a value computed for different inputs is ignored, with a warning
    detector = sample.targets[0].simulation.detector
    inputs = detector.get_thermal_inputs(units="e-/s")
    with pytest.warns(UserWarning):
        assert not detector.set_thermal_dark(reference * 2, units="e-/s", inputs=inputs[:-1] + (None,))

    # changing the optics temperature => recomputed, not reused.
    _ = detector.set_thermal_dark(reference, units="e-/s")
    simulation.update(optics__temperature=np.array(detector.thermaloptics.temperature) + 20) # (n, 1) as stored
    with warnings.catch_warnings():
        warnings.simplefilter("error") # no warning expected here
        hotter = detector.get_thermal_dark(units="e-/s", cached=True)
    assert not np.array_equal(hotter, reference)
    assert np.array_equal(hotter, detector.get_thermal_dark(units="e-/s"))
