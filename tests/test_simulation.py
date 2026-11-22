import numpy as np
import pytest

from slicersim.simulation import Simulation


@pytest.fixture
def simu(default_config):
    return Simulation.from_config(default_config)

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

def test_get_spectrum_method(simu):
    """ optimal (default) and aperture photometry of the point source """
    _, sig_default, var_default = simu.get_spectrum()
    _, sig_opt, var_opt = simu.get_spectrum(method="optimal")
    assert np.allclose(sig_default, sig_opt) and np.allclose(var_default, var_opt)

    # aperture: the point source cube and the variance cube summed within the aperture
    simu.update(aperture="0.2-arcsec")
    position, radius, _ = simu.get_aperture(position=simu.scene.pointsource_position)
    aper = simu.spectrograph.get_aperture(position=position, radius=radius)
    nramps = simu.get_parameter("nramps")

    sig_cube, _ = simu.get_cube(switch_off=["background", "host", "thermal",
                                            "dark", "thermal_dark", "roic_glow", "ron"],
                                per_ramp=True)
    _, var_cube = simu.get_cube(per_ramp=True)
    _, sig_ap, var_ap = simu.get_spectrum(method="aperture")
    assert np.allclose(sig_ap, sig_cube[:, aper].sum(axis=-1) * nramps)
    assert np.allclose(var_ap, var_cube[:, aper].sum(axis=-1) * nramps)

    # large aperture: all the point source flux
    # (without lsf: the lsf is applied before (optimal) or after (cube) the photon conversion)
    simu.update(aperture="1-arcsec")
    _, sig_opt, _ = simu.get_spectrum(method="optimal", apply_lsf=False)
    _, sig_ap, _ = simu.get_spectrum(method="aperture", apply_lsf=False)
    flagok = sig_opt > 0.01 * np.nanmax(sig_opt)
    assert np.allclose(sig_ap[flagok], sig_opt[flagok], rtol=1e-3)

    # stored method
    simu.update(method="aperture")
    assert np.allclose(simu.get_spectrum()[1], simu.get_spectrum(method="aperture")[1])
    simu.update(method="default")
    assert np.allclose(simu.get_spectrum()[1], sig_default)

    with pytest.raises(ValueError):
        simu.get_spectrum(method="psf")

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
    request_shape  = [20, 40]
    expected_output_shape = np.asarray(request_shape) * (1, simu.spectrograph._ANAMORPHOSE)
    simu.update(spatial_shape=request_shape)
    cube, cubevar = simu.get_cube()
    assert np.all(cube.shape[1:] == expected_output_shape), f"wrong cube shape after update of spectre shape {cube.shape[1:]=} != {expected_output_shape=}"


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
    simu.update(nramps=1, nmd=(64, 8, 0)) # 1 ramp by default
    datavolume_1ramp = simu.get_data_volume("GB")
    simu.update(nramps=2) # 2 ramps as test
    datavolume_2ramp = simu.get_data_volume("GB")

    assert (datavolume_1ramp>0.01) # 1 ramp should be at least 10 MB (500 likely)
    assert (datavolume_2ramp == 2*datavolume_1ramp)


def test_fetch_snr(simu):
    """ """
    requested_snr = 30

    # fetch snr.
    simu.update(redshift=0.2)
    config, snr, integration_time = simu.fetch_snr(requested_snr)
    assert np.isclose(snr, requested_snr, 2), "snr is far from that requested."

    simu.update(redshift=0.5)
    config_mid, snr_mid, integration_time_mid = simu.fetch_snr(requested_snr)
    assert np.isclose(snr_mid, requested_snr, 2), "snr is far from that requested."

    simu.update(redshift=0.8)
    config_high, snr_high, integration_time_high = simu.fetch_snr(requested_snr)
    assert np.isclose(snr_high, requested_snr, 2), "snr is far from that requested."

    # fetch snr.
    simu.update(redshift=1.5)
    config_long, snr_long, integration_time_long = simu.fetch_snr(requested_snr)
    assert np.isclose(snr_long, requested_snr, 2), "snr is far from that requested."

    assert integration_time <= integration_time_mid <= integration_time_high <= integration_time_long
    assert config["nramps"] <= config_mid["nramps"] <= config_high["nramps"] <= config_long["nramps"]


def test_get_scene_cubes(simu):
    """ """
    cubes = simu.get_scene_cubes(unit="flambda")

    # you have one scene per diffracted source of light.
    assert list(cubes.keys()) == ['pointsource', 'background', 'host_cube', 'thermal_cube']

    # Test pointsource is correct. comparable as unit=flambda
    pointsources = cubes["pointsource"]

    lbda_ref = simu.spectrograph.lbda
    spectrum_pointsource = pointsources.sum( (-2,-1) )
    _, spec_in = simu.scene.pointsource.get_spectrum(lbda_ref)
    assert np.isclose(spectrum_pointsource, spec_in, rtol=1e-4).all()

    # test flat background is indeed flat
    slicefull = cubes["background"].sum(0)

    assert len( np.unique(slicefull)) == 1

def test_conversion(simu):

    # test self-consistancy
    assert simu.convert_units("flambda", "flambda") == 1
    assert simu.convert_units("fphoton", "fphoton") == 1
    assert simu.convert_units("rate", "rate") == 1

    assert np.asarray(simu.convert_units("flambda", "adu") == 1/simu.convert_units("adu", "flambda")).all()
    assert np.isclose(simu.convert_units("fphoton", "adu"), 1/simu.convert_units("adu", "fphoton")).all()
    assert np.asarray(simu.convert_units("framerate", "adu") == 1/simu.convert_units("adu", "framerate")).all()

    # test basics
    assert (simu.convert_units("fphoton", "adu") > 1).all(), "rate to adu test failed."

    # combination
    assert simu.convert_units("rate", "adu") == simu.detector.exposure_time * simu.get_parameter("nramps"), "rate to adu test failed."

def test_pixel_variance(simu):
    """ """
    simu.update(nramps=1) # make sure this is at 1 ramp.
    pixel_variance = simu.get_pixel_variance()
    assert np.asarray(pixel_variance <= simu.detector.ron**2).all(), "pixel variance exceeds ron"
    assert np.asarray(pixel_variance >= 0).all(), "pixel variance is negative"


def test_background_spectrum(simu):
    """ """
    bkgd_spec_adu = simu.get_background_spectrum(unit="adu")
    assert bkgd_spec_adu.shape == simu.spectrograph.lbda.shape
    assert np.all(bkgd_spec_adu >= 0)

    bkgd_spec_flambda = simu.get_background_spectrum(unit="flambda")
    spec_adu = simu.convert_units("flambda", "adu", bkgd_spec_flambda)

    assert np.isclose(bkgd_spec_adu, spec_adu, rtol=0.1).all()


def test_fetch_snr_caching(simu):
    """ """
    _cached = simu.fetch_snr(30, use_cache=True)
    _not_cached = simu.fetch_snr(30, use_cache=False)
    assert _cached == _not_cached

    _cached = simu.fetch_snr(10, use_cache=True)
    _not_cached = simu.fetch_snr(10, use_cache=False)
    assert _cached == _not_cached

    simu.update(ron=50, dark=5e-3)
    _cached = simu.fetch_snr(10, use_cache=True)
    _not_cached = simu.fetch_snr(10, use_cache=False)
    assert _cached == _not_cached


def test_public_accessors(simu):
    """Public accessors for subclasses and add-ons."""
    assert simu.elements == ["scene", "telescope", "spectrograph", "detector", "extraction"]

    config = simu.input_config
    config["extraction"]["nramps"] = -1                 # a copy
    assert simu.input_config["extraction"]["nramps"] != -1
    fresh = simu.__class__.from_config(simu.input_config)
    assert fresh.cube_shape == simu.cube_shape

    cube = simu.get_projected_scene(cached=True)
    assert simu.get_projected_scene(cached=True) is cube
    simu.reset_cache()
    assert simu.get_projected_scene(cached=True) is not cube

    # cached spectrum (projected scene, thermal, psf) identical to the non-cached one
    _, _, variance = simu.get_spectrum(cached=False)
    _, _, variance_cached = simu.get_spectrum(cached=True)
    assert np.array_equal(variance, variance_cached)
    assert simu.spectrograph._cached_psf is not None
    simu.reset_cache()
    assert simu.spectrograph._cached_psf is None

    background = simu.scene.background
    simu.scene.set_background(None)
    assert simu.scene.background is None
    simu.scene.set_background(background)
    assert simu.scene.background is background


def test_slicer_one_pixel_per_voxel(simu):
    """ slicer: one voxel (spaxel x wavelength bin) is one detector pixel (see docs/anamorphosis.rst) """
    assert simu.spectrograph.type == "slicer"

    # empty cube with read-out noise only: each voxel has the read noise of one pixel.
    switch_off = [k for k in simu.variance_sources if k != "ron"]
    signal, variance = simu.get_cube(switch_off=switch_off, per_ramp=True)
    ron_variance = simu.detector.get_effective_ron(as_variance=True) * simu.detector.gain**2 # [ADU²]
    assert np.allclose(signal, 0)
    assert np.allclose(variance, ron_variance), "an empty voxel should have the read noise of 1 detector pixel, not 2"


# ================================ #
#  fetch_snr speed-ups: exactness  #
# ================================ #
SNR_PROP = dict(lbda_range=[4000, 6800], frame="rest")

def _snr_at(simu, nmd, nramps, **kwargs):
    """ band SNR of the simulation at the given read-out configuration (state left changed) """
    simu.update(nmd=nmd, nramps=nramps)
    return simu.get_band_snr(**(SNR_PROP | kwargs))


# tests for the sqrt(nramps) scaling used by fetch_snr
def test_snr_scales_as_sqrt_nramps(simu):
    """ signal and variance both scale with nramps: SNR(n) = sqrt(n) SNR(1) """
    for redshift in [0.5, 1.5]:
        simu.update(redshift=redshift)
        snr_1 = _snr_at(simu, (64, 8, 0), 1)
        for nramps in [2, 3, 7, 20]:
            assert np.isclose(_snr_at(simu, (64, 8, 0), nramps), np.sqrt(nramps) * snr_1, rtol=1e-12, atol=0)

    # saturation is per ramp: it does not depend on nramps.
    simu.update(redshift=0.01)
    assert np.isinf(_snr_at(simu, (64, 8, 0), 1)) and np.isinf(_snr_at(simu, (64, 8, 0), 5))


def test_snr_increases_with_exposure(simu):
    """ physical sanity: longer ramps or more ramps give a larger SNR """
    simu.update(redshift=1.)
    snrs_group = [_snr_at(simu, (ngroup, 8, 0), 1) for ngroup in [4, 8, 16, 32, 64]]
    snrs_ramps = [_snr_at(simu, (32, 8, 0), nramps) for nramps in [1, 2, 4, 8]]
    assert np.all(np.diff(snrs_group) > 0), "SNR should increase with the number of groups"
    assert np.all(np.diff(snrs_ramps) > 0), "SNR should increase with the number of ramps"


# tests for the func fetch_snr
def test_fetch_snr_multiramp_minimal(simu):
    """ several ramps: the smallest number of full ramps reaching the target is used """
    simu.update(redshift=1.5)
    max_group = simu.detector.max_group
    for target_snr in [5, 10, 20]:
        config, snr, exptime = simu.fetch_snr(target_snr, allow_smaller_ramps=False, nframe_per_group=8)
        nramps = config["nramps"]
        assert nramps >= 2 and tuple(config["nmd"]) == (max_group, 8, 0)

        # consistent with the simulation at that configuration
        assert np.isclose(_snr_at(simu, config["nmd"], nramps), snr, rtol=1e-12, atol=0)
        assert np.isclose(simu.observing_time, exptime)

        # minimal: reached with nramps, not with nramps-1
        assert snr >= target_snr
        assert nramps == 2 or _snr_at(simu, config["nmd"], nramps - 1) < target_snr


def test_fetch_snr_multiramp_finetuned(simu):
    """ allow_smaller_ramps: groups are reduced, never longer than full ramps """
    simu.update(redshift=1.5)
    for target_snr in [5, 10, 20]:
        config_full, snr_full, exptime_full = simu.fetch_snr(target_snr, allow_smaller_ramps=False, nframe_per_group=8)
        config, snr, exptime = simu.fetch_snr(target_snr, nframe_per_group=8)
        assert exptime <= exptime_full
        assert config["nmd"][0] <= simu.detector.max_group
        assert np.isclose(_snr_at(simu, config["nmd"], config["nramps"]), snr, rtol=1e-12, atol=0)
        assert snr > target_snr - 1, "reached snr far below target"


def test_fetch_snr_single_ramp(simu):
    """ bright targets: one ramp, returned values consistent with the simulation """
    for redshift, target_snr in [(0.1, 10), (0.5, 10), (0.01, 10)]: # 0.01: full ramp saturates
        simu.update(redshift=redshift, nmd=(16, 8, 0), nramps=1)
        config, snr, exptime = simu.fetch_snr(target_snr)
        assert config["nramps"] == 1
        assert np.isfinite(snr) and snr > target_snr - 1
        assert np.isclose(_snr_at(simu, config["nmd"], 1), snr, rtol=1e-12, atol=0)
        assert np.isclose(simu.observing_time, exptime)


def test_fetch_snr_reset_param(simu):
    """ reset_param: input configuration restored, or kept at the solution """
    simu.update(redshift=1.5, nmd=(16, 8, 0), nramps=1)
    config, _, _ = simu.fetch_snr(10)
    assert tuple(simu.get_parameter("nmd")) == (16, 8, 0) and simu.get_parameter("nramps") == 1

    config, _, _ = simu.fetch_snr(10, reset_param=False)
    assert tuple(simu.get_parameter("nmd")) == tuple(config["nmd"])
    assert simu.get_parameter("nramps") == config["nramps"]


# tests for the caches (projected scene, psf, electron rate, thermal) and incl_signal
def test_cached_results_identical(simu):
    """ cached results are bitwise those computed from scratch, for any read-out (incl. saturation) """
    for redshift in [0.01, 0.5, 1.5]:
        simu.update(redshift=redshift)
        simu.reset_cache(keep_thermal=False)
        _ = _snr_at(simu, (64, 8, 0), 1, cached=True)  # fills the caches with another read-out
        for nmd, nramps in [((64, 8, 0), 1), ((8, 4, 0), 3), ((2, 1, 0), 1), ((20, 8, 2), 2)]:
            simu.update(nmd=nmd, nramps=nramps)
            sig, var = simu.get_cube(cached=False)
            nsat = simu.detector.nsaturated_detpx
            sig_c, var_c = simu.get_cube(cached=True)
            assert np.array_equal(sig, sig_c, equal_nan=True) and np.array_equal(var, var_c)
            assert simu.detector.nsaturated_detpx == nsat
            for method in ["optimal", "aperture"]:
                spec = simu.get_spectrum(method=method, cached=False)
                spec_c = simu.get_spectrum(method=method, cached=True)
                assert all(np.array_equal(a, b) for a, b in zip(spec, spec_c))
            assert simu.get_band_snr(**SNR_PROP) == simu.get_band_snr(**SNR_PROP, cached=True)
        simu.reset_cache()

    # fetch_snr itself
    simu.update(redshift=1.5)
    assert simu.fetch_snr(10, use_cache=True) == simu.fetch_snr(10, use_cache=False)


def test_get_cube_incl_signal(simu):
    """ incl_signal=False only skips the signal cube: same variance and saturation """
    for redshift in [0.01, 1.]: # saturated and not
        simu.update(redshift=redshift, nmd=(64, 8, 0), nramps=2)
        for per_ramp in [True, False]:
            sig, var = simu.get_cube(per_ramp=per_ramp)
            nsat = simu.detector.nsaturated_detpx
            no_sig, var_nosig = simu.get_cube(per_ramp=per_ramp, incl_signal=False)
            assert no_sig is None
            assert np.array_equal(var, var_nosig)
            assert simu.detector.nsaturated_detpx == nsat
        assert (nsat > 0) == (redshift == 0.01)


def test_thermal_dark_inputs(simu):
    """ the thermal dark is reused only for unchanged inputs (and dispersed elements do not matter) """
    detector = simu.detector
    simu.reset_cache(keep_thermal=False)
    reference = detector.get_thermal_dark(units="e-/s", cached=True)
    inputs = detector.get_thermal_inputs(units="e-/s")
    dispersed = np.asarray(detector.thermaloptics.meta["dispersed"]).astype(bool)

    # nelements of a dispersed element (e.g. the IFU of a field): same inputs, same thermal dark
    # (n, 1) column, as stored: ThermalOptics.update uses np.atleast_2d
    nelements = np.array(detector.thermaloptics.nelements, dtype=float)
    nelements[np.argmax(dispersed)] += 2
    simu.update(optics__nelements=nelements)
    assert detector.get_thermal_inputs(units="e-/s") == inputs
    assert np.array_equal(detector.get_thermal_dark(units="e-/s"), reference)

    # temperature (integrated together with the others): new inputs, recomputed
    temperature = np.array(detector.thermaloptics.temperature, dtype=float) + 10
    simu.update(optics__temperature=temperature)
    assert detector.get_thermal_inputs(units="e-/s") != inputs
    hotter = detector.get_thermal_dark(units="e-/s", cached=True)
    assert np.all(hotter > reference), "hotter optics should emit more"
    assert np.array_equal(hotter, detector.get_thermal_dark(units="e-/s"))

    # kept by reset_cache (called by fetch_snr)
    _ = simu.fetch_snr(10)
    assert detector._cached_thermal is not None
