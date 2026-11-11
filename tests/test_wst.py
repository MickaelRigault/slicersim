"""Tests of the ground-based (WST) slicer IFS: atmosphere, spectrograph, CCD, simulation."""
import numpy as np
import pytest

from slicersim.atmosphere import Atmosphere, get_seeing_fwhm
from slicersim.wst import WST_INSTRUMENTS, WSTTarget, wst_etc

LBDA = np.linspace(3000, 11000, 8001)
FLUX = np.full_like(LBDA, 1e-17)  # flat F_lambda [erg/s/cm²/Å]


@pytest.fixture(scope="module")
def blue():
    return WSTTarget(LBDA, FLUX, channel="blue")


def test_seeing_fwhm():
    lbda = np.array([4000., 5000., 9000.])
    fwhm = get_seeing_fwhm(lbda, 0.8, airmass=1.0, diameter=12.)
    assert np.all(np.diff(fwhm) < 0)                     # λ^-1/5
    assert fwhm[1] < 0.8                                 # outer-scale correction
    np.testing.assert_allclose(get_seeing_fwhm(lbda, 0.8, airmass=1.5, diameter=12.) / fwhm,
                               1.5 ** 0.6)
    np.testing.assert_allclose(get_seeing_fwhm(5500, 0.8, diameter=8.2), 0.6555, atol=1e-4)


def test_atmosphere():
    atm = Atmosphere("wst_ifs_blue_sky.ecsv")
    assert atm.airmasses == [1.0, 1.2, 1.5, 2.0]
    assert atm.moons == ["brightsky", "darksky", "greysky"]
    lbda = np.linspace(3600, 6200, 100)                  # beyond the table: edge values
    trans = atm.get_transmission(lbda)
    assert np.all((trans > 0) & (trans <= 1)) and np.all(np.isfinite(atm.get_emission(lbda)))

    dark = atm.get_emission(lbda)
    atm.update(moon="brightsky")
    assert np.all(atm.get_emission(lbda) > dark)
    atm.update(airmass=2.0)
    assert np.all(atm.get_transmission(lbda) < trans)
    with pytest.raises(ValueError):
        atm.update(airmass=1.3)                          # not tabulated
    with pytest.raises(ValueError):
        atm.update(moon="fullmoon")


@pytest.mark.parametrize("channel, step, lmin, nlbda", [("blue", 0.64, 3700, 3751),
                                                         ("red", 0.97, 6000, 3712)])
def test_configuration(channel, step, lmin, nlbda):
    sim = WSTTarget(LBDA, FLUX, channel=channel).simulation
    sp = sim.spectrograph
    assert sp.nlbda == nlbda
    np.testing.assert_allclose(np.diff(sp.lbda), step, atol=1e-6)
    np.testing.assert_allclose(sp.lbda[0], lmin, atol=1e-6)
    np.testing.assert_allclose(sp.get_resolving_power(), sp.lbda / (2.5 * step), rtol=1e-6)
    assert sim.telescope.surface == pytest.approx(82.)
    assert sp.type == "slicer" and tuple(sp.spx_shape) == (25, 25)   # no anamorphosis
    assert np.all(sp.generate_thermal_signal() == 0)
    assert WST_INSTRUMENTS[channel] == f"wst_ifs_{channel}.toml"


def test_psf(blue):
    sp = blue.simulation.spectrograph
    seeing, optical, total = (sp.get_psf_fwhm([5000.], which=w)[0] for w in ("seeing", "optical", "total"))
    assert optical == pytest.approx(0.10)
    assert total == pytest.approx(np.hypot(seeing, optical))

    psf = sp.get_spatial_psf()
    assert psf.shape == (sp.nlbda, 25, 25)
    assert np.all((psf.sum(axis=(1, 2)) > 0.99) & (psf.sum(axis=(1, 2)) < 1))  # Moffat wings
    seeing_only = sp.get_spatial_psf(incl_telescope=False)
    assert np.all(seeing_only[:, 12, 12] > psf[:, 12, 12])                 # narrower: higher peak
    with pytest.raises(ValueError):
        sp.get_spatial_psf(incl_atmosphere=False, incl_telescope=False)

    blue.change_properties(moffat_truncation=2)
    try:
        np.testing.assert_allclose(sp.get_spatial_psf().sum(axis=(1, 2)), 1)
    finally:
        blue.change_properties(moffat_truncation=0)


def test_transmission_not_on_sky(blue):
    sp = blue.simulation.spectrograph
    trans = sp.get_atmospheric_transmission()
    assert np.all(trans < 1)
    np.testing.assert_allclose(sp.flambda2photon, sp.flambda2photon_sky * trans)
    sky = np.ones(sp.nlbda)
    np.testing.assert_allclose(sp.generate_background(sky, apply_lsf=False)[:, 0, 0],
                               sp.flambda2photon_sky * sp.spx_area)


def test_ccd():
    det = WSTTarget(LBDA, FLUX).simulation.detector
    assert det.dit == 1800. and det.integration_time == det.exposure_time == 1800.
    flux = np.array([0., 1., 10.])                                 # e-/s
    np.testing.assert_allclose(det.estimate_variance(flux),
                               (flux + 3. / 3600) * 1800 + 3. ** 2)
    det.update(dit=600.)
    assert det.integration_time == 600.


def test_pyetc_regression():
    """pyetc resolved-source path (snr_from_cube) gives SNR=23.0663 at 5000 Å."""
    target = WSTTarget(LBDA, FLUX, channel="blue", seeing=0.8, airmass=1.0, moon="darksky")
    target.change_properties(dit=1800., ndit=2, moffat_truncation=2)
    lbda, signal, variance = target.simulation.get_spectrum(method="box", box_hsize=2)
    k = np.argmin(np.abs(lbda - 5000))
    assert signal[k] / np.sqrt(variance[k]) == pytest.approx(23.0663, rel=2e-4)


def test_observing_conditions():
    target = WSTTarget(LBDA, FLUX, channel="red")
    sim = target.simulation
    band = [7300, 7700]
    snr = target.get_band_snr(band)
    for kwargs in (dict(seeing=1.2), dict(moon="greysky"), dict(airmass=1.5), dict(optical_fwhm=0.4)):
        target.change_properties(**kwargs)
        new_snr = target.get_band_snr(band)
        assert new_snr < snr
        snr = new_snr
    assert sim.observing_conditions == {"seeing": 1.2, "airmass": 1.5, "moon": "greysky"}
    assert sim.get_parameter("seeing", as_dict=False) == 1.2
    assert sim.get_parameter("optical_fwhm", as_dict=False) == 0.4

    target.change_properties(ndit=3)
    assert target.get_readout_config() == {"dit": 1800., "ndit": 3}
    sim.reset()
    assert sim.observing_conditions == {"seeing": 0.8, "airmass": 1.0, "moon": "darksky"}
    assert sim.get_parameter("optical_fwhm", as_dict=False) == pytest.approx(0.1)
    assert target.get_readout_config() == {"dit": 1800., "ndit": 1}


def test_extraction(blue):
    sim = blue.simulation
    for method in ("optimal", "aperture", "box"):
        lbda, signal, variance = sim.get_spectrum(method=method)
        assert np.all(np.isfinite(signal)) and np.all(variance > 0)
    var = sim.get_variance_contribution()
    assert {"dark", "ron", "pointsource", "background", "host"} <= set(var.columns)


def test_etc():
    target = WSTTarget(LBDA, FLUX * 0.1, channel="blue")
    sim = target.simulation
    band = [4800, 5200]
    config, snr, total = sim.fetch_snr(10, lbda_range=band, frame="obs", dit=1800.)
    assert snr >= 10 and total == config["nramps"] * 1800.
    assert sim.get_parameter("ndit") == 1                       # reset_param
    bright = WSTTarget(LBDA, FLUX, channel="blue").simulation   # SNR~15 in 1x1800 s
    config, snr, total = bright.fetch_snr(10, lbda_range=band, frame="obs", ndit=1)
    assert config["nramps"] == 1 and config["dit"] < 1800 and snr == pytest.approx(10, rel=1e-2)
    with pytest.warns(UserWarning, match="not reached"):
        sim.fetch_snr(1000, lbda_range=band, frame="obs", ndit=1)

    exptime, target = wst_etc(LBDA, FLUX * 0.1, 10, channel="blue", lbda_range=band)
    assert target.get_band_snr(band, per_resolution=False) >= 10
    assert exptime == target.get_readout_config()["ndit"] * 1800.
