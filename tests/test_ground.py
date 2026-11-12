"""Tests of the ground-based slicer IFS: atmosphere, spectrograph, CCD, simulation.

The instrument is a synthetic one, written in a temporary directory: this also
tests that the data files of a configuration are found next to it.
"""
import numpy as np
import pytest
from astropy.table import Table

from slicersim.atmosphere import Atmosphere, get_seeing_fwhm
from slicersim.ground import GroundTarget, ground_etc
from slicersim.iotools import read_config

LMIN, LMAX, STEP = 4000., 6000., 1.0
LBDA = np.linspace(3000, 7000, 4001)
FLUX = np.full_like(LBDA, 1e-17)  # flat F_lambda [erg/s/cm²/Å]

TOML = f'''
[atmosphere]
site = "nowhere"
sky_table = "test_sky.ecsv"
seeing = 0.8
airmass = 1.0
moon = "darksky"

[telescope]
diameter_ext = 8.0
diameter_int = 0.0
surface = 50.0
temperature = 0
emissivity = 0

[spectrograph]
name = "test_ifs"
spectral_range = [{LMIN:.0f}, {LMAX:.0f}]
dispersion_law = "test_dispersion.ecsv"
dispersion_resolution = 2.5
dispersion_scale = 0.8
spatial_scale = 0.25
spatial_shape = [15, 15]
lbda_ref = 5000

[spectrograph.throughput]
throughput = "test_throughput.csv"

[spectrograph.optics]
temperature = [0]
emissivity = [0]
nelements = [1]
fratio = [[1, 1]]
geometry = ["circular"]
dispersed = [false]

[spectrograph.psf.spatial]
psf_profile = "moffat"
optical_fwhm = 0.1
moffat_beta = 2.5
moffat_truncation = 0
psf_sigma_spectral = 0.3
guiding_sigma = 0.0

[detector]
shape = [4096, 4096]
QE = 1.0
gain = 1.0
saturation = 65535
ron = 3.0
dark = {3 / 3600:.6g}
pixel_size = 15.0
dit = 1800.0

[extraction]
nramps = 1
aperture_radius = 6
box_hsize = 2
'''


@pytest.fixture(scope="module")
def instrument(tmp_path_factory):
    """Path to a synthetic ground-based instrument configuration."""
    path = tmp_path_factory.mktemp("instrument")
    lbda = np.arange(LMIN, LMAX + STEP / 2, STEP)
    with open(path / "test_throughput.csv", "w") as f:
        f.write("wavelength,test_ifs\n")
        f.writelines(f"{l:.3f},0.3\n" for l in lbda)

    nodes = np.linspace(LMIN, LMAX, 11)
    dtab = Table({"wavelength": nodes / 1e4, "offset": (nodes - LMIN) / STEP})
    dtab["wavelength"].unit, dtab["offset"].unit = "micron", "pix"
    dtab.write(path / "test_dispersion.ecsv", format="ascii.ecsv")

    wave = np.arange(LMIN - 50, LMAX + 50, 0.5)
    band = np.exp(-0.5 * ((wave - 5500) / 5) ** 2)               # an absorption band
    lines = np.exp(-0.5 * ((wave - 5577) / 1) ** 2)               # a sky line
    sky = Table({"wavelength": wave})
    for airmass in (1.0, 1.5):
        sky[f"transmission_{airmass:.1f}"] = 0.95 ** airmass * (1 - 0.5 * airmass * band / 1.5)
        for moon, level in (("darksky", 1e-17), ("brightsky", 5e-17)):
            sky[f"emission_{moon}_{airmass:.1f}"] = level * airmass * (1 + 20 * lines)
    sky.write(path / "test_sky.ecsv", format="ascii.ecsv")

    toml = path / "test_ifs.toml"
    toml.write_text(TOML)
    return str(toml)


@pytest.fixture(scope="module")
def target(instrument):
    return GroundTarget(LBDA, FLUX, instrument)


def test_config_paths(instrument):
    config = read_config(instrument)
    directory = instrument.rsplit("/", 1)[0]
    assert config["atmosphere"]["sky_table"] == f"{directory}/test_sky.ecsv"
    assert config["spectrograph"]["throughput"]["throughput"] == f"{directory}/test_throughput.csv"
    assert config["atmosphere"]["moon"] == "darksky"                   # not a file: unchanged
    assert read_config("lazuli_cbe.toml")["spectrograph"]["throughput"]["throughput"] == "throughput_bol.csv"


def test_seeing_fwhm():
    lbda = np.array([4000., 5000., 9000.])
    fwhm = get_seeing_fwhm(lbda, 0.8, airmass=1.0, diameter=12.)
    assert np.all(np.diff(fwhm) < 0)                     # λ^-1/5
    assert fwhm[1] < 0.8                                 # outer-scale correction
    np.testing.assert_allclose(get_seeing_fwhm(lbda, 0.8, airmass=1.5, diameter=12.) / fwhm,
                               1.5 ** 0.6)
    np.testing.assert_allclose(get_seeing_fwhm(5500, 0.8, diameter=8.2), 0.6555, atol=1e-4)


def test_atmosphere(instrument):
    atm = Atmosphere.from_config(read_config(instrument)["atmosphere"])
    assert atm.airmasses == [1.0, 1.5]
    assert atm.moons == ["brightsky", "darksky"]
    lbda = np.linspace(3800, 6200, 100)                  # beyond the table: edge values
    trans = atm.get_transmission(lbda)
    assert np.all((trans > 0) & (trans <= 1)) and np.all(np.isfinite(atm.get_emission(lbda)))

    dark = atm.get_emission(lbda)
    atm.update(moon="brightsky")
    assert np.all(atm.get_emission(lbda) > dark)
    atm.update(airmass=1.5)
    assert np.all(atm.get_transmission(lbda) < trans)
    with pytest.raises(ValueError):
        atm.update(airmass=1.3)                          # not tabulated
    with pytest.raises(ValueError):
        atm.update(moon="fullmoon")


def test_configuration(target):
    sim = target.simulation
    sp = sim.spectrograph
    np.testing.assert_allclose(np.diff(sp.lbda), STEP, atol=1e-6)
    np.testing.assert_allclose(sp.lbda[[0, -1]], [LMIN, LMAX], atol=1e-6)
    np.testing.assert_allclose(sp.get_resolving_power(), sp.lbda / (2.5 * STEP), rtol=1e-6)
    assert sim.telescope.surface == pytest.approx(50.)
    assert sp.type == "slicer" and tuple(sp.spx_shape) == (15, 15)   # no anamorphosis
    assert np.all(sp.generate_thermal_signal() == 0)


def test_psf(target):
    sp = target.simulation.spectrograph
    seeing, optical, total = (sp.get_psf_fwhm([5000.], which=w)[0] for w in ("seeing", "optical", "total"))
    assert optical == pytest.approx(0.10)
    assert total == pytest.approx(np.hypot(seeing, optical))

    psf = sp.get_spatial_psf()
    assert psf.shape == (sp.nlbda, 15, 15)
    assert np.all((psf.sum(axis=(1, 2)) > 0.97) & (psf.sum(axis=(1, 2)) < 1))  # Moffat wings
    seeing_only = sp.get_spatial_psf(incl_telescope=False)
    assert np.all(seeing_only[:, 7, 7] > psf[:, 7, 7])                     # narrower: higher peak
    with pytest.raises(ValueError):
        sp.get_spatial_psf(incl_atmosphere=False, incl_telescope=False)

    target.change_properties(moffat_truncation=2)
    try:
        np.testing.assert_allclose(sp.get_spatial_psf().sum(axis=(1, 2)), 1)
    finally:
        target.change_properties(moffat_truncation=0)


def test_transmission_not_on_sky(target):
    sp = target.simulation.spectrograph
    trans = sp.get_atmospheric_transmission()
    assert np.all(trans < 1)
    np.testing.assert_allclose(sp.flambda2photon, sp.flambda2photon_sky * trans)
    sky = np.ones(sp.nlbda)
    np.testing.assert_allclose(sp.generate_background(sky, apply_lsf=False)[:, 0, 0],
                               sp.flambda2photon_sky * sp.spx_area)


def test_ccd(instrument):
    det = GroundTarget(LBDA, FLUX, instrument).simulation.detector
    assert det.dit == 1800. and det.integration_time == det.exposure_time == 1800.
    flux = np.array([0., 1., 10.])                                 # e-/s
    np.testing.assert_allclose(det.estimate_variance(flux),
                               (flux + 3. / 3600) * 1800 + 3. ** 2)
    det.update(dit=600.)
    assert det.integration_time == 600.


def test_observing_conditions(instrument):
    target = GroundTarget(LBDA, FLUX, instrument)
    sim = target.simulation
    band = [5200, 5800]
    snr = target.get_band_snr(band)
    for kwargs in (dict(seeing=1.2), dict(moon="brightsky"), dict(airmass=1.5), dict(optical_fwhm=0.4)):
        target.change_properties(**kwargs)
        new_snr = target.get_band_snr(band)
        assert new_snr < snr
        snr = new_snr
    assert sim.observing_conditions == {"seeing": 1.2, "airmass": 1.5, "moon": "brightsky"}
    assert sim.get_parameter("seeing", as_dict=False) == 1.2
    assert sim.get_parameter("optical_fwhm", as_dict=False) == 0.4

    target.change_properties(ndit=3)
    assert target.get_readout_config() == {"dit": 1800., "ndit": 3}
    sim.reset()
    assert sim.observing_conditions == {"seeing": 0.8, "airmass": 1.0, "moon": "darksky"}
    assert sim.get_parameter("optical_fwhm", as_dict=False) == pytest.approx(0.1)
    assert target.get_readout_config() == {"dit": 1800., "ndit": 1}


def test_box_extraction(target):
    """The box extraction sums the cubes over (2n+1)² spaxels."""
    sim = target.simulation
    for method in ("optimal", "aperture", "box"):
        lbda, signal, variance = sim.get_spectrum(method=method)
        assert np.all(np.isfinite(signal)) and np.all(variance > 0)
    _, s1, v1 = sim.get_spectrum(method="box", box_hsize=1)
    _, s2, v2 = sim.get_spectrum(method="box", box_hsize=2)
    assert np.all(s2 > s1) and np.all(v2 > v1)
    var = sim.get_variance_contribution()
    assert {"dark", "ron", "pointsource", "background", "host"} <= set(var.columns)


def test_etc(instrument):
    band = [4800, 5200]
    faint = GroundTarget(LBDA, FLUX * 0.1, instrument)
    sim = faint.simulation
    config, snr, total = sim.fetch_snr(10, lbda_range=band, frame="obs", dit=1800.)
    assert snr >= 10 and total == config["nramps"] * 1800.
    assert sim.get_parameter("ndit") == 1                       # reset_param
    with pytest.warns(UserWarning, match="not reached"):
        sim.fetch_snr(1000, lbda_range=band, frame="obs", ndit=1)

    bright = GroundTarget(LBDA, FLUX, instrument)
    snr1 = bright.get_band_snr(band, per_resolution=False)      # 1 x 1800 s
    config, snr, total = bright.simulation.fetch_snr(0.5 * snr1, lbda_range=band, frame="obs", ndit=1)
    assert config["nramps"] == 1 and config["dit"] < 1800 and snr == pytest.approx(0.5 * snr1, rel=1e-2)

    exptime, faint = ground_etc(faint, 10, lbda_range=band)
    assert faint.get_band_snr(band, per_resolution=False) >= 10
    assert exptime == faint.get_readout_config()["ndit"] * 1800.
