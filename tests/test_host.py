import warnings

import pytest
import numpy as np
import sncosmo
from astropy.cosmology import Planck18

from slicersim.scene import host
from slicersim.scene.sources import galaxy
from slicersim.scene.sources.blackbody import get_blackbody_flux
from slicersim.scene.sources.generic import generic_get_flux


def _has_galaxy_data():
    try:
        import lazuli_astrosim_data # noqa: F401
    except ImportError:
        return False
    return True

requires_galaxy_data = pytest.mark.skipif(not _has_galaxy_data(),
                                          reason="lazuli_astrosim_data is not installed")


def _rest_frame_absmag(lbda_obs_func, redshift, band="sdssr"):
    """ rest-frame absolute magnitude of an observed spectrum lbda_obs_func(lbda) """
    lbda_rest = np.linspace(2_000, 12_000, 5_000)
    flux_obs = lbda_obs_func(lbda_rest * (1 + redshift))
    flux_10pc = flux_obs * (1 + redshift) * 10**(0.4 * Planck18.distmod(redshift).value)
    return sncosmo.Spectrum(lbda_rest, flux_10pc).bandmag(band, "ab")


# default_config is defined as a fixture in conftest
@pytest.fixture
def default_host():
    config = host.get_host_config("sersic:bb", redshift=1)
    return host.Host.from_config( config )


# =============== #
#  modelname      #
# =============== #
@pytest.mark.parametrize("modelname, expected", [
    ("sersic:bb", ("sersic", "bb")),
    ("sersic:bb-6000", ("sersic", "bb-6000")),
    ("sersic:blackbody-6000", ("sersic", "blackbody-6000")),
    ("sersic:ngc0628", ("sersic", "ngc0628")),
    ("ngc0628", ("ngc0628", "ngc0628")),
    ("NGC0628", ("ngc0628", "ngc0628")),
    ("ngc0628:bb-4000", ("ngc0628", "bb-4000")),
    ])
def test_parse_modelname(modelname, expected):
    assert host.parse_modelname(modelname) == expected


@pytest.mark.parametrize("modelname", ["sersic", "bb-5000", "gauss:bb", "sersic:powerlaw", "sersic:bb5000"])
def test_parse_modelname_unknown(modelname):
    with pytest.raises(NotImplementedError):
        host.parse_modelname(modelname)


def test_get_host_config():
    config = host.get_host_config("sersic:bb", redshift=1)
    assert type(config) is dict
    assert "temperature" in config # as for bb
    assert "n" in config # as in sersic
    assert config["redshift"] == 1
    assert config["magabs"] == -21

    # temperature given in the model name
    assert "temperature" not in host.get_host_config("sersic:bb-6000")

    # ngc image: no sersic parameters
    config = host.get_host_config("ngc0628")
    assert config["refmodel"] == "ngc0628:ngc0628"
    assert "n" not in config and "r_kpc" not in config

    # kwargs overwrite
    assert host.get_host_config("sersic:bb", n=4)["n"] == 4

    # an observed magnitude replaces the default absolute one.
    config = host.get_host_config("sersic:bb", mag=22)
    assert config["mag"] == 22 and "magabs" not in config


# =============== #
#  Host           #
# =============== #
def test_instanciation(default_host):
    """ the temperature is read from the model name """
    assert isinstance(default_host, host.Host), "failed to instanciate the Host"
    assert default_host.meta["temperature"] == 5000

    bb6000 = host.Host.from_modelname("sersic:bb-6000")
    assert bb6000.meta["temperature"] == 6000


def test_get_cube(default_host):
    """ cube shape is (nlbda, ny, nx), with shape given as (nx, ny) """
    lbda = np.linspace(1_000, 30_000, 1_000)
    _, cube = default_host.get_cube(lbda)
    assert cube.ndim == 3, "cube is not 3D"
    assert cube.shape[0] == len(lbda), "cube wavelength axis does not match"

    # shape is given as (nx, ny), cube is (nlbda, ny, nx)
    _, cube = default_host.get_cube(lbda, shape=(30, 20), pixel_scale=0.05)
    assert cube.shape == (len(lbda), 20, 30)

    with pytest.raises(ValueError):
        default_host.get_cube(lbda, not_a_parameter=1)


def test_get_spectrum(default_host):
    lbda = np.linspace(4_000, 9_000, 100)
    _, spec = default_host.get_spectrum(lbda)
    _, spec_cube = default_host.get_spectrum(lbda, from_cube=True)
    assert spec.shape == lbda.shape
    # the grid contains only a fraction of the host
    assert np.all(spec_cube <= spec)


def test_update_spatial(default_host):
    """ updating a spatial parameter recomputes the cached image """
    image = default_host._current_spatial_model.copy()
    updated = default_host.update(n=4)
    assert updated == ["n"]
    assert not np.allclose(image, default_host._current_spatial_model)

    # a spectral parameter does not change the image
    image = default_host._current_spatial_model.copy()
    default_host.update(temperature=8000)
    assert np.allclose(image, default_host._current_spatial_model)


def test_from_sersic_and_spectrum():
    lbda_ref = np.linspace(2_000, 12_000, 500)
    flux_ref = np.ones_like(lbda_ref) * 1e-15
    this = host.Host.from_sersic_and_spectrum(lbda_ref, flux_ref, mag=20, r_eff=0.5,
                                              shape=(40, 40))
    lbda = np.linspace(4_000, 9_000, 50)
    assert this.get_cube(lbda)[1].shape == (50, 40, 40)

    this = host.Host.from_sersic_and_blackbody(5000, mag=20, r_eff=0.5, shape=(40, 40))
    assert this.get_cube(lbda)[1].shape == (50, 40, 40)


# =============== #
#  Spatial        #
# =============== #
def test_sersic_normalization():
    """ the profile is normalized to the total (analytic) flux """
    for n in [1, 4]:
        image = galaxy.get_sersic_profile(r_eff=0.3, n=n, ellip=0.3, theta=20,
                                          pixel_scale=0.05, shape=(400, 400))
        # n=4 has very extended wings.
        assert np.isclose(image.sum(), 1, atol=0.01 if n == 1 else 0.05)


def test_sersic_position():
    """ position is (dx, dy) in pixels from the grid center, image is (ny, nx) """
    image = galaxy.get_sersic_profile(r_eff=0.2, n=1, pixel_scale=0.1, shape=(41, 31),
                                      position=(5, -3))
    assert image.shape == (31, 41)
    y, x = np.indices(image.shape)
    xc, yc = (x * image).sum() / image.sum(), (y * image).sum() / image.sum()
    assert np.isclose(xc, 20 + 5, atol=0.05) and np.isclose(yc, 15 - 3, atol=0.05)


def test_sersic_redshift_size():
    """ r_kpc is converted to arcsec with the angular diameter distance """
    for z in [0.1, 1]:
        r_kpc = galaxy.get_sersic_profile(r_kpc=5, redshift=z, n=1, pixel_scale=0.05, shape=(60, 60))
        r_eff = galaxy.get_sersic_profile(r_eff=5 * Planck18.arcsec_per_kpc_proper(z).value,
                                          n=1, pixel_scale=0.05, shape=(60, 60))
        assert np.allclose(r_kpc, r_eff)


def test_image_projection():
    """ flux conservation, position and rotation of the galaxy image projection """
    image = np.zeros((101, 101))
    image[50, 50:71] = 1 # a bar along +x from the center
    image /= image.sum()
    center = (50, 50)

    # fully contained: flux is conserved.
    proj = galaxy.GalaxySpatialModel._image_projection(image, center, scale=0.5, shape=(40, 60))
    assert proj.shape == (40, 60)
    assert np.isclose(proj.sum(), 1)

    # rotation by 90 deg (counterclockwise): the bar goes along +y.
    proj = galaxy.GalaxySpatialModel._image_projection(image, center, scale=0.5, shape=(41, 41),
                                                       rotation=90)
    y, x = np.indices(proj.shape)
    assert np.isclose((x * proj).sum(), 20, atol=0.1)
    assert (y * proj).sum() > 22

    # galaxy out of the grid
    proj = galaxy.GalaxySpatialModel._image_projection(image, center, scale=0.5, shape=(10, 10),
                                                       position=(500, 500))
    assert proj.sum() == 0


def test_galaxy_pixel_scale():
    """ the physical size is conserved, the angular size scales as 1/D_A """
    gal = galaxy.GalaxySpatialModel(np.ones((10, 10)), pixel_scale=1., redshift=0.01, center=(4.5, 4.5))
    for z in [0.1, 1]:
        expected = (Planck18.angular_diameter_distance(0.01) / Planck18.angular_diameter_distance(z)).value
        assert np.isclose(gal.get_pixel_scale(z), expected)

    with pytest.raises(ValueError):
        gal.get_pixel_scale(0.005) # closer than the galaxy itself


# =============== #
#  Spectral       #
# =============== #
@pytest.mark.parametrize("redshift", [0.1, 0.9, 1.5])
def test_magabs_is_restframe(redshift):
    """ magabs is the rest-frame absolute magnitude, whatever the redshift (K-corrected). """
    absmag = _rest_frame_absmag(lambda l: get_blackbody_flux(l, 5000, magabs=-21, redshift=redshift),
                                redshift)
    assert np.isclose(absmag, -21, atol=1e-3)

    lbda_ref = np.linspace(1_000, 30_000, 3_000)
    flux_ref = get_blackbody_flux(lbda_ref, 7000, mag=0) * (lbda_ref / 5000)**-1
    absmag = _rest_frame_absmag(lambda l: generic_get_flux(l, lbda_ref, flux_ref, magabs=-21,
                                                           redshift=redshift),
                                redshift)
    assert np.isclose(absmag, -21, atol=1e-3)


def test_magabs_as_sncosmo():
    """ same convention as sncosmo set_source_peakabsmag (used for the point sources). """
    lbda_rest = np.linspace(1_000, 30_000, 3_000)
    phase = np.linspace(-20, 20, 9)
    flux = get_blackbody_flux(lbda_rest, 5000, mag=0)
    source = sncosmo.TimeSeriesSource(phase, lbda_rest, np.tile(flux, (len(phase), 1)))
    lbda = np.linspace(5_000, 15_000, 50)
    for z in [0.3, 0.9]:
        model = sncosmo.Model(source)
        model.set(z=z)
        model.set_source_peakabsmag(-21, "sdssr", "ab", cosmo=Planck18)
        assert np.allclose(get_blackbody_flux(lbda, 5000, magabs=-21, redshift=z),
                           model.flux(0, lbda), rtol=1e-4)


def test_mag_is_observed():
    """ mag is the observed magnitude in the observed band """
    lbda = np.linspace(3_000, 11_000, 2_000)
    flux = get_blackbody_flux(lbda, 5000, mag=22, redshift=0.9)
    assert np.isclose(sncosmo.Spectrum(lbda, flux).bandmag("sdssr", "ab"), 22, atol=1e-3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux_both = get_blackbody_flux(lbda, 5000, mag=22, magabs=-21, redshift=0.9)
    assert np.allclose(flux, flux_both)


# =============== #
#  NGC galaxies   #
# =============== #
@requires_galaxy_data
def test_ngc_host():
    lbda = np.linspace(4_000, 9_000, 50)
    for modelname in ["ngc0628", "NGC0628", "ngc0628:bb-6000", "sersic:ngc0628"]:
        this = host.Host.from_modelname(modelname, redshift=0.9)
        _, cube = this.get_cube(lbda, shape=(30, 20), pixel_scale=0.05)
        assert cube.shape == (50, 20, 30)
        assert np.all(np.isfinite(cube))

    this = host.Host.from_modelname("ngc0628", redshift=0.5)
    absmag = _rest_frame_absmag(lambda l: this.get_spectrum(l)[1], 0.5)
    assert np.isclose(absmag, -21, atol=1e-3)


@requires_galaxy_data
def test_ngc_angular_size():
    """ the galaxy gets smaller in arcsec as the redshift increases """
    this = host.Host.from_modelname("ngc0628")
    sizes = []
    for z in [0.3, 1]:
        image = this._model_func_spatial(redshift=z, shape=(116, 116), pixel_scale=0.04)
        y, x = np.indices(image.shape)
        xc, yc = (x * image).sum() / image.sum(), (y * image).sum() / image.sum()
        sizes.append(np.sqrt((((x - xc)**2 + (y - yc)**2) * image).sum() / image.sum()))

    assert sizes[1] < sizes[0]


# =============== #
#  Scene & Target #
# =============== #
@pytest.mark.parametrize("hostname", ["sersic:bb-5000",
                                      pytest.param("ngc0628", marks=requires_galaxy_data)])
def test_lazuli_supernova_host_redshift(hostname):
    """ the host follows the redshift of the point source """
    import slicersim
    target = slicersim.LazuliSupernova(redshift=0.9, host=hostname)
    scene = target.simulation.scene
    assert scene.has_element("host")
    assert scene.host.meta["redshift"] == 0.9
    assert target.get_properties("redshift", as_dict=False) == 0.9

    target.change_properties(redshift=0.5)
    assert scene.pointsource.meta["redshift"] == 0.5
    assert scene.host.meta["redshift"] == 0.5
    assert target.get_properties("redshift", as_dict=False) == 0.5

    # positions are independent
    target.change_properties(host__position=(3, -2))
    assert scene.host.meta["position"] == (3, -2)
    assert tuple(scene.pointsource.meta["position"]) != (3, -2)


def test_scene_host_dict_defaults():
    """ a host given as dict is completed by the get_host_config defaults """
    from slicersim.scene.scene import Scene
    scene = Scene.from_config({"host": {"refmodel": "sersic:bb-5000", "n": 4}})
    assert scene.host.meta["n"] == 4
    assert scene.host.meta["magabs"] == -21


# =============== #
#  Galaxy alone   #
# =============== #
@pytest.fixture
def lazuligalaxy():
    import slicersim
    return slicersim.LazuliGalaxy(redshift=0.5, model="sersic:bb-5000", position=(4, 2))


def test_galaxy_defaults(lazuligalaxy):
    sim = lazuligalaxy.simulation
    assert not sim.scene.has_element("pointsource")
    assert lazuligalaxy.get_properties(["aperture", "aperture__position"]) == {"aperture": "1-arcsec",
                                                                             "aperture__position": "host"}
    position, radius, optimal = sim.get_aperture()
    assert position == (4, 2) # host position
    assert np.isclose(radius, 1 / sim.spectrograph.spx_spatial_scale)
    assert not optimal


def test_aperture_parsing(lazuligalaxy):
    sim = lazuligalaxy.simulation
    scale = sim.spectrograph.spx_spatial_scale

    _, radius, optimal = sim.get_aperture("core")
    assert optimal and np.isclose(radius, sim._get_pointsource_radius_())

    _, radius, optimal = sim.get_aperture("0.5-arcsec")
    assert not optimal and np.isclose(radius, 0.5 / scale)

    for z in [0.5, 1]:
        lazuligalaxy.change_properties(redshift=z)
        _, radius, _ = sim.get_aperture("2-kpc")
        assert np.isclose(radius, 2 * Planck18.arcsec_per_kpc_proper(z).value / scale)

    for aperture in ["2-parsec", "big", "arcsec", 1]:
        with pytest.raises(ValueError):
            sim.get_aperture(aperture)


def test_aperture_position(lazuligalaxy):
    sim = lazuligalaxy.simulation
    # follows the host
    lazuligalaxy.change_properties(host__position=(-6, 1))
    assert sim.get_aperture()[0] == (-6, 1)

    # explicit
    lazuligalaxy.change_properties(aperture__position=(3, 3))
    assert sim.get_aperture()[0] == (3, 3)
    assert lazuligalaxy.get_properties("aperture__position", as_dict=False) == (3, 3)

    # back to the host
    lazuligalaxy.change_properties(aperture__position="host")
    assert sim.get_aperture()[0] == (-6, 1)


def test_short_name_position(lazuligalaxy):
    """ position is the host (or pointsource) position, not the aperture one. """
    import slicersim
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lazuligalaxy.change_properties(position=(1, 1))

        sn = slicersim.LazuliSupernova(redshift=0.5)
        sn.change_properties(position=(2, 2))

    assert tuple(lazuligalaxy.get_properties("host__position", as_dict=False)) == (1, 1)
    assert lazuligalaxy.get_properties("aperture__position", as_dict=False) == "host"
    assert tuple(sn.get_properties("pointsource__position", as_dict=False)) == (2, 2)


@pytest.mark.parametrize("aperture", ["1-arcsec", "2-kpc"])
def test_galaxy_spectrum_signal(lazuligalaxy, aperture):
    """ the aperture signal is the noise-free host cube summed within the aperture """
    sim = lazuligalaxy.simulation
    lazuligalaxy.change_properties(aperture=aperture, aperture__position=(2, -1))
    sig_cube, _ = sim.get_cube(switch_off=["background", "thermal", "dark", "thermal_dark", "roic_glow", "ron"],
                               per_ramp=True)
    position, radius, _ = sim.get_aperture()
    aper = sim.spectrograph.get_aperture(position=position, radius=radius)

    _, signal, variance = sim.get_spectrum()
    assert np.allclose(signal, sig_cube[:, aper].sum(axis=-1) * sim.get_parameter("nramps"))
    assert np.all(variance > 0)

    _, signal, _ = sim.get_spectrum(switch_off=["host"])
    assert np.all(signal == 0)


def test_galaxy_core(lazuligalaxy):
    """ the core is extracted as a point source """
    sim = lazuligalaxy.simulation
    lazuligalaxy.change_properties(aperture="core")
    position, radius, _ = sim.get_aperture()
    _, var_cube = sim.get_cube(per_ramp=True)

    _, _, variance = sim.get_spectrum()
    expected = sim.spectrograph.pointsource_variance(var_cube, position=position, radius=radius)
    assert np.allclose(variance, expected * sim.get_parameter("nramps"))


def test_galaxy_snr(lazuligalaxy):
    lazuligalaxy.change_properties(aperture="2-kpc")
    # discrete read-out modes: the target snr is only approached.
    _, snr = lazuligalaxy.setup_to_snr(5)
    assert np.isclose(snr, 5, rtol=0.3)
    assert np.isclose(lazuligalaxy.get_band_snr([4000, 6800], frame="rest"), snr)
    contrib = lazuligalaxy.get_variance_contribution()
    assert "host" in contrib.columns
