import numpy as np
import pytest

import slicersim
from slicersim import iotools
from slicersim.profiles import get_profilemodel
from slicersim.scene.extendedsource import (ExtendedSource,
                                            get_host_extendedsource,
                                            get_elliptical_flux,
                                            get_flat_flux)


# ================= #
#   Profile         #
# ================= #

@pytest.mark.parametrize("index, ellip", [(1., 0.), (4., 0.), (2.5, 0.4)])
def test_sersic_profile_normalized(index, ellip):
    """The normalized Sersic profile must integrate to 1 over the sky."""
    from scipy import integrate

    profile = get_profilemodel("sersic", position=(0, 0), normalized=True,
                               n=index, r_eff=1.5, ellip=0., theta=0.)
    total, _ = integrate.quad(lambda r: 2 * np.pi * r * profile(r, 0.), 0, np.inf,
                              limit=200)
    assert np.isclose(total, 1., rtol=1e-5)

    # elliptical profile: grid integration
    if ellip > 0:
        profile = get_profilemodel("sersic", position=(0, 0), normalized=True,
                                   n=index, r_eff=1.5, ellip=ellip, theta=0.3)
        x = np.linspace(-150, 150, 4001)
        dx = x[1] - x[0]
        img = profile(x[None, :], x[:, None])
        assert np.isclose(img.sum() * dx ** 2, 1., rtol=5e-2)


# ================= #
#   Spectra         #
# ================= #

def test_host_spectrum_normalization():
    """Built-in host spectra must match the requested total magnitude."""
    from sncosmo import Spectrum

    lbda = np.linspace(3500, 18000, 2000)

    flux = get_elliptical_flux(lbda, mag=21.5, band="sdssr")
    assert np.isclose(Spectrum(lbda, flux).bandmag("sdssr", "ab"), 21.5)
    assert np.all(flux > 0)

    # flat AB spectrum has the same magnitude in every band
    flux = get_flat_flux(lbda, mag=23.)
    assert np.isclose(Spectrum(lbda, flux).bandmag("sdssr", "ab"), 23., atol=1e-3)
    assert np.isclose(Spectrum(lbda, flux).bandmag("besselli", "ab"), 23., atol=1e-3)


def test_elliptical_spectrum_is_red():
    """The elliptical template must show a 4000A break."""
    lbda = np.linspace(3500, 18000, 2000)
    flux = get_elliptical_flux(lbda, mag=20)
    blue = np.mean(flux[lbda < 3800])
    red = np.mean(flux[(lbda > 4300) & (lbda < 4800)])
    assert red / blue > 1.5  # strong break


# ================= #
#  ExtendedSource   #
# ================= #

def test_extendedsource_from_config():
    """ExtendedSource setup from config helper and from a spectrum array."""
    config = get_host_extendedsource(spectrum="elliptical",
                                     index=2., r_eff=0.7, ellip=0.3,
                                     mag=21., position=[1, 2])
    host = ExtendedSource.from_config(config)

    assert host.profile_parameters == {"index": 2., "r_eff": 0.7,
                                       "ellip": 0.3, "theta": 0.}
    assert tuple(host.position) == (1, 2)
    lbda = np.linspace(4000, 17000, 500)
    _, spec = host.get_spectrum(lbda)
    assert np.isfinite(spec).all() and (spec > 0).all()

    # profile integrates to ~1 (grid, generous tolerance for n=2 wings)
    x = np.linspace(-80, 80, 3001)
    dx = x[1] - x[0]
    img = host.get_profile(x[None, :], x[:, None])
    assert np.isclose(img.sum() * dx ** 2, 1., rtol=5e-2)

    # from a user-provided spectrum
    lbda_ref = np.linspace(3000, 20000, 1000)
    host2 = ExtendedSource.from_config({"source": (lbda_ref, np.ones_like(lbda_ref) * 1e-16),
                                        "mag": 22., "band": "sdssr"})
    from sncosmo import Spectrum
    _, spec2 = host2.get_spectrum(lbda)
    assert np.isclose(Spectrum(lbda, spec2).bandmag("sdssr", "ab"), 22.)


def test_unknown_profile_raises():
    """ """
    with pytest.raises(NotImplementedError):
        get_host_extendedsource(profile="exponential-disk")


# ================= #
#   Simulation      #
# ================= #

def get_host_simulation(**host_kwargs):
    """ """
    host = {"scene": {"host": get_host_extendedsource(**host_kwargs)}}
    config = iotools.get_config(scene=["supernova.toml", host])
    return slicersim.Simulation.from_config(config)


def test_simulation_with_host():
    """Full simulation including a Sersic host."""
    simu = get_host_simulation(spectrum="elliptical", mag=20., index=4., r_eff=1.)

    assert simu.scene.has_element("host")

    # host mutable parameters are exposed and updatable
    assert "host.index" in simu.scene.mutable_parameters
    simu.update(host__mag=21., host__index=1.)
    assert simu.scene.host.meta["mag"] == 21.
    assert simu.scene.host.profile_parameters["index"] == 1.

    # scene cubes contain a finite, positive host cube
    cubes = simu.get_scene_cubes(unit="ph", apply_lsf=False)
    host_cube = cubes["host_cube"]
    assert np.isfinite(host_cube).all()
    assert host_cube.sum() > 0

    # photon conservation: FOV captures at most the total flux
    _, spec = simu.scene.get_element_spectrum("host")
    total_flux = spec * simu.spectrograph.flambda2photon * simu.get_parameter("nramps")
    fov_fraction = host_cube.sum(axis=(1, 2)) / total_flux
    assert np.all(fov_fraction <= 1.)
    assert np.all(fov_fraction > 0.3)  # r_eff=1" mostly inside the FOV

    # oversampled and rebinned cubes conserve photons
    c_over = simu.spectrograph.generate_structured_background(
        spec, apply_lsf=False, oversampling=3, as_oversampled=True,
        **simu.scene.host.profile_parameters)
    c_rebinned = simu.spectrograph.generate_structured_background(
        spec, apply_lsf=False, oversampling=3,
        **simu.scene.host.profile_parameters)
    assert np.isclose(c_over.sum(), c_rebinned.sum())


def test_host_contributes_to_signal_and_variance():
    """The host must add flux to the cube and to the extracted variance."""
    simu = get_host_simulation(spectrum="elliptical", mag=19., r_eff=0.5)

    cube_with = simu.get_projected_scene(apply_lsf=False)
    cube_without = simu.get_projected_scene(apply_lsf=False, switch_off=["host"])
    assert cube_with.sum() > cube_without.sum()

    _, _, var = simu.get_spectrum()
    _, _, var_nohost = simu.get_spectrum(switch_off=["host"])
    assert np.all(var >= var_nohost)
    assert var.sum() > var_nohost.sum()


def test_host_position_moves_profile():
    """An off-centered host peaks off-center in the cube."""
    simu = get_host_simulation(spectrum="flat", mag=18., index=1., r_eff=0.3,
                               position=[5, 3])
    _, spec = simu.scene.get_element_spectrum("host")

    cube = simu.spectrograph.generate_structured_background(
        spec, position=simu.scene.host.position, apply_lsf=False,
        **simu.scene.host.profile_parameters)
    img = cube.sum(axis=0)
    cy, cx = np.unravel_index(np.argmax(img), img.shape)
    center_y, center_x = (np.asarray(img.shape) - 1) / 2
    assert cx > center_x  # x=+5 spx
    assert cy > center_y  # y=+3 spx


def test_scene_without_host_unchanged():
    """A host-less scene keeps working, with a null host cube."""
    config = iotools.get_config(scene="supernova.toml")
    simu = slicersim.Simulation.from_config(config)

    assert not simu.scene.has_element("host")
    cubes = simu.get_scene_cubes(unit="ph", apply_lsf=False)
    assert np.all(cubes["host_cube"] == 0)

    _, spec = simu.scene.get_element_spectrum("host", fillna=0)
    assert np.all(spec == 0)
