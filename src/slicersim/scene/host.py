"""Scene: host module.

`Host` is the `~slicersim.scene.base.SceneElement` describing the host galaxy of the observed
target: a spatially extended source whose spectrum (reference spectrum
rescaled to a given magnitude) is modulated by a Sersic surface-brightness
profile, see `get_sersic_profile`.
"""
from copy import deepcopy
import numpy as np
from astropy.modeling.models import Sersic2D
from astropy.cosmology import Planck18 as cosmology

from .base import SceneElement
from ..utils import inspect_func

def get_sersic_profile(r_kpc=5, redshift=0.1, cosmology=cosmology,
                       r_eff=None,
                       n=1.0, ellip=0.0, theta=0.0, flux=1,
                       pixel_scale=0.1, shape=(10, 10), position=None,
                       oversample=4):
    """Render a Sersic profile onto a pixel grid, flux-normalized.

    Parameters
    ----------
    r_eff : float
        Effective (half-light) radius, in arcsec.
    n : float, optional
        Sersic index (1 = exponential disk, 4 = de Vaucouleurs).
        Default is 1.0.
    ellip : float, optional
        Ellipticity, 1 - b/a. Default is 0.0.
    theta : float, optional
        Position angle, in degrees (astropy math convention: from +x axis,
        counterclockwise — remap at the call site if you need on-sky PA).
        Default is 0.0.
    flux : float, optional
        Total integrated flux of the profile (integrated to infinity, so
        the part falling outside of the grid is lost). Default is 1.
    pixel_scale : float, optional
        Arcsec per pixel of the output grid. Default is 0.1.
    shape : tuple of int, optional
        Output image shape as ``(nx, ny)``. Default is ``(10, 10)``.
    position : tuple of float, optional
        Centroid offset ``(dx, dy)`` in pixels relative to the geometric
        center of the grid, where positive ``dx`` increases the column index
        and positive ``dy`` the row index. Default is None, i.e. ``(0, 0)``.
    oversample : int, optional
        Oversampling factor per axis, for approximating pixel integration.
        Default is 4.

    Returns
    -------
    numpy.ndarray
        2D image of shape ``(ny, nx)``, flux per pixel. It sums to ``flux``
        only if the profile is fully contained in the grid.
    """
    from scipy.special import gamma, gammaincinv

    if r_eff is None:
        if r_kpc is not None and redshift is not None:
            r_eff = r_kpc * cosmology.arcsec_per_kpc_proper( float(redshift) ).value
        else:
            raise ValueError("either r_eff or (r_kpc and redshift) must be given")

    nx, ny = shape
    center = ((nx - 1) / 2.0, (ny - 1) / 2.0)

    if position is None:
        position = (0.0, 0.0)

    dx, dy = position
    x0 = center[0] + dx
    y0 = center[1] + dy

    nx_os, ny_os = nx * oversample, ny * oversample

    # Convert everything to oversampled-pixel units
    r_eff_os = (r_eff / pixel_scale) * oversample
    x0_os = (x0 + 0.5) * oversample - 0.5
    y0_os = (y0 + 0.5) * oversample - 0.5

    y_grid, x_grid = np.mgrid[0:ny_os, 0:nx_os]

    model = Sersic2D(
        amplitude=1.0, r_eff=r_eff_os, n=n,
        x_0=x0_os, y_0=y0_os, ellip=ellip, theta=np.deg2rad(theta),
    )
    img_os = model(x_grid, y_grid)

    img = img_os.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))

    # analytic total flux of the (amplitude=1) profile, in pixel units.
    # normalizing by it (and not by img.sum()) conserves the surface
    # brightness when the galaxy extends beyond the grid.
    r_eff_pix = r_eff / pixel_scale
    b_n = gammaincinv(2 * n, 0.5)
    total = (2 * np.pi * n * np.exp(b_n) * b_n**(-2 * n) * gamma(2 * n)
             * (1 - ellip) * r_eff_pix**2)

    img *= flux / total
    return img


class Host(SceneElement):
    """Structured background representing the host galaxy of the observed target.

    Models a spatially extended source (e.g. the galaxy hosting a supernova).
    Its model function returns a datacube of shape (nlbda, ny, nx): a
    reference spectrum, rescaled to a given magnitude, multiplied by a
    Sersic profile (see `get_sersic_profile`).
    In a `~slicersim.simulation.Simulation`, the host is rendered on the spaxel grid
    (see `get_cube`) and convolved by the spatial PSF.

    Use `from_config` or `from_sersic_and_spectrum` to build it.

    .. note::
        Host galaxy support is currently in development; the interface is
        subject to change.
    """
    @classmethod
    def from_config(cls, config):
        """Generate a `Host` from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary. All entries are optional:

            - ``refmodel``: str, optional (default ``"blackbody-5000"``).
              Reference spectrum, either ``"blackbody-<temperature>"`` (in
              K) or ``"brown-<galname>"`` (Brown et al. galaxy template,
              requires ``wcc_etc``).
            - ``mag``: float, optional (default 20). AB magnitude of the host
              in ``band`` (see `from_sersic_and_spectrum`).
            - ``r_kpc``: float, optional. Effective radius in kpc. If given,
              ``redshift`` is used to convert it into ``r_eff`` (arcsec,
              Planck18 cosmology), overriding any ``r_eff`` entry.
            - ``redshift``: float, optional. Needed if ``r_kpc`` is given.

            Any other entry is passed to `from_sersic_and_spectrum` (e.g.
            ``band``, ``redshift``, ``r_eff``, ``n``, ``ellip``, ``theta``,
            ``pixel_scale``, ``shape``, ``position``, ``oversample``).
            The input dict is not modified.

        Returns
        -------
        Host
            An instance of the `Host` class.

        Warns
        -----
        UserWarning
            If ``r_kpc`` is given without ``redshift`` (``r_kpc`` is then
            ignored), or together with ``r_eff`` (``r_eff`` is then ignored).

        See Also
        --------
        from_sersic_and_spectrum : Build a `Host` from a spectrum and a
            Sersic profile.

        Examples
        --------
        >>> host = Host.from_config({"refmodel": "blackbody-5000", "mag": 21,
        ...                          "r_eff": 0.5, "n": 1, "ellip": 0.3,
        ...                          "shape": (20, 20)})
        """
        # do not affect the input config
        config = deepcopy(config)

        # build it.

        # Parse the model
        refmodel = config.pop("refmodel", "sersic:bb-5000")
        if ":" in refmodel:
            spatialmodel, spectralmodel = refmodel.split(":", 1)
            if spatialmodel == "sersic":
                spatial_func = get_sersic_profile
            else:
                raise NotImplementedError(f"spatial model '{spatialmodel}' is not recognized. Must be 'sersic'.")

            if spectralmodel.startswith("blackbody") or spectralmodel.startswith("bb"):
                from .sources.blackbody import get_blackbody_flux
                if "-" in spectralmodel:
                    temperature = float(spectralmodel.split("-")[1])
                    config["temperature"] = temperature

                spectral_func = get_blackbody_flux

            elif spectralmodel.startswith("brown-"):
                galname = spectralmodel.split("-")[1]
                from .sources.generic import generic_get_flux
                from astropy.io import fits
                from wcc_etc.io import get_any_astro_name
                galdata = fits.getdata( get_any_astro_name(galname) )
                config["lbda_ref"], config["flux_ref"] = galdata["wavelength"],galdata["flux"]
                spectral_func = generic_get_flux
            else:
                raise NotImplementedError(f"spectral model '{spectralmodel}' is not recognized. Must be 'blackbody-<temperature>' or 'brown-<galname>'.")

        else:
            raise NotImplementedError(f"refmodel '{refmodel}' is not recognized. Must be 'spatial:spectral'.")

        # Spatial source.
        return cls._from_spatial_and_spectral_func(spatial_func, spectral_func, **config)

    @classmethod
    def _from_spatial_and_spectral_func(cls, spatial_func, spectral_func, **kwargs):
        """ """
        this = cls(None, meta = kwargs )
        this._model_func_spectral = spectral_func
        this._model_func_spatial = spatial_func
        return this

    #
    # convinience from_ functions.
    #
    @classmethod
    def from_sersic_and_blackbody(cls, temperature, mag=None, band="sdssr", redshift=0.1,
                                 r_eff=0.1, n=1.0, ellip=0.0, theta=0.0,
                                 pixel_scale=0.1, shape=(10, 10), position=None,
                                 oversample=4,
                                 **kwargs):
        """Build a `Host` from a Sersic surface-brightness profile and a blackbody spectrum.

        Parameters
        ----------
        """
        meta = {key: values for key, values in locals().items()
                       if key not in ["cls" ,"kwargs"]
                } | kwargs

        return cls._from_spatial_and_spectral_func(spatial_func=get_sersic_profile,
                                                   spectral_func=get_blackbody_flux,
                                                   meta=meta)


    @classmethod
    def from_sersic_and_spectrum(cls, lbda, flux, mag=None, band="sdssr", redshift=0.1,
                                 r_eff=0.1, n=1.0, ellip=0.0, theta=0.0,
                                 pixel_scale=0.1, shape=(10, 10), position=None,
                                 oversample=4,
                                 **kwargs):
        """Build a `Host` from a Sersic surface-brightness profile and a reference spectrum.

        The host's spectral shape is taken from a reference spectrum
        (``lbda``, ``flux``), redshifted by ``redshift`` and rescaled to
        match ``mag`` in ``band``, and its spatial
        shape is a Sersic profile rendered onto a pixel grid (see
        `get_sersic_profile`). The resulting model function returns a
        datacube, i.e. the spectrum modulated by the spatial profile at
        every wavelength.

        Parameters
        ----------
        lbda : array_like
            Wavelength array of the reference spectrum, in Angstrom
            (rest-frame, i.e. before applying ``redshift``).
        flux : array_like
            Flux of the reference spectrum, sampled at ``lbda``, in
            erg/s/cm^2/A.
        mag : float or None, optional
            Total observer-frame AB magnitude of the host in ``band``. If
            None, the reference spectrum is used as is (no rescaling).
            Default is None.
        band : str, optional
            Bandpass name (must be known by ``sncosmo``) in which ``mag`` is
            defined. Default is ``"sdssr"``.
        redshift : float, optional
            Redshift applied to the reference spectrum wavelength axis (no
            flux dimming). It is fixed at construction and is not a mutable
            parameter. Default is 0.
        r_eff : float, optional
            Effective (half-light) radius of the Sersic profile, in arcsec.
            Default is 0.1.
        n : float, optional
            Sersic index (1 = exponential disk, 4 = de Vaucouleurs).
            Default is 1.0.
        ellip : float, optional
            Ellipticity of the Sersic profile, 1 - b/a. Default is 0.0.
        theta : float, optional
            Position angle of the Sersic profile, in degrees (astropy math
            convention: from +x axis, counterclockwise). Default is 0.0.
        pixel_scale : float, optional
            Arcsec per pixel of the output grid. Default is 0.1.
        shape : tuple of int, optional
            Output cube spatial shape as ``(nx, ny)``. Default is ``(10, 10)``.
        position : tuple of float, optional
            Centroid offset ``(dx, dy)`` in pixels relative to the geometric
            center of the grid. Default is None, i.e. ``(0, 0)``.
        oversample : int, optional
            Oversampling factor per axis used to render the Sersic profile.
            Default is 4.
        **kwargs
            Additional entries stored in the instance meta.

        Returns
        -------
        Host
            An instance of the `Host` class. Its mutable parameters are
            ``lbda``, ``mag``, ``band``, ``r_eff``, ``n``, ``ellip``,
            ``theta``, ``pixel_scale``, ``shape``, ``position`` and
            ``oversample``.
        """
        meta = {key: values for key, values in locals().items()
                       if key not in ["cls", "lbda", "flux","kwargs"]
                } | kwargs

        return cls._from_spatial_and_spectral_func(spatial_func=get_sersic_profile,
                                                   spectral_func=generic_get_flux,
                                                   **meta)


    # ========== #
    #  Methods   #
    # ========== #
    def _model_func(self, lbda, *args, **kwargs):
        """ """
        flux = self._model_func_spectral(lbda, **kwargs)
        spatial = self._current_spatial_model
        return flux[:, None, None] * spatial[None, :, :]

    def _update_spatial_model(self):
        """ """
        self._spatial_model = self._model_func_spatial(**self._meta_spatial)

    # ========== #
    #  Getter    #
    # ========== #
    def get_cube(self, lbda=None, **kwargs):
        """Get the host datacube.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, the default wavelength
            (``lbda`` attribute) is used. Default is None.
        **kwargs
            Overwrite the current model parameters for this call only,
            typically the grid the host is rendered on (``shape`` and
            ``pixel_scale``).

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        cube : numpy.ndarray
            Datacube of shape ``(nlbda, ny, nx)``, flux per pixel in
            erg/s/cm^2/A.

        Raises
        ------
        ValueError
            If ``lbda`` is None and no default wavelength is set.

        Examples
        --------
        Render the host on a 58x58 grid of 0.04 arcsec pixels:

        >>> lbda, cube = host.get_cube(lbda, shape=(58, 58), pixel_scale=0.04)  # doctest: +SKIP
        """
        if lbda is None:
            lbda = self._lbda
            if lbda is None:
                raise ValueError("no lbda given and None loading as self.lbda")

        model_kwargs = self._parse_model_kwargs_() | kwargs
        return lbda, self.model_func(lbda, **model_kwargs)


    def get_spectrum(self, lbda=None):
        """Get the host spectrum, integrated over its own grid.

        This is `get_cube` summed over the spatial axes, i.e. the flux
        falling within the host grid (``shape`` x ``pixel_scale`` in meta).

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, the default wavelength
            (``lbda`` attribute) is used. Default is None.

        Returns
        -------
        lbda : array_like
            Wavelength array in Angstrom.
        flux : numpy.ndarray
            Flux in erg/s/cm^2/A.
        """
        lbda, cube = self.get_cube(lbda)
        return lbda, np.sum(cube, axis=(-2, -1))


    # ============ #
    # properties   #
    # ============ #
    @property
    def _current_spatial_model(self):
        """ """
        if self._spatial_model is None:
            self._update_spatial_model()

        return self._spatial_model

    @property
    def _mutable_parameters_spatial(self):
        """ """
        return inspect_func(self._model_func_spatial )[0]

    @property
    def _mutable_parameters_spectral(self):
        """ """
        all_spectral = inspect_func(self._model_func_spectral )[0]
        return [key for key in all_spectral if key not in  ["lbda"] ]

    @property
    def mutable_parameters(self):
        """ """
        return list(self._mutable_parameters_spatial) + list(self._mutable_parameters_spectral)

    @property
    def _meta_spatial(self):
        """ """
        return {k: self.meta[k] for k in self._mutable_parameters_spatial if k in self.meta}

    @property
    def _meta_spectral(self):
        """ """
        return {k: self.meta[k] for k in self._mutable_parameters_spectral if k in self.meta}
