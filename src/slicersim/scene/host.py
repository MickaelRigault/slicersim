""" structured background """
import numpy as np
from astropy.modeling.models import Sersic2D

from .base import SceneElement

def get_sersic_profile(r_eff, n=1.0, ellip=0.0, theta=0.0, flux=1,
                       pixel_scale=0.1, shape=(10, 10), position=None,
                       oversample=4):
    """ Render a Sersic profile onto a pixel grid, flux-normalized.

    Parameters
    ----------
    flux : float
        Total integrated flux.
    r_eff : float
        Effective (half-light) radius, in arcsec.
    n : float
        Sersic index (1 = exponential disk, 4 = de Vaucouleurs).
    ellip : float
        Ellipticity, 1 - b/a.
    theta : float
        Position angle, in degrees (astropy math convention: from +x axis,
        counterclockwise — remap at the call site if you need on-sky PA).
    pixel_scale : float
        Arcsec per pixel of the output grid.
    shape : (int, int), optional
        Output image shape as (nx, ny).
    position : (float, float), optional
        Centroid offset (dx, dy) in pixels relative to the geometric center of
        `shape`, where positive dx moves right and positive dy moves down.
        Defaults to (0, 0).
    oversample : int
        Oversampling factor per axis, for approximating pixel integration.

    Returns
    -------
    numpy.ndarray
        2D image, shape (ny, nx), summing to `flux`.
    """
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

    img *= flux / img.sum()
    return img


class Host(SceneElement):
    """Structured background representing the host galaxy of the observed target.

    Models a spatially extended source (e.g. the galaxy hosting a supernova)
    whose surface-brightness distribution is wavelength-dependent.  The spatial
    profile at each wavelength is computed by a ``galsim``-compatible model
    object and rendered into the scene datacube.

    .. note::
        Host galaxy support is currently in development; the interface is
        subject to change.
    """
    @classmethod
    def from_sersic_and_spectrum(cls, lbda, flux, mag, band, redshift=0,
                                 r_eff=0.1, n=1.0, ellip=0.0, theta=0.0,
                                 pixel_scale=0.1, shape=(10, 10), position=None,
                                oversample=4, meta={},
                                **kwargs):
        """Build a `Host` from a Sersic surface-brightness profile and a reference spectrum.

        The host's spectral shape is taken from a reference spectrum
        (`lbda`, `flux`), rescaled to match `mag` in `band`, and its spatial
        shape is a Sersic profile rendered onto a pixel grid (see
        `get_sersic_profile`). The resulting model function returns a
        datacube, i.e. the spectrum modulated by the spatial profile at
        every wavelength.

        Parameters
        ----------
        lbda : array_like
            Wavelength array of the reference spectrum, in Angstrom
            (rest-frame, i.e. before applying `redshift`).
        flux : array_like
            Flux of the reference spectrum, sampled at `lbda`.
        mag : float or None
            Target magnitude of the host in `band`. If None, the reference
            spectrum is used as-is (no rescaling).
        band : str
            Bandpass name (must be known by sncosmo) used to normalize `mag`.
        redshift : float, optional
            Redshift applied to the reference spectrum wavelength axis.
            Default is 0.
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
        shape : (int, int), optional
            Output cube spatial shape as (nx, ny). Default is (10, 10).
        position : (float, float), optional
            Centroid offset (dx, dy) in pixels relative to the geometric
            center of `shape`. Default is None, i.e. (0, 0).
        oversample : int, optional
            Oversampling factor per axis used to render the Sersic profile.
            Default is 4.
        meta : dict, optional
            Extra metadata to store on the instance. Default is {}.
        **kwargs
            Additional metadata merged into `meta`.

        Returns
        -------
        Host
            An instance of the `Host` class.
        """
        from sncosmo import Spectrum
        # flux input
        meta["mag"] = mag
        meta["band"] = band
        meta["lbda_ref"] = lbda
        meta["flux_ref"] = flux

        # profile input (sersic)
        meta["r_eff"] = r_eff
        meta["n"] = n
        meta["ellip"] = ellip
        meta["theta"] = theta
        meta["pixel_scale"] = pixel_scale
        meta["shape"] = shape
        meta["position"] = position
        meta["oversample"] = oversample

        this = cls(None, meta= meta | kwargs)

        # internal function
        def _internal_get_flux(lbda, mag, band, r_eff=1., n=1.0, ellip=0.0, theta=0.0,
                            pixel_scale=0.1, shape=(10, 10), position=None,
                            oversample=4):
            """Internal model function: spectrum modulated by the Sersic profile.

            Parameters
            ----------
            lbda : array_like
                Wavelength array in Angstrom.
            mag : float or None
                Target magnitude. If None, the reference spectrum is used
                as-is (no rescaling).
            band : str
                Bandpass name.
            r_eff : float, optional
                Effective (half-light) radius, in arcsec. Default is 1.
            n : float, optional
                Sersic index. Default is 1.0.
            ellip : float, optional
                Ellipticity, 1 - b/a. Default is 0.0.
            theta : float, optional
                Position angle, in degrees. Default is 0.0.
            pixel_scale : float, optional
                Arcsec per pixel of the output grid. Default is 0.1.
            shape : (int, int), optional
                Output cube spatial shape as (nx, ny). Default is (10, 10).
            position : (float, float), optional
                Centroid offset (dx, dy) in pixels. Default is None.
            oversample : int, optional
                Oversampling factor per axis. Default is 4.

            Returns
            -------
            numpy.ndarray
                3D datacube of shape (len(lbda), ny, nx): the flux at each
                wavelength scaled by the (flux-normalized) Sersic profile.
            """
            if mag is None:
                flux_ratio = 1
            else:
                in_mag = Spectrum(meta["lbda_ref"]*(1+redshift), meta["flux_ref"]
                                  ).bandmag(band, "ab")
                flux_ratio = 10 ** (-0.4 * (mag - in_mag))

            flux_ = np.interp(lbda, meta["lbda_ref"]*(1+redshift), meta["flux_ref"],
                              left=np.nan, right=np.nan)
            flux_ *= flux_ratio

            profile = get_sersic_profile(r_eff=r_eff, n=n, ellip=ellip, theta=theta,
                                        pixel_scale=pixel_scale, shape=shape, position=position,
                                        oversample=oversample, flux=1)
            return flux_[:, None, None] * profile[None, :, :]

        this._model_func = _internal_get_flux
        return this
