"""Spatial models of galaxies, used by `~slicersim.scene.host.Host`.

- `get_sersic_profile`: analytic Sersic profile, whose size is given in kpc
  (``r_kpc`` at ``redshift``) or in arcsec (``r_eff``);
- `GalaxySpatialModel`: image of an actual (nearby) galaxy, e.g. an NGC
  galaxy, moved to any larger redshift while keeping its physical size.

Both return flux-normalized images: the fraction of the total galaxy flux
in each pixel.
"""
from copy import deepcopy

import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmology
from astropy.modeling.models import Sersic2D


# ============== #
#  Sersic model  #
# ============== #
def get_sersic_profile(r_kpc=5, redshift=0.1, cosmology=cosmology,
                       r_eff=None,
                       n=1.0, ellip=0.0, theta=0.0, flux=1,
                       pixel_scale=0.1, shape=(10, 10), position=None,
                       oversample=4):
    """Render a Sersic profile onto a pixel grid, flux-normalized.

    The size of the profile is either given in kpc (``r_kpc``, converted
    into arcsec at ``redshift``) or directly in arcsec (``r_eff``), the
    latter taking precedence.

    Parameters
    ----------
    r_kpc : float, optional
        Effective (half-light) radius, in kpc (proper). Used only if
        ``r_eff`` is None. Default is 5.
    redshift : float, optional
        Redshift used to convert ``r_kpc`` into arcsec. Default is 0.1.
    cosmology : astropy.cosmology.Cosmology, optional
        Cosmology used to convert ``r_kpc`` into arcsec. Default is Planck18.
    r_eff : float or None, optional
        Effective (half-light) radius, in arcsec. If given, ``r_kpc`` and
        ``redshift`` are ignored. Default is None.
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

    Raises
    ------
    ValueError
        If ``r_eff`` is None and ``r_kpc`` or ``redshift`` is None.
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

    model = Sersic2D( amplitude=1.0, r_eff=r_eff_os, n=n,
                      x_0=x0_os, y_0=y0_os, ellip=ellip,
                      theta=np.deg2rad(theta))
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

# ============== #
#  Image Galaxy  #
# ============== #
class GalaxySpatialModel:
    """Spatial model of an NGC galaxy from its ngcs image, to be observed at any redshift."""

    def __init__(self, image, pixel_scale, redshift, center, meta={}):
        self._image = np.asarray(image, dtype=float) / np.sum(image)  # normalized: total flux = 1
        self._pixel_scale = pixel_scale  # [arcsec]
        self._redshift = redshift        # of the galaxy itself
        self._center = center            # (x, y) pixel of the galaxy centre (0-based)

        # meta as in the rest of the package.
        self._meta =  deepcopy(meta)
        self._meta_in = deepcopy(meta)


    @classmethod
    def from_name(cls, name, filename=None):
        """Load the galaxy by its name alone.

        Parameters
        ----------
        name : str
            The name of the galaxy.
            Could be any NGC galaxy (accepting format "NGC0628" or "NGC 0628", "NGC628", "ngc 628")

        Returns
        -------
        GalaxySpatialModel
            The galaxy spatial model.
        """
        if filename is None:
            # .. is scene/
            from ...iotools import get_galaxy_filepath
            filename = get_galaxy_filepath(name, which="image")

        if not filename.is_file():
            raise FileNotFoundError(f"{filename} is not in the stored collection ({filename.parent})")

        image, header = fits.getdata(filename, header=True)
        # nan_to_num nan => 0 (-)inf => very large(negative) number
        image = np.clip( np.nan_to_num(image), 0, None)

        return cls(image,
                    pixel_scale=header["PIXSCALE"],
                    redshift=header["REDSHIFT"],
                    center=(header["XCENTER"] - 1, header["YCENTER"] - 1),
                    meta=dict(header)
                    )

    def __call__(self, redshift, **kwargs):
        """Shortcut to :meth:`project_to`.

        Parameters
        ----------
        redshift : float
            Redshift at which the galaxy is observed. Must be larger than
            the galaxy's own redshift.
        **kwargs
            Passed to :meth:`project_to` (e.g. ``position``, ``rotation``,
            ``shape``, ``pixel_scale``, ``cosmology``).

        Returns
        -------
        numpy.ndarray
            2D image of shape ``(ny, nx)``, the fraction of the total galaxy
            flux in each pixel.

        See Also
        --------
        project_to : project the galaxy image onto a pixel grid at a given redshift.
        """
        return self.project_to(redshift, **kwargs)

    # ============= #
    #   Method      #
    # ============= #
    def get_pixel_scale(self, redshift, cosmology=cosmology):
        """Image pixel scale [arcsec] when the galaxy is at `redshift` (same physical size)."""
        if redshift <= self.redshift:
            raise ValueError(f"redshift must be larger than the galaxy's own ({self.redshift})")

        return self.pixel_scale * (cosmology.angular_diameter_distance(self.redshift)
                                   / cosmology.angular_diameter_distance(redshift)).value

    # fixed named parameters: redshift, position, *shape*, *pixel_scale*, cosmology.
    def project_to(self, redshift, position=(0, 0),
                    rotation=0,
                    shape=(50, 100), # different than Lazuli wide and narrow to explicitely break if not set.
                    pixel_scale=0.04,
                    cosmology=cosmology):
        """The galaxy as observed at `redshift` on a grid of `pixel_scale` arcsec pixels.

        The galaxy keeps its physical size: its image pixel scale is
        rescaled by the ratio of angular diameter distances (see
        `get_pixel_scale`).

        Parameters
        ----------
        redshift : float
            Redshift at which the galaxy is observed. Must be larger than
            the galaxy's own redshift.
        position : tuple of float, optional
            ``(dx, dy)`` of the galaxy centre from the grid centre, in
            pixels. Default is ``(0, 0)``.
        rotation : float, optional
            Rotation of the galaxy, in degrees, counterclockwise.
            Default is 0.
        shape : tuple of int, optional
            Output grid shape as ``(nx, ny)`` (as `get_sersic_profile`).
            Default is ``(50, 100)``.
        pixel_scale : float, optional
            Arcsec per pixel of the output grid. Default is 0.04.
        cosmology : astropy.cosmology.Cosmology, optional
            Default is Planck18.

        Returns
        -------
        numpy.ndarray
            ``(ny, nx)`` fraction of the total galaxy flux in each pixel.
        """
        return self._image_projection(self.image, self.center,
                                        self.get_pixel_scale(redshift, cosmology) / pixel_scale,
                                        position=position, rotation=rotation, shape=shape[::-1])

    # internal tool
    @staticmethod
    def _image_projection(image, center, scale, position=(0, 0), rotation=0, shape=(58, 116), subsampling=8):
        """Project `image` onto a (ny, nx) grid, conserving flux.

        Each image pixel is split into k x k sub-pixels (k such that a sub-pixel is at most 1/subsampling of an
        output pixel) and each sub-pixel flux is added to the output pixel containing its centre.
        scale: image pixel size in output pixels. center: (x, y) image pixel placed at `position`.
        """
        ny, nx = shape
        theta = np.deg2rad(rotation)
        cos, sin = np.cos(theta), np.sin(theta)

        # 1. crop the image to what can fall onto the grid: a disk around the grid centre, any rotation
        dx, dy = -np.asarray(position, float) / scale  # grid centre from the galaxy centre [image pixels]
        x0, y0 = center[0] + cos * dx + sin * dy, center[1] - sin * dx + cos * dy
        radius = np.hypot(nx, ny) / 2 / scale + 1
        xmin, xmax = max(int(x0 - radius), 0), min(int(x0 + radius) + 2, image.shape[1])
        ymin, ymax = max(int(y0 - radius), 0), min(int(y0 + radius) + 2, image.shape[0])
        if xmin >= xmax or ymin >= ymax:
            return np.zeros(shape)
        sub = image[ymin:ymax, xmin:xmax]
        v, u = np.nonzero(sub)  # only the galaxy pixels (0 elsewhere)
        flux = sub[v, u]

        # 2. sub-pixels: offsets from each pixel centre [image pixels]
        k = max(1, int(np.ceil(subsampling * scale)))
        offset = (np.arange(k) + 0.5) / k - 0.5
        du, dv = np.meshgrid(offset, offset)
        u = (u + xmin - center[0])[:, None] + du.ravel()  # (npix, k*k), from the galaxy centre
        v = (v + ymin - center[1])[:, None] + dv.ravel()

        # 3. to output pixels: scale, rotate and shift
        x = (nx - 1) / 2 + position[0] + scale * (cos * u - sin * v)
        y = (ny - 1) / 2 + position[1] + scale * (sin * u + cos * v)
        ix, iy = np.floor(x + 0.5).astype(int), np.floor(y + 0.5).astype(int)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        weights = np.broadcast_to(flux[:, None] / k**2, ix.shape)

        return np.bincount((iy * nx + ix)[inside], weights=weights[inside], minlength=ny * nx).reshape(shape)


    # ============= #
    #  properties   #
    # ============= #
    @property
    def image(self):
        """The galaxy image, normalized to total flux = 1."""
        return self._image

    @property
    def pixel_scale(self):
        """The galaxy image pixel scale [arcsec]."""
        return self._pixel_scale

    @property
    def redshift(self):
        """The galaxy's own redshift (as initially given)."""
        return self._redshift

    @property
    def center(self):
        """The galaxy centre in the image (x, y) pixel coordinates (0-based)."""
        return self._center

    @property
    def meta(self):
        """Meta information about the galaxy."""
        return self._meta
