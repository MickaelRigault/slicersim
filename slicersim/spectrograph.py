"""
Spectrograph module, to simulate:

* mirror collecting surface and thermal emission
* spatial stage and MLA sampling
* spectral stage and dispersion law

Astrophysical scene is handled by :class:`mlaperf.scene.Scene` and detector by
:class:`mlaperf.detector.Detector`.

.. autosummary::

   Mirror
   Spectrograph
"""

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"

import warnings
from dataclasses import dataclass

import numpy as np

import astropy.units as u

from .utils import integ_gaussian2D_erf
from . import iotools


@dataclass
class Mirror:
    """
    Mirror data class.
    """

    surface: float              #: Collecting area [m²]
    temperature: float = 0.     #: Temperature [K]
    emissivity: float = 0.      #: Emissivity

    @staticmethod
    def get_surface(dext, dint=0):
        """
        Collecting area from outer and inner diameters.

        :param float dext: outer diameter [m]
        :param float dint: inner diameter [m]
        :return: collecting area
        """

        return np.pi/4 * (dext ** 2 - dint ** 2)

    def __str__(self):

        s = f"Mirror: {self.surface:.0f} m²"
        if self.temperature:
            s += f" at {self.temperature:.0f} K, emissivity: {self.emissivity:.2f}"
        else:
            s += ", no thermal emission"

        return s


@dataclass
class Camera:
    """
    Camera data class.
    """

    acceptance: float = 0.      #: Camera acceptance angle [rad]
    speed: float = np.inf       #: Camera speed (f-number)

    def __str__(self):

        return f"Camera: {np.degree(self.acceptance)} deg, f/{self.speed:.0f}"


class Spectrograph:
    """
    Spectrograph simulation.

    .. Warning:: the spectral PSF impacts the cross-dispersion profile,
                 but is not applied in the dispersion direction.
    """

    #: Mutable parameters (list)
    mutable_parameters = ('spectral_range', 'spectral_resolution',
                          'spectral_sigma', 'xdisp_sigma',
                          'spatial_sigma', 'guiding_sigma',
                          'spatial_scale', 'spatial_scale_insigma',
                          'spatial_shape', 'spatial_shape_insigma',
                          'mirror.temperature', 'mirror.emissivity',
                          'camera.acceptance', 'camera.speed',
                          )

    def __init__(self, config, verbose=False):
        """
        Initialize the spectrograph properties from `config` dictionary.

        :param dict config: spectrograph configuration dictionary
        :param bool verbose: verbose mode
        """

        config = self.rescale_parameters(**config)

        #: Spectrograph name
        self.name = config["name"]

        # First set the spectral domain, which is needed for all
        # chromatic quantities
        #: Wavelength domain `[wmin, wmax]` [Å]
        self.spectral_range = [ float(w) for w in config["spectral_range"] ]
        #: Constant input spectral resolution (actually resolving power)
        self.spectral_resolution = float(config.get("spectral_resolution", 0))
        #: Dispersion law filename (physical offset as a function of wavelength)
        self.dispersion_law = config.get("dispersion_law", '')
        #: Dispersion law scale
        self.dispersion_scale = float(config.get("dispersion_scale", 1))
        if self.spectral_resolution and self.dispersion_law:       # both: error
            raise ValueError("Cannot set 'spectral_resolution' AND "
                             "'dispersion_law' simultaneously.")
        if not (self.spectral_resolution or self.dispersion_law):  # none: error
            raise ValueError(
                "'spectral_resolution' OR 'dispersion_law' should be set.")
        if self.dispersion_law:                                 # dispersion law
            wname, dname = 'wavelength', 'offset'
            tab = iotools.read_ecsv(
                self.dispersion_law, colnames=[wname, dname],
                description='dispersion law' if verbose else '')
            assert tab[dname].unit == 'pix'
            #: Dispersion solution (wavelengths in Å, offset in pix)
            self.dsol = iotools.chromatic_interpolator(
                tab[wname].to(u.AA), tab[dname] * self.dispersion_scale,
                ext='extrapolate')
            #: Wavelength solution (wavelengths in Å, offset in pix)
            self.wsol = iotools.chromatic_interpolator(
                tab[wname].to(u.AA), tab[dname] * self.dispersion_scale,
                ext='extrapolate', inverse=True)
        elif self.spectral_resolution:                          # spectral res.
            self.dsol = self.wsol = None

        # Spectral domain: self.lbda[_edges]
        self.lbda = None        #: Wavelength at bin center (nlbda,)
        self.lbda_edges = None  #: Wavelength at bin edges (nlbda+1,)
        self.set_lbda()

        #: Chromatic (optical) spectral PSF on detector [px] (∝ λ at 1 µm)
        self.spectral_sigma = float(config["spectral_sigma"])
        #: Achromatic cross-dispersion width [px]
        self.xdisp_sigma = float(config["xdisp_sigma"])

        # Spatial domain: self.(nx,ny), self.(x,y)[_edges]
        #: MLA shape (nx, ny) [spx]
        self.ny, self.nx = self.spatial_shape = [
            int(_) for _ in config["spatial_shape"] ]
        self.y = self.x = None              #: Central coord. grids [spx] (ny, nx)
        self.y_edges = self.x_edges = None  #: Edge coord. grids [spx] (ny, nx)
        self.set_spaxels()

        #: Chromatic (optical) spatial PSF on MLA [arcsec] (∝ λ at 1 µm)
        self.spatial_sigma = float(config["spatial_sigma"])
        #: Achromatic (guiding) spatial PSF on MLA [arcsec]
        self.guiding_sigma = float(config["guiding_sigma"])
        #: MLA sampling [arcsec/spx]
        self.spatial_scale = float(config["spatial_scale"])

        # Throughput (constant or interpolated from file)
        try:
            # throughput is a constant
            self.throughput = float(config["throughput"])
            self.throughput_name = self.throughput_name_interp = None
        except ValueError:
            # throughput is a filename
            self.throughput_name = config["throughput"]  #: Throughput filename
            wname, tname = 'wavelength', 'throughput'
            tab = iotools.read_ecsv(self.throughput_name,
                                    colnames=[wname, tname],
                                    description='throughput' if verbose else '')
            #: Throughput interpolator (wavelengths in Å)
            self.throughput_interp = iotools.chromatic_interpolator(
                tab[wname].to(u.AA), tab[tname], ext='zeros')
            #: Throughput [% ph]
            self.throughput = self.throughput_interp(self.lbda)
            # This will be updated in update_lbda when needed

        #: Mirror
        self.mirror = Mirror(
            surface=Mirror.get_surface(
                dext=float(config['mirror']['diameter_ext']),
                dint=float(config['mirror']['diameter_int'])),
            temperature=float(config['mirror'].get('temperature', 0)),
            emissivity=float(config['mirror'].get('emissivity', 0)),
        )

        # #: Camera
        # self.camera = Camera(
        #     acceptance=float(config['camera'].get('acceptance', 0)),
        #     speed=float(config['camera'].get('speed', np.inf)))

        self.meta = config      #: Meta-parameters

    @classmethod
    def from_config(cls, config, verbose=False):
        """
        Initialize from spectrograph config.

        Added for consistency between classes as an alternative to
        :meth:`__init__`.

        :param dict config: spectrograph configuration dictionary
        :param bool verbose: verbose mode
        """

        return cls(config, verbose=verbose)

    @staticmethod
    def lbda_from_respow(spectral_range, res_power, npx=2):
        r"""
        Compute wavelength ramp for constant n-px resolving power.

        .. math::

           \frac{\Delta\lambda}{\lambda} &= \frac{1}{n\mathcal{R}} &\\
           &= \ln 10\,\Delta\log\lambda &\\
           \log\lambda_i^e &= \log\lambda_0 + i\times \Delta\log\lambda
           &\quad\text{(edges)} \\
           \log\lambda_i &= (\log\lambda_i + \log\lambda_{i+1})/2
           &\quad\text{(center)}

        :param 2-tuple spectral_range: spectral domain
        :param float res_power: resolving power R
        :param float npx: n-px resolution (i.e. n px per spectral elements)
        :return: mid and bin edge wavelengths [same units as input domain]
        """

        wmin, wmax = spectral_range
        dlog = 1/(2.302585092994046 * res_power * npx)  # ln(10)=2.303...
        npx = round((np.log10(wmax) - np.log10(wmin)) / dlog)
        loglbda_edges = np.linspace(np.log10(wmin), np.log10(wmax), npx + 1)
        loglbda_mid = (loglbda_edges[:-1] + loglbda_edges[1:])/2

        return 10**loglbda_mid, 10**loglbda_edges

    def set_lbda(self):
        """
        Set wavelength coordinates.

        Set :attr:`lbda` (:attr:`nlbda` mid wavelengths) and :attr:`lbda_egdes`
        (:attr:`nlbda` + 1 edge wavelengths).
        """

        if self.wsol is None:   # Compute from constant resolving power
            self.lbda, self.lbda_edges = self.lbda_from_respow(
                self.spectral_range, self.spectral_resolution)
        else:                   # Compute from wavelength solution
            wmin, wmax = self.spectral_range
            npx = round(self.dsol(wmax) - self.dsol(wmin))    # Total nb of px
            self.lbda = self.wsol(np.r_[:npx])                # λ at bin center
            self.lbda_edges = self.wsol(np.r_[:npx+1] - 0.5)  # λ at bin edge

    @property
    def nlbda(self):
        """Number of spectral pixels."""

        return len(self.lbda)

    def effective_resolution(self, npx=2, sigma=None, average=False):
        r"""
        Effective spectral resolution.

        .. math::

           R &= \frac{2}{n \delta\lambda} \\
           \delta\lambda &= \max(1, \sigma) \times \Delta\lambda

        where :math:`\Delta\lambda` is the spectral step [Å] and
        :math:`\sigma` is the spectral resolution [px].

        :param float npx: n-px resolution (i.e. n px per spectral elements)
        :param sigma: spectral PSF stddev override
        :param bool average: chromatic average (weighted by spectral step)
        :return: effective n-px wavelength resolution
                 (as a function of wavelength or averaged)
        """

        if sigma is None:
            sigma = self.get_spectral_sigma()  # (nlbda,)
        sigma = np.maximum(sigma, 1)

        dlbda = np.diff(self.lbda_edges)
        wres = self.lbda / (npx * sigma * dlbda)  # (nlbda,)

        if average:             # Chromatic average
            wres = np.average(wres, weights=dlbda)

        return wres

    def chromatic_average(self, quantity):
        """
        Chromatic average of a quantity (averaged over wavelength rather than px)

        :param array quantity: chromatic quantity (nlbda,)
        :return: chromatic average
        """

        return np.average(quantity, weights=np.diff(self.lbda_edges))

    def set_spaxels(self):
        """
        Set spaxel coordinates [spx] from MLA shape.

        Set `self.(x,y)[_edges]` from :attr:`spatial_shape`.
        """

        hnx, hny = (self.nx - 1)/2, (self.ny - 1)/2
        self.y, self.x = np.ogrid[-hny:hny:self.ny*1j,
                                  -hnx:hnx:self.nx*1j]  # Central coord. grids [spx]
        self.y_edges, self.x_edges = np.ogrid[
            -hny - 0.5:hny + 0.5:(self.ny + 1)*1j,
            -hnx - 0.5:hnx + 0.5:(self.nx + 1)*1j]      # Edge coord. grids [spx]

    @property
    def mla_extent(self):
        """MLA extent [spx]."""

        hx, hy = self.nx / 2, self.ny / 2  # Half total width [spx]

        return [-hx, hx, -hy, hy]

    def __str__(self):

        wmin, wmax = self.spectral_range
        avwres0 = self.effective_resolution(npx=2, sigma=0, average=True)
        avwres = self.effective_resolution(npx=2, average=True)
        wres = self.effective_resolution(npx=2)
        imin = np.argmin(wres)

        s = f"Spectrograph {self.name!r}:"
        s += f"\n  Spectral range: {wmin:_.0f}-{wmax:_.0f} Å, {self.nlbda} px"
        if self.wsol is not None:
            s += "\n  Spectral dispersion: " \
                f"{self.dispersion_law!r} ×{self.dispersion_scale} (R0~{avwres0:.0f})"
        else:
            s += f"\n  Fixed resolving power (2 px): {avwres0:.0f}"
        s += f"\n  Spectral PSF: chromatic σ={self.spectral_sigma:.2f} px at 1 µm, "
        s += f"x-disp. σ={self.xdisp_sigma:.2f} px"
        s += "\n  Resolving power (2-px + σ): " \
            f"R~{avwres:.0f} (λ-average), " \
            f"min={wres[imin]:.0f} at {self.lbda[imin]:_.0f} Å"
        shape = "×".join([ str(i) for i in self.spatial_shape ])
        s += f"\n  MLA: {shape} spx of {self.spatial_scale*1e3:.0f} mas"
        s += f"\n  Spatial PSF: chromatic σ={self.spatial_sigma*1e3:.0f} mas at 1 µm, "
        s += f"guiding σ={self.guiding_sigma*1e3:.0f} mas"

        if self.throughput_name:
            s += (f"\n  Total throughput: {self.throughput_name!r} "
                  f"(~{self.throughput.mean():.0%})")  # px-average
        else:
            s += f"\n  Total throughput: constant {self.throughput:.0%}"

        s += "\n  " + str(self.mirror)

        return s

    @property
    def flambda2photon(self):
        """Chromatic conversion factor from erg/s/cm²/Å to ph/s."""

        dlbda = np.diff(self.lbda_edges)  # Spectral step (nlbda,)
        hnu = 1.9864459e-08 / self.lbda   # Photon energy [erg] with lbda in [Å]

        # erg/s/cm²/Å * (cm² * throughput * Å / erg/ph) = ph/s
        return (self.mirror.surface * 1e4 *
                self.throughput * dlbda / hnu)  # (nlbda,) [ph/s]

    def rescale_parameters(self, **kwargs):
        """
        Convert parameters in relative units to absolute units.

        .. Warning:: if needed, the reference parameters
           (e.g. :attr:`spatial_sigma`) should be updated
           simultaneously to the normalized parameters (not during a
           subsequent call).
        """

        # Get sigma from kwargs, or self, or raise.
        def _get_sigma(name):        # 'spatial_sigma' or 'spectral_sigma'
            if hasattr(self, name):  # Initialized spectrograph
                return kwargs.get(name, getattr(self, name))
            else:                    # Config dictionary
                return kwargs[name]

        spatial_sigma = _get_sigma("spatial_sigma")
        spectral_sigma = _get_sigma('spectral_sigma')

        new_kw = {}             # native or rescaled parameters

        # treat spatial_scale first, as it will impact some other parameters
        if 'spatial_scale_insigma' in kwargs:
            # spatial_scale in units of spatial_sigma
            spatial_scale = new_kw["spatial_scale"] = (
                kwargs['spatial_scale_insigma'] * spatial_sigma)
            # print("Rescaling spatial_scale: "
            #       f"{v}×{spatial_sigma}={new_kw['spatial_scale']}")
        else:
            spatial_scale = _get_sigma("spatial_scale")  # initial value

        for k, v in kwargs.items():
            if not k.endswith('_insigma'):  # Parameter in absolute units
                new_kw[k] = v               # Left untouched
            elif k == "spatial_scale_insigma":  # see above
                continue
            elif k == "spatial_shape_insigma":
                # convert spatial_shape in units of spatial_sigma to spx
                new_kw["spatial_shape"] = [
                    round(vv * spatial_sigma / spatial_scale) for vv in v ] # (ny, nx)
                # print("Rescaling spatial_shape: "
                #       f"{v}×{spatial_sigma}/{spatial_scale}={new_kw['spatial_shape']}")
            elif k == "aperture_radius_insigma":
                # convert aperture_radius in units of spatial_sigma to spx
                new_kw["aperture_radius"] = v * spatial_sigma / spatial_scale
                # print("Rescaling aperture_radius: "
                #       f"{v}×{spatial_sigma}/{spatial_scale}={new_kw['aperture_radius']}")
            elif k == "xdisp_width_insigma":
                # xdisp_width in units of spectral_sigma
                new_kw["xdisp_width"] = round(v * spectral_sigma)
                # print("Rescaling xdisp_width: "
                #       f"{v}×{spectral_sigma}={new_kw['xdisp_width']}")
            else:
                raise KeyError("Unknown relative parameter", k)

        return new_kw

    def update(self, **kwargs):
        """
        Update any mutable attribute of the spectrograph.
        """

        kwargs = self.rescale_parameters(**kwargs)

        updates = {}
        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            if v is None:        # Skip
                continue

            if '.' not in k:         # Simple key
                setattr(self, k, v)  # Update attribute
            else:                    # Chained key: key1.key2
                k1, k2 = k.split('.')
                setattr(getattr(self, k1), k2, v)

            if k in ('spectral_range', 'spectral_resolution'):
                # Simulation.update is in charge of updating other chromatic
                # quantities if 'spectral_range' or 'spectral_resolution' are
                # modified.
                if k == "spectral_resolution" and self.wsol is not None:
                    warnings.warn("Switching to constant resolving power.")
                    updates['dispersion_law'] = ''
                    self.wsol = self.dsol = None
                self.update_lbda()  # Update spectral attributes

            if k == 'spatial_shape':
                self.set_spaxels()  # update spatial attributes

            if '.' not in k:    # Simple key
                updates[k] = v  # Keep track of updated parameters
            else:               # Chained key: key1.key2
                k1, k2 = k.split('.')
                updates[k1] = dict(k2=v)

        # Update the metadata
        self.meta = {**self.meta, **updates}

    def update_lbda(self):
        """
        Update wavelength and chromatic components.
        """

        self.set_lbda()         # Update wavelengths
        # Update throughput if needed
        if self.throughput_interp is not None:
            self.throughput = self.throughput_interp(self.lbda)

    def generate_background(self, spectrum):
        """
        Generate a photon flux cube from uniform scene background spectrum.

        :param spectrum: uniform scene background spectrum [erg/s/cm²/Å/arcsec²]
        :return: (nlbda, ny, nx) photon flux cube [ph/s/spx]
        """

        # erg/s/cm²/Å/arcsec² * cm² * Å / erg/ph * arcsec² = ph/s/spx
        flux = spectrum * self.flambda2photon * self.spatial_scale**2

        return np.full((self.nlbda, self.ny, self.nx),
                       flux[:, np.newaxis, np.newaxis])  # (nlbda, ny, nx)

    @staticmethod
    def get_chromatic_sigma(lbda, chromatic_sigma, constant_sigma,
                            wref=10_000., xdims=0):
        """
        Get total PSF, including chromatic and constant components.

        The total (Gaussian) stddev is the quadratic sum of two components:

        - the chromatic stddev, proportional to wavelength,
          normalized at `wref`,
        - the achromatic, constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        :param lbda: wavelength
        :param chromatic_sigma: chromatic (linear) stddev
        :param constant_sigma: achromatic (constant) stddev
        :param float wref: reference wavelength (same unit as `lbda`)
        :param int xdims: extra dimensions to be appended
        :return: total stddev as function of wavelength
        """

        lmin, lmax = np.array(lbda)[[0, -1]]  # 1st and last wavelengths
        assert lmin > wref/3 and lmax < wref*3, \
            "Input and reference wavelengths probably not in same units."

        sigma = np.hypot(constant_sigma,
                         chromatic_sigma * (lbda / wref))  # [px]
        if xdims:
            sigma = sigma.reshape(sigma.shape + (1,) * xdims)

        return sigma

    def get_spectral_sigma(self, xdims=0, xdisp_sigma=None):
        """
        Get spectral PSF stddev [px].

        The total (Gaussian) spectral PSF is made of two components:

        - the optical (chromatic) component, with stddev proportional
          to wavelength, normalized at wref=1 µm,
        - the achromatic component, with constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        :param int xdims: extra dimensions to be appended
        :param float xdisp_sigma: constant sigma override [px]
                                  (None for default)
        :return: total sigma [px]
        """

        if xdisp_sigma is None:
            xdisp_sigma = self.xdisp_sigma

        return self.get_chromatic_sigma(self.lbda,
                                        self.spectral_sigma,
                                        xdisp_sigma,
                                        wref=10_000, xdims=xdims)

    def get_spatial_sigma(self, xdims=0, guiding_sigma=None):
        """
        Get spatial PSF stddev [arcsec].

        The total (Gaussian) spatial PSF is made of two components:

        - the optical (chromatic) component, with stddev proportional
          to wavelength, normalized at wref=1 µm,
        - the guiding (achromatic) component, with constant stddev.

        If needed, the 1D vector can be embedded in a N-dim array of
        shape `(nlbda,) + (1,)*xdims`.

        :param int xdims: extra dimensions to be appended
        :param float guiding_sigma: guiding sigma override [arcsec]
                                    (None for default)
        :return: total sigma [arcsec]
        """

        if guiding_sigma is None:
            guiding_sigma = self.guiding_sigma

        return self.get_chromatic_sigma(self.lbda,
                                        self.spatial_sigma,
                                        guiding_sigma,
                                        wref=10_000, xdims=xdims)

    def get_spatial_psf(self, position=(0, 0)):
        """
        Get normalized 2D spatial PSF on the MLA.

        This uses the exact 2D Gaussian PSF integration over the spx.

        :param 2-tuple position: point source position in MLA [spx]
        :return: normalized PSF (nlbda, ny, nx)
        """

        sigmas = self.get_spatial_sigma(xdims=2) / self.spatial_scale
        psf = integ_gaussian2D_erf(
            (self.x_edges, self.y_edges),  # ((1, nx), (ny, 1)) [spx]
            sigmas,                        # (nlbda, 1, 1) [spx]
            position,                      # [spx]
            normed=True)                   # sum(axis=(1, 2)) = 1

        return psf                         # (nlbda, ny, nx)

    def generate_point_source(self, spectrum, position=(0, 0)):
        """
        Generate a photon flux cube from a point source spectrum.

        :param spectrum: point source spectrum [erg/s/cm²/Å]
        :param 2-tuple position: point source position in MLA [spx]
        :return: (nlbda, ny, nx) photon flux cube [ph/s/spx]
        """

        # Spatial PSF
        psf = self.get_spatial_psf(position=position)      # (nlbda, ny, nx)

        # erg/s/cm²/Å / erg/ph * cm² * Å = ph/s
        flux = spectrum * self.flambda2photon      # (nlbda,) [ph/s]

        return np.reshape(flux, (-1, 1, 1)) * psf  # Point source (nlbda, ny, nx)

    def generate_thermal(self):
        """
        Generate a photon flux cube from mirror thermal emission.

        :return: (nlbda, ny, nx) photon flux cube [ph/s/spx]
        """

        signal = self.thermal_signal()  # (nlbda,)

        return np.full((self.nlbda, self.ny, self.nx),
                       signal[:, np.newaxis, np.newaxis])  # (nlbda, ny, nx)

    def point_source_variance(self,
                              varcube, position=(0, 0), radius=5, optimal=True,
                              verbose=False):
        """
        Point-source extracted variance from variance cube.

        It is evaluated from a plain summation over the aperture, or from an
        optimal extraction, i.e. inverse-variance weighted least-square fit of
        the PSF profile (assumed Gaussian).  As the extraction is assumed to do
        a perfect job on the signal, only the variance depends on instrumental
        parameters and aperture definition.

        :param varcube: variance cube (nlbda, ny, nx) [ADU²]
        :param 2-tuple position: point source position in MLA [spx]
        :param float radius: MLA aperture [spx]
        :param bool optimal: optimal vs. plain extraction
        :return: spx spectrum [ADU] and variance [ADU²]
        """

        # Spatial PSF
        psf = self.get_spatial_psf(position=position)  # (nlbda, ny, nx)

        # Aperture
        x0, y0 = position
        r = np.hypot(self.x - x0, self.y - y0)  # (ny, nx) [spx]
        aper = (r <= radius)
        nspx = np.count_nonzero(aper)           # Selected spx
        if verbose:
            print(f"Aperture of {radius=} spx: {nspx} spx selected")

        # If cube is (nlbda, ny, nx), cube[:, aper] is (nlbda, nspx)
        if optimal:  # Optimal extraction variance = 1/sum(psf**2/var)
            variance = 1 / (psf[:, aper]**2 / varcube[:, aper]).sum(axis=-1)
        else:        # Plain summation: variance = sum(variance)
            variance = varcube[:, aper].sum(axis=-1)

        return variance                  # (nlbda,) [ADU²]

    @property
    def omega(self):
        """Spaxel solid angle [sr]."""

        hspx = self.spatial_scale / 2  # [arcsec]
        hspx *= 4.84813681109536e-06   # [rad]

        return np.pi * np.sin(hspx)**2

    def thermal_signal(self, domains=None, temperature=None, emissivity=None):
        """
        Mirror thermal signal [ph/s/spx/Δλ].

        :param domains: (nlbda, 2) list of spectral domains [Å],
                        or spectral px by default
        :param float temperature: mirror temperature [K], or default one
        :param float emissivity: mirror emissivity, or default one
        :return: thermal signal in ph/s/spx/Δλ
        """

        from .thermal import thermal_signal

        if domains is None:
            domains = np.vstack([self.lbda_edges[:-1],
                                 self.lbda_edges[1:]]).T  # (nlbda, 2) [Å]
        if temperature is None:
            temperature = self.mirror.temperature  # Mirror temperature [K]

        if emissivity is None:
            emissivity = self.mirror.emissivity    # Mirror emissivity

        omega = self.omega                         # Spx solid angle [sr]
        signal = np.array([
            thermal_signal(omega,
                           self.mirror.surface,    # Collecting area [m²]
                           domain_mu,              # Spectral bin [µm]
                           temperature, emissivity)
            for domain_mu in (domains * 1e-4) ])   # Convert from Å to µm

        return signal                              # [ph/s/spx/Δλ]


def plot_spectral_resolution(spectro, ax=None):
    """
    Plot effective spectral resolution.
    """

    wres = spectro.effective_resolution(npx=2)
    mwres = spectro.effective_resolution(npx=2, average=True)
    dlbda = np.diff(spectro.lbda_edges)           # [Å]
    sigma = spectro.get_spectral_sigma() * dlbda  # [Å]

    lbda_mu = spectro.lbda / 1e4  # [µm]

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(tight_layout=True)

    ax.plot(lbda_mu, wres, c='C00')
    ax.set(xlabel="Wavelength [µm]",
           title="Resolving power (2 δλ): " +
           f"λ-mean={mwres:.0f}, min={wres.min():.0f}")
    ax.axhline(mwres, ls='--', c='C00')
    ax.set_ylabel(r"Resolving power $\mathcal{R} = \lambda/(2\delta\lambda)$",
                  c='C00')
    ax.tick_params(axis='y', labelcolor='C00');
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(lbda_mu, np.maximum(dlbda, sigma), c='C01')
    ax2.plot(lbda_mu, dlbda, c='C01', ls=':',
             label=f"Δλ ({spectro.dispersion_law!r}×{spectro.dispersion_scale})")
    ax2.plot(lbda_mu, sigma, c='C01', ls='--',
             label=f"σ ({spectro.spectral_sigma} λ/µm & {spectro.xdisp_sigma} px)")
    ax2.set_ylabel("δλ [Å]", color='C01')
    ax2.tick_params(axis='y', labelcolor='C01');
    ax2.legend()

    return ax


if __name__ == "__main__":

    from mlaperf.iotools import get_config
    config = get_config("instrument.toml")

    spectro = Spectrograph(config["spectrograph"])
    print(spectro)
