"""Ground-based slicer spectrograph: seeing-limited PSF and atmospheric transmission.

`GroundSlicerSpectrograph` specialises `~slicersim.spectrograph.SlicerSpectrograph`
for a spectrograph behind a ground-based telescope, with an
`~slicersim.atmosphere.Atmosphere` providing the seeing-induced part of the PSF,
the atmospheric transmission and the sky emission.

.. autosummary::

   GroundSlicerSpectrograph
"""

import numpy as np

from .atmosphere import moffat_density
from .spectrograph import SlicerSpectrograph

__all__ = ["GroundSlicerSpectrograph"]


class GroundSlicerSpectrograph(SlicerSpectrograph):
    """Slicer spectrograph behind a ground-based telescope.

    Differences with `~slicersim.spectrograph.SlicerSpectrograph`:

    - **spatial PSF**: circular Moffat (``moffat_beta``) whose FWHM is the
      quadratic sum of two components (pyetc/ESO-ETC approximation):

      * the **seeing-induced** FWHM, chromatic, given by the
        `~slicersim.atmosphere.Atmosphere` (seeing, airmass, telescope
        diameter), see `~slicersim.atmosphere.get_seeing_fwhm`;
      * the **optical** FWHM (``optical_fwhm``), achromatic image quality of
        the telescope and instrument.

      See `get_psf_fwhm`. ``moffat_truncation`` (if > 0) emulates pyetc's
      truncation of the PSF to a box of ~``moffat_truncation`` FWHM,
      renormalized to unit flux (0: exact Moffat, the default);
    - **atmospheric transmission**: applied to astrophysical sources
      (`flambda2photon`), not to the sky emission (`generate_background`);
    - **LSF**: ``dispersion_resolution`` is the Gaussian *FWHM* in pixels;
    - **no anamorphosis**: 1 detector pixel per voxel.

    Parameters
    ----------
    *args, **kwargs
        See `~slicersim.spectrograph.Spectrograph`.
    atmosphere : ~slicersim.atmosphere.Atmosphere, optional
        Atmosphere above the telescope. If None, no seeing and no
        transmission. Default is None.
    """
    _ANAMORPHOSE = 1
    _SPECTROGRAPH_TYPE = "slicer"

    #: PSF parameters of the instrument, in ``[spectrograph.psf.spatial]``
    PSF_KEYS = ["optical_fwhm", "moffat_beta", "moffat_truncation"]
    mutable_parameters = SlicerSpectrograph.mutable_parameters + PSF_KEYS

    def __init__(self, *args, atmosphere=None, **kwargs):
        """Initialize the spectrograph (see the class docstring)."""
        self.atmosphere = atmosphere
        self._atmosphere_off = False
        self._psf_cache = {}
        super().__init__(*args, **kwargs)
        if self.atmosphere is not None:
            self.mutable_parameters = self.mutable_parameters + \
                [f"atmosphere.{k}" for k in self.atmosphere.mutable_parameters]

    @classmethod
    def from_config(cls, config, telescope=None, atmosphere=None):
        """Create the spectrograph from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Spectrograph configuration dictionary, see
            `~slicersim.spectrograph.Spectrograph.from_config`. ``[psf.spatial]`` may contain
            ``optical_fwhm`` (arcsec, default 0), ``moffat_beta`` (default
            2.5) and ``moffat_truncation`` (default 0).
        telescope : ~slicersim.telescope.Telescope, optional
            Telescope feeding the spectrograph. Default is None.
        atmosphere : ~slicersim.atmosphere.Atmosphere, optional
            Atmosphere above the telescope. Default is None.

        Returns
        -------
        GroundSlicerSpectrograph
            The spectrograph instance.
        """
        init_prop, _ = cls._parse_config(config)
        return cls(telescope=telescope, atmosphere=atmosphere, **init_prop)

    @staticmethod
    def _parse_config(config):
        """Parse the configuration, adding the Moffat PSF parameters.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        init_prop : dict
            Keyword arguments for the class constructor.
        input_config : dict
            Deep copy of the input configuration.
        """
        init_prop, input_config = SlicerSpectrograph._parse_config(config)
        spatial = input_config["psf"]["spatial"]
        init_prop["spatial_psf"] |= {
            "optical_fwhm": float(spatial.get("optical_fwhm", 0.)),
            "moffat_beta": float(spatial.get("moffat_beta", 2.5)),
            "moffat_truncation": float(spatial.get("moffat_truncation", 0) or 0),
        }
        return init_prop, input_config

    def update(self, reset_others=False, **kwargs):
        """Update mutable parameters, including ``atmosphere.<param>``.

        See `~slicersim.spectrograph.Spectrograph.update`.

        Parameters
        ----------
        reset_others : bool, optional
            If True, the parameters not given are reset to their initial
            values. Default is False.
        **kwargs
            Parameters to update, e.g. ``optical_fwhm``, ``moffat_beta`` or
            ``atmosphere.seeing`` (also ``atmosphere__seeing``).

        Returns
        -------
        list of str
            Names of the components that changed (``"atmosphere"`` included).
        """
        atmosphere_updates = {}
        others = {}
        for k, v in kwargs.items():
            name = k.replace("__", ".")
            if name.startswith("atmosphere."):
                atmosphere_updates[name.replace("atmosphere.", "")] = v
            else:
                others[k] = v

        what_changed = super().update(reset_others=reset_others, **others) if others else []
        if atmosphere_updates and self.atmosphere is not None:
            self.atmosphere.update(reset_others=reset_others, **atmosphere_updates)
            what_changed.append("atmosphere")

        return what_changed

    # ---------- atmosphere ---------- #
    def get_atmospheric_transmission(self, lbda=None):
        """Atmospheric transmission (1 without atmosphere).

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength in Angstrom. If None, `lbda` is used. Default is None.

        Returns
        -------
        numpy.ndarray
            Transmission, of shape ``(nlbda,)``.
        """
        if lbda is None:
            lbda = self.lbda
        if self.atmosphere is None or self._atmosphere_off:
            return np.ones(np.shape(lbda))
        return self.atmosphere.get_transmission(lbda)

    @property
    def flambda2photon(self):
        """Conversion from erg/s/cm²/Å (top of the atmosphere) to ph/s.

        Includes the atmospheric transmission, see
        `get_atmospheric_transmission`.
        """
        return super().flambda2photon * self.get_atmospheric_transmission()

    @property
    def flambda2photon_sky(self):
        """Conversion from erg/s/cm²/Å (at the telescope) to ph/s.

        Without atmospheric transmission, e.g. for the sky emission.
        """
        return SlicerSpectrograph.flambda2photon.fget(self)

    def generate_background(self, spectrum, oversampling=None, apply_lsf=True):
        """Generate a photon flux cube from the uniform sky emission.

        The sky emission is not attenuated by the atmospheric transmission.

        Parameters
        ----------
        spectrum : array_like
            Sky surface brightness, of shape ``(nlbda,)``, in
            erg/s/cm²/Å/arcsec².
        oversampling : int, optional
            Spatial oversampling factor of the output cube. Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.

        Returns
        -------
        numpy.ndarray
            Photon flux cube, of shape ``(nlbda, ny, nx)``, in ph/s/spx.
        """
        flux = spectrum * self.flambda2photon_sky * self.spx_area
        bkgd_cube = np.full((self.nlbda, *self.get_spectrograph_shape(oversampling=oversampling)),
                            flux[:, np.newaxis, np.newaxis])
        if apply_lsf:
            bkgd_cube = self.apply_line_spread_function(bkgd_cube)
        return bkgd_cube

    # ---------- LSF ---------- #
    def get_lsf_dispersion(self, as_="scale"):
        """Get the Gaussian LSF dispersion, in wavelength bins (pixels).

        Contrary to `~slicersim.spectrograph.Spectrograph.get_lsf_dispersion`, the
        `dispersion_resolution` is the LSF FWHM, hence
        ``sigma = dispersion_resolution / 2.3548``.

        Parameters
        ----------
        as_ : str or None, optional
            ``"scale"`` or ``"sigma"`` (default), or ``"resolution"``/None
            (full resolution element).

        Returns
        -------
        float
            LSF dispersion in wavelength bins.
        """
        if as_ in ["scale", "sigma"]:
            return self.get_dispersion_resolution() / (2 * np.sqrt(2 * np.log(2)))
        return super().get_lsf_dispersion(as_=as_)

    # ---------- PSF ---------- #
    def get_psf_fwhm(self, lbda=None, which="total"):
        """Get the FWHM of the spatial PSF, in arcsec.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength in Angstrom. If None, `lbda` is used. Default is None.
        which : str, optional
            - ``"seeing"``: seeing-induced FWHM (0 without atmosphere);
            - ``"optical"``: telescope + instrument FWHM (``optical_fwhm``);
            - ``"total"``: quadratic sum of both. Default.

        Returns
        -------
        numpy.ndarray
            FWHM in arcsec, of shape ``(nlbda,)``.

        Raises
        ------
        ValueError
            If ``which`` is unknown.
        """
        if lbda is None:
            lbda = self.lbda
        lbda = np.atleast_1d(lbda)

        if self.atmosphere is None:
            seeing = np.zeros(lbda.shape)
        else:
            seeing = self.atmosphere.get_seeing_fwhm(lbda, diameter=self.telescope.diameter_ext)
        optical = np.full(lbda.shape, self.spatial_psf["optical_fwhm"])

        if which == "seeing":
            return seeing
        if which == "optical":
            return optical
        if which == "total":
            return np.hypot(seeing, optical)
        raise ValueError(f"unknown {which=}: 'seeing', 'optical' or 'total'")

    def _moffat_image(self, fwhm, position, oversampling, truncation):
        """Moffat PSF integrated over the spaxels.

        Parameters
        ----------
        fwhm : float
            Moffat FWHM in arcsec.
        position : tuple of float
            ``(x, y)`` position in spaxels.
        oversampling : int
            Spatial oversampling factor.
        truncation : float
            If > 0, pyetc-like truncation (see the class docstring).

        Returns
        -------
        numpy.ndarray
            PSF image, of shape ``(ny, nx)``.
        """
        ny, nx = self.spx_shape
        scale = float(self.spx_spatial_scale)
        x = ((np.arange(nx * oversampling) + 0.5) / oversampling - nx / 2 - position[0]) * scale
        y = ((np.arange(ny * oversampling) + 0.5) / oversampling - ny / 2 - position[1]) * scale
        r2 = x[None, :] ** 2 + y[:, None] ** 2
        img = moffat_density(r2, fwhm, self.spatial_psf["moffat_beta"]) * (scale / oversampling) ** 2
        img = img.reshape(ny, oversampling, nx, oversampling).sum(axis=(1, 3))
        if truncation:
            # pyetc.etc.moffat(): odd box of ns spaxels around the source, renormalized
            ns = int((truncation * fwhm / scale + 1) / 2) * 2 + 1
            xc = np.arange(nx) + 0.5 - nx / 2 - position[0]
            yc = np.arange(ny) + 0.5 - ny / 2 - position[1]
            inbox = (np.abs(xc)[None, :] <= ns / 2) & (np.abs(yc)[:, None] <= ns / 2)
            img = np.where(inbox, img, 0)
            img /= img.sum()
        return img

    def get_spatial_psf(self, profile="default", position=(0, 0), oversampling=None,
                        as_oversampled=False, incl_atmosphere=True, incl_telescope=True,
                        nlbda_nodes=16, **kwargs):
        """Get the normalized 2D spatial PSF on the spaxel grid.

        For the ``"default"`` (or ``"moffat"``) profile, the PSF is a Moffat
        with the FWHM of `get_psf_fwhm`, integrated over the spaxels. It is
        computed at ``nlbda_nodes`` wavelengths and linearly interpolated in
        between (the seeing FWHM varies as λ^-1/5), and cached. Other profiles
        fall back to `~slicersim.spectrograph.SlicerSpectrograph.get_spatial_psf`.

        Parameters
        ----------
        profile : str, optional
            ``"default"`` or ``"moffat"``, or any profile of
            `~slicersim.spectrograph.Spectrograph.get_spatial_psf`. Default is ``"default"``.
        position : tuple of float, optional
            ``(x, y)`` point source position, in spaxels. Default is (0, 0).
        oversampling : int, optional
            Spatial oversampling (at least 5, default 10).
        as_oversampled : bool, optional
            Not implemented for the Moffat profile. Default is False.
        incl_atmosphere : bool, optional
            If False, the seeing-induced component is removed. Default is True.
        incl_telescope : bool, optional
            If False, the optical component is removed. Default is True.
        nlbda_nodes : int, optional
            Number of wavelengths where the PSF is computed. Default is 16.
        **kwargs
            Goes to `~slicersim.spectrograph.Spectrograph.get_spatial_psf` for other profiles.

        Returns
        -------
        numpy.ndarray
            Normalized PSF, of shape ``(nlbda, ny, nx)``.

        Raises
        ------
        NotImplementedError
            If ``as_oversampled`` is True for the Moffat profile.
        ValueError
            If both the seeing and the optical components are removed.
        """
        if profile not in ("default", "moffat"):
            return super().get_spatial_psf(profile=profile, position=position,
                                           oversampling=oversampling, as_oversampled=as_oversampled,
                                           incl_telescope=incl_telescope, **kwargs)
        if as_oversampled:
            raise NotImplementedError("oversampled Moffat PSF not implemented.")
        if oversampling is None or oversampling < 5:
            oversampling = 10

        which = {(True, True): "total", (True, False): "seeing",
                 (False, True): "optical"}.get((incl_atmosphere, incl_telescope))
        if which is None:
            raise ValueError("incl_atmosphere and incl_telescope cannot be both False.")

        psf = self.spatial_psf
        atm = self.atmosphere.meta if self.atmosphere is not None else {}
        key = (tuple(np.round(np.asarray(position, dtype=float), 6)), oversampling, nlbda_nodes, which,
               tuple(np.asarray(self.spx_shape).tolist()), float(self.spx_spatial_scale),
               self.nlbda, float(self.lbda[0]), float(self.lbda[-1]),
               float(self.telescope.diameter_ext), atm.get("seeing"), atm.get("airmass"),
               *[psf[k] for k in self.PSF_KEYS])
        if key in self._psf_cache:
            return self._psf_cache[key]

        lbda_nodes = np.linspace(self.lbda[0], self.lbda[-1], nlbda_nodes)
        nodes = np.array([self._moffat_image(fwhm, position, oversampling, psf["moffat_truncation"])
                          for fwhm in self.get_psf_fwhm(lbda_nodes, which=which)])
        idx = np.clip(np.searchsorted(lbda_nodes, self.lbda) - 1, 0, nlbda_nodes - 2)
        weight = ((self.lbda - lbda_nodes[idx]) / (lbda_nodes[idx + 1] - lbda_nodes[idx]))[:, None, None]
        cube = (1 - weight) * nodes[idx] + weight * nodes[idx + 1]

        if len(self._psf_cache) > 8:
            self._psf_cache.clear()
        self._psf_cache[key] = cube
        return cube

    def get_box_aperture(self, position=(0, 0), hsize=2):
        """Square aperture of ``(2*hsize+1)²`` spaxels (pyetc ``square_fixed``).

        Parameters
        ----------
        position : tuple of float, optional
            ``(x, y)`` position, in spaxels; the box is centered on the
            nearest spaxel. Default is (0, 0).
        hsize : int, optional
            Half size of the box, in spaxels. Default is 2.

        Returns
        -------
        numpy.ndarray
            ``(ny, nx)`` boolean mask.
        """
        (x, y), _ = self.get_spaxel_centroids()
        x0, y0 = np.round(position)
        return (np.abs(x - x0) <= hsize + 1e-6) & (np.abs(y - y0) <= hsize + 1e-6)
