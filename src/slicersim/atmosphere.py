"""
Atmosphere module: seeing, transmission and sky emission of a ground-based site.

This module models what the atmosphere adds to a ground-based observation:

- the **seeing-induced** blurring of the spatial PSF (`get_seeing_fwhm`), and
  the seeing-limited PSF profile (`moffat_density`). The optical (telescope +
  instrument) contribution is *not* included here: it belongs to the
  instrument (see `~slicersim.ground_spectrograph.GroundSlicerSpectrograph`),
  which combines both;
- the atmospheric **transmission**, applied to astrophysical sources only;
- the sky **emission**, a spatially flat background.

Transmission and emission are tabulated (e.g. from the ESO SkyCalc model) for a
discrete set of airmasses and sky (moon) conditions, see `Atmosphere`.

.. autosummary::

   get_seeing_fwhm
   moffat_density
   read_sky_table
   Atmosphere
"""

import warnings
from copy import deepcopy

import numpy as np

from . import iotools

__all__ = ["Atmosphere", "get_seeing_fwhm", "moffat_density", "read_sky_table"]

def get_seeing_fwhm(lbda, seeing, airmass=1.0, diameter=8.2,
                    outer_scale=46., r0_ref=0.188, lbda_ref=5000.):
    r"""Seeing-induced FWHM of the PSF (ESO ETC model).

    The atmospheric FWHM follows the Kolmogorov scaling with wavelength and
    airmass, corrected for the finite outer scale of the turbulence
    (Tokovinin 2002, as implemented in the ESO ETC and in pyetc):

    .. math::

       \mathrm{FWHM}_\mathrm{atm} &=
       s \left(\frac{\lambda}{\lambda_\mathrm{ref}}\right)^{-1/5}
       X^{3/5} \sqrt{1 + F_\mathrm{Kolb}\, 2.183\, (r_0/L_0)^{0.356}}, \\
       F_\mathrm{Kolb} &= \frac{1}{1 + 300\, D/L_0} - 1

    It does *not* include the optical (telescope + instrument) image quality.

    Parameters
    ----------
    lbda : float or array_like
        Wavelength(s) in Angstrom.
    seeing : float
        Seeing FWHM in arcsec at ``lbda_ref`` and zenith.
    airmass : float, optional
        Airmass of the observation. Default is 1.
    diameter : float, optional
        Telescope primary mirror diameter in m. Default is 8.2 (VLT).
    outer_scale : float, optional
        Turbulence outer scale :math:`L_0` in m. Default is 46 (ESO ETC).
    r0_ref : float, optional
        Fried parameter :math:`r_0` in m used in the outer-scale correction.
        Default is 0.188 (ESO ETC).
    lbda_ref : float, optional
        Reference wavelength of ``seeing``, in Angstrom. Default is 5000.

    Returns
    -------
    float or numpy.ndarray
        Seeing-induced FWHM in arcsec.

    Examples
    --------
    >>> round(float(get_seeing_fwhm(5500, 0.8, diameter=8.2)), 3)
    0.656
    """
    fkolb = 1 / (1 + 300 * diameter / outer_scale) - 1
    return (seeing * (np.asarray(lbda) / lbda_ref) ** (-0.2) * airmass ** (+0.6)
            * np.sqrt(1 + fkolb * 2.183 * (r0_ref / outer_scale) ** 0.356))


def moffat_density(r2, fwhm, beta):
    r"""Normalized circular Moffat profile.

    .. math::

       I(r) &= \frac{\beta - 1}{\pi \alpha^2}\left(1 + \frac{r^2}{\alpha^2}\right)^{-\beta}, \\
       \alpha &= \frac{\mathrm{FWHM}}{2\sqrt{2^{1/\beta} - 1}}

    which integrates to 1 over the plane.

    Parameters
    ----------
    r2 : array_like
        Squared distance to the center, in the unit of ``fwhm`` squared.
    fwhm : float
        Full width at half maximum.
    beta : float
        Moffat shape parameter (> 1).

    Returns
    -------
    numpy.ndarray
        Surface density, in the inverse unit of ``r2``.
    """
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    return (beta - 1) / (np.pi * alpha ** 2) * (1 + r2 / alpha ** 2) ** (-beta)


def read_sky_table(filename):
    """Read a tabulated sky (transmission and emission) file.

    The ECSV file has a ``wavelength`` column [Angstrom], one
    ``transmission_<airmass>`` column per airmass (e.g. ``transmission_1.2``)
    and one ``emission_<moon>_<airmass>`` column per sky condition and airmass
    (e.g. ``emission_darksky_1.2``) in erg/s/cm^2/A/arcsec^2.

    Parameters
    ----------
    filename : str
        File name, looked for in the configuration directory if it has no
        directory (see `~slicersim.iotools.expand_path`).

    Returns
    -------
    astropy.table.Table
        The sky table.
    """
    from astropy.table import Table
    return Table.read(iotools.expand_path(filename), format="ascii.ecsv")


class Atmosphere:
    """Atmosphere above a ground-based telescope.

    Parameters
    ----------
    sky_table : str or astropy.table.Table
        Tabulated transmission and sky emission, see `read_sky_table`.
    seeing : float, optional
        Seeing FWHM in arcsec at 5000 A and zenith. Default is 0.8.
    airmass : float, optional
        Airmass; must be one of the tabulated airmasses. Default is 1.
    moon : str, optional
        Sky condition (e.g. ``"darksky"``, ``"greysky"``, ``"brightsky"``);
        must be one of the tabulated conditions. Default is ``"darksky"``.
    meta : dict, optional
        Additional metadata (e.g. ``site``, ``outer_scale``, ``r0_ref``,
        ``lbda_ref`` used by `get_seeing_fwhm`). Default is an empty dict.

    Attributes
    ----------
    mutable_parameters : list of str
        Parameters that can be changed after initialization with `update`.
    """
    mutable_parameters = ["seeing", "airmass", "moon"]

    def __init__(self, sky_table, seeing=0.8, airmass=1.0, moon="darksky", meta={}):
        """Initialize the Atmosphere (see the class docstring for parameters)."""
        if isinstance(sky_table, str):
            meta = meta | {"sky_table": sky_table}
            sky_table = read_sky_table(sky_table)

        self._sky_table = sky_table
        self._lbda_table = np.asarray(sky_table["wavelength"], dtype=float)

        self._meta = deepcopy(meta) | {"seeing": float(seeing), "airmass": float(airmass),
                                       "moon": moon}
        self._check_conditions(self._meta)
        self._meta_in = deepcopy(self._meta)

    @classmethod
    def from_config(cls, config):
        """Build the atmosphere from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary (e.g. the ``[atmosphere]`` section of a
            configuration file). It must contain ``"sky_table"`` and may
            contain ``"seeing"``, ``"airmass"``, ``"moon"`` and any other
            metadata (e.g. ``"site"``).

        Returns
        -------
        Atmosphere
            The atmosphere instance.
        """
        config = dict(config)
        sky_table = config.pop("sky_table")
        kwargs = {k: config.pop(k) for k in ["seeing", "airmass", "moon"] if k in config}
        return cls(sky_table, meta=config | {"sky_table": sky_table}, **kwargs)

    def __str__(self):
        """Return a human-readable summary of the observing conditions."""
        return (f"Atmosphere ({self.meta.get('site', 'unknown site')}): seeing {self.seeing}\", "
                f"airmass {self.airmass}, {self.moon}")

    # ============== #
    #   Methods      #
    # ============== #
    def update(self, reset_others=False, **kwargs):
        """Update the observing conditions.

        Parameters
        ----------
        reset_others : bool, optional
            If True, parameters not given in ``kwargs`` are reset to their
            initial values. Default is False.
        **kwargs
            ``seeing``, ``airmass`` and/or ``moon``. None values are ignored.

        Raises
        ------
        ValueError
            If the airmass or the moon condition is not tabulated.

        Warns
        -----
        UserWarning
            If a given parameter is not mutable (it is then ignored).
        """
        updates = {}
        for k, v in kwargs.items():
            if k not in self.mutable_parameters:
                warnings.warn(f"{k} is not an Atmosphere mutable parameter.")
                continue
            if v is None:
                continue
            updates[k] = v if k == "moon" else float(v)

        meta = (self._meta_in if reset_others else self._meta) | updates
        self._check_conditions(meta)
        self._meta = meta

    def get_seeing_fwhm(self, lbda, diameter):
        """Seeing-induced PSF FWHM in arcsec (see `get_seeing_fwhm`).

        Parameters
        ----------
        lbda : float or array_like
            Wavelength(s) in Angstrom.
        diameter : float
            Telescope primary mirror diameter in m.

        Returns
        -------
        float or numpy.ndarray
            Seeing-induced FWHM in arcsec (optical contribution excluded).
        """
        options = {k: self.meta[k] for k in ["outer_scale", "r0_ref", "lbda_ref"] if k in self.meta}
        return get_seeing_fwhm(lbda, self.seeing, airmass=self.airmass,
                               diameter=diameter, **options)

    def get_transmission(self, lbda):
        """Atmospheric transmission at the current airmass.

        Parameters
        ----------
        lbda : array_like
            Wavelength(s) in Angstrom. Outside the tabulated domain, the edge
            values are used.

        Returns
        -------
        numpy.ndarray
            Transmission (between 0 and 1).
        """
        column = self._sky_table[f"transmission_{self.airmass:.1f}"]
        return np.interp(lbda, self._lbda_table, np.asarray(column, dtype=float))

    def get_emission(self, lbda):
        """Sky emission at the current airmass and sky condition.

        Parameters
        ----------
        lbda : array_like
            Wavelength(s) in Angstrom. Outside the tabulated domain, the edge
            values are used.

        Returns
        -------
        numpy.ndarray
            Sky surface brightness in erg/s/cm^2/A/arcsec^2.
        """
        column = self._sky_table[f"emission_{self.moon}_{self.airmass:.1f}"]
        return np.interp(lbda, self._lbda_table, np.asarray(column, dtype=float))

    def _check_conditions(self, meta):
        """Make sure the observing conditions are tabulated."""
        if not np.any(np.isclose(meta["airmass"], self.airmasses)):
            raise ValueError(f"airmass={meta['airmass']} not tabulated: {self.airmasses}")
        meta["airmass"] = float(self.airmasses[np.argmin(np.abs(np.subtract(self.airmasses, meta["airmass"])))])
        if meta["moon"] not in self.moons:
            raise ValueError(f"moon={meta['moon']!r} not tabulated: {self.moons}")

    # ================ #
    #   Properties     #
    # ================ #
    @property
    def meta(self):
        """Atmosphere metadata (observing conditions included)."""
        return self._meta

    @property
    def seeing(self):
        """Seeing FWHM in arcsec at the reference wavelength and zenith."""
        return self._meta["seeing"]

    @property
    def airmass(self):
        """Airmass."""
        return self._meta["airmass"]

    @property
    def moon(self):
        """Sky (moon) condition."""
        return self._meta["moon"]

    @property
    def airmasses(self):
        """Tabulated airmasses."""
        return sorted({float(name.split("_")[-1]) for name in self._sky_table.colnames
                       if name.startswith("transmission_")})

    @property
    def moons(self):
        """Tabulated sky (moon) conditions."""
        return sorted({name.split("_")[1] for name in self._sky_table.colnames
                       if name.startswith("emission_")})
