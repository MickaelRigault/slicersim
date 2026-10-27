"""Spectrograph models: wavelength sampling, spaxels, PSF, LSF and throughput.

This module provides the `Spectrograph` base class and its `SlicerSpectrograph`
and `MLASpectrograph` flavours, which convert input scene spectra (in
erg/s/cm²/Å) into photon flux cubes (in ph/s/spaxel) sampled on the
spectrograph wavelength and spaxel grids. It also provides `OpticsThroughput`,
which handles the chromatic throughput of the optical elements.
"""

import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
import pandas

from . import iotools
from .profiles import build_pixels
from .thermal import ThermalOptics
from .utils import recursive_get  # , bin_array


# ================ #
#                  #
#   Spectrograph   #
#                  #
# ================ #
def lbda_from_resolution_power(spectral_range, res_power, npx_resolution=2):
    r"""Compute a wavelength ramp with a constant resolving power.

    The bins are log-uniformly spaced such that a resolution element
    (:math:`\lambda/\mathcal{R}`) spans ``npx_resolution`` pixels.

    Parameters
    ----------
    spectral_range : tuple of float
        Spectral domain ``(wmin, wmax)``, in Angstrom.
    res_power : float
        Resolving power :math:`\mathcal{R} = \lambda/\Delta\lambda`.
    npx_resolution : float, optional
        Number of pixels per spectral resolution element. Default is 2.

    Returns
    -------
    lbda : numpy.ndarray
        Mid-bin wavelengths (geometric mean of the edges), in Angstrom.
    lbda_edges : numpy.ndarray
        Bin edge wavelengths (``len(lbda) + 1`` entries), in Angstrom.

    Notes
    -----
    .. math::

       \frac{\Delta\lambda}{\lambda} &= \frac{1}{n\mathcal{R}} \\
       &= \ln 10\,\Delta\log\lambda \\
       \log\lambda_i^e &= \log\lambda_0 + i\times \Delta\log\lambda
       &\quad\text{(edges)} \\
       \log\lambda_i &= (\log\lambda_i^e + \log\lambda_{i+1}^e)/2
       &\quad\text{(center)}
    """

    wmin, wmax = spectral_range
    # ln(10)=2.303...
    dlog = 1 / (2.302585092994046 * res_power * npx_resolution)
    npx = round((np.log10(wmax) - np.log10(wmin)) / dlog)
    loglbda_edges = np.linspace(np.log10(wmin), np.log10(wmax), npx + 1)
    loglbda_mid = (loglbda_edges[:-1] + loglbda_edges[1:]) / 2  # mean

    return 10**loglbda_mid, 10**loglbda_edges


def build_lbda(spectral_range, spectral_resolution=None,
                   wsol=None, dsol=None,
                   npx_resolution=2):
    """Build the wavelength coordinates.

    This computes the mid-bin wavelengths (``lbda``) and the bin edge
    wavelengths (``lbda_edges``), either from a constant resolving power
    (if ``wsol`` is None, see `lbda_from_resolution_power`) or from a
    wavelength/dispersion solution.

    Parameters
    ----------
    spectral_range : tuple of float
        Spectral domain ``(wmin, wmax)``, in Angstrom.
    spectral_resolution : float, optional
        Resolving power R. Used (and required) only if ``wsol`` is None.
        Default is None.
    wsol : callable, optional
        Wavelength solution (pixel -> wavelength in Angstrom).
        Default is None.
    dsol : callable, optional
        Dispersion solution (wavelength in Angstrom -> pixel). Required if
        ``wsol`` is given. Default is None.
    npx_resolution : float, optional
        Number of pixels per spectral resolution element. Only used if
        ``wsol`` is None. Default is 2.

    Returns
    -------
    lbda : numpy.ndarray
        Mid-bin wavelengths, in Angstrom.
    lbda_edges : numpy.ndarray
        Bin edge wavelengths, in Angstrom.
    """
    if wsol is None:  # Compute from constant resolving power
        lbda, lbda_edges = lbda_from_resolution_power(np.asarray(spectral_range, dtype="float32"),
                                                          spectral_resolution,
                                                          npx_resolution=npx_resolution)
    else:  # Compute from wavelength solution
        wmin, wmax = spectral_range
        npx = round(dsol(wmax) - dsol(wmin)) + 1  # Total nb of px
        lbda = wsol(np.r_[:npx])  # λ at bin center
        lbda_edges = wsol(np.r_[:npx + 1] - 0.5)  # λ at bin edge

    return lbda, lbda_edges

def build_lbda_from_config(config):
    """Build wavelength coordinates from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Spectrograph configuration dictionary. The following keys are used:

        - ``"spectral_range"``: ``(wmin, wmax)`` in Angstrom.
        - ``"dispersion_law"``: ecsv file with ``wavelength`` and ``offset``
          (in pix) columns. If given, it defines the wavelength solution.
        - ``"dispersion_scale"``: scaling applied to the dispersion law
          (default 1).
        - ``"spectral_resolution"``: resolving power, used if no
          ``"dispersion_law"`` is given.
        - ``"dispersion_resolution"``: number of pixels per resolution
          element (default 2.1).

    Returns
    -------
    lbda : numpy.ndarray
        Mid-bin wavelengths, in Angstrom.
    lbda_edges : numpy.ndarray
        Bin edge wavelengths, in Angstrom.

    Raises
    ------
    ValueError
        If neither ``"spectral_resolution"`` nor ``"dispersion_law"`` is set.
    """
    spectral_range = config.get("spectral_range", None)

    dispersion_law = config.get("dispersion_law", None)
    dispersion_scale = float(config.get("dispersion_scale", 1))

    spectral_resolution = config.get("spectral_resolution", None)
    npx_resolution = config.get("dispersion_resolution", 2.1)  # number of pixel per resolution element.

    # Set by dispersion law ?
    if dispersion_law is not None:
        # => ok, you have a dispersion law
        wname, dname = 'wavelength', 'offset'
        tab = iotools.read_ecsv(dispersion_law, colnames=[wname, dname])
        assert tab[dname].unit == 'pix'

        # dispersion solution (wavelengths in Å, offset in pix)
        dsol = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[dname] * dispersion_scale * npx_resolution / 2, # 2. is the expected nyquist limit
            ext='extrapolate')

        # wavelength solution (wavelengths in Å, offset in pix)
        wsol = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[dname] * dispersion_scale * npx_resolution / 2,# 2. is the expected nyquist limit
            ext='extrapolate', inverse=True)

    elif spectral_resolution is None:
        raise ValueError("'spectral_resolution' OR 'dispersion_law' should be set.")
    else:
        wsol = dsol = None
        dispersion_scale = float(dispersion_scale)

    lbda, lbda_edges = build_lbda(spectral_range, wsol=wsol, dsol=dsol,
                                  spectral_resolution=spectral_resolution,
                                  npx_resolution=npx_resolution)

    return lbda, lbda_edges

def build_spaxels_from_config(config):
    """Build spaxel coordinates from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary, containing (possibly nested) the
        ``"spatial_shape"`` and ``"spatial_scale"`` (in arcsec) entries.

    Returns
    -------
    dict
        Spaxel information, with keys ``"shape"`` (spaxel grid shape) and
        ``"spx_scale"`` (spaxel size in arcsec).
    """
    # Spaxel spatial scale (size of a single spaxel in arcsec)
    spatial_shape = recursive_get(config, "spatial_shape")
    spx_spatial_scale = recursive_get(config, "spatial_scale")

    return {"shape": spatial_shape,
            "spx_scale": spx_spatial_scale}

def build_throughput_from_config(config):
    """Build throughput from a configuration dictionary.

    Parameters
    ----------
    config : dict or float or str
        Throughput configuration. If a dict with a ``"throughput"`` key, that
        entry is used, otherwise the input itself is. The throughput can then
        be a number (constant throughput), a dict (see
        `OpticsThroughput.from_config`) or a file name (see
        `OpticsThroughput.from_filename`).

    Returns
    -------
    float or OpticsThroughput
        Constant throughput, or an `OpticsThroughput` instance (callable,
        function of wavelength in Angstrom).

    Raises
    ------
    ValueError
        If the throughput is None.
    """
    throughput =  config.get("throughput", config)

    if throughput is None:
        raise ValueError("No given 'throughput' in the input config.")

    # is that simply a float or something ?
    try:
        throughput = float(throughput)

    except (ValueError, TypeError):
        pass

    else: # float worked, so let's return it.
        return throughput

    # if we are here, throughput is not simply a float.
    if isinstance(throughput, dict):
        # it is itself a config. Let's use the dedicated object
        return OpticsThroughput.from_config(throughput)

    return OpticsThroughput.from_filename(throughput)


# ============== #
#  Spectrograph  #
# ============== #
class Spectrograph:
    """Generic integral field spectrograph.

    It holds the wavelength sampling, the spaxel grid, the spatial PSF, the
    throughput and the internal (thermal) optics of the spectrograph, and
    converts scene spectra (in erg/s/cm²/Å) into photon flux cubes
    (in ph/s/spaxel), see e.g. `generate_pointsource`.

    You likely want to create it using the `from_config` constructor.

    Parameters
    ----------
    lbda : array_like
        Mid-bin wavelengths, in Angstrom.
    telescope : ~slicersim.telescope.Telescope, optional
        Telescope feeding the spectrograph. Default is None.
    spaxels : dict, optional
        Spaxel information, passed to `set_spaxels`:
        ``{"shape": (ny, nx), "spx_scale": float}`` with ``spx_scale`` in
        arcsec. Default is ``{}``.
    throughput : float, callable or OpticsThroughput, optional
        Throughput of the spectrograph, see `set_throughput`.
        Default is None.
    spatial_psf : dict, optional
        Spatial PSF information, with keys ``"sigma_spectral"`` (chromatic
        sigma in arcsec at `lbda_ref`), ``"guiding_sigma"`` (in arcsec) and
        ``"profile"``. Default is ``{}``.
    lbda_edges : array_like, optional
        Wavelength bin edges, in Angstrom (``len(lbda) + 1`` entries).
        Default is None.
    optics : ~slicersim.thermal.ThermalOptics, optional
        Internal optics of the spectrograph (thermal emission).
        Default is ``{}``.
    dispersion_resolution : float, optional
        Number of pixels per spectral resolution element (spot size), see
        `dispersion_resolution`. Default is 2.
    meta : dict, optional
        Metadata (typically the input configuration). Default is ``{}``.
    """
    _SPECTROGRAPH_TYPE = "Unknown"

    #: Mutable parameters (list)
    mutable_parameters = [  # 'spectral_range', 'spectral_resolution', # lbda
        "dispersion_resolution", "spotsize", "dispersion_scale",
        "which_throughput",
        'xdisp_sigma_spectral', 'xdisp_sigma',  # xdisp_profile,
        'psf_sigma_spectral', 'guiding_sigma',  # psf_profile
        'spatial_scale', 'spatial_shape',  # spaxels
        'spx_scale', 'spx_shape',  # spaxels
        ]

    def __init__(self, lbda, telescope=None,
                 spaxels={}, throughput=None,
                 spatial_psf={},
                 lbda_edges=None,
                 optics={},
                 dispersion_resolution=2,
                 meta={}):
        """Initialize the spectrograph.

        See the class docstring for the parameters. You likely want to
        create it using the `from_config` constructor.
        """
        # wavelength
        self.lbda = lbda
        self.lbda_edges = lbda_edges

        # mirror
        self.telescope = telescope

        # optics
        self.optics = optics

        # Spaxels
        self.set_spaxels(**spaxels)

        # Throughput: flaot or func, see get_throughput()
        self.set_throughput(throughput)

        self.spatial_psf = spatial_psf

        # affect the instance
        telescope_mutable = []
        self.mutable_parameters = self.mutable_parameters + \
                                  ([f"telescope.{k}" for k in self.telescope.mutable_parameters] if self.telescope is not None else []) +\
                                  ([f"optics.{k}" for k in self.optics.mutable_parameters] if self.optics is not None else []) +\
                                  ([f"throughput.{k}" for k in self.throughput.mutable_parameters] if hasattr(self.throughput, "mutable_parameters") else [])

        meta["dispersion_resolution"] = dispersion_resolution
        self._meta_in = meta.copy()  #: Meta-parameters as input
        self._meta = meta.copy()  #: Meta-parameters as used

    @classmethod
    def from_config(cls, config, telescope=None):
        """Create a spectrograph from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Spectrograph configuration dictionary (e.g. the ``"spectrograph"``
            entry of `~slicersim.iotools.get_config`). It must contain the
            ``"throughput"``, ``"optics"`` and ``"psf"`` entries, and the
            wavelength and spaxel information (see `build_lbda_from_config`
            and `build_spaxels_from_config`).
        telescope : ~slicersim.telescope.Telescope, optional
            Telescope feeding the spectrograph. Default is None.

        Returns
        -------
        Spectrograph
            An instance of the calling class.
        """
        # spectrograph that inherit Spectrograph simply need to update _parse_config()
        init_prop, _ = cls._parse_config(config)

        return cls(telescope=telescope, **init_prop)

    @staticmethod
    def _parse_config(config):
        """Parse the configuration dictionary.

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
        # make sure the input config is not changed
        input_config = deepcopy(config)

        # wavelengths
        lbda, lbda_edges = build_lbda_from_config(input_config)

        # Spaxels
        spaxels = build_spaxels_from_config(input_config)

        # throughput
        throughput = build_throughput_from_config(input_config["throughput"])

        # thermal optics
        optics = ThermalOptics.from_config(input_config["optics"])

        # in coming PSF
        psf = {"sigma_spectral": input_config["psf"]["spatial"]["psf_sigma_spectral"],
               "guiding_sigma": float(input_config["psf"]["spatial"]["guiding_sigma"]),
               "profile": input_config["psf"]["spatial"]["psf_profile"]}

        init_prop = {"lbda": lbda,
                     "lbda_edges": lbda_edges,
                     "spaxels": spaxels,
                     "throughput": throughput,
                     "spatial_psf": psf,
                     "optics": optics,
                     "dispersion_resolution": input_config.get("dispersion_resolution", 2),
                     "meta": input_config}

        return init_prop, input_config

    @staticmethod
    def build_lbda_from_config(config):
        """Build wavelength coordinates from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        lbda : numpy.ndarray
            Mid-bin wavelengths, in Angstrom.
        lbda_edges : numpy.ndarray
            Bin edge wavelengths, in Angstrom.

        See Also
        --------
        slicersim.spectrograph.build_lbda_from_config : The function called.
        """
        return build_lbda_from_config(config)

    def update(self, reset_others=False, **kwargs):
        """Update any mutable attribute of the spectrograph.

        The method accepts the django-like format, such that ``a__b`` is
        understood as ``a.b``. For instance, ``update(optics__temperature=220)``
        is equivalent to ``update(**{'optics.temperature': 220})``.

        Notes on the wavelength sampling (``lbda``):

        - changing ``dispersion_resolution`` redefines ``self.lbda`` such that
          the resolving power is unchanged;
        - changing ``spotsize`` updates ``self.dispersion_resolution``
          *without* updating ``lbda``, effectively changing the resolving power.

        Parameters
        ----------
        reset_others : bool, optional
            If True, the parameters not given are reset to their initial
            values (meta and telescope). Default is False.
        **kwargs
            Parameters to update, e.g. any of `mutable_parameters`, or
            ``telescope.<param>``, ``optics.<param>``, ``throughput.<param>``.
            ``throughput`` itself replaces the throughput (see
            `OpticsThroughput.from_config`). None values are ignored.

        Returns
        -------
        list of str
            Names of the components that changed, among ``"telescope"``,
            ``"optics"``, ``"throughput"``, ``"psf"``, ``"spaxels"``,
            ``"lbda"`` and ``"meta"``.

        Warns
        -----
        UserWarning
            If a parameter is not mutable or cannot be parsed. It is then
            ignored.
        """
        NAME_ALT = {"spx_shape": "spatial_shape",
                    "spx_scale": "spatial_scale"}

        updates = {}
        telescope_updates = {}
        optics_updates = {}
        lbda_updates = {}
        psf_updates = {}
        throughput_updates = {}
        spaxel_updates = {}

        # == Filling the update == #
        for k, v in kwargs.items():
            k = NAME_ALT.get(k, k).replace("__", ".") # accept django like format

            # Nothing to do, skip
            if v is None:
                continue

            # special case: changing the throughput structure
            if k == "throughput": # no . at the end.
                new_throughput = OpticsThroughput.from_config(v)
                self.set_throughput(new_throughput)
                continue

            # telescope
            if k.startswith("telescope."):
                telescope_updates[k.replace("telescope.", "")] = v
                continue

            # optics
            if k.startswith("optics."):
                optics_updates[k.replace("optics.", "")] = v
                continue

            # throughput
            if k.startswith("throughput."):
                throughput_updates[k.replace("throughput.", "")] = v
                continue

            # This test comes after the submodule to allow some flexibilities
            if k not in self.mutable_parameters:
                warnings.warn(f"Parameter {k!r} is not mutable.")
                continue

            # lbda
            if k in ["dispersion_resolution", "spotsize"]:
                # update the meta (for lsf and instrumental psf)
                updates["dispersion_resolution"] = v
                # and update the wavelengths
                if k == "dispersion_resolution":
                    # you changed the actual dispersion resolution, so let's change lbda
                    lbda_updates["dispersion_resolution"] = v

            elif k == "dispersion_scale":
                lbda_updates["dispersion_scale"] = v

            # not allowed to change.

            # change PSF
            elif k in self.meta["psf"]["spatial"]:
                psf_updates[k.replace("psf_", "")] = v

            # spaxels
            elif k in ('spatial_scale', 'spatial_scale_insigma',
                       'spatial_shape', 'spatial_shape_insigma'):
                spaxel_updates[k] = v

            # generic change
            elif k in ["which_throughput"]:
                updates[k] = v
            else:
                warnings.warn(f"{k=} is unparsed")

        what_changed = []
        # update telescope
        if telescope_updates:
            what_changed.append("telescope")
            self.telescope.update(**telescope_updates, reset_others=reset_others)

        # update internal optics
        if optics_updates:
            what_changed.append("optics")
            self.optics.update(**optics_updates)

        # update internal optics
        if throughput_updates:
            what_changed.append("throughput")
            self.throughput.update(**throughput_updates)

        # psf
        if psf_updates:
            what_changed.append("psf")
            self.spatial_psf = self.spatial_psf | psf_updates

        # lbda
        if spaxel_updates:
            what_changed.append("spaxels")
            spaxels = build_spaxels_from_config(self._meta | spaxel_updates)
            updates |= {NAME_ALT.get(k, k):v for k,v in spaxels.items()}
            self.set_spaxels(**spaxels)

        if lbda_updates:
            what_changed.append("lbda")
            self.update_lbda(*build_lbda_from_config(self._meta | lbda_updates))

        if updates:
            what_changed.append("meta")

        # the rest if any
        if reset_others:
            self._meta = self._meta_in | updates
        else:
            self._meta = self._meta | updates

        return what_changed

    def update_lbda(self, lbda, lbda_edges):
        """Update the wavelength coordinates.

        Parameters
        ----------
        lbda : array_like
            Mid-bin wavelengths, in Angstrom.
        lbda_edges : array_like
            Bin edge wavelengths, in Angstrom.
        """
        self.lbda = lbda
        self.lbda_edges = lbda_edges

    def set_spaxels(self, shape, spx_scale):
        """Set the spaxel information.

        Parameters
        ----------
        shape : tuple of int
            Spaxel grid shape ``(ny, nx)``.
        spx_scale : float or tuple of float
            Spaxel size in arcsec, either a float or ``(scale_y, scale_x)``.
        """
        self._spaxels = {"shape": shape, "spx_scale": spx_scale}
        self._spaxel_coords = {}

    def set_throughput(self, throughput):
        """Set the throughput of the spectrograph.

        Parameters
        ----------
        throughput : float, array_like, callable, pandas.Series, pandas.DataFrame or OpticsThroughput
            Throughput of the system (between 0 and 1):

            - float: constant throughput.
            - array_like: must broadcast with ``self.lbda``.
            - callable: such that ``throughput(lbda)`` returns the throughput
              at ``lbda`` (in Angstrom).
            - pandas.Series: throughput with wavelength (in Angstrom) as
              index. It is converted into a cubic interpolator (zero outside
              the given range).
            - pandas.DataFrame: its first column is used as a
              pandas.Series.
            - `OpticsThroughput`: detailed per-element throughput.
        """
        if isinstance(throughput, pandas.DataFrame):
            throughput = throughput.iloc[:, 0] # convert as serie

        # Serie => func
        if isinstance(throughput, pandas.Series):
            throughput = iotools.chromatic_interpolator(
                throughput.index, throughput.values, ext='zeros')

        self.throughput = throughput

    # --------- #
    #  GETTER   #
    # --------- #
    def get_lbda(self, units=None, oversample=None):
        """Get the wavelength array.

        Parameters
        ----------
        units : str or astropy.units.Unit, optional
            Unit of the returned array. None (default) means Angstrom.
        oversample : int, optional
            Oversampling factor of the wavelength binning (linear
            interpolation between the native wavelengths). Default is None.

        Returns
        -------
        numpy.ndarray
            Wavelength array, in Angstrom unless ``units`` is given.
        """
        lbda = self.lbda.copy() # in AA
        if units is not None:
            lbda *= units.AA.to(units)

        if oversample is not None:
            nlbda = len(lbda)
            lbda = np.interp( np.arange(nlbda, step=1/oversample), np.arange(nlbda), lbda)

        return lbda

    def get_fieldname(self):
        """Get the name of the field matching the spaxel configuration.

        The field is inferred from the current ``spatial_shape`` and
        ``spatial_scale`` of the spectrograph (see `meta`).

        Returns
        -------
        str
            Field name (e.g. ``"narrow"`` or ``"wide"`` for Lazuli), or
            ``"unknown"`` if the configuration does not match any known field.

        Raises
        ------
        NotImplementedError
            If the spectrograph is not Lazuli (see `name`).
        """
        if self.name.lower() == "lazuli":
            from .lazuli import spectroconfig_to_fieldname
            field_name = spectroconfig_to_fieldname(spatial_shape = self.meta["spatial_shape"],
                                              spatial_scale = self.meta["spatial_scale"])

        else:
            raise NotImplementedError(f"get_fieldname() is not implemented for {self.name} spectrograph.")

        return field_name

    def get_throughput(self, lbda=None, **kwargs):
        """Get the spectrograph throughput.

        When the throughput is callable (e.g. `OpticsThroughput`), it is
        called with a ``which`` argument set to:

        - ``meta["which_throughput"]`` if given (see `update`);
        - otherwise, for Lazuli, the current field name (see `get_fieldname`);
        - otherwise None, i.e. the product of all throughput curves.

        Parameters
        ----------
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, ``self.lbda`` is used.
            Ignored if the throughput is not callable. Default is None.
        **kwargs
            Goes to the throughput call (e.g.
            `OpticsThroughput.get_throughput`). A given ``which`` overrides
            the one described above.

        Returns
        -------
        numpy.ndarray or float
            Throughput as a function of wavelength, or the constant
            throughput.

        Warns
        -----
        UserWarning
            If ``lbda`` is given while the throughput is not callable.
        """
        # internal tool
        which_throughput = {"which": self.meta.get("which_throughput", None)}
        if which_throughput["which"] is None and (self.name.lower() == "lazuli"):
            # throughput requires the fieldname.
            which_throughput = {"which" : self.get_fieldname()}

        if callable(self.throughput):
            if lbda is None:
                lbda = self.lbda

            return self.throughput(lbda, **(which_throughput | kwargs) )

        # float
        if lbda is not None:
            warnings.warn("self.throughput is not callable, input lbda is ignored.")

        return self.throughput

    def get_spectrograph_shape(self, oversampling=None):
        """Get the spatial shape of the spaxel grid.

        Parameters
        ----------
        oversampling : int, optional
            Spatial oversampling factor. Default is None (no oversampling).

        Returns
        -------
        tuple of int or numpy.ndarray
            Spaxel grid shape ``(ny, nx)``, multiplied by ``oversampling``
            if given.
        """
        if oversampling is None or oversampling == 1:
            return self.spx_shape

        cube_shape = (np.asarray(self.spx_shape) * oversampling).astype(int)
        return cube_shape

    def get_resolving_power(self):
        r"""Get the resolving power :math:`\mathcal{R} = \lambda/\Delta\lambda`.

        It is computed as ``lbda / np.diff(lbda_edges) / dispersion_resolution``,
        i.e. the resolution element is ``dispersion_resolution`` pixels wide.

        Returns
        -------
        numpy.ndarray
            Resolving power, of shape ``(nlbda,)``.
        """
        return self.lbda / np.diff(self.lbda_edges) / self.get_dispersion_resolution()

    def get_dispersion_resolution(self, which=None):
        """Get the effective dispersion resolution, in pixels.

        Parameters
        ----------
        which : str, optional
            Which axis to return:

            - None: the resolution as stored.
            - ``"wavelength"``, ``"lsf"``, ``"wave"``, ``"lbda"`` or ``"y"``:
              the spectral axis.
            - ``"spatial"``, ``"psf"`` or ``"x"``: the spatial axis.

            Default is None.

        Returns
        -------
        float or dict
            The dispersion resolution in pixels. If ``which`` is None, the
            value as stored, which may be a float, a 2-element array or a dict
            with the ``"wavelength"`` and ``"spatial"`` entries. Otherwise,
            the resolution of the requested axis.

        Raises
        ------
        ValueError
            If the stored dispersion resolution can be parsed neither as a
            scalar nor as a 2-element array, or if ``which`` is not one of the
            accepted names.

        Notes
        -----
        2 pixels is optimal: below that the PSF is unresolved, above it the
        PSF is oversampled and adds unnecessary pixel noise.

        When the stored resolution is a 2-element array and ``which`` is given,
        the full ``{"wavelength", "spatial"}`` dict is returned instead of the
        requested axis alone.
        """
        dispersion_resolution = self.dispersion_resolution
        if which is None:
            return dispersion_resolution

        if type(dispersion_resolution) is not dict:
            if np.ndim(dispersion_resolution) == 0: # float-like
                dispersion_resolution = {"wavelength": dispersion_resolution,
                                            "spatial": dispersion_resolution}
            elif np.ndim(dispersion_resolution) == 1 and len(dispersion_resolution) == 2: # array like
                disp_wave, disp_spatial = dispersion_resolution
                return {"wavelength": disp_wave,
                        "spatial": disp_spatial}
            else:
                raise ValueError(f"I cannot parse input dispersion resolution {dispersion_resolution}")


        if which in ["wavelength", "lsf", "wave", "lbda", "y"]:
            return dispersion_resolution["wavelength"]
        elif which in ["spatial", "psf", "x"]:
            return dispersion_resolution["spatial"]
        else:
            raise ValueError(f"I cannot parse {which=}")

    def get_spaxel_centroids(self, in_arcsec=False, squeeze=False, oversampling=None):
        """Get the spaxel centroid coordinates.

        Parameters
        ----------
        in_arcsec : bool, optional
            If True, return centroids in arcsec, otherwise in spaxels.
            Default is False.
        squeeze : bool, optional
            If True, squeeze the output arrays. Default is False.
        oversampling : int or list of int, optional
            Spatial oversampling factor. Default is None (native grid).

        Returns
        -------
        centroids : tuple of numpy.ndarray
            ``(x, y)`` centroid coordinates (see
            `~slicersim.profiles.build_pixels`).
        oversampling : int or list of int
            Oversampling factor actually used.
        """
        # numpy and all() test enables to have list/array oversampling (or float/int)
        if oversampling is None or (np.asarray(oversampling) == self.spaxels["oversampling"]).all():
            spaxels_coords = self.spaxels
        else:
            spaxels_coords = build_pixels(self.spx_shape, oversampling=oversampling)

        x, y = spaxels_coords["centroids"] # dense if oversampling
        if in_arcsec:
            if np.ndim(self.spx_spatial_scale) == 0:
                spx_y = spx_x = self.spx_spatial_scale
            else:
                spx_y, spx_x = self.spx_spatial_scale
            # not /= not to change
            x = x * spx_x # / spaxels_coords["oversampling"]  # in arcsec
            y = y * spx_y # / spaxels_coords["oversampling"]  # in arcsec

        if squeeze:
            x, y = np.squeeze(x), np.squeeze(y)

        #if corr_anamorphose:
        #    y = bin_array(y, self._ANAMORPHOSE, np.mean)

        return (x, y), spaxels_coords["oversampling"]

    def get_psf_sigma_spectral(self, guiding_sigma=None,
                                   incl_telescope=True,
                                   in_spaxels=False, lbda=None):
        """Get the spatial PSF standard deviation (in arcsec or in spaxels).

        The total (Gaussian) spatial PSF is the quadratic sum of two
        components:

        - the optical (chromatic) component, with stddev proportional
          to wavelength, normalized at `lbda_ref` (see `psf_sigma_spectral`),
        - the guiding (achromatic) component, with constant stddev.

        Parameters
        ----------
        guiding_sigma : float, optional
            Guiding sigma override in arcsec. If None,
            ``self.spatial_psf["guiding_sigma"]`` is used. Set
            ``guiding_sigma=0`` for no guiding contribution. Default is None.
        incl_telescope : bool, optional
            If True, include the optical (chromatic) component. If False, only
            the guiding component remains. Default is True.
        in_spaxels : bool, optional
            If True, return sigma in spaxels, otherwise in arcsec.
            Default is False.
        lbda : array_like, optional
            Wavelength array in Angstrom. If None, ``self.lbda`` is used.
            Default is None.

        Returns
        -------
        numpy.ndarray
            Total sigma in arcsec or spaxels, of shape ``(nlbda, 1, 1)``.
        """
        if lbda is None:
            lbda = self.lbda

        if guiding_sigma is None:
            # no default in .get() for it to break if not specified
            guiding_sigma = self.spatial_psf.get("guiding_sigma")

        if incl_telescope:
            chromatic_sigma = self.psf_sigma_spectral
        else:
            chromatic_sigma = 0

        sigma = self._get_chromatic_sigma(lbda,
                                          chromatic_sigma=chromatic_sigma,
                                          constant_sigma=guiding_sigma,
                                          wref=self.lbda_ref)

        # broadcast as (nlbda, 1, 1)
        sigma = sigma[..., None, None]

        if in_spaxels:
            # for broadcasting reasons /= does not work.
            sigma = sigma/ self.spx_spatial_scale

        return sigma

    def get_additonal_spatial_scatter(self, guiding_sigma=None,
                                          incl_instrument=True,
                                          in_spaxels=True):
        """Get the scatter induced by the telescope jitter and instrumental PSF.

        The current implementation assumes Gaussian scatter.

        Parameters
        ----------
        guiding_sigma : float, optional
            Gaussian sigma (in arcsec) of the jitter. If None, the internally
            stored ``spatial_psf["guiding_sigma"]`` is used. Set
            ``guiding_sigma=0`` for no guiding scatter. Default is None.
        incl_instrument : bool, optional
            If True, also include the instrumental PSF (see
            `get_instrumental_psf`) along the ``y`` (slice) direction. This
            only applies to slicer spectrographs. Default is True.
        in_spaxels : bool, optional
            If True, return the scatter in spaxels, otherwise in arcsec.
            Default is True.

        Returns
        -------
        numpy.ndarray or None
            ``(sigma_x, sigma_y)`` Gaussian sigma, in spaxels or in arcsec
            (see ``in_spaxels``). None if there is no scatter.
        """
        if guiding_sigma is None:
            guiding_sigma = self.spatial_psf["guiding_sigma"]

        if guiding_sigma is not None:
            # make sure it is 2d (RA, Dec)
            guiding_sigma = np.full((2,), guiding_sigma) / self.spx_spatial_scale # in spaxels


        # Note:
        # Slicer: instrumental PSF affects the LSF along the "x" direction
        #     and the PSF along the "y" direction.
        #     => The effective "sigma" this thus guiding_sigma & (0, inst_psf[1])
        # MLA: it affects the LSF and the x-dispersion profile.
        #     Not the spatial PSF
        #     => the effective "sigma" thus is just guiding_sigma
        if self._SPECTROGRAPH_TYPE == "slicer" and incl_instrument:
            inst_psf = self.get_instrumental_psf(in_spaxels=True) # in spaxels
            if inst_psf is not None: # if None, nothing to do.
                # make sure the inst_psf is 2d (x and y)
                inst_psf = np.full((2,), inst_psf)
                # and remove the "axis=0" contribution (see LSF)
                inst_psf[0] = 0

            if guiding_sigma is None and inst_psf is None:
                effective_sigma = None
            elif guiding_sigma is None:
                effective_sigma = inst_psf
            elif inst_psf is None:
                effective_sigma = guiding_sigma
            else:
                effective_sigma = np.hypot(guiding_sigma, inst_psf)
        else:
            effective_sigma = guiding_sigma

        # spaxels
        if in_spaxels:
            return effective_sigma

        # arcsec
        return effective_sigma * self.spx_spatial_scale

    def get_spatial_psf(self, profile="default", position=(0, 0),
                        guiding_sigma=None, oversampling=5,
                        as_oversampled=False,
                        incl_instrument=True,
                        incl_telescope=True,
                        **kwargs):
        """Get the normalized 2D spatial PSF on the spaxel grid.

        This takes the PSF generated by the telescope, includes the jitter
        and adds the contribution of the instrument internal optics.

        Parameters
        ----------
        profile : str, optional
            Profile to be used:

            - ``"default"``, ``"gaussian"``, ``"normal"`` or ``"norm"``:
              Gaussian profile using exact erf integration over the spaxels.
            - ``"airy"``, ``"mirror"``, ``"telescope"`` or ``"airydisk"``:
              Airy disk generated by the telescope mirror.
            - ``"Gaussian2D"`` or ``"gaussian2d"``: Gaussian profile evaluated
              on the oversampled grid.

            Default is ``"default"``.
        position : tuple of float, optional
            ``(x, y)`` location of the point source with respect to the
            grid center, in spaxels. Default is ``(0, 0)``.
        guiding_sigma : float, optional
            Gaussian sigma (in arcsec) of the jitter. If None, the stored
            ``spatial_psf["guiding_sigma"]`` is used. Default is None.
        oversampling : int, optional
            Oversampling factor used to evaluate the PSF (not used by the
            erf-integrated Gaussian profiles unless ``as_oversampled`` is
            True). If None, 3 is used. Set ``oversampling=1`` for no
            oversampling. Default is 5.
        as_oversampled : bool, optional
            If True, return the PSF on the oversampled grid. Default is False.
        incl_instrument : bool, optional
            If True, include the instrumental PSF contribution (see
            `get_additonal_spatial_scatter`). Default is True.
        incl_telescope : bool, optional
            If True, include the telescope PSF and jitter. If False, the
            guiding sigma is forced to 0. Default is True.
        **kwargs
            For the Gaussian (erf) profile, additional arguments passed to
            `~slicersim.profiles.get_gaussian2d`. Ignored otherwise.

        Returns
        -------
        numpy.ndarray
            Normalized PSF, of shape ``(nlbda, ny, nx)``.

        Raises
        ------
        ValueError
            If ``profile`` is not implemented.

        Warns
        -----
        UserWarning
            If ``incl_telescope`` is False while ``guiding_sigma`` is
            positive.
        """
        from . import profiles
        if oversampling is None:
            oversampling = 3 # like ~10 (9) subpixels

        if not incl_telescope: # no telescope, no jitter
            if guiding_sigma>0:
                warnings.warn("incl_telescope is False but guiding_sigma is none 0. We are forcing guiding_sigma=0 ; no telescope, no jitter.")

            guiding_sigma = 0

        # to solve the numpy vs. plot confusion
        position_xy = position

        # effective_sigma is 2d (x-scatter, y-scatter) | x == dispersion y=slice
        effective_sigma = self.get_additonal_spatial_scatter(guiding_sigma=guiding_sigma,
                                                            incl_instrument=incl_instrument)
        #
        # Gaussian, special as convolution with additional gaussian scatter is analytic
        #
        if profile in ["default", "gaussian", "normal", "norm"]:
            # adding gaussian_sigma in arcsec (in_spaxels=False)| e.g. jitter
            sigmas = self.get_psf_sigma_spectral(in_spaxels=True,
                                                 guiding_sigma=0, # guiding is included after
                                                 incl_telescope=incl_telescope
                                                ) # in spaxels

            # adding gaussian scatter (e.g. jitter)
            if effective_sigma is not None:
                # effective_sigma is in spaxel
                sigmas = np.hypot(sigmas, effective_sigma) # in spaxels

            prop = dict(in_arcsec=False, squeeze=False, oversampling=1)
            if as_oversampled:  # this is for test as one doesn't need to oversample since it uses erf functions.
                # warnings.warn(f"changin oversampling to {oversampling=}")
                prop["oversampling"] = oversampling

            (xx, yy), oversampling = self.get_spaxel_centroids(**prop)

            psf = profiles.get_gaussian2d(xx, yy, sigma=sigmas, mean=position_xy, **kwargs)
            # remark: if oversampled, the flux need to be *summed*, not *averaged*
            #         as the energy is conserved in the current structure.
            #         => smaller pixel (oversampled) -> less flux.
            #
            return psf

        # ----------------------------------------------------- #
        #
        # Generic PSF,
        #    any profile convolved with additional
        #    gaussian scatter (like jitter)
        #
        # ----------------------------------------------------- #

        # This part of the code works in arcsec.
        position_xy = np.asarray(position_xy) * self.spx_spatial_scale # in spaxels => arcsec

        # to accomodate with non-square spaxels (like slicer)
        # we work in arcsec, not in spaxels.
        if profile in ["airy", "mirror", "telescope", "airydisk"]:
            if incl_telescope:
                radius = self.telescope.get_airy_radius(self.lbda)  # in arcsec
            else:
                radius = np.ones(self.lbda.shape) * 1e-4 # very close to zeros but non zero to avoid issues.

            # assumed symetric on x and y
            psf_func = profiles.get_profilemodel("airy", position=position_xy,
                                                  radius=radius[:, None, None],
                                                  normalized=True)


        elif profile in ["Gaussian2D", "gaussian2d", "gaussian"]:
            sigmas = self.get_psf_sigma_spectral(in_spaxels=False,
                                                 guiding_sigma=0, # explicitely null, see after.
                                                 incl_telescope=incl_telescope
                                                ) # in arcsec
            psf_func = profiles.get_profilemodel("Gaussian2D", position=position_xy,
                                                  sigma=sigmas,
                                                  normalized=True)

        else:
            raise ValueError(f"psf profile {profile=} not implemented")

        # coordinates including oversampling for exact PSF profile and guiding convolution
        # x and y are in arcsec as radius is given in arcsec.
        (xx, yy), oversampling = self.get_spaxel_centroids(in_arcsec=True, squeeze=False,
                                                           oversampling=oversampling)
        psf = psf_func(xx, yy)
        # Since we are working in arcsec, let's make sure this goes away.
        psf *= self.spx_area

        # Note: Normalisation have been tested. See:
        # - psf_scipy = self.get_spatial_psf("normal") # works in spaxels
        # - psf_profile = self.get_spatial_psf("Gaussian2D") # works in arcsec
        #   they must both agree within numerical errors.


        # oversampling: The PSF intensity need to be devided by oversampling area
        #               to conserve energy. Hence the rebinning needs to sum (not average)
        if oversampling is not None:
            if np.ndim(oversampling) == 1:
                oversampling_y, oversampling_x = oversampling
            else:
                oversampling_y = oversampling_x = oversampling

            psf /= oversampling_y*oversampling_x

        # effective_sigma is either None or [sigma_x, sigma_y]
        if effective_sigma is not None and np.any(effective_sigma > 0):
            # effective_sigma is already in spaxels.
            psf = self._apply_additional_scatter(psf, guiding=effective_sigma, oversampling=oversampling,
                                          in_units="spaxels")

        if oversampling != 1 and not as_oversampled:
            # in the current structure, you need to sum oversampled pixels as energy is conserved.
            psf = self._remove_oversampling(psf, oversampling=oversampling, func=np.nansum)

        #if self.type == "slicer":
        #    psf = self._correct_anamorphose_(psf)

        return psf  # (nlbda, ny, nx)

    def get_instrumental_psf(self, spotsize=None, expected_size=2,
                                 in_spaxels=True):
        """Get the contribution of the spectrograph to the total spot size.

        This contribution is null if ``spotsize <= expected_size``. Otherwise,
        it is the Gaussian sigma
        ``extra_pixels = np.sqrt(spotsize**2 - expected_size**2) / 2``,
        further divided by the anamorphosis factor for slicer spectrographs.

        Parameters
        ----------
        spotsize : float, optional
            Total size (in pixels) of a point source on the detector, i.e.
            the total contribution of telescope and spectrograph. If None,
            `get_dispersion_resolution` is used. Default is None.
        expected_size : float, optional
            Expected geometric footprint (in pixels) of the PSF on the
            detector. Default is 2.
        in_spaxels : bool, optional
            If True, return the extra scattering in spaxels (spaxel = pixel
            for a slicer along the slice), otherwise in arcsec.
            Default is True.

        Returns
        -------
        float or numpy.ndarray or None
            Extra scattering (Gaussian sigma), in spaxels or arcsec.
            None if ``spotsize <= expected_size``.
        """
        if spotsize is None:
            spotsize = self.get_dispersion_resolution()

        if spotsize <= expected_size:
            return None # nothing to do | should a warning be added ? This should not happen

        # additional contribution (assumed gaussian) for the effective spotsize
        # The divided by 2 is because extra_pixel is consistent with a gaussian sigma.
        extra_pixels = np.sqrt(spotsize**2 - expected_size**2) / 2.

        # now including anamorphose:
        if self.type == "slicer":
            extra_pixels /= self._ANAMORPHOSE # anamorphose is > 1, so this reduces the extra_pixels

        # get the scatter scale in pixels/spaxels
        # pixels/spaxels: same for a slicer in that direction
        if in_spaxels:
            return extra_pixels

        # or get this scale in arcsec
        return extra_pixels * self.spx_spatial_scale # in arsec

    # - internal tricks
    def _apply_additional_scatter(self, image, guiding, oversampling=1, in_units="arcsec"):
        """Apply guiding error convolution.

        Parameters
        ----------
        image : array_like
            Image to convolve.
        guiding : float
            Guiding error [arcsec or spaxel, see in_units].
        oversampling : int, optional
            Oversampling factor. Default is 1.
        in_units : str, optional
            Units of ``guiding``: ``"arcsec"`` or ``"spaxels"``.
            Default is ``"arcsec"``.

        Returns
        -------
        array_like
            Convolved image.
        """
        from scipy.ndimage import gaussian_filter
        # gaussian convolution | guiding_sigma is given in arcsec
        # works by itself if spx_spatial_scale is a list or a float.
        if in_units == "arcsec":
            guiding = guiding/self.spx_spatial_scale

        elif in_units not in ["spaxels", "spx", "pixels"]:
            raise ValueError(f"in_units should be arcsec or spaxels, {in_units=} given.")

        # guiding is in spaxels/pixels at this stage.
        sigma_guiding_pixels = guiding * oversampling  # in spaxels
        return gaussian_filter(image, sigma_guiding_pixels, axes=(-2, -1))

    @staticmethod
    def _remove_oversampling(image, oversampling, func=np.mean):
        """Remove oversampling from an image.

        Parameters
        ----------
        image : array_like
            Oversampled image.
        oversampling : int
            Oversampling factor.
        func : callable, optional
            Function to use for downsampling. Default is `numpy.mean`.

        Returns
        -------
        array_like
            Downsampled image.
        """
        from astropy import nddata

        if np.ndim(image) == 2:
            block_size = (oversampling, oversampling)
        elif np.ndim(image) == 3:
            block_size = (1, oversampling, oversampling)
        else:
            raise ValueError(f"image ndim must be 2 or 3 {np.ndim(image)=} given.")

        return nddata.block_reduce(image, block_size=block_size, func=func)

    #  internal tricks

    def cube_to_slice(self, cube, lbda_range, func=np.nansum, squeeze=False):
        """Collapse a cube into images over wavelength ranges.

        Parameters
        ----------
        cube : array_like
            Data cube of shape ``(nlbda, ny, nx)``.
        lbda_range : tuple of float or list of tuple
            Wavelength range(s) ``(wmin, wmax)``, in Angstrom (bounds
            included).
        func : callable, optional
            Function merging the wavelengths of a range into one image,
            called as ``func(subcube, axis=0)``. Default is `numpy.nansum`.
        squeeze : bool, optional
            If True, squeeze the output array. Default is False.

        Returns
        -------
        numpy.ndarray
            Images, of shape ``(n_ranges, ny, nx)``.
        """
        slices = []
        for wmin, wmax in np.atleast_2d(lbda_range):
            flag_lbda = ((self.lbda >= wmin) & (self.lbda <= wmax))
            slices.append(func(cube[flag_lbda], axis=0))

        slices = np.asarray(slices)
        if squeeze:
            slices = np.squeeze(slices)

        return slices

    def get_nea(self, position=(0, 0), nea_spatial=None, nea_pixels=None): # pragma: no cover
        """Noise Equivalent Area (PSF -> Spaxel -> detector).

        Parameters
        ----------
        position : tuple of float, optional
            Position of the PSF in spaxels. Ignored if ``nea_spatial`` is
            given. Default is ``(0, 0)``.
        nea_spatial : float or array_like, optional
            Noise equivalent area of the spatial PSF (in spaxels).
            If None, `get_nea_spatial` is used. Default is None.
        nea_pixels : float or array_like, optional
            Noise equivalent area of a spaxel/slice caused by the
            cross-dispersion (in pixels). If None, ``get_nea_pixels()`` is
            used (only defined for `SlicerSpectrograph` and
            `MLASpectrograph`). Default is None.

        Returns
        -------
        float or numpy.ndarray
            Total noise equivalent area, in pixels.
        """

        # NEA_Spatial PSF on the MLA (2D) / Slicer (1d) | in slice / spaxel
        if nea_spatial is None:
            nea_spatial = self.get_nea_spatial(position=position)

        # NEA_pixel of 1 spaxel / slice on the dectetor
        if nea_pixels is None:
            nea_pixels = self.get_nea_pixels()

        return nea_spatial * nea_pixels

    def get_nea_telescope_airy(self, position=(0, 0), in_spaxels=True): # pragma: no cover
        """Get the noise equivalent area of the telescope Airy disk.

        Parameters
        ----------
        position : tuple of float, optional
            Position of the PSF in spaxels. Default is ``(0, 0)``.
        in_spaxels : bool, optional
            If True, return the NEA in spaxels. Only True is currently
            supported. Default is True.

        Returns
        -------
        numpy.ndarray
            NEA of the Airy disk as a function of wavelength.

        See Also
        --------
        slicersim.telescope.Telescope.get_nea_airy : The method called.
        """
        if in_spaxels:
            norm_scale = self.spx_spatial_scale

        return self.telescope.get_nea_airy(self.lbda, norm_scale=norm_scale, position=position)

    # ------------- #
    #   Others      #
    # ------------- #
    def pointsource_variance(self, varcube, position=(0, 0), radius=5,
                              psf_profile="default",
                              optimal=True, verbose=False):
        """Point-source extracted variance from variance cube.

        It is evaluated from a plain summation over the aperture, or from an
        optimal extraction, i.e. inverse-variance weighted least-square fit of
        the PSF profile (assumed Gaussian). As the extraction is assumed to do
        a perfect job on the signal, only the variance depends on instrumental
        parameters and aperture definition.

        Parameters
        ----------
        varcube : array_like
            Variance cube, of shape ``(nlbda, ny, nx)``, in ADU².
        position : tuple of float, optional
            ``(x, y)`` point source position, in spaxels.
            Default is ``(0, 0)``.
        radius : float or tuple of float, optional
            Aperture radius in spaxels, either a float (circular aperture) or
            ``(radius_y, radius_x)`` (elliptical aperture). Default is 5.
        psf_profile : str, optional
            PSF profile to use (see `get_spatial_psf`). Default is
            ``"default"``.
        optimal : bool, optional
            If True, use optimal extraction, otherwise plain summation.
            Default is True.
        verbose : bool, optional
            If True, print the number of spaxels in the aperture.
            Default is False.

        Returns
        -------
        numpy.ndarray
            Extracted variance spectrum, of shape ``(nlbda,)``, in ADU².

        Raises
        ------
        ValueError
            If ``radius`` cannot be parsed.
        """
        # Spatial PSF
        psf = self.get_spatial_psf(position=position, profile=psf_profile)  # (nlbda, ny, nx)

        # Aperture
        aper = self.get_aperture(position=position, radius=radius)
        if verbose:
            print(f"Aperture of {radius=} spx: {np.count_nonzero(aper)} spx selected")

        # If cube is (nlbda, ny, nx), cube[:, aper] is (nlbda, nspx)
        if optimal:  # Optimal extraction variance = 1/sum(psf**2/var)
            variance = 1 / (psf[:, aper] ** 2 / varcube[:, aper]).sum(axis=-1)
        else:  # Plain summation: variance = sum(variance)
            variance = varcube[:, aper].sum(axis=-1)

        return variance  # (nlbda,) [ADU²]

    def aperture_extraction(self, cube, varcube, position=(0, 0), radius=5,
                            psf_profile="default", optimal=False):
        """Extracted signal and variance of an extended source within an aperture.

        Contrary to `pointsource_variance`, the signal is measured from the
        (noise-free) signal cube, using the same weights as the variance:

        - plain summation (``optimal=False``): ``signal = sum(cube)`` and
          ``variance = sum(varcube)`` over the aperture;
        - optimal extraction (``optimal=True``): inverse-variance weighted
          least-square fit of the PSF profile, i.e. the flux of the point
          source that best matches the cube at ``position``:
          ``signal = sum(psf*cube/var) / sum(psf**2/var)`` and
          ``variance = 1 / sum(psf**2/var)``.

        Parameters
        ----------
        cube : array_like
            Signal cube, of shape ``(nlbda, ny, nx)``.
        varcube : array_like
            Variance cube, of shape ``(nlbda, ny, nx)``, in the square of the
            ``cube`` unit.
        position : tuple of float, optional
            ``(x, y)`` aperture center, in spaxels. Default is ``(0, 0)``.
        radius : float or tuple of float, optional
            Aperture radius in spaxels, see `get_aperture`. Default is 5.
        psf_profile : str, optional
            PSF profile used for the optimal extraction (see
            `get_spatial_psf`). Default is ``"default"``.
        optimal : bool, optional
            If True, use optimal extraction, otherwise plain summation.
            Default is False.

        Returns
        -------
        signal, variance : numpy.ndarray
            Extracted signal and variance, of shape ``(nlbda,)``.

        See Also
        --------
        pointsource_variance : Variance of a point source.
        """
        aper = self.get_aperture(position=position, radius=radius)

        # If cube is (nlbda, ny, nx), cube[:, aper] is (nlbda, nspx)
        if optimal: # PSF-weighted, as a point source.
            psf = self.get_spatial_psf(position=position, profile=psf_profile)[:, aper]  # (nlbda, nspx)
            inv_variance = (psf ** 2 / varcube[:, aper]).sum(axis=-1)
            signal = (psf * cube[:, aper] / varcube[:, aper]).sum(axis=-1) / inv_variance
            variance = 1 / inv_variance
        else:
            signal = cube[:, aper].sum(axis=-1)
            variance = varcube[:, aper].sum(axis=-1)

        return signal, variance

    def get_aperture(self, position=(0, 0), radius=5):
        """Spaxels within a circular (or elliptical) aperture.

        Parameters
        ----------
        position : tuple of float, optional
            ``(x, y)`` aperture center, in spaxels. Default is ``(0, 0)``.
        radius : float or tuple of float, optional
            Aperture radius in spaxels, either a float (circular aperture) or
            ``(radius_y, radius_x)`` (elliptical aperture). Default is 5.

        Returns
        -------
        numpy.ndarray
            ``(ny, nx)`` boolean mask, True for the spaxels whose centroid is
            within the aperture.

        Raises
        ------
        ValueError
            If ``radius`` cannot be parsed.
        """
        if np.ndim(radius) == 0:
            radius_y = radius_x = radius
        elif np.ndim(radius) == 1 and len(radius) == 2:
            radius_y, radius_x = radius
        else:
            raise ValueError(f"cannot parse input {radius=}")

        x0, y0 = position
        (x, y), oversampling = self.get_spaxel_centroids()
        # y /= 2
        r_radius = np.hypot((x - x0) / radius_x, (y - y0) / radius_y)  # (ny, nx) [spx]
        return r_radius <= 1

    def get_lsf_dispersion(self, as_="scale"):
        """Get the Gaussian LSF dispersion, in wavelength bins (pixels).

        Parameters
        ----------
        as_ : str or None, optional
            How the dispersion is defined:

            - ``"scale"`` or ``"sigma"``: the Gaussian sigma (i.e. ``scale``
              in the scipy/numpy definition), that is half the
              `dispersion_resolution`.
            - ``"resolution"`` or None: the full resolution element, i.e.
              `get_dispersion_resolution`.

            Default is ``"scale"``.

        Returns
        -------
        float
            LSF dispersion in wavelength bins.

        Raises
        ------
        NotImplementedError
            If ``as_`` is not one of the accepted values.
        """
        if as_ is None or as_ in ["resolution"]:
            disp_ = self.get_dispersion_resolution()

        elif as_ in ["scale", "sigma"]:
            disp_ = self.get_dispersion_resolution() / 2.
        else:
            raise NotImplementedError(f"{as_=} is not implemented, use: scale or resolution")

        return disp_

    def apply_line_spread_function(self, fluxes, **kwargs):
        """Apply the line spread function to a flux cube.

        The LSF is a Gaussian of sigma ``get_lsf_dispersion(as_="scale")``
        wavelength bins.

        Parameters
        ----------
        fluxes : array_like
            Flux spectrum ``(nlbda,)`` or cube ``(nlbda, ny, nx)``.
        **kwargs
            Additional arguments passed to `scipy.ndimage.gaussian_filter1d`
            (``axis=0`` by default).

        Returns
        -------
        numpy.ndarray
            Fluxes convolved with the LSF, with the same shape as ``fluxes``.
        """
        from scipy.ndimage import gaussian_filter1d

        # wavelength is the first direction of fluxes
        lsf_sigma = self.get_lsf_dispersion(as_="scale")

        # apply a gaussian filter to input fluxes ; wavelength axis is 0 by default
        return gaussian_filter1d(fluxes, lsf_sigma, **({"axis": 0} | kwargs))

    # ------------- #
    #    generate   #
    # ------------- #
    def get_array_as_cube(self, array, oversampling=None):
        """Broadcast an array into a spectrograph-shaped cube.

        Parameters
        ----------
        array : float or array_like
            Value(s) filling the cube. Accepted shapes are a scalar (uniform
            cube), ``(nlbda, )`` (a spectrum, spatially uniform) or the
            spatial shape ``(ny, nx)`` (an image, uniform in wavelength).
        oversampling : int, optional
            Spatial oversampling factor of the output cube. Default is None,
            i.e. the native spaxel shape.

        Returns
        -------
        numpy.ndarray
            Cube of shape ``(nlbda, ny, nx)``.

        Raises
        ------
        ValueError
            If ``array`` has more than 3 dimensions, or if its shape matches
            neither ``(nlbda, )`` nor the spatial shape of the spectrograph.

        See Also
        --------
        get_spectrograph_shape : The spatial shape the cube is built on.
        """
        spatial_shape = self.get_spectrograph_shape(oversampling=oversampling)

        array = np.asarray(array)
        # float as input
        if array.ndim == 0:
            pass # this broadcasts good to go

        elif array.ndim == 1:
            if array.shape == (self.nlbda,):
                array = array[:, None, None]
            else:
                raise ValueError("cannot parse the input shape of array.")

        elif array.ndim == 2:
            if array.shape == spatial_shape:
                array = array[None, :]
            else:
                raise ValueError("cannot parse the input shape of array.")
        elif array.ndim > 3:
            raise ValueError("cannot parse the input shape of array.")

        cube = np.full((self.nlbda, *spatial_shape),  # lbda, y, x
                        array)  # (nlbda, ny, nx)
        #if self.type == "slicer":
        #    cube = self._correct_anamorphose_(cube)

        return cube

    # Generic
    def generate_fullof_cube(self, array, oversample=True):
        """Project an array as a cube (not implemented).

        Placeholder: this currently does nothing and returns None. See
        `get_array_as_cube` instead.

        Parameters
        ----------
        array : array_like
            Array to project.
        oversample : bool, optional
            Unused. Default is True.
        """


    # Point Souce
    def generate_pointsource(self, spectrum, position=(0, 0),
                              psf_profile="default",
                              oversampling=None,
                              as_oversampled=False,
                              apply_lsf=True, **kwargs):
        """Generate a photon flux cube from a point source spectrum.

        Parameters
        ----------
        spectrum : array_like
            Point source spectrum, of shape ``(nlbda,)``, in erg/s/cm²/Å.
        position : tuple of float, optional
            ``(x, y)`` point source position, in spaxels.
            Default is ``(0, 0)``.
        psf_profile : str, optional
            PSF profile to use (see `get_spatial_psf`). Default is
            ``"default"``.
        oversampling : int, optional
            Spatial oversampling factor of the PSF (see `get_spatial_psf`).
            Default is None.
        as_oversampled : bool, optional
            If True, return the oversampled cube. Default is False.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to `get_spatial_psf`.

        Returns
        -------
        numpy.ndarray
            Photon flux cube, of shape ``(nlbda, ny, nx)``, in ph/s/spx.
        """
        # Spatial PSF
        psf = self.get_spatial_psf(profile=psf_profile, position=position,
                                    as_oversampled=as_oversampled,
                                    oversampling=oversampling,
                                    **kwargs
                                    )  # (nlbda, ny, nx)

        # erg/s/cm²/Å / erg/ph * cm² * Å = ph/s
        flux = spectrum * self.flambda2photon  # (nlbda,) [ph/s]

        psf_cube = np.reshape(flux, (-1, 1, 1)) * psf  # Point source (nlbda, ny, nx)
        if apply_lsf:
            psf_cube = self.apply_line_spread_function(psf_cube)

        # no self._correct_anamorphose_(psf_cube)
        # as this is already including inside get_spatial_psf

        return psf_cube

    # Flat Background
    def generate_background(self, spectrum, oversampling=None,
                            apply_lsf=True):
        """Generate a photon flux cube from uniform scene background spectrum.

        Parameters
        ----------
        spectrum : array_like
            Uniform scene background spectrum, of shape ``(nlbda,)``, in
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
        # erg/s/cm²/Å/arcsec² * cm² * Å / erg/ph * arcsec² = ph/s/spx
        flux = spectrum * self.flambda2photon * self.spx_area

        bkgd_cube = np.full((self.nlbda, *self.get_spectrograph_shape(oversampling=oversampling)),  # y, x
                             flux[:, np.newaxis, np.newaxis])  # (nlbda, ny, nx)
        if apply_lsf:
            bkgd_cube = self.apply_line_spread_function(bkgd_cube)

        #if self.type == "slicer":
        #    bkgd_cube = self._correct_anamorphose_(bkgd_cube)

        return bkgd_cube

    # Structured background
    def generate_structured_background(self, cube, psf_profile="default",
                                       apply_lsf=True, **kwargs):
        """Generate a photon flux cube from a structured background scene.

        The input cube is converted into photons and convolved, wavelength
        per wavelength, by the spatial PSF (see `get_spatial_psf`).

        Parameters
        ----------
        cube : array_like
            (nlbda, ny, nx) structured background (e.g. host) flux per
            spaxel in erg/s/cm²/Å, on the spaxel grid (see `spx_shape`).
        psf_profile : str, optional
            PSF profile to use. Default is "default".
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        **kwargs
            Goes to `get_spatial_psf`.

        Returns
        -------
        numpy.ndarray
            Photon flux cube, of shape ``(nlbda, ny, nx)``, in ph/s/spx.
        """
        from scipy.signal import fftconvolve

        # PSF kernel centered on the pixel fftconvolve(mode="same") uses as
        # origin, i.e. index (n-1)//2, which is offset by -0.5 spaxel from the
        # grid center for even-sized axes.
        ny, nx = self.spx_shape
        kernel_center = ((nx - 1) // 2 - (nx - 1) / 2,
                         (ny - 1) // 2 - (ny - 1) / 2) # (x, y) [spaxels]
        psf = self.get_spatial_psf(profile=psf_profile, position=kernel_center,
                                   **kwargs)  # (nlbda, ny, nx)

        # erg/s/cm²/Å / erg/ph * cm² * Å = ph/s
        flux = cube * self.flambda2photon[:, np.newaxis, np.newaxis]
        # This means each wavelengthes are independent.
        host_cube = fftconvolve(flux, psf, mode="same", axes=(-2, -1))
        host_cube = np.clip(host_cube, 0, None) # remove fft numerical noise (<0)

        if apply_lsf:
            host_cube = self.apply_line_spread_function(host_cube)

        return host_cube

    # Themal (pre-dispersor)
    def _get_optics_dispersed_signal(self, lbda_bin):
        """Thermal signal of the optical surfaces sitting before the disperser.

        Only the surfaces flagged as "dispersed" in the optics metadata
        contribute here: their thermal emission goes through the disperser and
        is therefore spread along the spectral axis, unlike the emission of the
        surfaces located after it.

        Parameters
        ----------
        lbda_bin : array_like
            (nlbda, 2) list of spectral domains in Angstrom.

        Returns
        -------
        numpy.ndarray
            Signal summed over the dispersed surfaces, in ph/s/spx/Δλ,
            of shape (nlbda, ).

        See Also
        --------
        generate_thermal_signal : Public entry point using this contribution.
        """
        all_surface_radiations = self.optics.get_signal(lbda_bin,
                                                        area=self.telescope.surface,
                                                        solid_angle=self.omega)

        dispersed_radiations = all_surface_radiations[ self.optics.meta["dispersed"] ]
        return np.sum(dispersed_radiations, axis=0)

    def _get_telescope_dispersed_signal(self, lbda_bin):
        """Thermal signal of the telescope mirrors, seen through the disperser.

        Parameters
        ----------
        lbda_bin : array_like
            (nlbda, 2) list of spectral domains in Angstrom.

        Returns
        -------
        numpy.ndarray
            Signal summed over all telescope mirrors, in ph/s/spx/Δλ,
            of shape (nlbda, ).

        See Also
        --------
        generate_thermal_signal : Public entry point using this contribution.
        """
        return self.telescope.get_thermal_signal(lbda_bin,
                                                 solid_angle=self.omega,  # Spx solid angle [sr]
                                                 as_sum=True)

    def generate_thermal_signal(self, lbda_bin=None,
                                as_cube=False, oversampling=None,
                                apply_lsf=True, as_sum=True):
        """Get the dispersed thermal signal, in ph/s/spx/Δλ.

        This is the thermal emission that goes through the disperser: the
        telescope mirrors and the spectrograph optical surfaces located
        before the disperser.

        Parameters
        ----------
        lbda_bin : array_like, optional
            ``(nlbda, 2)`` list of spectral domains in Angstrom.
            If None, `lbda_bin` is used. Default is None.
        as_cube : bool, optional
            If True, return a ``(nlbda, ny, nx)`` cube, otherwise a
            ``(nlbda,)`` spectrum. Default is False.
        oversampling : int, optional
            Spatial oversampling factor, used if ``as_cube`` is True.
            Default is None.
        apply_lsf : bool, optional
            If True, apply the line spread function. Default is True.
        as_sum : bool, optional
            Unused: the contributions are always summed. Default is True.

        Returns
        -------
        numpy.ndarray
            Thermal signal in ph/s/spx/Δλ, of shape ``(nlbda,)`` or
            ``(nlbda, ny, nx)`` (see ``as_cube``).
        """
        if lbda_bin is None:
            lbda_bin = self.lbda_bin

        # Telescope
        signal = self._get_optics_dispersed_signal(lbda_bin=lbda_bin) \
               + self._get_telescope_dispersed_signal(lbda_bin=lbda_bin)


        #signal =
        # output formating
        if as_cube:
            signal = np.full((self.nlbda, *self.get_spectrograph_shape(oversampling=oversampling)),
                             signal[..., np.newaxis, np.newaxis])  # (nlbda, ny, nx)

        if apply_lsf:
            signal = self.apply_line_spread_function(signal)

        #if self.type == "slicer":
        #    signal = self._correct_anamorphose_(signal)

        return signal  # [ph/s/spx/Δλ]

    # Empty cube
    def get_empty_cube(self, filled=0, oversampling=None):
        """Get a uniform data cube.

        Parameters
        ----------
        filled : float, optional
            Value to fill the cube with. Default is 0.
        oversampling : int, optional
            Spatial oversampling factor. Default is None.

        Returns
        -------
        numpy.ndarray
            Cube of shape ``(nlbda, ny, nx)`` filled with ``filled``.
        """
        # no apply LSF as zeros...
        ny, nx = self.get_spectrograph_shape(oversampling=oversampling)

        cube =  filled * np.ones((self.nlbda, ny, nx))
        #if self.type == "slicer":
        #    cube = self._correct_anamorphose_(cube)

        return cube

    def _correct_anamorphose_(self, cube):
        """
        in slicer, there are 2 detector pixels in the width.
        In slicersim, spectrograph work at the half-width level for error treatment (1spaxel=>1^pixel)
        Here we should just then re-bin the slice by 2 as this sampling does not exist in reality.
        cubes are: (lbda, width, height). width should be merged.
        """
        warnings.warn("Anamorphose correction is disabled.")
        return cube
        #return bin_array(cube, (1, *self._ANAMORPHOSE), function=np.sum)

    # ------------ #
    #   GETTER     #
    # ------------ #
    def get_nea_spatial(self, position=(0, 0), in_spaxels=True, guiding_sigma=None): # pragma: no cover
        """Get the noise equivalent area of the spatial PSF.

        That is, the equivalent number of spaxels contributing noise.

        Parameters
        ----------
        position : tuple of float, optional
            Position of the PSF, in spaxels. Default is ``(0, 0)``.
        in_spaxels : bool, optional
            If True, return the area in spaxels, otherwise in arcsec².
            Default is True.
        guiding_sigma : float, optional
            Guiding sigma override in arcsec (see `get_psf_sigma_spectral`).
            Default is None.

        Returns
        -------
        numpy.ndarray
            Noise equivalent area as a function of wavelength.

        Raises
        ------
        NotImplementedError
            If the spatial PSF profile is not Gaussian.
        """
        from .nea import get_2dnorm_nea

        if self.spatial_psf["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian spatial PSF profile implemented, but: {self.spatial_psf['profile']=}")

        sigma_at_spectro = self.get_psf_sigma_spectral(in_spaxels=in_spaxels,
                                                       guiding_sigma=guiding_sigma)

        return get_2dnorm_nea(sigma_at_spectro, mean=position)

    #
    # - tools
    #
    def chromatic_average(self, quantity):
        """Compute the chromatic average of a quantity.

        The average is weighted by the wavelength bin widths, i.e. averaged
        over wavelength rather than over pixels.

        Parameters
        ----------
        quantity : array_like
            Chromatic quantity, of shape ``(nlbda,)``.

        Returns
        -------
        float
            Chromatic average.
        """
        return np.average(quantity, weights=np.diff(self.lbda_edges))

    #
    # - Internal
    #
    @staticmethod
    def _get_chromatic_sigma(lbda, chromatic_sigma,
                             constant_sigma,
                             wref):
        """Get total PSF, including chromatic and constant components.

        The total (Gaussian) stddev is the quadratic sum of two components:

        - the chromatic stddev, proportional to wavelength,
          normalized at ``wref``,
        - the achromatic, constant stddev.

        Parameters
        ----------
        lbda : array_like
            Wavelength.
        chromatic_sigma : float
            Chromatic (linear) stddev at ``wref``.
        constant_sigma : float
            Achromatic (constant) stddev.
        wref : float
            Reference wavelength (same unit as ``lbda``).

        Returns
        -------
        array_like
            Total stddev as function of wavelength.
        """

        lmin, lmax = np.array(lbda)[[0, -1]]  # 1st and last wavelengths
        assert lmin > wref / 3 and lmax < wref * 3, \
            "Input and reference wavelengths probably not in same units."

        return np.hypot(constant_sigma,
                        chromatic_sigma * (lbda / wref))  # [px]

    # -------- #
    #  SHOW    #
    # -------- #
    def show_nea(self, ax=None, position=(0, 0), legend=True): # pragma: no cover
        """Show the total noise equivalent area as a function of wavelength.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        position : tuple of float, optional
            Position of the PSF in spaxels. Default is ``(0, 0)``.
        legend : bool, optional
            If True, show the legend. Default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # data
        nea_spatial = self.get_nea_spatial(in_spaxels=True, position=position)
        nea_no_guiding = self.get_nea_spatial(guiding_sigma=0, in_spaxels=True, position=position)
        nea_telescope = self.get_nea_telescope_airy(in_spaxels=True, position=position)

        ax.plot(self.lbda, self.get_nea(position=position, nea_spatial=nea_spatial),
                color="#194D80", label="total")
        ax.plot(self.lbda, self.get_nea(position=position, nea_spatial=nea_no_guiding),
                color="#194D80", ls="--", label="without guiding")
        ax.plot(self.lbda, self.get_nea(position=position, nea_spatial=nea_telescope),
                color="#F8AD05", label="airy from telescope")
        ax.legend(fontsize="small", frameon=False)

        ax.set_xlabel(r"wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("NEA [in pixels]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)

        return fig

    def show_nea_spatial(self, ax=None, position=(0, 0), legend=True): # pragma: no cover
        """Show the spatial noise equivalent area as a function of wavelength.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        position : tuple of float, optional
            Position of the PSF in spaxels. Default is ``(0, 0)``.
        legend : bool, optional
            If True, show the legend. Default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # data
        nea = self.get_nea_spatial(in_spaxels=True, position=position)
        nea_no_guiding = self.get_nea_spatial(guiding_sigma=0, in_spaxels=True, position=position)
        nea_telescope = self.get_nea_telescope_airy(in_spaxels=True, position=position)

        ax.plot(self.lbda, nea, color="#194D80", label="total")
        ax.plot(self.lbda, nea_no_guiding, color="#194D80", ls="--", label="without guiding")
        ax.plot(self.lbda, nea_telescope, color="#F8AD05", label="airy from telescope")

        ax.set_xlabel(r"wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("NEA [in spaxels]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)

        return fig

    def show_fwhm(self, ax=None, legend=True, guiding_arcsec=None, in_arcsec=False, show_band=True): # pragma: no cover
        """Show the full width at half maximum (FWHM) of the spatial PSF.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.
        legend : bool, optional
            If True, show the legend. Default is True.
        guiding_arcsec : float, optional
            Guiding sigma in arcsec. If None, the stored
            ``spatial_psf["guiding_sigma"]`` is used. Default is None.
        in_arcsec : bool, optional
            If True, show the FWHM in arcsec, otherwise in spaxels.
            Default is False.
        show_band : bool, optional
            If True, highlight the under-sampled (FWHM < 2 spaxels) region.
            Default is True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        in_spaxels = not in_arcsec
        norm_scale = 1 if in_arcsec else self.spx_spatial_scale
        norm_sampling = self.spx_spatial_scale if in_arcsec else 1
        # sigma
        sigma_at_mla = self.get_psf_sigma_spectral(in_spaxels=in_spaxels, guiding_sigma=guiding_arcsec)
        sigma_at_mla_no_guiding = self.get_psf_sigma_spectral(in_spaxels=in_spaxels, guiding_sigma=0)
        radius = self.telescope.get_airy_radius(self.lbda, norm_scale=norm_scale)

        ax.plot(self.lbda, 2.35 * sigma_at_mla, color="#194D80", label="total scatter")
        ax.plot(self.lbda, 2.35 * sigma_at_mla_no_guiding, color="#194D80", ls="--", label="without guiding")
        ax.plot(self.lbda, 0.8 * radius, color="#F8AD05", label="airy from telescope")

        if show_band:
            ax.axhspan(2 * norm_sampling, 2.35 * norm_sampling, color="tab:orange", alpha=0.05, lw=0)
            _ylow, _ = ax.get_ylim()
            ax.axhline(2 * norm_sampling, color="tab:red", alpha=1, ls="--", lw=0.5)
            ax.axhspan(0, 2 * norm_sampling, color="tab:red", alpha=0.05, lw=0)
            ax.set_ylim(_ylow)

        ax.set_xlabel(r"wavelength [$\AA$]", fontsize="large")
        ax.set_ylabel("FWHM [in spaxels]" if in_spaxels else "FWHM [in arcsec]", fontsize="large")
        if legend:
            ax.legend(fontsize="small", frameon=False)

        return fig

    def show_psf(self, lbda_range, profile="default",
                 guiding_arcsec=None, axes=None,
                 position=(0, 0), oversampling=5,
                 in_arcsec=False,
                 norm="log", **kwargs): # pragma: no cover
        """Show the spatial PSF, integrated over a wavelength range.

        Three panels are shown: the oversampled PSF without guiding, the
        oversampled PSF with guiding, and the PSF on the spaxel grid.

        Parameters
        ----------
        lbda_range : tuple of float
            Wavelength range ``(wmin, wmax)`` to integrate over, in Angstrom.
        profile : str, optional
            PSF profile to use (see `get_spatial_psf`). Default is
            ``"default"``.
        guiding_arcsec : float, optional
            Guiding sigma in arcsec. If None, the stored
            ``spatial_psf["guiding_sigma"]`` is used. Default is None.
        axes : list of matplotlib.axes.Axes, optional
            List of 3 axes to plot on. If None, a new figure and axes are
            created. Default is None.
        position : tuple of float, optional
            Position of the PSF in spaxels. Default is ``(0, 0)``.
        oversampling : int, optional
            Oversampling factor. Default is 5.
        in_arcsec : bool, optional
            If True, show the extent in arcsec, otherwise in spaxels.
            Default is False.
        norm : str, optional
            Normalization for the color scale (``"linear"`` or ``"log"``).
            Default is ``"log"``.
        **kwargs
            Additional arguments passed to `matplotlib.axes.Axes.imshow`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        from matplotlib import colors

        if norm is None or norm in ["linear"]:
            norm = colors.Normalize
        elif norm in ["log"]:
            norm = colors.LogNorm

        extent = np.asarray(self.spaxel_extent)
        if in_arcsec:
            extent *= self.spx_spatial_scale

        if guiding_arcsec is None:
            guiding_arcsec = self.spatial_psf["guiding_sigma"]

        if axes is None:
            import matplotlib.pyplot as plt
            fig, (ax, axg, axsl) = plt.subplots(nrows=3,
                                                figsize=(5, 7),
                                                gridspec_kw={"hspace": 0.04})
        else:
            (ax, axg, axsl) = axes
            fig = ax.figure

        # highly-resolved cube
        psf_cube = self.get_spatial_psf(profile=profile, guiding_sigma=0,  # no guiding error yet
                                        oversampling=oversampling,
                                        as_oversampled=True,
                                        position=position)

        # perfect model
        psf_slice = self.cube_to_slice(psf_cube, lbda_range, squeeze=True)
        ax.imshow(psf_slice, norm=norm(), extent=extent, **kwargs)
        ax.set_xticklabels([])

        # adding guiding
        psf_cube = self._apply_additional_scatter(psf_cube, guiding=guiding_arcsec, oversampling=oversampling,
                                                  in_units="arcsec")
        psf_slice = self.cube_to_slice(psf_cube, lbda_range, squeeze=True)
        axg.imshow(psf_slice, norm=norm(), extent=extent, **kwargs)
        axg.set_xticklabels([])

        # removing oversampling
        psf_cube = self._remove_oversampling(psf_cube, oversampling=oversampling)
        psf_slice = self.cube_to_slice(psf_cube, lbda_range, squeeze=True)
        axsl.imshow(psf_slice, norm=norm(), extent=extent, **kwargs)

        return fig

    # ================= #
    #  Properties       #
    # ================= #
    @property
    def spaxel_extent(self):
        """Extent ``[-hx, hx, -hy, hy]`` of the spaxel grid, in spaxels."""
        hy, hx = np.asarray(self.spx_shape) / 2  # Half total width [spx]
        return [-hx, hx, -hy, hy]

    @property
    def flambda2photon(self):
        """Chromatic conversion factor from erg/s/cm²/Å to ph/s.

        It includes the telescope collecting surface, the spectrograph
        throughput (see `get_throughput`) and the wavelength bin widths.
        """
        dlbda = np.diff(self.lbda_edges)  # Spectral step (nlbda,)
        hnu = 1.9864459e-08 / self.lbda  # Photon energy [erg] with lbda in [Å]

        # erg/s/cm²/Å * (cm² * throughput * Å / erg/ph) = ph/s
        return (self.telescope.surface * 1e4 *
                self.get_throughput() * dlbda / hnu)  # (nlbda,) [ph/s]

    # Spaxels
    @property
    def _hspaxels(self):
        """Spaxel properties {shape: (N,M), spx_scale: float [in arcsec]}."""
        return self._spaxels

    @property
    def spaxels(self):
        """Spaxel coordinates and properties (see `~slicersim.profiles.build_pixels`)."""
        # build from self._spaxels
        if self._spaxel_coords is None or len(self._spaxel_coords) == 0:
            self._spaxel_coords = build_pixels(self._spaxels["shape"], oversampling=1)

        return self._spaxel_coords

    @property
    def spx_shape(self):
        """Shape ``(ny, nx)`` of the spaxel grid."""
        return self.spaxels["shape"]

    @property
    def spx_centroids(self):
        """Spaxel ``(x, y)`` centroid coordinates, in spaxels."""
        return self.spaxels["centroids"]

    @property
    def spx_edges(self):
        """Spaxel ``(x, y)`` edge coordinates, in spaxels."""
        return self.spaxels["edges"]

    @property
    def spx_area(self):
        """Area of a spaxel, in arcsec²."""
        if np.ndim(self.spx_spatial_scale) == 0:
            spx_spatial_scale_y = spx_spatial_scale_x = self.spx_spatial_scale
        else:
            spx_spatial_scale_y, spx_spatial_scale_x = self.spx_spatial_scale
        return spx_spatial_scale_y * spx_spatial_scale_x

    @property
    def spx_spatial_scale(self):
        """Spatial scale of a spaxel, in arcsec.

        If not set, it is derived from the telescope Airy radius at
        `lbda_ref`.
        """
        spx_spatial_scale = self._spaxels.get("spx_scale", None)
        if spx_spatial_scale is None:
            from .profiles import airyradius_to_gaussiansigma
            warnings.warn("Setting spaxels scale from airy")
            radius_airy = self.telescope.get_airy_radius(self.lbda_ref)
            spx_spatial_scale = airyradius_to_gaussiansigma(radius_airy, on="fwhm")

        return spx_spatial_scale

    @property
    def dispersion_resolution(self):
        """Effective dispersion resolution (spot size), in pixels.

        2 is optimal: below that, the PSF is unresolved; above it, the PSF is
        oversampled, hence producing unnecessary pixel noise.
        """
        # - mapper -
        return self.meta.get("dispersion_resolution", 2)

    @property
    def nlbda(self):
        """Number of spectral pixels (wavelength bins)."""
        return len(self.lbda)

    @property
    def lbda_bin(self):
        """Wavelength bins ``(lbda_min, lbda_max)``, of shape ``(nlbda, 2)``, in Angstrom."""
        return np.vstack([self.lbda_edges[:-1],
                          self.lbda_edges[1:]]).T  # (nlbda, 2) [Å]

    @property
    def meta(self):
        """Metadata of the instance."""
        return self._meta

    @property
    def name(self):
        """Name of the spectrograph (empty string if not set)."""
        return self.meta.get("name", "")

    @property
    def lbda_ref(self):
        """Reference wavelength of the chromatic PSF, in Angstrom (default 10000)."""
        # 1micron by default
        return self.meta.get("lbda_ref", 10_000)

    @property
    def omega(self):
        """Spaxel solid angle, in steradians."""
        # spx_spatial_scale is in arcsec (e.g. 0.04 for 40mas spaxels)
        hspx = self.spx_spatial_scale * u.arcsecond.to("rad")  # [rad]
        if np.ndim(hspx) == 0:
            hspx_y = hspx_x = hspx
        else:
            hspx_y, hspx_x = hspx

        # Case: square | could remove np.sin that has no impact
        return np.sin(hspx_y) * np.sin(hspx_x)

        # Case: circular
        # return np.pi * np.sin(hspx_y/2) * np.sin(hspx_x/2)

    @property
    def skyarea(self):
        """Full sky area covered by the spaxels (not implemented, always None)."""

    @property
    def psf_sigma_spectral(self):
        """Chromatic PSF sigma at `lbda_ref`, in arcsec.

        If not set (None or ``"default"``), it is derived from the telescope
        Airy radius at `lbda_ref`.
        """
        if self.spatial_psf["sigma_spectral"] in (None, "default"):
            # 2.9 is the airy equivalent.
            from .profiles import airyradius_to_gaussiansigma
            radius_airy = self.telescope.get_airy_radius(self.lbda_ref)
            self.spatial_psf["sigma_spectral"] = airyradius_to_gaussiansigma(radius_airy, on="fwhm")

        return self.spatial_psf["sigma_spectral"].item()

    @property
    def type(self):
        """Type of the spectrograph (e.g. ``"slicer"`` or ``"mla"``)."""
        return self._SPECTROGRAPH_TYPE



class SlicerSpectrograph(Spectrograph):
    """Image-slicer spectrograph, with anamorphosis.

    This class inherits from `Spectrograph` and modifies the spaxel
    definition to account for the anamorphosis introduced by the slicer: the
    number of spaxels along ``x`` (along the slices) is multiplied by the
    anamorphosis factor (``_ANAMORPHOSE = 2``), each slice width being
    sampled by 2 detector pixels.

    See `Spectrograph` for the parameters.
    """
    _ANAMORPHOSE = 2
    _SPECTROGRAPH_TYPE = "slicer"

    def set_spaxels(self, shape, spx_scale):
        """Set the spaxel information, accounting for the anamorphosis.

        Parameters
        ----------
        shape : tuple of int
            Slicer grid shape ``(ny, nx)``. The stored ``nx`` is multiplied
            by the anamorphosis factor.
        spx_scale : float or tuple of float
            Spaxel size in arcsec, either a float or ``(scale_y, scale_x)``.
        """
        shape = np.asarray(shape) * (1, self._ANAMORPHOSE)
        spx_scale = np.asarray(spx_scale)
        self._spaxels = {"shape": shape, "spx_scale": spx_scale}
        self._spaxel_coords = {}

    def get_nea_pixels(self):
        """Get the noise equivalent area of a slice (per wavelength), in pixels.

        Returns
        -------
        int
            Noise equivalent area in pixels (1 for a slicer).
        """
        return 1


class MLASpectrograph(Spectrograph):
    """Micro-lens array (MLA) spectrograph.

    No anamorphosis, but each spaxel trace has a cross-dispersion profile.

    Parameters
    ----------
    xdispersion : dict, optional
        Cross-dispersion information, with keys ``"sigma_spectral"``
        (chromatic sigma at `~Spectrograph.lbda_ref`, in pixels),
        ``"sigma"`` (constant sigma, in pixels) and ``"profile"``.
        Default is ``{}``.
    *args
        Goes to `Spectrograph`.
    **kwargs
        Goes to `Spectrograph`.
    """
    _ANAMORPHOSE = None
    _SPECTROGRAPH_TYPE = "mla"

    def __init__(self, xdispersion={}, *args, **kwargs): # pragma: no cover
        """Initialize the MLA spectrograph.

        See the class docstring for the parameters.
        """
        _ = super().__init__(*args, **kwargs)

        # an MLA has a cross-dispersion
        self.xdispersion = xdispersion

    # this is what from_config needs
    @classmethod
    def _parse_config(cls, config):# pragma: no cover
        """Parse the configuration dictionary.

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

        init_prop, config = super()._parse_config(config)

        # PSF at the detector level
        xdispersion = {"sigma_spectral": float(config["psf"]["detector"]["xdisp_sigma_spectral"]),
                       "sigma": float(config["psf"]["detector"]["xdisp_sigma"]),
                       "profile": config["psf"]["detector"]["xdisp_profile"]}
        init_prop["xdispersion"] = xdispersion
        return init_prop, config

    def update(self, reset_others=False, **kwargs): # pragma: no cover
        """Update any mutable attribute of the spectrograph.

        Cross-dispersion parameters (``xdisp_*`` entries of the detector PSF
        configuration) are handled here, the rest goes to
        `Spectrograph.update`.

        Parameters
        ----------
        reset_others : bool, optional
            If True, reset other parameters to their initial values.
            Currently not forwarded (False is always used). Default is False.
        **kwargs
            Parameters to update.

        Returns
        -------
        list of str
            Names of the components that changed (see
            `Spectrograph.update`).
        """
        # do xdispersion stuffs
        xdisp_updates = {}
        for k, v in kwargs.items():
            if k in self.meta["psf"]["detector"]:
                xdisp_updates[k.replace("xdisp_", "")] = v
                _ = kwargs.pop(k)  # remove them

        self.xdispersion = self.xdispersion | xdisp_updates
        # and the rest (normal spectrograph)
        return super().update(reset_others=False, **kwargs)

    def get_xdisp_sigma_spectral(self, xdims=0, xdisp_sigma=None): # pragma: no cover
        """Get the cross-dispersion stddev, in pixels.

        The total (Gaussian) cross-dispersion profile is the quadratic sum
        of two components:

        - the optical (chromatic) component, with stddev proportional
          to wavelength, normalized at `~Spectrograph.lbda_ref`,
        - the achromatic component, with constant stddev (including the
          instrumental PSF, see `~Spectrograph.get_instrumental_psf`).

        If needed, the 1D vector can be embedded in a N-dim array of
        shape ``(nlbda,) + (1,)*xdims``.

        Parameters
        ----------
        xdims : int, optional
            Extra dimensions to be appended. Default is 0.
        xdisp_sigma : float, optional
            Constant sigma override in pixels. If None,
            ``xdispersion["sigma"]`` is used. Default is None.

        Returns
        -------
        numpy.ndarray
            Total sigma in pixels.
        """
        if xdisp_sigma is None:
            xdisp_sigma = self.xdispersion["sigma"]

        # adding the internal PSF
        inst_psf = self.get_instrumental_psf(in_spaxels=True)
        if inst_psf is not None and inst_psf>0:
            xdisp_sigma = np.hypot(xdisp_sigma, inst_psf)

        return self._get_chromatic_sigma(self.lbda,
                                         chromatic_sigma=self.xdispersion["sigma_spectral"],
                                         constant_sigma=xdisp_sigma,
                                         wref=self.lbda_ref,
                                         xdims=xdims)

    def get_nea_spatial(self, position=(0, 0), in_spaxels=True, guiding_sigma=None): # pragma: no cover
        """Get the noise equivalent area of the spatial PSF.

        That is, the equivalent number of spaxels contributing noise.

        Parameters
        ----------
        position : tuple of float, optional
            Position of the PSF, in spaxels. Default is ``(0, 0)``.
        in_spaxels : bool, optional
            If True, return the area in spaxels, otherwise in arcsec².
            Default is True.
        guiding_sigma : float, optional
            Guiding sigma override in arcsec (see `get_psf_sigma_spectral`).
            Default is None.

        Returns
        -------
        numpy.ndarray
            Noise equivalent area as a function of wavelength.

        Raises
        ------
        NotImplementedError
            If the spatial PSF profile is not Gaussian.
        """
        from .nea import get_2dnorm_nea

        if self.spatial_psf["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian spatial PSF profile implemented, but: {self.spatial_psf['profile']=}")

        sigma_at_mla = self.get_psf_sigma_spectral(in_spaxels=in_spaxels,
                                                   guiding_sigma=guiding_sigma)
        return get_2dnorm_nea(sigma_at_mla, mean=position)

    def get_nea_pixels(self): # pragma: no cover
        """Get the noise equivalent area of a spaxel trace, in pixels.

        Returns
        -------
        numpy.ndarray
            Noise equivalent area in pixels, as a function of wavelength.

        Raises
        ------
        NotImplementedError
            If the cross-dispersion profile is not Gaussian.
        """
        from .nea import get_1dnorm_nea
        if self.xdispersion["profile"] not in ("normal", "norm", "gaussian"):
            raise NotImplementedError(f"only gaussian xdispersion PSF profile implemented, but: {self.xdispersion['profile']=}")

        sigma_xdisp_at_detector = self.get_xdisp_sigma_spectral()  # in pixels
        return get_1dnorm_nea(sigma_xdisp_at_detector)

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def xdisp_sigma_spectral(self): # pragma: no cover
        """Chromatic cross-dispersion sigma in pixels."""
        return self.xdispersion["sigma_spectral"]


# ================ #
#                  #
#   ThroughPut     #
#                  #
# ================ #

class OpticsThroughput:
    """Throughput of the optical elements.

    This class handles the throughput of the different optical elements
    of the spectrograph. The throughput of each element is defined by a curve
    (throughput vs. wavelength) and a number of optics of that type.

    Parameters
    ----------
    curves : dict
        Dictionary of throughput curves for each optical element.
        The keys are the element names and the values are callables
        (e.g., interpolators) that return the throughput as a function
        of wavelength.
    noptics : int or dict, optional
        Number of optics for each curve (the curve is raised to that power).
        If a dict, it must have the same keys as ``curves``. If None, it is
        not set and must be in ``meta``. Default is None.
    meta : dict, optional
        Metadata dictionary. Default is ``{}``.

    Attributes
    ----------
    mutable_parameters : list of str
        List of mutable parameters.
    """
    mutable_parameters = ["noptics"]
    _ELEMENTS_ADDUP = None

    def __init__(self, curves, noptics=None, meta={}):
        """Initialize the OpticsThroughput object.

        See the class docstring for the parameters.
        """
        self._curves = curves
        if noptics is not None:
            meta["noptics"] = noptics

        self._meta_in = deepcopy(meta)
        self._meta = deepcopy(meta)

    @classmethod
    def from_curves(cls, curves, noptics=1, lbda=None, names=None):
        """Create an OpticsThroughput instance from throughput arrays.

        Parameters
        ----------
        curves : array_like
            2D array ``[curve_1, curve_2, ...]`` of shape
            ``(nelements, nlbda)``: one throughput curve per element, all
            sampled at ``lbda``.
        noptics : int or list of int, optional
            Number of optics for each curve. If a list, it must have one
            entry per curve. None means 1. Default is 1.
        lbda : array_like, optional
            Wavelengths (in Angstrom) at which the curves are sampled,
            of shape ``(nlbda,)``. Default is None.
        names : list of str, optional
            Name of each element. Default is None.

        Returns
        -------
        OpticsThroughput
            An OpticsThroughput instance.

        Raises
        ------
        ValueError
            If the number of ``names`` or ``noptics`` entries does not match
            the number of curves.

        See Also
        --------
        from_curvesdf : Same, from a pandas.DataFrame.
        """
        # reformat as dataframe
        curves = np.asarray(curves)
        nelements, nlbda = curves.shape

        if names is None:
            names = np.arange(len)
        elif (nnames:=len(names)) != nelements:
            raise ValueError(f"{nnames} names but {nelements} elements")

        if noptics is None:
            noptics = 1

        if not isinstance(noptics, int):
            if (n_noptics := len(noptics)) != nelements:
                raise ValueError(f"{n_noptics} noptics details but {nelements} elements")

        # as dataframe
        curves = pandas.DataFrame(curves.T, index=lbda, columns=names)
        return cls.from_curvesdf(curves, noptics=noptics)

    @classmethod
    def from_curvesdf(cls, curves, noptics=1, ext="zeros", **kwargs):
        """Create an OpticsThroughput instance from a pandas DataFrame.

        Parameters
        ----------
        curves : pandas.DataFrame
            DataFrame with wavelength (in Angstrom) as index and one column
            per optical element.
        noptics : int, list of int or dict, optional
            Number of optics for each curve. If an int, the same number is
            applied to all elements. If a dict, it must have the same keys
            as the columns of ``curves``. Default is 1.
        ext : str, optional
            Extrapolation method of the interpolators (see
            `~slicersim.iotools.chromatic_interpolator`). Default is
            ``"zeros"``.
        **kwargs
            Additional arguments passed to the constructor (e.g. ``meta``).

        Returns
        -------
        OpticsThroughput
            An OpticsThroughput instance.
        """
        elements = {name_: iotools.chromatic_interpolator(curve_.index, curve_.values,
                                                              ext=ext)
                    for name_, curve_ in curves.T.iterrows()}

        if not isinstance(noptics, dict):
            names = curves.columns
            noptics = dict(zip(names,np.broadcast_to(noptics, np.shape( list(elements.values())))))

        this = cls(elements, noptics=noptics, **kwargs)
        this._input_curves = curves
        return this

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create an OpticsThroughput instance from a configuration dictionary.

        The configuration dictionary must contain a ``"throughput"`` key
        pointing to a file (csv or ecsv) with the throughput curves, see
        `from_filename`. An optional ``"noptics"`` key (default 1) sets the
        number of optics per curve. The full configuration is stored as
        `meta`.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        **kwargs
            Goes to `from_filename`.

        Returns
        -------
        OpticsThroughput
            An OpticsThroughput instance.

        Raises
        ------
        ValueError
            If ``config`` has no ``"throughput"`` entry.
        """
        throughput_file = config.get("throughput", None) # e.g. "throughput.cvs"
        if throughput_file is None:
            raise ValueError("no 'throughput' in input config")

        noptics = config.get("noptics", 1)
        return cls.from_filename(throughput_file, noptics=noptics, meta=config, **kwargs)

    @classmethod
    def from_filename(cls, filename, noptics=1, meta={}, **kwargs):
        """Create an OpticsThroughput instance from a data file.

        A csv file must have a ``wavelength`` column (in Angstrom, used as
        index) and one column per optical element. An ecsv file must have
        ``wavelength`` and ``throughput`` columns (single element named
        ``"spectro"``). The curves are extrapolated outside the tabulated
        range.

        Parameters
        ----------
        filename : str
            Path of the throughput file (.ecsv or .csv). If it does not
            exist as given, it is looked for in the package configuration
            directory (see `~slicersim.iotools.expand_path`).
        noptics : int or dict, optional
            Number of optics associated to each curve. Default is 1.
        meta : dict, optional
            Metadata stored with the instance. Default is ``{}``.
        **kwargs
            Additional arguments passed to `pandas.read_csv` (csv files
            only).

        Returns
        -------
        OpticsThroughput
            An OpticsThroughput instance.
        """
        import os
        if not os.path.isfile(filename): # name of config file most likely.
            filename = iotools.expand_path(filename) # make sure it is fullpath

        if str(filename).endswith(".ecsv"):
            curves_df = cls._read_ecsv(filename, as_dataframe=True) # backward compatibility
        else:
            curves_df = pandas.read_csv(filename, index_col="wavelength", **kwargs)

        options = {"ext": "extrapolate"}
        return cls.from_curvesdf(curves_df, noptics, meta=meta, **options)

    @staticmethod
    def _read_csv(filename):
        """Read a csv file containing throughput data.

        Parameters
        ----------
        filename : str
            Path to the csv file (see `~slicersim.iotools.expand_path`).

        Returns
        -------
        pandas.DataFrame
            Throughput data.
        """
        fullpath = iotools.expand_path(filename)
        return pandas.read_csv(fullpath)

    @staticmethod
    def _read_ecsv(filename, as_dataframe=False):
        """Read an ecsv file containing throughput data.

        Parameters
        ----------
        filename : str
            Path to the ecsv file.
        as_dataframe : bool, optional
            If True, return a pandas DataFrame. If False, return a
            callable interpolator. Default is False.

        Returns
        -------
        pandas.DataFrame or callable
            Throughput data as a DataFrame or an interpolator.
        """
        wname, tname = 'wavelength', 'throughput'
        # load the inout data
        tab = iotools.read_ecsv(filename,
                                colnames=[wname, tname])
        if as_dataframe:
            return pandas.DataFrame({"spectro": tab[tname]}, index=tab[wname].to(u.AA))

        # create the interpolator (wavelengths in Å)
        throughput = iotools.chromatic_interpolator(
            tab[wname].to(u.AA), tab[tname], ext='zeros')

        return throughput

    # -------- #
    #  Generic  #
    # -------- #
    def __call__(self, lbda, **kwargs):
        """Shortcut to `get_throughput`."""
        return self.get_throughput(lbda, **kwargs)

    # -------- #
    #  Methods #
    # -------- #
    def update_curve(self, name, curve, ext='zeros'):
        """Replace the throughput curve of one optical element.

        Parameters
        ----------
        name : str
            Name of the optical element whose curve is replaced. It must
            already be known (see `names`).
        curve : pandas.Series
            New throughput curve, indexed by wavelength in Angstrom.
        ext : str, optional
            How to extrapolate outside the tabulated wavelength range
            (``"extrapolate"``, ``"zeros"``, ``"raise"`` or ``"const"``, see
            `~slicersim.iotools.chromatic_interpolator`). Default is
            ``"zeros"``.

        Raises
        ------
        ValueError
            If ``name`` is not one of the known curves.

        Notes
        -----
        This replaces the curve of an existing element, it cannot add a new
        one. The number of optics associated to ``name`` is left untouched,
        use `update` to change it.
        """
        if name not in self._curves:
            raise ValueError(f"unknown curve: {name=}.")

        elements = iotools.chromatic_interpolator(curve.index, curve.values, ext=ext)
        self._curves |= {name: elements}

    def update(self, **kwargs):
        """Update the number of optics for one or more elements.

        Parameters
        ----------
        **kwargs
            Either ``noptics=value`` (replaces the whole ``noptics`` entry)
            or ``noptics.<name>=value`` to change the number of optics of
            a single element. The django-like format is accepted, e.g.
            ``noptics__mirror=2``.

        Warns
        -----
        UserWarning
            If a key cannot be parsed, refers to an unknown element or is not
            mutable. It is then ignored.
        """
        for key, value in kwargs.items():
            # accept django-like format
            key = key.replace("__", ".")

            # change a specific noptics
            if "." in key:
                noptics_, which = key.split(".")
                if noptics_ != "noptics":
                    warnings.warn(f"cannot parse input format; noptics.`which` expected, {key} given")
                    continue
                if which not in self.names:
                    warnings.warn(f"{which} (from {key}) is not a known name ({self.names})")
                    continue

                # ok, let's change just the one you want.
                self.meta["noptics"][which] = value

            # change it all
            elif key not in self.mutable_parameters:
                warnings.warn(f"unmutable key {key} ; ignored")
            else:
                self.meta[key] = value

    def get_throughput(self, lbda, which=None, per_element=False, incl_noptics=True, ignore=None):
        """Get the total throughput of the optical system.

        Parameters
        ----------
        lbda : array_like
            Wavelength array in Angstrom.
        which : str, optional
            If given, only return the throughput of this element (see
            `get_element_throughput`). Default is None.
        per_element : bool, optional
            If True, return the throughput of each element separately.
            Ignored if ``which`` is given. Default is False.
        incl_noptics : bool, optional
            If True, each curve is raised to the power of its number of
            optics. Default is True.
        ignore : list of str, optional
            Element names to ignore. Not used if ``which`` is given.
            Default is None.

        Returns
        -------
        numpy.ndarray or list of numpy.ndarray
            Total throughput (product over the elements), or list of
            throughputs per element.
        """
        if which is not None:
            return self.get_element_throughput(which, lbda, incl_noptics=incl_noptics)

        curves_per_element = [self.get_element_throughput(name_, lbda, incl_noptics)
                              for name_ in self.names if (ignore is None or name_ not in ignore)]
        if per_element:
            return curves_per_element

        return np.prod(curves_per_element, axis=0)

    def get_element_throughput(self, which, lbda, incl_noptics=True):
        """Get the throughput of a single optical element.

        Parameters
        ----------
        which : str
            Name of the element.
        lbda : array_like
            Wavelength array in Angstrom.
        incl_noptics : bool, optional
            If True, the curve is raised to the power of the element number
            of optics. Default is True.

        Returns
        -------
        numpy.ndarray
            Throughput of the element.

        Raises
        ------
        ValueError
            If the stored curve is an array whose length does not match
            ``lbda``.
        """
        curve = self.curves[which]
        noptics = self.noptics[which]
        if callable(curve):
            curve = curve(lbda)
            curve[(curve<0) & (curve>=-1e-3)] = 0 # interpolation error accepted.

        elif (ncurve:=len(curve)) != (nlbda:=len(lbda)):
            raise ValueError(f"curve dimension {ncurve} doesn't match input lbda's {nlbda}")

        # at this put curve is an array of lbda's dimension.
        if incl_noptics:
            return np.asarray(curve)**noptics

        return np.asarray(curve)

    def show(self, lbda, ax=None): # pragma: no cover
        """Show the throughput of the optical system.

        This method plots the cumulative throughput of each element and the
        total throughput as a function of wavelength.

        Parameters
        ----------
        lbda : array_like
            Wavelength array in Angstrom.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
            Default is None.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        import matplotlib.pyplot as plt

        # data to show
        element_throughputs = self.get_throughput(lbda, per_element=True, incl_noptics=True)
        total_throughputs = self.get_throughput(lbda) # could simply np.prod, but so it test self consistancy.

        # plotting
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 3))
        else:
            fig = ax.figure

        current_throughput = 1
        for name, throughput  in zip(self.names, element_throughputs):
            next_throughput = current_throughput*throughput
            ax.fill_between(lbda, current_throughput, next_throughput,
                            alpha=0.2, label=name)
            current_throughput = next_throughput

        ax.plot(lbda, total_throughputs, color="k")

        ax.legend(ncols=len(self.names), frameon=False, loc="best")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Wavelength", fontsize="large")
        ax.set_ylabel("Throughput", fontsize="large")

        return fig

    # ============= #
    #  Properties   #
    # ============= #
    @property
    def curves(self):
        """Throughput curve (callable of wavelength in Angstrom) per optical element."""
        return self._curves

    @property
    def names(self):
        """Names of the optical elements (dict keys view)."""
        return self.curves.keys()

    @property
    def noptics(self):
        """Number of optics per element (i.e., per curve)."""
        return self.meta["noptics"]

    @property
    def meta(self):
        """Metadata of the instance."""
        return self._meta
