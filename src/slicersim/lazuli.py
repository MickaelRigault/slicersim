"""Lazuli-specific targets and exposure time calculators.

Lazuli is a two-field integral field spectrograph: a narrow field (40 mas
spaxels) and a wide field (80 mas spaxels) observe adjacent sky regions at two
samplings and are imaged side by side on a single detector. The classes here
specialise the generic targets of `~slicersim.target` (e.g.
`LazuliSupernova`, `LazuliKilonova`, `LazuliCalSpec`, `LazuliTarget`) with
the Lazuli instrument configuration and add the handling of the two fields, from
switching between them (`VirtualLazuliTarget.change_spectrograph`) to
projecting both onto one detector image (`VirtualLazuliTarget.to_image`).

The module also exposes two convenience exposure time calculators,
`lazuli_sn_etc` for SN Ia models and `lazuli_etc` for an arbitrary input
spectrum.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from .iotools import get_config
from .simulation import Simulation
from .target import BlackBody, CalSpec, Kilonova, PowerLaw, Supernova, Target

__all__ = [
    "LazuliBlackBody",
    "LazuliCalSpec",
    "LazuliKilonova",
    "LazuliPowerLaw",
    "LazuliSupernova",
    "LazuliTarget",
    # exposure time calculators shortcuts
    "lazuli_etc",
    "lazuli_sn_etc",
]


SPECTROGRAPH_FIELD = {"narrow": {'spatial_shape': [58, 58], 'spatial_scale': 0.04,
                                    'ifu_nelements': 4,
                                    "which_throughput": "narrow",  # limit throughput column entry
                                    },

                        "wide": {'spatial_shape': [58, 58], 'spatial_scale': 0.08,
                                'ifu_nelements': 6,
                                "which_throughput": "wide",  # limit throughput column entry
                                },

                        "hspace": 0.06 # in arcsec like spatial_scale
                        }


def spectroconfig_to_fieldname(spatial_shape, spatial_scale, default = "unknown"):
    """Get the Lazuli field name matching a spectrograph spaxel configuration.

    Parameters
    ----------
    spatial_shape : list or tuple of int
        Spaxel grid shape (ny, nx).
    spatial_scale : float
        Spaxel size in arcsec.
    default : str, optional
        Name returned if no field of ``SPECTROGRAPH_FIELD`` matches the input
        configuration. Default is ``"unknown"``.

    Returns
    -------
    str
        Name of the matching field (``"narrow"`` or ``"wide"``), ``default``
        otherwise.

    See Also
    --------
    VirtualLazuliTarget.get_spectrograph_field : Field of a Lazuli target.
    ~slicersim.spectrograph.Spectrograph.get_fieldname : Field of a spectrograph.

    Examples
    --------
    >>> spectroconfig_to_fieldname([58, 58], 0.08)
    'wide'
    """
    field = default
    for this_field, this_config in SPECTROGRAPH_FIELD.items():
        if not isinstance(this_config, dict): # e.g. hspace
            continue

        if list(this_config["spatial_shape"]) == list(spatial_shape) and \
            this_config["spatial_scale"] == spatial_scale:
            field = this_field
            break

    return field


def lazuli_sn_etc(model, redshift, snr, per_resolution=True,
                phase=0,
                lbda_range=[4000, 6800], frame='rest',
                statistic=np.nanmean,
                max_group=None, nmd=None, time_details=False,
                **kwargs):
    """Compute the exposure time needed to reach a given SNR on a SN Ia.

    This builds a `LazuliSupernova` with the requested model and parameters,
    sets the detector read-out mode, then finds the read-out configuration
    (MACC mode and number of ramps) reaching the target signal-to-noise ratio
    (see `~slicersim.target.VirtualTarget.setup_to_snr`) and returns the
    corresponding exposure time.

    Parameters
    ----------
    model : str
        SN Ia model, the model parameters being given as ``**kwargs``:

        - ``"salt"``: SALT2-extended, parameters ``x1``, ``c`` (and ``MBmax``).
          Any sncosmo SALT source name (e.g. ``"salt3"``) also works.
        - ``"twin"``: Twins-Embedding, parameters ``xi1``, ``xi2``, ``xi3``
          and ``color``.
    redshift : float
        Redshift of the supernova.
    snr : float
        Target signal-to-noise ratio.
    per_resolution : bool, optional
        If True, ``snr`` is given per spectral resolution element, otherwise
        per wavelength bin. Default is True.
    phase : float, optional
        Rest-frame phase of the supernova in days with respect to maximum
        light. Default is 0.
    lbda_range : list of float, optional
        Wavelength range ``[min, max]`` in Angstrom over which the SNR is
        computed. Default is ``[4000, 6800]``.
    frame : str, optional
        Reference frame of ``lbda_range``: ``"rest"`` (rest-frame) or
        ``"obs"`` (observer-frame). Default is ``"rest"``.
    statistic : callable, optional
        Function used to summarise the SNR over ``lbda_range``.
        Default is `numpy.nanmean`.
    max_group : int, optional
        Maximum number of groups in a single ramp. If None, the current
        instrument configuration is kept. Default is None.
    nmd : tuple of int, optional
        Detector MACC mode ``(n_groups, n_frames_per_group, n_drops)``. If
        None, the current instrument configuration is kept. Default is None.
    time_details : bool, optional
        If True, return the full timing details instead of the total
        exposure time. Default is False.
    **kwargs
        Parameters of the SN model (see ``model``) and other point source
        parameters (e.g. ``position``), passed to `LazuliSupernova`.

    Returns
    -------
    exptime : float or dict
        Total exposure time in seconds (including all ramps), or, if
        ``time_details`` is True, a dict of detector times in seconds
        (``"integration_time"``, ``"exposure_time"``, ``"tframe"``,
        ``"tgroup"`` and ``"total_exptime"``).
    target : LazuliSupernova
        The target, configured to the read-out mode reaching ``snr``.

    See Also
    --------
    lazuli_etc : Same for an arbitrary input spectrum.

    Examples
    --------
    >>> exptime, sn = lazuli_sn_etc("salt", redshift=0.5, snr=15, x1=0, c=0)  # doctest: +SKIP
    >>> readout = sn.get_readout_config()  # {"nmd": ..., "nramps": ...}  # doctest: +SKIP
    """
    # create a target of specified conditions
    target = LazuliSupernova(model=model, redshift=redshift, phase=phase, **kwargs)

    # specify the detector read-out mode | None are ignored.
    target.change_detector(nmd=nmd, max_group=max_group)

    # setup the instrument to the requested signal to noise
    _ = target.setup_to_snr(snr, per_resolution=per_resolution,
                            lbda_range=lbda_range, frame=frame,
                            statistic=statistic, inplace=True)

    # get the exposure time
    exptime = target.get_exposure_time(full=time_details)

    return exptime, target

def lazuli_etc(lbda, flux, snr, per_resolution=True,
                   mag=None, band="bessellb",
                   lbda_range=[4000, 6800], frame='rest',
                   statistic=np.nanmean,
                   max_group=None, nmd=None, time_details=False):
    """Compute the exposure time needed to reach a given SNR on any spectrum.

    This builds a `LazuliTarget` point source from the input spectrum, sets
    the detector read-out mode, then finds the read-out configuration (MACC
    mode and number of ramps) reaching the target signal-to-noise ratio (see
    `~slicersim.target.VirtualTarget.setup_to_snr`) and returns the
    corresponding exposure time.

    Parameters
    ----------
    lbda : array_like
        Wavelengths of the spectrum in Angstrom. Should cover the Lazuli
        bandpass.
    flux : array_like
        Flux of the spectrum, in erg/s/cm^2/A if ``mag`` is None, in
        arbitrary units otherwise.
    snr : float
        Target signal-to-noise ratio.
    per_resolution : bool, optional
        If True, ``snr`` is given per spectral resolution element, otherwise
        per wavelength bin. Default is True.
    mag : float, optional
        If given, ``flux`` is rescaled so that the spectrum has this
        magnitude in ``band``. Default is None.
    band : str, optional
        Bandpass of ``mag`` (any sncosmo bandpass name). Ignored if ``mag``
        is None. Default is ``"bessellb"``.
    lbda_range : list of float, optional
        Wavelength range ``[min, max]`` in Angstrom over which the SNR is
        computed. Default is ``[4000, 6800]``.
    frame : str, optional
        Reference frame of ``lbda_range``: ``"rest"`` (rest-frame) or
        ``"obs"`` (observer-frame). Default is ``"rest"``.
    statistic : callable, optional
        Function used to summarise the SNR over ``lbda_range``.
        Default is `numpy.nanmean`.
    max_group : int, optional
        Maximum number of groups in a single ramp. If None, the current
        instrument configuration is kept. Default is None.
    nmd : tuple of int, optional
        Detector MACC mode ``(n_groups, n_frames_per_group, n_drops)``. If
        None, the current instrument configuration is kept. Default is None.
    time_details : bool, optional
        If True, return the full timing details instead of the total
        exposure time. Default is False.

    Returns
    -------
    exptime : float or dict
        Total exposure time in seconds (including all ramps), or, if
        ``time_details`` is True, a dict of detector times in seconds
        (``"integration_time"``, ``"exposure_time"``, ``"tframe"``,
        ``"tgroup"`` and ``"total_exptime"``).
    target : LazuliTarget
        The target, configured to the read-out mode reaching ``snr``.

    See Also
    --------
    lazuli_sn_etc : Same for SN Ia models.

    Notes
    -----
    The input spectrum has no redshift, so ``frame="rest"`` and
    ``frame="obs"`` select the same wavelengths.

    Examples
    --------
    >>> lbda = np.arange(3_000, 20_000, 0.5)
    >>> flux = np.ones_like(lbda)
    >>> exptime, target = lazuli_etc(lbda, flux, snr=10, mag=22, band="bessellb")  # doctest: +SKIP
    """
    # create a target of specified conditions
    target = LazuliTarget(lbda, flux, mag=mag, band=band)

    # specify the detector read-out mode | None are ignored.
    target.change_detector(nmd=nmd, max_group=max_group)

    # setup the instrument to the requested signal to noise
    _ = target.setup_to_snr(snr, per_resolution=per_resolution,
                    lbda_range=lbda_range, frame=frame,
                    statistic=statistic, inplace=True)

    # get the exposure time
    exptime = target.get_exposure_time(full=time_details)

    return exptime, target

class VirtualLazuliTarget:
    """Mixin adding the Lazuli instrument and its two fields to a target.

    This class is not meant to be used directly: it is combined with one of
    the generic targets of `~slicersim.target` (e.g. `LazuliSupernova`
    inherits from `VirtualLazuliTarget` and `~slicersim.target.Supernova`).
    It sets the Lazuli instrument configuration and adds the handling of the
    narrow and wide fields (`change_spectrograph`, `get_spectrograph_field`,
    `get_cube`, `to_image`).

    Parameters
    ----------
    simulation : slicersim.simulation.Simulation, optional
        The simulation the target wraps. Default is None.
    """
    _INSTRUMENT = 'lazuli_cbe.toml'

    @classmethod
    def from_scene(cls, scene=None, **kwargs):
        """Load the instance from a scene configuration.

        Parameters
        ----------
        scene : dict or str, optional
            Scene configuration (see `~slicersim.scene.get_scene`), in the
            format::

                scene = {"scene": {"pointsource": {},  # PSF
                                   "background": {},   # spatially flat
                                   "host": {}}}        # structured background

            or the name of a scene configuration file. Default is None.
        **kwargs
            Goes to `~slicersim.iotools.get_config`, overriding the
            ``scene`` and the Lazuli ``instrument`` configuration.

        Returns
        -------
        VirtualLazuliTarget
            An instance of the class.
        """
        # create the simulator
        config = get_config( **(cls._DEFAULT_CONFIG | {"instrument": cls._INSTRUMENT, "scene": scene} | kwargs) )
        simulation = Simulation.from_config(config)
        return cls(simulation=simulation)

    # =============== #
    #   Methods       #
    # =============== #
    def change_configuration(self, which):
        """Change the Lazuli instrument configuration.

        The corresponding configuration file (``lazuli_eol.toml`` or
        ``lazuli_cbe.toml``) is read and the simulation updated with it.

        Parameters
        ----------
        which : str
            Name of the configuration: ``"eol"`` (end of life) or ``"cbe"``
            (current best estimate); ``"bol"`` (beginning of life) is an
            alias of ``"cbe"``.

        Returns
        -------
        None
        """
        from .iotools import read_config
        if which == "eol":
            config = read_config("lazuli_eol.toml")

        elif which in ["bol","cbe"]:
            config = read_config("lazuli_cbe.toml")

        return self.simulation.update_from_config(config)

    def change_spectrograph(self, field=None, spatial_shape=None, spatial_scale=None):
        """Change the spectrograph configuration.

        Parameters
        ----------
        field : str, optional
            Lazuli field to use (see ``SPECTROGRAPH_FIELD``):

            - ``"narrow"``: finely sampled field, with 0.04 arcsec spaxels.
            - ``"wide"``: coarser field, with 0.08 arcsec spaxels.

            If None, only ``spatial_shape`` and ``spatial_scale`` are
            applied. Default is None.
        spatial_shape : tuple of int, optional
            Spaxel grid shape (e.g. ``(40, 40)``), overriding that of
            ``field`` if any. Default is None.
        spatial_scale : float or list of float, optional
            Spaxel size in arcsec, overriding that of ``field`` if any. If a
            float, spaxels are square; if a list, ``(x, y)``.
            Default is None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``field`` is not a known Lazuli field.
        """
        # overwrites the current one.
        if field is not None:
            config = SPECTROGRAPH_FIELD.get(field, None)
            if config is None:
                raise ValueError(f"cannot parse the given field {field=} | {SPECTROGRAPH_FIELD} expected")
        else:
            config = {}

        if "ifu_nelements" in config:
            # get all spectrograph optics nelements
            nelements_spectro = self.simulation.get_parameter("optics.nelements")
            # ifu is the 2nd (so [1]) optics.
            nelements_spectro[1] = config.pop("ifu_nelements")
            config["spectrograph__optics__nelements"] = nelements_spectro

        # manual setting if any
        if spatial_shape is not None:
            config["spatial_shape"] = spatial_shape

        if spatial_scale is not None:
            config["spatial_scale"] = spatial_scale

        return self.simulation.update(**config)

    # GETTER
    def get_readout_config(self):
        """Get the current MACC mode and number of ramps.

        Returns
        -------
        dict
            Read-out configuration, with keys:

            - ``"nmd"``: ``(n_groups, n_frames_per_group, n_drops)``.
            - ``"nramps"``: number of ramps, each ramp following the
              ``nmd`` MACC mode.
        """
        return self.get_properties(["nmd", "nramps"])


    def get_spectrograph_field(self):
        """Get the current spectrograph field.

        Returns
        -------
        field : str
            Name of the current field (``"narrow"`` or ``"wide"``), or
            ``"unknown"`` if the spectrograph matches neither (see
            `spectroconfig_to_fieldname`).
        config : dict
            Current ``"spatial_shape"`` and ``"spatial_scale"`` (arcsec) of
            the spectrograph.
        """
        keys_to_check = ["spatial_shape", "spatial_scale"]
        current_config = self.get_properties(keys_to_check)
        field = spectroconfig_to_fieldname(**current_config)
        return field, current_config

    def get_cube(self, which="both", **kwargs):
        """Get the simulated cubes, for one or both Lazuli fields.

        Parameters
        ----------
        which : str, optional
            Which cube should be computed:

            - ``"both"``: both narrow and wide fields.
            - ``"current"``: only the current setup.
            - ``"narrow"``: only the narrow field.
            - ``"wide"``: only the wide field.

            Default is ``"both"``.
        **kwargs
            Goes to `~slicersim.simulation.Simulation.get_cube`.

        Returns
        -------
        tuple
            ``(cube, varcube)``, the cube (ADU) and variance cube (ADU^2) of
            the requested field, or, if ``which="both"``,
            ``((cube_narrow, varcube_narrow), (cube_wide, varcube_wide))``.

        Raises
        ------
        ValueError
            If ``which`` is not ``"current"`` while the spectrograph is set
            up neither as the narrow nor the wide field.
        NotImplementedError
            If the scene contains a host (not yet supported).

        Notes
        -----
        When switching field, the point source position is converted to the
        spaxel units of the other field. The original spectrograph setup and
        position are restored afterwards.
        """
        # just get the current cube.
        if which == "current":
            return self.simulation.get_cube(**kwargs)

        # let's see which configuration you have
        current_field, current_config = self.get_spectrograph_field()
        if current_field not in ["narrow", "wide"]:
            raise ValueError(f"field is neither narrow nor wide ({current_field}). Only which='current' available. {which=}")

        # check if  point source exists
        # Work will be needed here for the host.
        if self.simulation.scene.has_element("pointsource"):
            source_original_position = self.simulation.get_parameter("pointsource__position")
        else:
            source_original_position = None

        if self.simulation.scene.has_element("host"):
            host_original_position = self.simulation.get_parameter("host__position")
        else:
            host_original_position = None

        # => you want the other one, or both.
        #    First, let's get the position in each field.
        if source_original_position is not None:
            source_pos_narrow, source_pos_wide = self._get_field_positions(position=source_original_position)

        if host_original_position is not None:
            host_pos_narrow, host_pos_wide = self._get_field_positions(position=host_original_position)

        # let's loop over fields, update the position
        # to that of interest
        if which in ["both", "narrow"]:
            self.change_spectrograph("narrow")
            if source_original_position is not None:
                self.simulation.update(pointsource__position = source_pos_narrow)

            if host_original_position is not None:
                self.simulation.update(host__position = host_pos_narrow)

            cubes_narrow = self.simulation.get_cube(**kwargs)
        else:
            cubes_narrow = None

        if which in ["both", "wide"]:
            self.change_spectrograph("wide")
            if source_original_position is not None:
                self.simulation.update(pointsource__position = source_pos_wide)

            if host_original_position is not None:
                self.simulation.update(host__position = host_pos_wide)

            cubes_wide = self.simulation.get_cube(**kwargs)
        else:
            cubes_wide = None

        # revert back to original config (could be )
        self.change_spectrograph(field=current_field, **current_config)
        if source_original_position is not None:
            self.simulation.update(pointsource__position = source_original_position)

        if host_original_position is not None:
            self.simulation.update(host__position = host_original_position)

        if which == "both":
            return cubes_narrow, cubes_wide

        elif which == "narrow":
            return cubes_narrow

        elif which == "wide":
            return cubes_wide

    def to_image(self, mapper, cubes=None, **kwargs):
        """Project the narrow and wide field cubes onto a single detector image.

        Both fields are generated, then each is projected onto the slices it
        occupies: the narrow field fills the first ``nslices_narrow`` slices
        and the wide field the ones immediately above. The two projections are
        summed into one image.

        Parameters
        ----------
        mapper : slicersim.mapper.SlicerMapper
            Mapper describing where each slice falls on the detector.
        cubes : tuple of numpy.ndarray, optional
            ``(cube_narrow, cube_wide)`` to project, to avoid regenerating
            them. If None, both are generated with
            ``get_cube(which="both")``. Default is None.
        **kwargs
            Goes to `get_cube`. Ignored if ``cubes`` is given.

        Returns
        -------
        numpy.ndarray
            2D detector image, summing the narrow and wide field
            contributions.

        Notes
        -----
        The slice identifiers are reversed (``[::-1]``) before being handed to
        the mapper, because the mapper numbers slices from the top of the
        detector while the cubes are ordered from the bottom.
        """
        # Generate cube
        if cubes is None:
            (cube_narrow, _), (cube_wide, _) = self.get_cube(which="both", **kwargs)
        else:
            cube_narrow, cube_wide = cubes

        nslices_narrow = cube_narrow.shape[-2]
        nslices_wide = cube_wide.shape[-2]

        # Project to detector
        lbda = self.simulation.spectrograph.lbda
        # [::-1] as top <-> bottom definition inversion: 1 is top for Tim, 1 is first (lower) for me
        img_narrow = mapper.project_slice(np.arange(1, nslices_narrow+1)[::-1], cube_narrow, lbda)
        img_med = mapper.project_slice(np.arange(nslices_narrow+1, nslices_narrow+1+nslices_wide)[::-1], cube_wide, lbda)
        img_ = np.sum([img_med, img_narrow], axis=0)
        return img_

    def get_detector_image(self, mapper, cubes=None, **kwargs):
        """Project the narrow and wide field cubes onto a single detector image.

        .. deprecated:: 1.1.1
            Use `to_image` instead, which this simply forwards to.

        Parameters
        ----------
        mapper : slicersim.mapper.SlicerMapper
            Mapper describing where each slice falls on the detector.
        cubes : tuple of numpy.ndarray, optional
            ``(cube_narrow, cube_wide)`` to project. Default is None.
        **kwargs
            Goes to `to_image`.

        Returns
        -------
        numpy.ndarray
            2D detector image.

        Warns
        -----
        UserWarning
            Always, since this method is deprecated in favour of `to_image`.
        """
        import warnings
        warnings.warn("get_detector_image is deprecated. use to_image() instead")
        return self.to_image(mapper, cubes=cubes, **kwargs)

    def _get_field_positions(self, position=None, field=None):
        """ get the 'position' parameters for each of the two fields (narrow and wide).

        These position are in the respective slicer units, and could directly be used
        using the position= arguments one the corresponding field has been set.
        """
        if position is None:
            position = np.asarray(self.simulation.get_parameter("position"))
        else:
            # make sure it is not a list of tuple and that it's indeed float-kind.
            position = np.asarray( position ).astype("float32")

        if field is None:
            field, _ = self.get_spectrograph_field()

        # information for the fine field
        nx_narrow, ny_narrow = SPECTROGRAPH_FIELD["narrow"]["spatial_shape"]
        nx_wide, ny_wide = SPECTROGRAPH_FIELD["wide"]["spatial_shape"]

        # multiplying factory between mid and large
        narrow_to_wide = SPECTROGRAPH_FIELD["narrow"]["spatial_scale"] / SPECTROGRAPH_FIELD["wide"]["spatial_scale"]

        # these are in unit of respective slice width
        # they share "x=0"
        hspace_in_narrow_spaxel = SPECTROGRAPH_FIELD["hspace"] / SPECTROGRAPH_FIELD["narrow"]["spatial_scale"]
        narrow_bottom = -ny_narrow/2 - hspace_in_narrow_spaxel
        wide_top = +ny_wide/2

        if field == "narrow":
            position_narrow = position
            centroid_narrow_for_med = np.asarray([0,
                wide_top - narrow_bottom * narrow_to_wide])
            offset_from_centroid_for_med = position * narrow_to_wide
            # narrow is on top
            position_wide = centroid_narrow_for_med + offset_from_centroid_for_med

        elif field  == "wide":
            position_wide = position
            centroid_wide_for_narrow = np.asarray([0, narrow_bottom - wide_top / narrow_to_wide])
            offset_from_centroid_for_narrow = position / narrow_to_wide
            # narrow is on top
            position_narrow = centroid_wide_for_narrow + offset_from_centroid_for_narrow

        else:
            raise ValueError(f"in_which should be either narrow or wide ; {field=} given")

        return position_narrow, position_wide

    @staticmethod
    def _get_fieldlayout_(fig=None, left=0.1, bottom=0.1, right=0.9, top=0.9): # pragma: no cover
        """ get matploblib's axes corresponding to the Lazuli layout.

        Parameters
        ----------
        fig: matplotlib.Figure
            the figure you want to axes to be generated into.
            If None, a new figure will be generated.

        Returns
        -------
        fig, (ax_narrow, ax_wide)
        """
        if fig is None:
            fig = plt.figure(figsize=(8,8))

        width = right-left
        height = top-bottom

        width_med = width # full width
        height_med = height*0.66

        axfine = fig.add_axes([left+width/4, bottom+height_med, width_med/2, height_med/2])
        axmed = fig.add_axes([left, bottom, width_med, height_med])
        return fig, (axfine, axmed)

    def show_cube(self, cubes, lbda_range, axes=None,
                ref_field="narrow",
                **kwarg):
        """Show the narrow and wide field cubes, collapsed over a wavelength range.

        Each cube is summed over the wavelength bins within ``lbda_range``,
        and the two resulting images are displayed following the Lazuli
        layout: the narrow field on top of the wide field, separated by the
        gap between the two fields (``SPECTROGRAPH_FIELD["hspace"]``).

        Parameters
        ----------
        cubes : tuple of numpy.ndarray
            ``(cube_narrow, cube_wide)``, the cubes of each field, with
            wavelength as first axis (e.g. the cubes returned by
            ``get_cube(which="both")``, without their variance).
        lbda_range : list of float
            Wavelength range ``[min, max]`` in Angstrom (observer-frame) over
            which the cubes are summed. Bounds are excluded.
        axes : tuple of matplotlib.axes.Axes, optional
            ``(ax_narrow, ax_wide)``, the axes to draw the narrow and wide
            fields into. If None, a new figure is created with the Lazuli
            layout (see ``_get_fieldlayout_``). Default is None.
        ref_field : str, optional
            Field whose image sets the colour normalisation (99.8 percentile
            interval with an asinh stretch) shared by both images:
            ``"narrow"`` or ``"wide"``. Default is ``"narrow"``.
        **kwarg
            Goes to `matplotlib.axes.Axes.imshow`, overriding the default
            display properties (``origin``, ``norm``, ``interpolation``,
            ``aspect`` and ``cmap``).

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the axes.

        Notes
        -----
        The wide field axes is repositioned right below the narrow field
        axes, so its original position is changed even when ``axes`` is
        given.

        Examples
        --------
        >>> (cube_n, _), (cube_w, _) = target.get_cube(which="both")  # doctest: +SKIP
        >>> fig = target.show_cube((cube_n, cube_w), lbda_range=[5000, 6000])  # doctest: +SKIP
        """

        import matplotlib.pyplot as plt
        from matplotlib import colors
        from astropy.visualization import ImageNormalize, AsinhStretch, PercentileInterval

        if axes is None:
            fig = plt.figure(figsize=[8, 7])
            fig, (axn, axw) = self._get_fieldlayout_(fig=fig)
        else:
            axn, axw = axes
            fig = axn.figure


        cube_narrow, cube_wide = cubes
        lbda = self.simulation.spectrograph.lbda
        flag_slice = (lbda>lbda_range[0]) & (lbda<lbda_range[1])

        slice_narrow = cube_narrow[flag_slice].sum(axis=0)
        slice_wide = cube_wide[flag_slice].sum(axis=0)

        norm = ImageNormalize(slice_narrow if ref_field == "narrow" else slice_wide,
                            interval=PercentileInterval(99.8), stretch=AsinhStretch(0.02))

        prop = dict( origin="lower", norm=norm,
                    interpolation="nearest", aspect="equal",
                    cmap="cividis")


        axn.imshow(slice_narrow, **(prop | kwarg))
        axw.imshow(slice_wide, **(prop | kwarg))

        axw.spines[["left", "right", "bottom", "top"]].set_visible(False)
        axn.spines[["left", "right", "bottom", "top"]].set_visible(False)
        #axn.set_zorder(9)

        bbox_n = axn.get_position()
        bbox_w = axw.get_position()
        spacing = 0.06/0.04  *bbox_n.height/58
        print(spacing)
        axw.set_position([bbox_w.x0, bbox_n.y0-bbox_w.height-spacing, bbox_w.width, bbox_w.height])


        yticks_n = np.asarray([1] + list(np.arange(0, 59, 10)[1:]))
        ytick_loc = cube_narrow.shape[1]-yticks_n
        axn.set_yticks(ytick_loc, labels=yticks_n)

        yticks_n = np.asarray([cube_narrow.shape[1]+1] + list(np.arange(70, 59+cube_narrow.shape[1], 10)))
        axw.set_yticks(ytick_loc, labels=yticks_n)

        [ax_.tick_params(labelsize="small",
                        color="0.5",
                        labelcolor="0.5") for ax_ in fig.axes]
        [ax_.set_xticks([]) for ax_ in fig.axes]
        axn.set_ylabel("slice index", color="0.5")
        axn.text(1.05, 0.5, "narrow field", transform=axn.transAxes, rotation=-90,
                va="center", ha="left")
        axn.text(1.0, 0.5, r"$2.3^{\prime\prime} \times 4.6^{\prime\prime}$", transform=axn.transAxes, rotation=-90,
                va="center", ha="left", fontsize="small")

        axw.text(1.025, 0.5, "wide field", transform=axw.transAxes, rotation=-90,
                va="center", ha="left")
        axw.text(1.0, 0.5, r"$4.6^{\prime\prime} \times 9.2^{\prime\prime}$", transform=axw.transAxes, rotation=-90,
                va="center", ha="left", fontsize="small")

        return fig

# ============ #
#  Specifics   #
# ============ #
# Supernovae
class LazuliSupernova( VirtualLazuliTarget, Supernova ):
    """A Type Ia supernova observed with Lazuli.

    The supernova is a point source on top of the zodiacal background,
    observed by default in the narrow field.

    Parameters
    ----------
    model : str, optional
        The supernova model:

        - ``"salt"``: SALT2-extended, parameters ``x1``, ``c`` and ``MBmax``
          [default]. Any sncosmo SALT source name (e.g. ``"salt3"``) also
          works.
        - ``"twin"``: Twins-Embedding, parameters ``xi1``, ``xi2``, ``xi3``
          and ``color``.
    **kwargs
        Point source parameters, e.g. ``redshift``, ``phase`` (rest-frame
        days from maximum light), ``position`` (spaxels) and the model
        parameters above. ``instrument`` and ``background`` can also be
        given (see `~slicersim.scene.get_scene`).

    See Also
    --------
    lazuli_sn_etc : Exposure time to reach a given SNR, in one call.

    Examples
    --------
    >>> sn = LazuliSupernova(redshift=1.0, x1=0, c=0.2, phase=1.5)  # doctest: +SKIP
    >>> config, snr = sn.setup_to_snr(20)  # doctest: +SKIP
    >>> exptime = sn.get_exposure_time()  # in seconds  # doctest: +SKIP
    >>> lbda, flux, variance = sn.get_spectrum(unit="flambda")  # doctest: +SKIP
    """

# Kilonovae
class LazuliKilonova( VirtualLazuliTarget, Kilonova ):
    """A kilonova observed with Lazuli.

    The spectrum comes from the POSSIS radiative transfer models (Bulla 2019,
    2023) for a given viewing angle, see
    `~slicersim.scene.sources.kilonova.get_kilonova_flux`.

    Parameters
    ----------
    model : str, optional
        ``"bulla23"`` [default] or ``"bulla19"``.
    **kwargs
        Point source parameters, mainly ``redshift``, ``phase`` (days since
        merger), ``theta`` (viewing angle in degrees), ``magabs`` or
        ``magobs`` (peak magnitude in ``band``) and ``position`` (spaxels).

    Examples
    --------
    >>> kn = LazuliKilonova(redshift=0.05, phase=1.4, theta=30)  # doctest: +SKIP
    """

# CalSpec Stars
class LazuliCalSpec( VirtualLazuliTarget, CalSpec  ):
    """An HST CalSpec spectrophotometric standard star observed with Lazuli.

    Parameters
    ----------
    name : str
        Name (or short name) of the CalSpec star, e.g. ``"bd_17"`` or
        ``"gd_71"``.
    **kwargs
        Point source parameters, e.g. ``position`` (spaxels).

    Examples
    --------
    >>> star = LazuliCalSpec("bd_17")  # doctest: +SKIP
    """

# Blackbody point source
class LazuliBlackBody( VirtualLazuliTarget, BlackBody ):
    """A black body point source observed with Lazuli.

    Parameters
    ----------
    temperature : float
        Black body temperature in Kelvin.
    **kwargs
        Point source parameters: ``mag`` and ``band`` (normalisation, e.g.
        ``band="sdssr"``), ``magsys`` and ``position`` (spaxels).

    Examples
    --------
    >>> bb = LazuliBlackBody(temperature=5_000, mag=20, band="sdssr")  # doctest: +SKIP
    """


# Power-law point source
class LazuliPowerLaw( VirtualLazuliTarget, PowerLaw ):
    """A power-law point source observed with Lazuli.

    Parameters
    ----------
    alpha : float
        Power-law index, see `~slicersim.scene.sources.generic.get_powerlaw_flux`.
    **kwargs
        Point source parameters: ``mag`` and ``band`` (normalisation),
        ``lambda_ref`` (Angstrom) and ``position`` (spaxels).

    Examples
    --------
    >>> pl = LazuliPowerLaw(alpha=-2, mag=20, band="sdssr")  # doctest: +SKIP
    """


# Generic object
class LazuliTarget( VirtualLazuliTarget, Target  ):
    """Any spectrum observed with Lazuli, as a point source.

    Parameters
    ----------
    lbda : array_like
        Wavelengths in Angstrom. Should cover the Lazuli bandpass.
    flux : array_like
        Flux in erg/s/cm^2/A, or in arbitrary units when ``mag`` is given.
    mag : float, optional
        If given, ``flux`` is rescaled to this magnitude in ``band``.
        Default is None.
    band : str, optional
        Bandpass of ``mag`` (any sncosmo bandpass name).
        Default is ``"bessellb"``.
    **kwargs
        Other point source parameters, e.g. ``position`` (spaxels).

    See Also
    --------
    lazuli_etc : Exposure time to reach a given SNR, in one call.

    Examples
    --------
    >>> lbda = np.arange(3_000, 20_000, 0.5)  # doctest: +SKIP
    >>> target = LazuliTarget(lbda, np.ones_like(lbda), mag=20, band="lsstg")  # doctest: +SKIP
    """

# Generic object
class LazuliFlat( VirtualLazuliTarget, Target  ):
    """A spatially uniform source observed with Lazuli.

    The scene contains neither point source nor host: the input spectrum is
    used as a spatially flat background filling the field.

    .. warning::
        Instantiation currently fails with a `TypeError`, as the parent
        initialisation resolves to `~slicersim.target.Target`, which expects
        ``lbda`` and ``flux``. See `Lazuli3DFlat` for flat-field
        calibration exposures.

    Parameters
    ----------
    lbda : array_like
        Wavelengths in Angstrom of the reference spectrum.
    flux : array_like
        Flux of the reference spectrum, in erg/s/cm^2/A per spaxel.
    amplitude : float, optional
        Scaling factor applied to ``flux``. It is a mutable parameter of the
        scene, so the flat level can be changed afterwards without rebuilding
        the target. Default is 1.
    **kwargs
        Goes to `~slicersim.simulation.Simulation.from_config`.
    """
    def __init__(self, lbda, flux, amplitude=1,
                 **kwargs):
        """Initialize the LazuliFlat.

        Parameters
        ----------
        lbda : array_like
            Wavelengths in Angstrom of the reference spectrum.
        flux : array_like
            Flux of the reference spectrum.
        amplitude : float, optional
            Scaling factor applied to ``flux``. Default is 1.
        **kwargs
            Goes to `~slicersim.simulation.Simulation.from_config`.
        """

        def model_flux(wave, amplitude):
            """Interpolate the reference flat spectrum onto a wavelength array.

            Parameters
            ----------
            wave : array_like
                Wavelength array in Angstrom.
            amplitude : float
                Scaling factor applied to the interpolated flux.

            Returns
            -------
            numpy.ndarray
                The scaled flat flux. Wavelengths outside the range of the
                reference spectrum are NaN.
            """
            return np.interp(wave, lbda, flux, left=np.nan, right=np.nan) * amplitude

        # build the scene config
        scene = {"scene":{"pointsource": None,
                           "host": None,
                           "background": {"name": "generic",
                                          "func": model_flux,
                                          "amplitude": amplitude},
                         }
                 }

        config = get_config(scene=scene, instrument=self._INSTRUMENT)
        simulation = Simulation.from_config(config, **kwargs)

        super().__init__(simulation=simulation)

from .calibration import Flat3DCalibration


class Lazuli3DFlat(VirtualLazuliTarget, Flat3DCalibration):
    """Lazuli 3D flat-field calibration target.

    Combines the Lazuli instrument configuration of `VirtualLazuliTarget` with
    the spatially uniform scene of
    `~slicersim.calibration.Flat3DCalibration`, to simulate the flat-field
    exposures used to calibrate the wavelength response of the two fields.

    The calibration source is the on-board QTH lamp, approximated by a
    blackbody (see `from_qth`), optionally seen through the Fabry-Perot etalon
    that turns its continuum into a comb of spectral features used for
    wavelength calibration (see `from_febryperot`).

    It is normally built with `from_qth` or `from_febryperot`. The
    spectrograph is set up in the narrow field by default; use
    `~VirtualLazuliTarget.change_spectrograph` to switch field.

    Parameters
    ----------
    simulation : slicersim.simulation.Simulation, optional
        The simulation the target wraps. Default is None.

    Examples
    --------
    >>> flat = Lazuli3DFlat.from_qth()  # doctest: +SKIP
    >>> cube, varcube = flat.get_cube(which="current")  # doctest: +SKIP
    """
    _QTH_TEMPERATURE = 3000
    @classmethod
    def from_qth(cls, temperature=None, mag=12, band='sdssr',
                        **kwargs):
        """Initialize the flat as observing the QTH lamp.

        The lamp is modelled as a blackbody (see
        `~slicersim.calibration.Flat3DCalibration.from_blackbody`).

        Parameters
        ----------
        temperature : float, optional
            Temperature of the lamp in Kelvin. If None, the class default
            ``_QTH_TEMPERATURE`` (3000 K) is used. Default is None.
        mag : float, optional
            Magnitude the lamp spectrum is normalised to. Default is 12.
        band : str, optional
            Name of the bandpass used for the normalisation (any sncosmo
            bandpass name). Default is ``"sdssr"``.
        **kwargs
            Goes to
            `~slicersim.calibration.Flat3DCalibration.from_blackbody`
            (e.g. ``magsys``, ``lbda_range``, ``lbda_bin``).

        Returns
        -------
        Lazuli3DFlat
            An instance of the class.

        See Also
        --------
        from_febryperot : The same lamp seen through the Fabry-Perot etalon.
        """
        if temperature is None:
            temperature = cls._QTH_TEMPERATURE

        return cls.from_blackbody(temperature=temperature, mag=mag, band=band, **kwargs)

    @classmethod
    def from_febryperot(cls, fp_throughput="lazuli_fp_transmission.csv",
                            temperature=None, mag=12, band='sdssr', **kwargs):
        """Initialize the flat as observing the QTH lamp through the Fabry-Perot.

        Parameters
        ----------
        fp_throughput : str or callable, optional
            Fabry-Perot transmission. If a str, the path to a csv file with
            the wavelength in Angstrom as index and a ``"transmission"``
            column; it is looked up in the package config directory when it
            is not found locally. If a callable, the transmission as a
            function of wavelength in Angstrom.
            Default is ``"lazuli_fp_transmission.csv"``.
        temperature : float, optional
            Temperature of the lamp in Kelvin. Currently ignored: the class
            default ``_QTH_TEMPERATURE`` (3000 K) is always used.
            Default is None.
        mag : float, optional
            Magnitude the lamp spectrum is normalised to, before the
            Fabry-Perot transmission is applied. Default is 12.
        band : str, optional
            Name of the bandpass used for the normalisation (any sncosmo
            bandpass name). Default is ``"sdssr"``.
        **kwargs
            Goes to
            `~slicersim.calibration.Flat3DCalibration.from_febryperot`.

        Returns
        -------
        Lazuli3DFlat
            An instance of the class.

        Warns
        -----
        UserWarning
            When ``fp_throughput`` is read from a file, flagging the temporary
            extrapolation patch that fills wavelengths outside the tabulated
            range with the first tabulated transmission value.

        See Also
        --------
        from_qth : The same lamp, without the Fabry-Perot etalon.
        """
        if type(fp_throughput) is str:
            if not os.path.isfile(fp_throughput):
                from .iotools import expand_path
                fp_throughput = expand_path(fp_throughput)

            import warnings

            import pandas
            from scipy import interpolate
            fp_throughput = pandas.read_csv(fp_throughput, index_col=0)
            warnings.warn("TMP extraploation patch to be removed.")
            fp_throughput = interpolate.interp1d(fp_throughput.index, fp_throughput["transmission"], bounds_error=False,
                                                 fill_value=fp_throughput["transmission"].iloc[0]
            )

        if temperature is None:
            temperature = cls._QTH_TEMPERATURE

        return super().from_febryperot(temperature=cls._QTH_TEMPERATURE, fp_throughput=fp_throughput,
                                        mag=mag, band=band, **kwargs)
