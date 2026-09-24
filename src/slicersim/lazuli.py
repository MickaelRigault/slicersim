""" This module handles the Lazuli-specific targets and exposure time calculators.

Lazuli is a two-field integral field spectrograph: a narrow field (40 mas
spaxels) and a wide field (80 mas spaxels) observe the same sky region at two
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

from .iotools import get_config
from .simulation import Simulation
from .target import CalSpec, Kilonova, Supernova, Target

__all__ = [
    "LazuliBlackBody",
    "LazuliCalSpec",
    "LazuliKilonova",
    "LazuliSupernova",
    "LazuliTarget",
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



def lazuli_sn_etc(model, redshift, snr, per_resolution=True,
                phase=0,
                lbda_range=[4000, 6800], frame='rest',
                statistic=np.nanmean,
                max_group=None, nmd=None, time_details=False,
                **kwargs):
    """Calculate the exposure time to achieve a specified Signal-to-Noise Ratio (SNR).

    This function creates a Supernova with a specified model, configures the
    detector read-out mode needed to achieve the desired SNR,
    and calculates the exposure time .

    Parameters
    ----------
    model : str
        The kind of supernova requested. Specify parameters with kwargs.

        - salt: SN Ia - parameters: x1, c
        - twin: SN Ia - parameters: xi1, xi2, xi3, color
    redshift : float
        The redshift of the supernova.
    snr : float
        The target Signal-to-Noise Ratio to achieve.
    per_resolution : bool, optional
        If True, the SNR is calculated per resolution element. Default is True.
    phase : float, optional
        The phase (with respect to manimum light) of the supernova. Default is 0.
    lbda_range : list of float, optional
        The wavelength range in Angstroms over which to calculate the SNR.
        Default is [4000, 6800].
    frame : str, optional
        The reference frame for lbda_range. Options are 'rest' (rest-frame) or
        'obs' (observer-frame). Default is 'rest'.
    statistic : function, optional
        The statistical function to use for calculating the SNR.
        Default is `numpy.nanmean`.
    max_group : int, optional
        Specify the maximum number of groups in a single ramp for the detector
        read-out mode. If None, default configuration parameters are used.
        Default is None.
    nmd : int, optional
        Specify the detector MACC mode (n, m, d).
        n: number of groups
        m: number of frames per group
        d: number of drops between groups
        Default is None.
    time_details : bool, optional
        If True, returns detailed exposure time information. Default is False.
    **kwargs
        Parameters of the SN model (see `model`).

    Returns
    -------
    exptime : float or dict
        The calculated exposure time required to achieve the target SNR.
        (see `time_details`).
    target : LazuliTarget
        The configured target object with the specified conditions.
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
    """Calculate the exposure time to achieve a specified Signal-to-Noise Ratio (SNR).

    This function creates a target with specified spectral conditions, configures
    the detector read-out mode, and calculates the exposure time needed to
    achieve the desired SNR.

    Parameters
    ----------
    lbda : array_like
        The wavelength of the spectrum in Angstroms.
    flux : array_like
        The flux values of the spectrum. See `mag` for the unit.
    snr : float
        The target Signal-to-Noise Ratio to achieve.
    per_resolution : bool, optional
        If True, the SNR is calculated per resolution element. Default is False.
    mag : float, optional
        Specify the desired magnitude of the target. If None, the input flux is
        assumed to be in erg/s/cm2/A. If given, the input flux will be
        multiplied to reach the desired magnitude in the given band (see `band`).
        Default is None.
    band : str, optional
        The photometric band for the magnitude. Ignored if `mag` is None.
        Default is "bessellb".
    lbda_range : list of float, optional
        The wavelength range in Angstroms over which to calculate the SNR.
        Default is [4000, 6800].
    frame : str, optional
        The reference frame for `lbda_range`. Options are 'rest' (rest-frame) or
        'obs' (observer-frame). Default is 'rest'.
    statistic : function, optional
        The statistical function to use for calculating the SNR.
        Default is `numpy.nanmean`.
    max_group : int, optional
        Specify the maximum number of groups in a single ramp for the detector
        read-out mode. If None, default configuration parameters are used.
        Default is None.
    nmd : int, optional
        Specify the detector MACC mode (n, m, d).
        n: number of groups
        m: number of frames per group
        d: number of drops between groups
        Default is None.
    time_details : bool, optional
        If True, returns detailed exposure time information. Default is False.

    Returns
    -------
    exptime : float or dict
        The calculated exposure time required to achieve the target SNR.
        (see `time_details`).
    target : LazuliTarget
        The configured target object with the specified conditions.
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
    """A virtual class to build Lazuli Target (see child classes).

    This class provides a generic interface to the `slicersim.Simulation`
    object. It is not intended to be used directly, but rather to be inherited
    by other classes that define specific targets.

    Parameters
    ----------
    simulation : slicersim.Simulation, optional
        The simulation object. Default is None.

    """
    _INSTRUMENT = 'lazuli_cbe.toml'

    def __init__(self, simulation=None, field="narrow"):
        """Initialize the VirtualLazuliTarget.

        Parameters
        ----------
        simulation : slicersim.Simulation, optional
            The simulation object. Default is None.
        field : str, optional
            Lazuli field the spectrograph is configured for, "narrow" or
            "wide". If None, the spectrograph is left as the simulation
            defines it. Default is "narrow".
        """
        # set it.
        self._simulation = simulation
        if field is not None:
            self.change_spectrograph(field)

    @classmethod
    def from_scene(cls, scene=None, **kwargs):
        """Load the instance from a scene configuration.

        Parameters
        ----------
        scene : dict, optional
            Scene configuration with the given format (see
            `slicersim.scene.get_scene`)::

                scene = {"scene": {"pointsource": {},  # PSF
                                   "background": {},   # spatially flat
                                   "host": {}}}        # structured background

            Default is None.
        **kwargs
            Goes to `iotools.get_config()` and updates the configuration.

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
        """ change the intrumental configuration

        This will look for lazuli_{which}.csv and update the current configuration with that one.

        Parameters
        ----------
        which: str
            name of the configuration.

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
            Field mode to use:

            - "narrow": well-spatially sampled grid (~[58/2, 58] with 40mas spaxels)
            - "wide": coarser grid field (~[58/2, 58] with 80mas spaxels)

            Default is None.
        spatial_shape : tuple of float, optional
            Manually set the grid shape (e.g., (40, 40)).
            This is on top of `field` if any. Default is None.
        spatial_scale : float or list, optional
            Manually set spaxel size in arcsec.
            If float, this is assumed to be squared. If list, (x, y).
            This is on top of `field` if any. Default is None.
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
        """Get the current MACC (nmd) mode and the number of ramps.

        Returns
        -------
        dict
            Readout configuration:

            - nmd: (ngroup, nframe_per_group, ndrop)
            - nramps: number of ramps (1-ramp = 1-nmd)
        """
        return self.get_properties(["nmd", "nramps"])

    def get_spectrograph_field(self):
        """Get the current spectrograph field configuration.


        Returns
        -------
        str
            The name of the field mode if it exists, otherwise "manual".
        dict
            The configuration of the field.

        """
        keys_to_check = ["spatial_shape", "spatial_scale"]
        current_config = self.get_properties(keys_to_check)
        field = "manual"
        for this_field, this_config in SPECTROGRAPH_FIELD.items():
            field_config = {k: this_config.get(k, None) for k in keys_to_check}
            if field_config == current_config:
                field = this_field
                break

        return field, current_config

    def get_cube(self, which="both", **kwargs):
        """Get the simulated cubes, for one or both Lazuli fields.

        Parameters
        ----------
        which : str, optional
            Which cube should be computed:

            - "both": both narrow and wide field.
            - "current": only the current setup.
            - "narrow": only the narrow field.
            - "wide": only the wide field.

            Default is "both".
        **kwargs
            Goes to `simulation.Simulation.get_cube`.

        Returns
        -------
        tuple
            ``(cube, varcube)``, the cube and variance cube of the requested
            field, or, if ``which="both"``,
            ``((cube_narrow, varcube_narrow), (cube_wide, varcube_wide))``.

        Raises
        ------
        ValueError
            If `which` is not "current" while the spectrograph is set up
            neither as the narrow nor the wide field.
        NotImplementedError
            If the scene contains a host (not yet supported).
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
            original_position = self.simulation.get_parameter("position")
        else:
            original_position = None

        if self.simulation.scene.has_element("host"):
            raise NotImplementedError("simulation with 'host' component is not yet supported")

        # => you want the other one, or both.
        #    First, let's get the position in each field.
        if original_position is not None:
            pos_narrow, pos_wide = self._get_field_positions()

        # let's loop over fields, update the position
        # to that of interest
        if which in ["both", "narrow"]:
            self.change_spectrograph("narrow")
            if original_position is not None:
                self.simulation.update(position = pos_narrow)

            cubes_narrow = self.simulation.get_cube(**kwargs)
        else:
            cubes_narrow = None

        if which in ["both", "wide"]:
            self.change_spectrograph("wide")
            if original_position is not None:
                self.simulation.update(position = pos_wide)
            cubes_wide = self.simulation.get_cube(**kwargs)
        else:
            cubes_wide = None

        # revert back to original config (could be )
        self.change_spectrograph(field=current_field, **current_config)
        if original_position is not None:
            self.simulation.update(position = original_position)

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
        cubes : tuple, optional
            ``(cube_narrow, cube_wide)`` to project, to avoid regenerating
            them. If None, both are generated with
            ``get_cube(which="both")``. Default is None.
        **kwargs
            Goes to `get_cube`.

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

        .. deprecated::
            Use `to_image` instead, which this simply forwards to.

        Parameters
        ----------
        mapper : slicersim.mapper.SlicerMapper
            Mapper describing where each slice falls on the detector.
        cubes : tuple, optional
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
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8,8))

        width = right-left
        height = top-bottom

        width_med = width # full width
        height_med = height*0.66

        axfine = fig.add_axes([left+width/4, bottom+height_med, width_med/2, height_med/2])
        axmed = fig.add_axes([left, bottom, width_med, height_med])
        return fig, (axfine, axmed)

# ============ #
#  Specifics   #
# ============ #
# Supernovae
class LazuliSupernova( VirtualLazuliTarget, Supernova ):
    """Lazuli class for Supernovae.

    Parameters
    ----------
    model : str, optional
        The supernova model to use. Default is "salt".
    **kwargs
        Goes to `scene.get_scene()`.

    See Also
    --------
    LazuliKilonova : The same, for kilonovae.
    """
    def __init__(self, model="salt", **kwargs):
        """Initialize the LazuliSupernova.

        Parameters
        ----------
        model : str, optional
            The supernova model to use (see `scene.get_scene`, it is passed
            as ``source="snia-{model}"``):

            - "salt": SALT2-extended (parameters: x1, c, MBmax)
            - any sncosmo salt source name (e.g. "salt3")
            - "twin": Twins-Embedding

            Default is "salt".
        **kwargs
            Goes to `scene.get_scene()`, i.e. to the point source
            configuration (e.g. redshift, phase, position, x1, c).
        """
        from .scene import get_scene
        scene = get_scene(source=f"snia-{model}", **kwargs)
        config = get_config( **( self._DEFAULT_CONFIG | {"instrument": self._INSTRUMENT} | {"scene": scene}) )
        simulation = Simulation.from_config(config)

        super().__init__(simulation=simulation)

# Kilonovae
class LazuliKilonova( VirtualLazuliTarget, Kilonova ):
    """Lazuli class for Kilonovae.

    The kilonova spectrum is computed from POSSIS radiative transfer
    models (Bulla 2019, 2023; see `scene.sources.kilonova`), for a given
    viewing angle `theta`, and normalized either to a peak absolute magnitude
    (`magabs`) or to a peak observed magnitude (`magobs`).

    Parameters
    ----------
    model : str, optional
        The kilonova model to use:

        - "bulla23": POSSIS grid from Bulla (2023) [default]
        - "bulla19": POSSIS model from Bulla (2019)
    **kwargs
        Goes to `scene.get_scene()`.

    See Also
    --------
    LazuliSupernova : The same, for supernovae.
    slicersim.scene.sources.kilonova.get_kilonova_flux : The spectral model.

    Examples
    --------
    >>> import slicersim
    >>> target = slicersim.LazuliKilonova(redshift=0.1, phase=1.4, theta=30)
    >>> _ = target.setup_to_snr(10, lbda_range=[6000, 9000], frame="obs")
    >>> lbda, flux, variance = target.get_spectrum(unit="flambda")
    """
    def __init__(self, model="bulla23", **kwargs):
        """Initialize the LazuliKilonova.

        Parameters
        ----------
        model : str, optional
            The kilonova model to use (it is passed to `scene.get_scene`
            as ``source="kilonova-{model}"``):

            - "bulla23": POSSIS grid from Bulla (2023) [default]
            - "bulla19": POSSIS model from Bulla (2019)
        **kwargs
            Goes to `scene.get_scene()`, i.e. to the point source
            configuration. Main parameters are (see
            `scene.sources.kilonova.get_kilonova_flux`):

            - redshift: redshift of the kilonova (default 0.2)
            - phase: days since merger (default 1.4)
            - theta: viewing angle in degrees (default 0, i.e. pole-on)
            - magabs: peak absolute magnitude in `band` (default -15.8)
            - magobs: peak observed magnitude in `band` (overrides `magabs`)
            - band: bandpass for the normalization (default "sdssr")
            - position: position in the IFU in spaxels (default [1, 0.5])
        """
        from .scene import get_scene
        scene = get_scene(source=f"kilonova-{model}", **kwargs)
        config = get_config( **( self._DEFAULT_CONFIG | {"instrument": self._INSTRUMENT} | {"scene": scene}) )
        simulation = Simulation.from_config(config)

        super().__init__(simulation=simulation)

# Blackbody point source
class LazuliBlackBody( VirtualLazuliTarget, Target ):
    """Lazuli class for blackbody point sources.

    The blackbody spectrum is generated by `scene.sources.blackbody.get_blackbody_flux`
    (based on `astropy.modeling.models.BlackBody`) and normalized to the requested
    magnitude in the given band.

    Parameters
    ----------
    temperature : float, optional
        Temperature of the blackbody in Kelvin. Default is 6000.
    mag : float, optional
        Target magnitude in the given band. Default is 20.
    band : str, optional
        Name of the bandpass (from sncosmo). Default is "sdssr".
    magsys : str, optional
        Name of the magnitude system (see sncosmo). Default is "ab".
    position : list, optional
        Position in the MLA in spaxels. Default is [1, 0.5].
    background : str or dict, optional
        Background to use. Default is "zodi".
    **kwargs
        Goes to `simulation.Simulation.from_config()`.

    """
    def __init__(self, temperature=6000, mag=20,
                     band="sdssr", magsys="ab",
                     position=[1, 0.5], background="zodi",
                     **kwargs):
        """Initialize the LazuliBlackBody.

        Parameters
        ----------
        temperature : float, optional
            Temperature of the blackbody in Kelvin. Default is 6000.
        mag : float, optional
            Target magnitude in the given band. Default is 20.
        band : str, optional
            Name of the bandpass (from sncosmo). Default is "sdssr".
        magsys : str, optional
            Name of the magnitude system (see sncosmo). Default is "ab".
        position : list, optional
            Position in the MLA in spaxels. Default is [1, 0.5].
        background : str or dict, optional
            Background to use. Default is "zodi".
        **kwargs
            Goes to `simulation.Simulation.from_config()`.
        """
        # build the scene config | background (str or dict) is merged in by get_config
        scene = {"scene": {"pointsource": {"name": "blackbody",
                                           "source": "blackbody",
                                           "temperature": temperature,
                                           "mag": mag,
                                           "band": band,
                                           "magsys": magsys,
                                           "position": position},
                           "host": None,
                          }}

        config = get_config(scene=[scene, background], instrument=self._INSTRUMENT)
        simulation = Simulation.from_config(config, **kwargs)

        super().__init__(simulation=simulation)

# CalSpec Stars
class LazuliCalSpec( VirtualLazuliTarget, CalSpec  ):
    """Lazuli class for CalSpec stars.

    Parameters
    ----------
    name : str
        Name of the CalSpec star.
    background : str, optional
        Background to use. Default is "zodi".
    **kwargs
        Goes to `simulation.Simulation.from_source()`.

    """
    def __init__(self, name, background="zodi",
                 **kwargs):
        """Initialize the LazuliCalSpec.

        Parameters
        ----------
        name : str
            Name of the CalSpec star.
        background : str, optional
            Background to use. Default is "zodi".
        **kwargs
            Goes to `simulation.Simulation.from_source()`.
        """
        lbda, flux, _ = self._SOURCES.get_spectrum(name)
        simulation = Simulation.from_source(lbda, flux, background=background,
                                            instrument=self._INSTRUMENT,
                                            **kwargs)
        super().__init__(simulation=simulation)

    @classmethod
    def from_name(cls, name, **kwargs):
        """Build a `LazuliCalSpec` from the name of the star.

        Parameters
        ----------
        name : str
            Name of the CalSpec star.
        **kwargs
            Goes to `simulation.Simulation.from_source()`.

        Returns
        -------
        LazuliCalSpec
            An instance of the class.

        """
        # this is actually a wrapper of the init
        return cls(name, **kwargs)

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def source_names(self):
        """List of available CalSpec sources."""
        return self._SOURCES.source.index.values.astype(str)

# Generic object
class LazuliTarget( VirtualLazuliTarget, Target  ):
    """Lazuli class for generic targets.

    Parameters
    ----------
    lbda : array_like
        Wavelength array.
    flux : array_like
        Flux array.
    mag : float, optional
        Magnitude of the target. Default is None.
    band : str, optional
        Photometric band for the magnitude. Default is "bessellb".
    background : str, optional
        Background to use. Default is "zodi".
    **kwargs
        Goes to `simulation.Simulation.from_source()`.

    """
    def __init__(self, lbda, flux, mag=None, band="bessellb",
                     background="zodi",
                 **kwargs):
        """Initialize the LazuliTarget.

        Parameters
        ----------
        lbda : array_like
            Wavelength array.
        flux : array_like
            Flux array.
        mag : float, optional
            Magnitude of the target. Default is None.
        band : str, optional
            Photometric band for the magnitude. Default is "bessellb".
        background : str, optional
            Background to use. Default is "zodi".
        **kwargs
            Goes to `simulation.Simulation.from_source()`.
        """
        simulation = Simulation.from_source(lbda, flux, background=background,
                                                mag=mag, band=band,
                                                **kwargs)
        super().__init__(simulation=simulation)

# Generic object
class LazuliFlat( VirtualLazuliTarget, Target  ):
    """Lazuli class for flat sources (no pointsource)

    Parameters
    ----------
    lbda : array_like
        Wavelength array.
    flux : array_like
        Flux array.
    **kwargs
        Goes to `simulation.Simulation.from_source()`.

    """
    def __init__(self, lbda, flux, amplitude=1,
                 **kwargs):
        """Initialize the LazuliTarget.

        Parameters
        ----------
        lbda : array_like
            Wavelength array.
        flux : array_like
            Flux array.
        amplitude : float, optional
            Scaling factor applied to `flux`. It is a mutable parameter of the
            resulting scene, so the flat level can be changed afterwards
            without rebuilding the target. Default is 1.
        **kwargs
            Goes to `simulation.Simulation.from_config()`.
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
                reference spectrum are `numpy.nan`.
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

    Parameters
    ----------
    simulation : slicersim.Simulation, optional
        The simulation object. Default is None.
    field : str, optional
        Lazuli field the spectrograph is configured for, "narrow" or "wide".
        Default is "narrow".

    Attributes
    ----------
    _QTH_TEMPERATURE : float
        Blackbody temperature in Kelvin assumed for the QTH lamp.
    """
    _QTH_TEMPERATURE = 3000
    @classmethod
    def from_qth(cls, temperature=None, mag=12, band='sdssr',
                        **kwargs):
        """ Initialize the flat as observing the 3D flat QTH lamp
        This method make use the generic .from_blackbody class method.
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
            the wavelength in Angstrom as index and a "transmission" column;
            it is looked up in the package config directory when it is not
            found locally. If a callable, the transmission as a function of
            wavelength in Angstrom.
            Default is "lazuli_fp_transmission.csv".
        temperature : float, optional
            Temperature of the lamp in Kelvin. Default is None.
            If None`_QTH_TEMPERATURE` is used.
        mag : float, optional
            Magnitude the lamp spectrum is normalised to. Default is 12.
        band : str, optional
            Name of the bandpass used for the normalisation (must be known by
            `sncosmo`). Default is "sdssr".
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
            When `fp_throughput` is read from a file, flagging the temporary
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
