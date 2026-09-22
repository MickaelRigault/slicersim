""" This modules handles calibration exposures """
import numpy as np

from .target import VirtualTarget


class VirtualFlat(VirtualTarget):
    """Abstract base class for flat-field calibration targets.

    Subclasses must implement :meth:`get_spectrum` and :meth:`setup_to_snr`
    for their specific flat-field geometry.
    """

    def get_spectrum(self, *args, **kwargs):
        """Not implemented for the generic VirtualFlat class.

        Raises
        ------
        NotImplementedError
            Always raised; subclasses must override this method.
        """
        raise NotImplementedError("This functionality is not implemented for the generic VirtualFlat class")

    def setup_to_snr(self, *args, **kwargs):
        """Not implemented for the generic VirtualFlat class.

        Raises
        ------
        NotImplementedError
            Always raised; subclasses must override this method.
        """
        raise NotImplementedError("This functionality is not implemented for the generic VirtualFlat class")


class Flat3DCalibration(VirtualFlat):
    """3D flat-field calibration target built from a spatially uniform background source.

    Wraps a :class:`~slicersim.simulation.Simulation` whose scene contains only a
    background component (no point source, no host), allowing the full IFU spatial
    field to be illuminated uniformly for wavelength-dependent flat-field calibration.
    """

    @classmethod
    def from_spectrum(cls, lbda, flux, instrument=None, **kwargs):
        """Build a 3D flat calibration target from a reference spectrum.

        The spectrum is loaded as a spatially uniform
        `~slicersim.scene.background.Background`, and the scene is built with
        neither point source nor host, so that the full IFU field is
        illuminated uniformly.

        Parameters
        ----------
        lbda : array_like
            Wavelength array in Angstrom.
        flux : array_like
            Flux of the reference spectrum in erg/s/cm^2/A.
        instrument : str or dict, optional
            Instrument configuration. If None, the class default
            ``_INSTRUMENT`` is used. Default is None.
        **kwargs
            Additional parameters passed to
            `~slicersim.scene.background.Background.from_spectrum`, overriding
            the defaults ``mag=None``, ``band="bessellb"`` and ``meta={}``.

        Returns
        -------
        Flat3DCalibration
            An instance of the class, wrapping the corresponding simulation.

        Raises
        ------
        ValueError
            If no instrument is given and the class defines no default
            ``_INSTRUMENT``.

        Notes
        -----
        ``mag`` defaults to None, so the reference spectrum keeps its native
        amplitude and is not photometrically renormalised.
        """
        from .scene.background import Background
        from .scene.scene import Scene
        from .simulation import Simulation
        if instrument is None:
            if hasattr(cls, "_INSTRUMENT"):
                instrument = cls._INSTRUMENT
            else:
                raise ValueError("No instrument specified and no default instrument available")

        # Create a scene with solely a background
        prop_default = dict(mag=None, band="bessellb", meta={})
        background = Background.from_spectrum(lbda, flux, **(prop_default| kwargs) ) # mag None measn no
        scene = Scene(background=background,
                        host=None, pointsource=None)

        # create the simulation for this scene and the specified instrument.
        simulation = Simulation.from_scene_and_instconfig(scene=scene, config=instrument)

        # return the target
        return cls.from_simulation(simulation=simulation)

    @classmethod
    def from_blackbody(cls, temperature, instrument=None,
                        mag=15, band='sdssr', magsys='ab',
                        lbda_range=[3_000, 20_000], lbda_bin=2000,
                        **kwargs):
        """Build a 3D flat calibration target from a blackbody lamp.

        Parameters
        ----------
        temperature : float
            Temperature of the blackbody in Kelvin.
        instrument : str or dict, optional
            Instrument configuration. If None, the class default
            ``_INSTRUMENT`` is used. Default is None.
        mag : float, optional
            Magnitude the blackbody spectrum is normalised to. Default is 15.
        band : str, optional
            Name of the bandpass used for the normalisation (must be known by
            `sncosmo`). Default is "sdssr".
        magsys : str, optional
            Name of the magnitude system (see `sncosmo`). Default is "ab".
        lbda_range : list, optional
            (min, max) wavelength range in Angstrom.
            Default is [3_000, 20_000].
        lbda_bin : int, optional
            Number of wavelength samples. Default is 2000.
        **kwargs
            Goes to `from_spectrum`.

        Returns
        -------
        Flat3DCalibration
            An instance of the class.

        See Also
        --------
        slicersim.scene.sources.blackbody.get_blackbody_flux : The blackbody source.
        """
        from .scene.sources.blackbody import get_blackbody_flux
        lbda_ = np.linspace(*lbda_range, lbda_bin)
        flux_ = get_blackbody_flux(lbda_, temperature, mag=mag, band=band, magsys=magsys)
        return cls.from_spectrum(lbda_, flux_, instrument=instrument, **kwargs)

    @classmethod
    def from_febryperot(cls, temperature, fp_throughput,
                        instrument=None,
                        mag=15, band='sdssr', magsys='ab',
                        lbda_range=[3_000, 20_000], lbda_bin=2000,
                        **kwargs):
        """Build a 3D flat calibration target from a blackbody seen through a Fabry-Perot.

        The blackbody continuum is multiplied by the Fabry-Perot transmission,
        which turns it into the comb of spectral features used for wavelength
        calibration.

        Parameters
        ----------
        temperature : float
            Temperature of the blackbody in Kelvin.
        fp_throughput : callable
            Fabry-Perot transmission as a function of wavelength in Angstrom.
        instrument : str or dict, optional
            Instrument configuration. If None, the class default
            ``_INSTRUMENT`` is used. Default is None.
        mag : float, optional
            Magnitude the blackbody spectrum is normalised to, before the
            Fabry-Perot transmission is applied. Default is 15.
        band : str, optional
            Name of the bandpass used for the normalisation (must be known by
            `sncosmo`). Default is "sdssr".
        magsys : str, optional
            Name of the magnitude system (see `sncosmo`). Default is "ab".
        lbda_range : list, optional
            (min, max) wavelength range in Angstrom.
            Default is [3_000, 20_000].
        lbda_bin : int, optional
            Number of wavelength samples. Default is 2000.
        **kwargs
            Goes to `from_spectrum`.

        Returns
        -------
        Flat3DCalibration
            An instance of the class.

        See Also
        --------
        from_blackbody : Same source, without the Fabry-Perot etalon.
        """
        from .scene.sources.blackbody import get_blackbody_flux
        # source is a backbody
        lbda_ = np.linspace(*lbda_range, lbda_bin)
        flux_bb = get_blackbody_flux(lbda_, temperature, mag=mag, band=band, magsys=magsys)

        # going through the FP
        flux_ = flux_bb * fp_throughput(lbda_)
        return cls.from_spectrum(lbda_, flux_, instrument=instrument, **kwargs)
