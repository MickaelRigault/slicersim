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
        """ """
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
        """ """
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
        """ """
        from .scene.sources.blackbody import get_blackbody_flux
        # source is a backbody
        lbda_ = np.linspace(*lbda_range, lbda_bin)
        flux_bb = get_blackbody_flux(lbda_, temperature, mag=mag, band=band, magsys=magsys)

        # going through the FP
        flux_ = flux_bb * fp_throughput(lbda_)
        return cls.from_spectrum(lbda_, flux_, instrument=instrument, **kwargs)
