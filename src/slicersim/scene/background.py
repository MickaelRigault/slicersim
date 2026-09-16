import numpy as np
import warnings
from .base import SceneElement
from ..utils import inspect_func


def get_background_model_func(name):
    """Get the model function for a background name.

    Parameters
    ----------
    name : str
        Name of the background model.
        - "zodi"

    Returns
    -------
    callable
        The model function for the given background.
    """
    if "zodi" in name:
        model_func = zodiacal_spectrum

    elif name in ["bb", "blackbody"]:
        from ..thermal import get_source_radiation
        model_func = get_source_radiation

    else:
        raise NotImplementedError(f"background: {name} is not implemented")

    return model_func

def zodiacal_spectrum(lbda, scale=1, model="Aldering01.BB5800"):
    """Sky background (zodi) spectrum at the North Ecliptic Pole.

    Two models are implemented from Aldering 2001 (LBNL-51157):
    - Black-Body approximation (`Aldering01.BB5800`)
    - Truncated Power Law (`Aldering01.TPL`)

    Parameters
    ----------
    lbda : array_like
        Wavelength in Angstrom.
    scale : float, optional
        Scaling factor. Default is 1.
    model : str, optional
        Model name. Default is "Aldering01.BB5800".

    Returns
    -------
    array_like
        Flux in erg/s/cm^2/A/arcsec^2.

    See Also
    --------
    :ads:`Scaramella+22 <2022A&A...662A.112E>`

    Examples
    --------
    .. plot::

       import matplotlib.pyplot as plt
       from slicersim.spectrograph import Spectrograph
       from slicersim.scene.background import zodiacal_spectrum
       lbda, _ = Spectrograph.lbda_from_respow([4000, 17000], 75)
       fig, ax = plt.subplots()
       ax.plot(lbda / 1e4,
               np.log10(zodiacal_spectrum(lbda, model="Aldering01.BB5800")),
               ds='steps-mid', label="BB5800")
       ax.plot(lbda / 1e4,
               np.log10(zodiacal_spectrum(lbda, model="Aldering01.TPL")),
               ds='steps-mid', label="TPL")
       ax.legend()
       ax.set(xlabel="Wavelength [µm]",
              ylabel="log(Flux [fλ/arcsec²])",
              title="Zodiacal spectrum (Aldering01)")
    """

    if model == "Aldering01.BB5800":
        # h*c/(k*5800 K) = 24806.498 Å
        zodi = 1e-17 * (lbda * 1e-4) ** -4.64 / (np.exp(24806.498 / lbda) - 1)

    elif model == "Aldering01.TPL":
        zodi = 10 ** (-17.755 - 0.740 * (lbda * 1e-4 - 0.61))
        zodi[lbda < 6100] = 10 ** (-17.755)

    else:
        raise NotImplementedError(f"Unknown model {model!r}.")

    return scale * zodi  # Account for scale factor


# ================ #
#                  #
#     Class        #
#                  #
# ================ #
class Background(SceneElement):
    """Class for background scene elements."""

    @classmethod
    def from_config(cls, config):
        """Build the class from a configuration file.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
            It must contain the key "name".

        Returns
        -------
        Background
            An instance of the Background class.
        """
        func = config.get("func", None)
        name = config.get("name", None)
        if name is None and func is None:
            raise ValueError("neither name nor func in config. One needed")

        # no function given, look for one
        if func is None:
            func = get_background_model_func(name)

        inputnames, default = inspect_func(func)
        if inputnames[0] not in ["lbda", "wave", "wavelength"]:
            warnings.warn(f"first argument of modeling function should be this wavelength, it is called {inputnames[0]} in the given model. It could mean there is a problem.")

        return cls(func, meta=default | config)

    @classmethod
    def from_spectrum(cls, lbda_, flux_, mag=20, band="bessellb",
                       meta={}):
        """Generate a `PointSource` from a spectrum.

        Parameters
        ----------
        lbda_ : array_like
            Wavelength array in Angstrom of the reference spectrum.
        flux_ : array_like
            Flux of the reference spectrum.
        mag : float, optional
            Default magnitude of the target. Default is 20.
        band : str, optional
            Name of the bandpass (must be known by sncosmo). Default is "bessellb".
        meta : dict, optional
            Meta information carried by the object. Default is {}.

        Returns
        -------
        PointSource
            An instance of the `PointSource` class.
        """
        from sncosmo import Spectrum
        meta["mag"] = mag
        meta["band"] = band
        meta["flux_ref"] = flux_
        meta["lbda_ref"] = lbda_
        this = cls(None, meta=meta)

        # internal function
        def _internal_get_flux(lbda, mag, band):
            """Internal function to get the flux of the spectrum.

            Parameters
            ----------
            lbda : array_like
                Wavelength array in Angstrom.
            mag : float
                Target magnitude.
            band : str
                Bandpass name.

            Returns
            -------
            array_like
                The flux of the spectrum.
            """
            if mag is None:
                flux_ratio = 1
            else:
                in_mag = Spectrum(meta["lbda_ref"], meta["flux_ref"]
                                  ).bandmag(band, "ab")
                flux_ratio = 10 ** (-0.4 * (mag - in_mag))

            flux_ = np.interp(lbda, meta["lbda_ref"], meta["flux_ref"],
                              left=np.nan, right=np.nan)
            return flux_ * flux_ratio

        this._model_func = _internal_get_flux
        return this
