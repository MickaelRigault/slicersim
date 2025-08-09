import numpy as np

from .base import SceneElement
from ..utils import inspect_func


def def get_background_model_func(name):
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
        name = config.get("name", None)
        if name is None:
            raise ValueError("no name in config. One needed")

        model_func = get_background_model_func(name)
        _, default = inspect_func(model_func)
        return cls(model_func, meta=default | config)