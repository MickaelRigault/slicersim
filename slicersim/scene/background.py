import numpy as np
from astropy.modeling.models import GeneralSersic2D
from scipy.special import gamma
from .base import SceneElement


def get_background_model_func(name):
    """ top level method to parse name into model_func """
    if "zodi" in name:
        model_func = zodiacal_spectrum
    else:
        raise NotImplementedError(f"background: {name} is not implemented")
    
    return model_func

def zodiacal_spectrum(lbda, scale=1, model="Aldering01.BB5800"):
    """
    Sky background (zodi) spectrum at the North Ecliptic Pole.

    Two models are implemented from Aldering 2001 (LBNL-51157):

    * Black-Body approximation (`Aldering01.BB5800`)
    * Truncated Power Law (`Aldering01.TPL`)

    :param lbda: wavelength [Å]
    :param float scale: scaling factor
    :param str model: model name
    :return: flux in erg/s/cm²/Å/arcsec²

    See also :ads:`Scaramella+22 <2022A&A...662A.112E>`.

    .. plot::

       import matplotlib.pyplot as plt
       from mlaperf.spectrograph import Spectrograph
       from mlaperf.scene import zodiacal_spectrum
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
        zodi = 1e-17 * (lbda * 1e-4)**-4.64 / (np.exp(24806.498 / lbda) - 1)
        
    elif model == "Aldering01.TPL":
        zodi = 10**(-17.755 - 0.740*(lbda * 1e-4 - 0.61))
        zodi[lbda < 6100] = 10**(-17.755)
        
    else:
        raise NotImplementedError(f"Unknown model {model!r}.")

    return scale * zodi         # Account for scale factor
      

#def normalizedSersic(surfMag=22, r_eff=1, n=4, x_0=0, y_0=0, ellip=0, theta=0.0, c = 0):
#    
#    mod = GeneralSersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=x_0, y_0=y_0, ellip=ellip, theta=theta, c=c)
#    
#    return normProfile


# ================ #
#                  #
#     Class        #
#                  #
# ================ #
from ..utils import inspect_func
class Background( SceneElement ):
    
    @classmethod
    def from_config(cls, config):
        """ """
        name = config.get("name", None)
        if name is None:
            raise ValueError("no name in config. One needed")

        model_func = get_background_model_func(name)
        _, default = inspect_func(model_func)
        return cls(model_func, meta= default | config)
