"""
Computation of thermal signal (solid angles, black body spectrum).
"""

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"


import numpy as np

def cos2arctan(x):
    r"""
    .. math::

       \cos^2(\arctan x) = \frac{1}{1 + x^2}.
    """

    return 1/(1 + x**2)

def sinarctan(x):
    r"""
    .. math::

       \sin(\arctan x) = \frac{x}{\sqrt{1 + x^2}}.
    """

    return x / (1 + x**2)**0.5

def omega_camera(cam_accept):
    r"""
    Solid angle of camera:

    .. math::

       \Omega &= 2\pi\int_{\pi}^{\theta_1} \sin\theta\,\cos\theta\,\d\theta \\
              &= \pi\left(1 - \cos^2(\arctan x_1)\right)

    with :math:`\alpha` camera acceptance (maximum incidence angles),
    :math:`x_1 = 1/(2 \alpha)`, :math:`\theta_1 = \arctan x_1` and
    :math:`\cos^2(\arctan x) = 1 / (1 + x^2)`.

    :param cam_acceptance: camera acceptance
    :return: camera solid angle
    """

    omega = np.pi * (1 - cos2arctan(0.5 / cam_accept))

    return omega

def omega_pupil(cam_speed, cam_accept):
    r"""
    Solid angle of pupil:

    .. math::

       \Omega &= 2\pi\int_{\theta_2}^{\theta_1} \cos\theta\,\d\theta \\
              &= 2\pi\left(\sin(\arctan x_1) - \sin(\arctan x_2)\right)

    with :math:`\alpha` camera acceptance, :math:`\beta` camera speed,
    :math:`x_1 = 1/(2 \alpha)`, :math:`x_2 = 1/(2 \beta)`,
    :math:`\theta_i = \arctan x_i` and
    :math:`\sin(\arctan x) = x / \sqrt{1 + x^2}`.

    :param cam_speed: camera speed
    :param cam_acceptance: camera acceptance
    :return: solid angle
    """

    sol = sinarctan(0.5/cam_accept) - sinarctan(0.5/cam_speed)

    return 2*np.pi * sol

def omega_slit(cam_speed):
    r"""
    Solid angle of the slit:

    .. math::

       \Omega &= 2\pi\int_{0}^{\theta_2} \cos\theta\,\d\theta \\
              &= 2\pi\sin(\arctan x_2)

    with :math:`\beta` camera speed, :math:`x_2 = 1/(2 \beta)`,
    :math:`\theta_2 = \arctan x_2` and
    :math:`\sin(\arctan x) = x / \sqrt{1 + x^2}`.

    :param cam_speed: camera speed
    :return: solid angle
    """

    sol = sinarctan(0.5/cam_speed)

    return 2*np.pi * sol

def omega_tel(half_spx):
    r"""
    Solid angle of telescope:

    .. math::

       \Omega &= 2\pi\int_{0}^{\theta_0} \sin\theta\,\cos\theta\,\d\theta \\
              &= \pi\sin^2(\theta_0)

    with :math:`\theta_0` the semi angular size of a spaxel.

    :param half_spx: *half* angular size of a spaxel [rad]
    :return: solid angle
    """

    omega = np.pi * np.sin(half_spx)**2

    return omega

def nphot_BB(lbda_mu, T):
    r"""
    Black-body spectral radiance [photon/s/sr/m²/µm].

    Return the number of photon/s/sr/m²/µm for a given wavelength [µm] and
    temperature [K]:

    .. math::

       BB_T^{\gamma}(\lambda) = \frac{2c}{\lambda^4}
       \left[\exp\left(\frac{hc}{\lambda k_B T}\right) - 1\right]^{-1}
    """

    c = 299_792_458           # [m/s]
    hc_over_kB = 0.014387774  # [K.m]
    l = 2 * c / ((lbda_mu * 1e-6)**4 *
                 (np.exp(hc_over_kB / (lbda_mu*1e-6 * T)) - 1))  # [ph/s/sr/m²/m]

    return l * 1e-6  # [photon/s/sr/m²/µm]

def thermal_signal(omega, area, spectral_band, temperature, emissivity):
    r"""
    Thermal signal [photons/s].

    .. math::

       f = S\Omega\,E \times
       \int_{\lambda_1}^{\lambda_2} BB_T^{\gamma}(\lambda)\,\d\lambda

    :param omega: solid angle [sr]
    :param area: area [m²]
    :param spectral_band: spectral band [µm] (could be array of)
    :param temperature: temperature [K]
    :param emissivity: emissivity
    """

    from scipy.integrate import quad

    if np.ndim(spectral_band) == 1:
        wmin, wmax = spectral_band
        flux, _ = quad(nphot_BB, wmin, wmax, args=(temperature,))
    elif np.ndim(spectral_band) == 2:
        flux = np.asarray([quad(nphot_BB, wmin, wmax, args=(temperature,))[0]
                               for wmin, wmax in spectral_band])
    else:
        raise ValueError(f"ndim of input spectral_band must be 1 or 2 {np.ndim(spectral_band)=} ")
    
    return flux * omega * area * emissivity  # [photon/s] integrated over bandwidth

def dark_current(det_cutoff, det_temp, px_size, type='HgCdTe'):
    """
    Dark current (thermal noise) model [e/s/px].

    :param float det_cutoff: detector cut-off (upper) wavelength [µm]
    :param float det_temp: detector temperature [K]
    :param float px_size: pixel size [µm]
    :param str type: HgCdTe or InAs
    :return: dark current [e/s/px]

    References: Tennant et al. 2008JEMat..37.1406T; Tennant,
    2010JEMat..39.1030T; O’Loughlin, PhD 2020
    """

    if type == 'HgCdTe':
        # Tennant+08 original parameters for HgCdTe
        J0 = 8367.00001853855              # [A/cm²]
        Pwr = 0.544071281108481
        C = -1.16239134096245
        lamb_scale = 0.200847413564122     # [µm]
        lamb_threshold = 4.63513642316149  # [µm]
    elif type == 'InAs':
        # O'Loughlin parameters for InAs
        J0 = 5315.034051              # [A/cm²]
        Pwr = 0.569561634
        C = -1.140270099
        lamb_scale = 0.21507853       # [µm]
        lamb_threshold = 4.843406496  # [µm]
    else:
        raise NotImplementedError(f"Unknown detector type {type!r}.")

    kB = 1.380649e-23             # Boltzmann constant [J/K]
    q = 1.602176634e-19           # electron charge [C]
    amp2e = 6.2415091e+18         # A to e/s

    apx = (px_size * 1e-4)**2     # [cm²]

    lamb_e = np.where(
        det_cutoff >= lamb_threshold,
        det_cutoff,
        det_cutoff / (1 - (lamb_scale/det_cutoff - lamb_scale/lamb_threshold)**Pwr))
    J = J0 * np.exp((C * 1.24 * q / (lamb_e * kB * det_temp)))  # [A/cm²]

    return J * apx * amp2e      # [e/s/px]
