"""Scene elements: point source, backgrounds and host.

The scene is made of an optional point source (e.g. a supernova or a
kilonova, see `slicersim.scene.sources`), a spatially flat background, and an
optional structured background, the host galaxy (see `slicersim.scene.host`).
A scene can also be a galaxy alone (no point source). Use `get_scene` to
build a scene configuration and `~slicersim.scene.scene.Scene` to load it.
"""
import warnings
from .pointsource import PointSource  # noqa: F401
from .scene import Scene  # noqa: F401

def get_scene(source, background="zodi", host={}, **kwargs):
    """Get a scene configuration.

    This function generates a dictionary that defines a scene containing a
    point source, a background, and a host galaxy. The scene can then be used
    to generate a simulation.

    Parameters
    ----------
    source : str, tuple or None
        The kind of point source, optionally followed by the model to use,
        separated by "-" (``"{kind}-{model}"``). For instance, ``"snia-salt"``
        means a SN Ia modeled with SALT, ``"kilonova-bulla19"`` a kilonova
        modeled with the Bulla (2019) POSSIS model.
        Available kinds and models are:

        - ``snia``: ``salt`` [default], ``twin``, or any ``sncosmo`` SALT
          source name (see
          `~slicersim.scene.sources.supernovae.get_snia_pointsource`).
        - ``kilonova``: ``bulla23`` [default], ``bulla19`` (see
          `~slicersim.scene.sources.kilonova.get_kilonova_pointsource`).
        - ``calspec``: any HST CalSpec star name (e.g. ``bd_17``, ``gd71``,
          ``p177d``). No default (see
          `~slicersim.scene.sources.stars.get_calspec_pointsource`).
        - ``blackbody``: temperature in K (e.g. ``"blackbody-6000"``,
          ``"blackbody-10000"``), see
          `~slicersim.scene.sources.blackbody.get_blackbody_flux`.
        - ``powerlaw``: power-law index ``alpha`` of F_lambda (e.g.
          ``"powerlaw--2"`` for alpha=-2, ``"powerlaw-0.5"`` for alpha=0.5),
          see `~slicersim.scene.sources.generic.get_powerlaw_flux`.

        Hence ``"snia"`` is equivalent to ``"snia-salt"`` and ``"kilonova"``
        to ``"kilonova-bulla23"``.
        Alternatively, ``source`` can be a ``(lbda, flux)`` tuple (or list)
        defining a tabulated spectrum, with ``lbda`` in Angstrom (see
        `~slicersim.scene.pointsource.PointSource.from_spectrum`).
        If None, the scene has no point source: it is a galaxy alone, given
        by ``host`` (e.g. `~slicersim.target.Galaxy`).
    background : str or dict, optional
        The background configuration (see
        `~slicersim.scene.background.Background.from_config`). ``"zodi"`` is
        a shortcut for the Aldering (2001) zodiacal light model
        (``"Aldering01.BB5800"``, scaled by 2). Default is ``"zodi"``.
    host : str, dict or None, optional
        The host galaxy, as a model name ``"spatial:spectral"`` (e.g.
        ``"ngc0628"`` or ``"sersic:bb-5000"``, see
        `~slicersim.scene.host.parse_modelname`) or a configuration dict
        (model name as ``"refmodel"``). Missing entries take the defaults of
        `~slicersim.scene.host.get_host_config`, and the host takes the
        redshift of the point source. An empty dict or None means no host.
        If ``source`` is None, the host entries are completed by
        ``kwargs``, and a host without model name is the default
        ``"sersic:bb-5000"`` galaxy. Default is an empty dict.
    **kwargs
        Additional parameters of the point source configuration (e.g.
        ``redshift``, ``phase``, ``position``, or model specific parameters
        such as ``x1`` and ``c`` for SALT, ``theta`` and ``magabs`` for
        kilonovae, ``mag`` and ``band`` for blackbody, power-law or CalSpec
        sources). If ``source`` is None, they are galaxy (host) parameters
        instead (e.g. ``redshift``, ``magabs``, ``position``, ``r_kpc``).

    Returns
    -------
    dict
        A dictionary ``{"scene": {"pointsource": ..., "background": ...,
        "host": ...}}`` defining the scene.

    Raises
    ------
    NotImplementedError
        If ``source`` is a str of unknown kind, or neither a str nor a
        ``(lbda, flux)`` tuple or list.

    Examples
    --------
    >>> scene = get_scene("kilonova-bulla23", redshift=0.05, theta=30)
    >>> scene["scene"]["pointsource"]["source"]
    'bulla23'

    A supernova in NGC 0628, and a galaxy alone:

    >>> scene = get_scene("snia", redshift=0.9, host="ngc0628")
    >>> scene["scene"]["host"]
    'ngc0628'
    >>> scene = get_scene(None, host="sersic:bb-6000", redshift=0.5, r_kpc=3)
    >>> scene["scene"]["pointsource"] is None, scene["scene"]["host"]["refmodel"]
    (True, 'sersic:bb-6000')
    """
    def _parse_source_(source):
        """Get the model part of "{kind}-{model}", None if not given."""
        # maxsplit=1: model names may contain "-" (e.g. salt2-extended)
        _, *source = source.split("-", 1)
        if len(source) == 0:
           return None

        return source[0]

    # Name short cuts
    if source is None:
        pointsource = None
        if host is not None:
            if type(host) == str:
                host = {"refmodel": host} | kwargs
            elif type(host) == dict:
                host = host  | kwargs
            else:
                raise NotImplementedError(f"Cannot parse host {host!r} for none pointsource scene. Must be a str or a dict.")

    elif type(source) == str:
        # transient
        if "snia" in source:
            from .sources.supernovae import get_snia_pointsource
            # default is SALT.
            pointsource = get_snia_pointsource(source=_parse_source_(source), **kwargs)

        elif "kilonova" in source:
            from .sources.kilonova import get_kilonova_pointsource
            pointsource = get_kilonova_pointsource(source=_parse_source_(source), **kwargs)

        elif "calspec" in source:
            from .sources.stars import get_calspec_pointsource
            source = _parse_source_(source) # get_calspec_pointsource need a non-None entry
            pointsource = get_calspec_pointsource(source=source, **kwargs)

        elif "blackbody" in source:
            from .sources.blackbody import get_blackbody_flux
            temperature = _parse_source_(source)
            if temperature is not None:
                temperature = float(temperature)

            pointsource = {"name": "blackbody",
                           "model_func": get_blackbody_flux,
                           "temperature": temperature} | kwargs

        elif "powerlaw" in source:
            from .sources.generic import get_powerlaw_flux
            alpha = _parse_source_(source)
            if alpha is not None:
                alpha = float(alpha)
            pointsource = {"name": "powerlaw",
                           "model_func": get_powerlaw_flux,
                           "alpha": alpha} | kwargs

        else:
            raise NotImplementedError(f"Unknown source name: {source!r}.'snia', 'kilonova', 'calspec', 'blackbody' implemented.")

    elif type(source) in (tuple, list):
        lbda, flux = source
        pointsource = {"source": [lbda, flux]} | kwargs

    else:
        raise NotImplementedError(f"Unknown source {source!r}. Must be a str or a (lbda, flux) tuple or list.")

    # background
    if background == "zodi":
        background = {'name': 'zodi', 'model': 'Aldering01.BB5800', 'scale': 2.0}

    return {"scene": {"pointsource": pointsource,
                      "background": background,
                      "host": host
                    }}


def get_sn_scene(model="salt", background="zodi", host={}, **kwargs):
    """Get a scene configuration for a supernova model.

    .. deprecated:: 1.3.1
        Use ``get_scene(source="snia-{model}", ...)`` instead.

    This function generates a dictionary that defines a scene containing a
    supernova, a background, and a host galaxy. The scene can then be used
    to generate a simulation.

    Parameters
    ----------
    model : str, optional
        The name of the supernova model to use. Can be ``"salt"``,
        ``"twin"`` or any other ``sncosmo`` SALT source name.
        Default is ``"salt"``.
    background : str or dict, optional
        The background configuration, see `get_scene`. Default is ``"zodi"``.
    host : dict, optional
        A dictionary defining the host galaxy. Default is an empty dict.
    **kwargs
        Additional keyword arguments to pass to the supernova model.

    Returns
    -------
    dict
        A dictionary defining the scene, see `get_scene`.

    Warns
    -----
    DeprecationWarning
        Always, since this function is deprecated in favour of `get_scene`.
    """
    warnings.warn(f"get_sn_scene is deprecated. Use get_scene(source='snia-{model}', ...) instead.", DeprecationWarning)
    return get_scene(source=f"snia-{model}", background=background, host=host, **kwargs)
