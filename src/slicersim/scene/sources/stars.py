"""This module builds point sources from stellar spectra.

Stars are defined by their tabulated spectrum (e.g. the HST CalSpec
spectrophotometric standards, see `~slicersim.scene.sources.calspec`) rather
than by a model function.
"""


def get_calspec_pointsource(source, **kwargs):
    """Get the point source configuration of a CalSpec standard star.

    The spectrum of the star is downloaded from the HST CalSpec archive
    (see `~slicersim.scene.sources.calspec.CalSpecSource.get_spectrum`)
    and stored as a tabulated ``(lbda, flux)`` spectrum in the configuration.

    Parameters
    ----------
    source : str
        Name of the CalSpec star (e.g. "gd71", "p177d", "bd_17").
        The start of the name is enough as long as it is not ambiguous,
        e.g. "bd_17" for "bd_17d4708".
    **kwargs
        Additional parameters of the point source configuration, for instance:

        - mag: magnitude of the star in `band`. If not given, the measured
          CalSpec flux level is used.
        - band: bandpass (from sncosmo) in which `mag` is defined
          (default "bessellb").
        - position: position of the point source in spaxels.

    Returns
    -------
    dict
        Configuration dictionary for the point source, ready to be passed
        to `~slicersim.scene.pointsource.PointSource.from_config`. Its
        ``"source"`` entry is the ``[lbda, flux]`` spectrum of the star,
        with ``lbda`` in Angstrom and ``flux`` in erg/s/cm^2/A.

    See Also
    --------
    slicersim.scene.get_scene : Build a full scene around a CalSpec star
        (``source="calspec-{name}"``).
    slicersim.lazuli.LazuliCalSpec : Lazuli target of a CalSpec star.

    Examples
    --------
    >>> config = get_calspec_pointsource("gd71", mag=20, band="sdssr")  # doctest: +SKIP
    >>> lbda, flux = config["source"]  # doctest: +SKIP
    """
    from .calspec import calspecsource
    lbda, flux, *_ = calspecsource.get_spectrum(source)
    return {"source": [lbda, flux]} | kwargs
