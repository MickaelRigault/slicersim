

def get_calspec_pointsource(source, **kwargs):
    """ """
    from .calspec import calspecsource
    lbda, flux, *_ = calspecsource.get_spectrum(source)
    return {"source": [lbda, flux]} | kwargs
