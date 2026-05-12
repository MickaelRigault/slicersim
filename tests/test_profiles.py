import numpy as np

from slicersim import profiles

def test_get_gaussian2d():
    """ """
    xx = np.arange(0, 13)
    yy = np.arange(0, 9)

    sigma = np.asarray([[2, 5], [4, 5], [5, 2]])
    mu = np.asarray([3, 6])
    
    kernel = profiles.get_gaussian2d(xx, yy, sigma=sigma, mean=mu)
    kernel_1d = profiles.get_gaussian2d(xx, yy, sigma=sigma[0], mean=mu)

    assert kernel_1d.shape == (yy.shape[0], xx.shape[0])
    assert kernel.shape == (sigma.shape[0], yy.shape[0], xx.shape[0])
    assert np.isclose(kernel[0], kernel_1d).all()
