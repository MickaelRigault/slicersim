import numpy as np

from slicersim import utils

def test_recursive_get():
    """ """
    dd = {"a": {"ab": 12,
                "ac": 13},
          "b": 2}
    assert utils.recursive_get(dd, "ab") == 12
    assert utils.recursive_get(dd, "b") == 2

def test_mesh_kwargs():
    """ """
    kwargs = dict(scale=[0.02, 0.05], sigma=[0.03, 0.06], toto=[1,2,3])
    
    df_out = utils.mesh_kwargs(**kwargs)
    excepted_ncol = len(list(kwargs.keys()))
    excepted_nrows = np.prod([len(v) for v in kwargs.values()])
    
    assert list(df_out.columns) == list(kwargs.keys())
    assert df_out.values.shape == (excepted_nrows, excepted_ncol)

def test_inspect_func():
    """ """
    def test_func(a, ith=4, **kwargs):
        pass
        
    keys, default_values = utils.inspect_func(test_func) 
    assert list(keys) == ["a", "ith"]
    assert default_values == {"ith":4}


def test_restride_bin_unbin():
    """ """
    assert utils.restride(np.ones((6, 8)), 2).shape == (3, 4, 2, 2)
    assert utils.restride(np.ones((6, 8)), (3, 2)).shape == (2, 4, 3, 2)

    assert utils.bin_array(np.arange(4*6).reshape(4, 6), 2, np.mean).shape == (2, 3)
    assert utils.unbin_array(np.arange(2*3).reshape(2, 3), 2).shape == (4, 6)


def test_erf_functions():
    """ """
    # 1D case
    x_edges = np.linspace(-10, 10, 41); sigma = 2
    assert np.isclose( utils.integ_gaussian1D_erf(x_edges, sigma=sigma, normed=True).sum(), 1)
    assert np.isclose( utils.integ_gaussian1D_erf(x_edges, sigma=sigma, normed=False).sum(), sigma * (2*np.pi)**0.5)

    # 2D case
    y, x = np.ogrid[-5:5.1, -6:6.1]
    sigma = np.array([1])
    assert np.isclose( utils.integ_gaussian2D_erf((x, y), sigma=sigma, normed=True).sum(), 1)
    assert np.isclose( utils.integ_gaussian2D_erf((x, y), sigma=sigma, normed=False).sum(), sigma**2 * 2*np.pi)


def test_complete_dims():
    """ """
    assert utils.complete_dims(1, 2, False).shape == (1, 1)
    assert utils.complete_dims([[0, 1, 2]], -4, False).shape == (1, 3, 1, 1)
