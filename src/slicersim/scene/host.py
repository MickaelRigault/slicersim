""" structured background """
import numpy as np
from .base import SceneElement

class Host(SceneElement):
    """Structured background representing the host galaxy of the observed target.

    Models a spatially extended source (e.g. the galaxy hosting a supernova)
    whose surface-brightness distribution is wavelength-dependent.  The spatial
    profile at each wavelength is computed by a ``galsim``-compatible model
    object and rendered into the scene datacube.

    .. note::
        Host galaxy support is currently in development; the interface is
        subject to change.
    """

    def __init__(self, galmodel, meta={}):
        """ Initialize the Host object.

        Parameters
        ----------
        galmodel : object
            Galaxy model (e.g. a galsim object) used to generate images
            at given wavelengths.

        meta : dict, optional
            Metadata associated with the host object. Default is an
            empty dict.

        Returns
        -------
        None
        """
        self._galmodel = galmodel
        self._meta = meta.copy()
        self._meta_in = meta.copy()

    @classmethod
    def from_galsim(cls, galsim_obj, meta={}):
        """ Create a Host instance from a galsim object.

        Parameters
        ----------
        galsim_obj : object
            A galsim object representing the galaxy model.

        meta : dict, optional
            Metadata associated with the host object. Default is an
            empty dict.

        Returns
        -------
        Host
            A new instance of Host initialized with the given galsim
            object.
        """
        return cls(galmodel=galsim_obj, meta=meta)

    # ================ #
    #   Methods        #
    # ================ #
    def get_cube(self, lbda, nx, ny, scale=1):
        """ Build a datacube of the host galaxy model.

        Parameters
        ----------
        lbda : array-like
            Wavelength array defining the spectral axis of the cube.

        nx : int
            Number of pixels along the x-axis (columns) of the cube.

        ny : int
            Number of pixels along the y-axis (rows) of the cube.

        scale : float, optional
            Pixel scale used when drawing the image. Default is 1.

        Returns
        -------
        numpy.ndarray
            A 3D array of shape (len(lbda), ny, nx) containing the
            host galaxy datacube.
        """
        cube = np.zeros((len(lbda), ny, nx), dtype=np.float64)
        self.fill_cube(cube, lbda, scale=scale, inplace=True)
        return cube

    def fill_cube(self, cube, lbda, scale=1, inplace=False):
        """ Fill a datacube with the host galaxy model evaluated at each wavelength.

        Parameters
        ----------
        cube : numpy.ndarray
            3D array to be filled with the host galaxy model, of shape
            (len(lbda), ny, nx).

        lbda : array-like
            Wavelength array corresponding to the first axis of the cube.

        scale : float, optional
            Pixel scale used when drawing the image. Default is 1.

        inplace : bool, optional
            If True, modify the input cube in place. If False, operate
            on and return a copy of the cube. Default is False.

        Returns
        -------
        numpy.ndarray or None
            The filled cube if `inplace` is False, otherwise None.
        """
        nlbda, ny, nx = cube.shape
        if len(lbda) != nlbda:
            raise ValueError("wavelength array and cube shape do not match")

        if not inplace:
            cube = cube.copy()

        for wave_slice, lbda_i in zip(cube, lbda):
            mono_galaxy = self.galmodel.evaluateAtWavelength(lbda_i/10.) # expected in nm by GalSim
            image = mono_galaxy.drawImage(nx=nx, ny=ny, scale=scale)
            wave_slice += image.array.astype(np.float64)

        if not inplace:
            return cube

    # ================ #
    #   Properties     #
    # ================ #
    @property
    def galmodel(self):
        """ Galaxy model used to generate images at given wavelengths. """
        return self._galmodel
