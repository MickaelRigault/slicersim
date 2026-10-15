""" This module handles the mapping between the slicer and the detector.

The geometry of an image slicer is not analytic: which detector pixel a given
(slice, position along the slice, wavelength) triplet lands on is known from
spot-diagram calibration data. `SlicerMapper` turns that data into
interpolators and exposes them as the two operations the simulation needs:
projecting a simulated scene cube onto a detector image, and de-projecting a
detector image back onto the sky plane. It also describes the geometry of the
individual slices (contours, extent, footprint on the detector).
"""

import numpy as np
import pandas
from itertools import pairwise
from astropy import units as u
from scipy.interpolate import LinearNDInterpolator

from .utils import mesh_kwargs, unbin_array


class SlicerMapper:
    """Map between detector pixel coordinates and IFS slicer coordinates.

    The mapping is derived from spot calibration data that relates each
    slice, field position along the slice (from -1 to 1) and wavelength to a
    physical ``(x, y)`` position on the detector. Linear interpolators built
    from this data are used to project scene cubes onto the detector and to
    de-project detector images back to the sky plane.

    Internally, wavelengths are stored in Angstrom and detector positions in
    mm, with (0, 0) at the detector center.

    Parameters
    ----------
    data : pandas.DataFrame
        Calibration data, with one row per spot and columns for the slice
        id, field position, wavelength and ``(x, y)`` detector position
        (column names can be set through ``**kwargs``).
    detector_shape : tuple of int, optional
        Shape of the detector in pixels ``(ny, nx)``. Default is
        (4096, 4096).
    pixel_size : float, optional
        Size of a detector pixel, in micrometers. Default is 10.
    xy_units : str, optional
        Unit of the ``(x, y)`` coordinates in ``data`` (an astropy unit
        name). Default is ``"mm"``.
    **kwargs
        Column names and wavelength unit of ``data``: ``wavelength_units``
        (default ``"micrometer"``), ``x`` (default ``"x"``), ``y`` (default
        ``"y"``), ``sliceid`` (default ``"slice"``), ``fieldpos`` (default
        ``"fieldpos"``) and ``wavelength`` (default ``"lbda"``).

    See Also
    --------
    from_data : Build the mapper from a calibration file.
    """

    _LBDA_UNITS = "angstrom"
    _XY_UNITS = "mm"
    def __init__(self, data,
                 detector_shape=(4096, 4096),
                 pixel_size=10, # in micrometer
                 xy_units="mm", **kwargs
                ):
        """Initialize the mapper (see the class docstring for parameters)."""
        self._data = data
        self._sliceposwave, self._xy, self._metakey = self._get_interp_structures_(data,
                                                                                    xy_units=xy_units,
                                                                                    **kwargs)
        self._detector_shape = np.asarray(detector_shape, dtype="int")
        self._pixel_size = float(pixel_size) # in micro

    @classmethod
    def from_data(cls, data, xy_units="mm", **kwargs):
        """Create a `SlicerMapper` from a calibration file or DataFrame.

        Parameters
        ----------
        data : str or pandas.DataFrame
            Path to a space-separated CSV file (e.g. ``mapping_spotdata.csv``
            in ``PACKAGE_PATH``) or DataFrame containing the calibration
            data.
        xy_units : str, optional
            Unit of the ``(x, y)`` coordinates. Default is ``"mm"``.
        **kwargs
            Goes to `SlicerMapper`.

        Returns
        -------
        SlicerMapper
            Mapper built from the calibration data.
        """
        if type(data) in [str, np.str_]:
            data = pandas.read_csv(data, sep=" ")

        return cls(data, xy_units=xy_units,  **kwargs)

    @classmethod
    def _get_interp_structures_(cls, data,
                                    wavelength_units="micrometer",
                                    xy_units = "micrometer",
                                    x="x", y="y",
                                    sliceid="slice",
                                    fieldpos="fieldpos",
                                     wavelength="lbda"):
        """Extract and prepare interpolation structures from input data.

        Parameters
        ----------
        data : pandas.DataFrame
            Input calibration data.
        wavelength_units : str, optional
            Units of wavelength in the input data. Default is "micrometer".
        xy_units : str, optional
            Units of xy coordinates in the input data. Default is "micrometer".
        x : str, optional
            Column name for x coordinates. Default is "x".
        y : str, optional
            Column name for y coordinates. Default is "y".
        sliceid : str, optional
            Column name for slice IDs. Default is "slice".
        fieldpos : str, optional
            Column name for field positions. Default is "fieldpos".
        wavelength : str, optional
            Column name for wavelengths. Default is "lbda".

        Returns
        -------
        sliceposwave : numpy.ndarray
            Array of shape (N, 3) containing [sliceid, fieldpos, wavelength]
            values, wavelengths in Angstrom.
        xy : numpy.ndarray
            Array of shape (N, 2) containing [x, y] physical coordinates in mm.
        metakey : dict
            Dictionary storing the column name mappings for reference.
        """
        data = data.copy()
        # for record
        metakey = {k: v for k,v in locals().items() if k not in ["cls", "data"]}

        # wavelength
        lbda_units_convert = getattr(u, wavelength_units).to(cls._LBDA_UNITS) # make sure unit make sense
        data["lbda"] = data[wavelength].astype(float)*lbda_units_convert # in requested units
        sliceposwave = data[ [sliceid, fieldpos, "lbda"] ].astype(float).values

        # spot location
        xy_units_convert = getattr(u, xy_units).to(cls._XY_UNITS) # make sure unit make sense
        xy = data[[x, y]].astype(float).values * xy_units_convert # in requested units
        return sliceposwave, xy, metakey

    # =============== #
    #   method        #
    # =============== #
    def project_lazulitarget(self, lazulitarget, psf_profile="airy", switch_off=[],
                                 **kwargs):
        """Project a Lazuli target simulation onto the detector plane.

        The cubes of both the narrow (slices 1 to 58) and wide (slices 59
        to 116) fields are simulated and projected, and the resulting images
        are summed.

        Parameters
        ----------
        lazulitarget : slicersim.lazuli.VirtualLazuliTarget
            Lazuli target, e.g. a `~slicersim.lazuli.LazuliSupernova`.
        psf_profile : str, optional
            PSF profile used to simulate the cubes. Default is ``"airy"``.
        switch_off : list of str, optional
            Scene contributions to switch off. Default is ``[]``.
        **kwargs
            Goes to `~slicersim.lazuli.VirtualLazuliTarget.get_cube`.

        Returns
        -------
        numpy.ndarray
            2D detector image combining both the narrow and wide fields.
        """

        # generate the cubes (one per channel), you can remove contributions.
        (cube_fine, _), (cube_wide, _) = lazulitarget.get_cube(which='both',
                                                               psf_profile=psf_profile,
                                                               switch_off=switch_off,
                                                                **kwargs)

        # get the wavelength array
        lbda = lazulitarget.simulation.spectrograph.lbda

        # fine grid are slices between 1 and 59 | wide between 59 and 117
        # [::-1] as top <-> bottom definition inversion
        # project them one at the time.
        img_fine = self.project_slice(np.arange(1, 59)[::-1], cube_fine, lbda)
        img_med = self.project_slice(np.arange(59, 117)[::-1], cube_wide, lbda)
        # combine them to get the full image.
        return np.sum([img_med, img_fine], axis=0)

    def project_lazulicube(self, cube, which, lbda, fill_value=0, **kwargs):
        """Project a data cube onto the detector plane.

        Parameters
        ----------
        cube : numpy.ndarray
            Data cube to project, of shape ``(nlbda, 58, nspatial)``.
        which : str
            Which slices to project: ``"fine"`` or ``"narrow"`` for slices
            1-58, ``"medium"`` or ``"wide"`` for slices 59-116.
        lbda : numpy.ndarray
            Wavelengths in Angstrom of the cube spectral axis.
        fill_value : float, optional
            Value of the pixels with no projected signal. Default is 0.
        **kwargs
            Goes to `project_slice`.

        Returns
        -------
        numpy.ndarray
            2D detector image.

        Raises
        ------
        ValueError
            If ``which`` is not one of the available options.
        """
        if which in ["fine", "narrow"]:
            index_ = np.arange(1, 59)[::-1]
        elif which in ["medium", "wide"]:
            index_ = np.arange(59, 117)[::-1]
        else:
            raise ValueError(f"{which=} is not available (narrow or wide")

        return self.project_slice(index_, cube, lbda, fill_value=fill_value, **kwargs)

    def deproject_cubeof(self, key, deproj_df=None,
                            nslices=58, nfieldpos=58, nlbda=571):
        """Extract a cube of deprojected data from the deprojection DataFrame.

        Parameters
        ----------
        key : str
            Column name to extract from the deprojected DataFrame.
        deproj_df : pandas.DataFrame or None, optional
            Deprojected DataFrame, as returned by `deproject_data`. If None,
            the one stored by ``deproject_data(..., inplace=True)`` is used.
            Default is None.
        nslices : int, optional
            Number of slices. Default is 58.
        nfieldpos : int, optional
            Number of field positions. Default is 58.
        nlbda : int, optional
            Number of wavelength points. Default is 571.

        Returns
        -------
        numpy.ndarray
            3D array of shape ``(nlbda, nfieldpos, nslices)``.

        Raises
        ------
        ValueError
            If no deprojected DataFrame is provided or stored.

        Notes
        -----
        The ``key`` column is reshaped as ``(nslices, nfieldpos, nlbda)``
        before moving the axes, while `deproject_data` orders its rows with
        the field position varying the slowest, then the slice id, then the
        wavelength.
        """
        if deproj_df is None:
            if self._deprojected_df is None:
                raise ValueError("No deprojected given, none in self.")

            deproj_df = self._deprojected_df

        cube = deproj_df[key].array.reshape(nslices, nfieldpos, nlbda)
        # move lbda as first axis
        cube = np.moveaxis(cube, -1, 0)
        # replace location of sliceid and fieldpos
        cube = np.moveaxis(cube, 1, -1)

        return np.asarray(cube)


    def deproject_data(self, sliceid, fieldpos, lbda, key=None, inplace=False):
        """Interpolate the calibration data on a slice-field-wavelength grid.

        For each combination of ``sliceid``, ``fieldpos`` and ``lbda``, the
        detector position is computed, and the ``key`` columns of `data` are
        linearly interpolated (in the detector plane) at that position.

        Parameters
        ----------
        sliceid : array_like
            Slice ids.
        fieldpos : array_like
            Field positions along the slice (from -1 to 1).
        lbda : array_like
            Wavelengths in Angstrom.
        key : str or list of str or None, optional
            Column name(s) of `data` to interpolate. If None, all the columns
            are used. Default is None.
        inplace : bool, optional
            If True, also store the result in the mapper, for later use by
            `deproject_cubeof`. Default is False.

        Returns
        -------
        pandas.DataFrame
            One row per grid point (see `~slicersim.utils.mesh_kwargs` for
            the ordering), with columns ``sliceid``, ``fieldpos``, ``lbda``,
            ``xpixel``, ``ypixel`` and the interpolated ``key`` column(s).
        """
        if key is None:
            key = self.data.columns

        df = self._deproject_data(sliceid=sliceid, fieldpos=fieldpos, lbda=lbda,
                                 key=key)

        # store deprojected data in place.
        if inplace:
            self._hdeprojected_df = df

        return df

    def _deproject_data(self, sliceid, fieldpos, lbda, key):
        """Internal method to perform deprojection using interpolation.

        Parameters
        ----------
        sliceid : array_like
            Slice IDs to deproject.
        fieldpos : array_like
            Field positions to deproject.
        lbda : array_like
            Wavelengths to deproject.
        key : str or list
            Column name(s) to interpolate.

        Returns
        -------
        pandas.DataFrame
            DataFrame with deprojected data and pixel positions.
        """
        df = mesh_kwargs(sliceid = sliceid,
                         fieldpos = fieldpos,
                         lbda = lbda)

        # interpolation happens xy in the physical space (see self._xy)
        xy = self.get_pixel_positions(df)
        xy_physical = self.pixels_to_physical(xy)

        # store the pixel position
        df["xpixel"], df["ypixel"] = xy.T

        # get the corresponding values
        df[key] = LinearNDInterpolator(self._xy, self.data[key])( xy_physical )
        return df

    def project_slice(self, sliceid, sliceimg, lbda, oversample=(5, 2), fill_value=0):
        """Project slice image(s) onto the detector.

        The slice images are oversampled (by repetition), each oversampled
        element is mapped to its detector pixel, and the values falling into
        the same pixel are summed.

        Parameters
        ----------
        sliceid : int or array_like
            Id(s) of the slice(s), matching the second axis of ``sliceimg``.
        sliceimg : numpy.ndarray
            Image of the slices, of shape ``(nlbda, nslices, nspatial)``,
            the spatial axis spanning field positions -1 to 1.
        lbda : numpy.ndarray
            Wavelengths in Angstrom of the spectral axis of ``sliceimg``.
        oversample : tuple of int, optional
            ``(spectral_oversampling, spatial_oversampling)`` factors applied
            to the image before projection. Default is (5, 2).
        fill_value : float, optional
            Initial value of the detector pixels, onto which the projected
            values are added. If NaN, the pixels with no (or zero) signal are
            set to NaN. Default is 0.

        Returns
        -------
        numpy.ndarray
            2D image of shape ``detector["shape"]``.

        Raises
        ------
        ValueError
            If the shape of ``sliceimg`` does not match the lengths of
            ``sliceid`` or ``lbda``.
        """
        # general format
        sliceid = np.atleast_1d(sliceid)

        nlbda, *nslices, nspatial = sliceimg.shape
        # generalized format
        nslices = 0 if len(nslices) == 0 else nslices[0]

        # test input
        if nslices != len(sliceid):
            raise ValueError(f"Format do not match between {sliceimg.shape=} and {len(sliceid)=}")

        if nlbda != len(lbda):
            raise ValueError(f"Format do not match between {sliceimg.shape=} and {len(lbda)=}")

        # ok, good to go
        # oversampled slice_positions
        slice_positions = np.linspace(-1, 1, nspatial * oversample[1])

        # oversampled lbda
        lbda = np.interp( np.arange(nlbda, step=1/oversample[0]), np.arange(nlbda), lbda)

        # get mesh
        df_in = mesh_kwargs(sliceid=sliceid,
                            slicepos=slice_positions,
                            wavelength=lbda)

        # xy are the "position" in the detector
        pixels = self.get_pixel_positions(df_in)

        # flag entries out of interpolation boundaries.
        flag_nan = np.isnan(pixels).any(axis=1)

        # these are the corresponding slice value (.T as lbda axis first.)
        # but do not affect the slice dimension
        if nslices == 1 and len(sliceimg.shape)==2:
            slice_oversampled = unbin_array(sliceimg, oversample)
        else:
            slice_oversampled = unbin_array(sliceimg, (oversample[0], 1, oversample[1]))

        value_pixels_t = slice_oversampled.T.flatten()

        # build the corresponding image
        if np.isnan(fill_value):
            fill_zero_to_nan = True
            fill_value = 0
        else:
            fill_zero_to_nan = False

        ## 1. image full of nan
        image = np.full(self.detector["shape"], fill_value=fill_value, dtype="float")

        # .T[::-1] as numpy vs. mpl conventions: pixels[:,1], pixels[:,0]
        # flag_nan remove entries outsize of interpolation boundaries.
        x_, y_ = pixels[~flag_nan].astype(int).T[::-1]
        image[x_, y_] += value_pixels_t[~flag_nan]

        if fill_zero_to_nan:
            image[image==0] = np.nan

        # the detector image
        return image

    def get_slice_contours(self, sliceid, lbda_range=[3_500, 17_000],
                          out_format="numpy", units="pixels",
                          slice_edge=[-1, 1], nbin_pos=5, nbin_lbda=10,
                          combined=False):
        """Get the contour edges of specified slices.

        The contour is the convex hull of the detector positions of a
        ``nbin_pos x nbin_lbda`` grid of field positions and wavelengths.

        Parameters
        ----------
        sliceid : int or array_like
            Slice id(s) to get contours for.
        lbda_range : list of float, optional
            Wavelength range ``[min, max]`` in Angstrom.
            Default is ``[3_500, 17_000]``.
        out_format : str, optional
            Output format: ``"shapely"`` or ``"geometry"`` for a shapely
            polygon, ``"numpy"`` or ``"array"`` for the vertex coordinates.
            Default is ``"numpy"``.
        units : str, optional
            Units of the output coordinates: ``"pixels"`` (or ``"pxl"``,
            ``"pixel"``) for detector pixels, ``"physical"`` or ``"mm"`` for
            mm from the detector center. Default is ``"pixels"``.
        slice_edge : list of float, optional
            Range of field positions along the slice. Default is ``[-1, 1]``.
        nbin_pos : int, optional
            Number of points along the field position. Default is 5.
        nbin_lbda : int, optional
            Number of points along wavelength. Default is 10.
        combined : bool, optional
            If True and several slices are given, return a single contour
            enclosing all of them. If False, return one contour per slice.
            Default is False.

        Returns
        -------
        ``shapely.Polygon`` or numpy.ndarray or list
            Contour polygon, or array of its ``(x, y)`` vertices of shape
            ``(nvertices, 2)``, depending on ``out_format``. If several
            slices are given and ``combined`` is False, a list with one such
            result per slice.

        Raises
        ------
        ValueError
            If ``units`` or ``out_format`` are invalid.

        Notes
        -----
        ``nbin_pos`` and ``nbin_lbda`` are not propagated when ``combined`` is
        False and several slices are given (the defaults are used).
        """
        from shapely import geometry  # new dependecy

        if not combined and len(np.atleast_1d(sliceid))>1:
            return [self.get_slice_contours(sliceid_,
                                            lbda_range=lbda_range,
                                            out_format=out_format,
                                            units=units,
                                            slice_edge=slice_edge,
                                            combined=False)
                             for sliceid_ in sliceid]

        # slicer position are in units of slice (-1, 0, 1)
        slicepos = np.linspace(*slice_edge, nbin_pos)
        wavelength = np.linspace(*lbda_range, nbin_lbda)

        # get the mesh of all this information: lbda & slicepos
        df_in = mesh_kwargs(sliceid=sliceid, slicepos=slicepos,
                            wavelength=wavelength)

        # xy are the "position" in the detector
        xy = self.interp_3d_to_2d(df_in)

        # which "units" you want this position to be in ?
        if units in ["pixels", "pxl", "pixel"]:
            xy = self.physical_to_pixels(xy)

        elif units not in ["physical", "mm"]:
            raise ValueError(f"only physical or pixels units implemented: {units:} given")
        # else: means physical so in mm

        # shapely geometry of the points edge
        geom = geometry.MultiPoint(xy).convex_hull

        # which "format" do you want the output to be ?
        if out_format in ["shapely", "geometry"]:
            return geom
        elif out_format in ["array", "numpy"]:
            return np.asarray(geom.exterior.xy).T
        else:
            raise ValueError(f"Only array or geometry output format implemented: {out_format:} given")

    def get_slice_inbox(self, sliceid, lbda_range=[3500, 17000], units='pixels'):
        """Get the bounding box of specified slice(s).

        Parameters
        ----------
        sliceid : int or array_like
            Slice id(s) to get the bounding box for.
        lbda_range : list of float, optional
            Wavelength range ``[min, max]`` in Angstrom.
            Default is ``[3500, 17000]``.
        units : str, optional
            Units of the output: ``"pixels"`` or ``"physical"``/``"mm"``.
            Default is ``"pixels"``.

        Returns
        -------
        numpy.ndarray
            For ``N`` slices, array of shape ``(2, N, 2)``: axis 0 is
            ``(x, y)``, axis 1 the slice and axis 2 ``(min, max)``.

        See Also
        --------
        get_slice_contours : The slice contours used to get the box.
        """
        sliceid = np.atleast_1d(sliceid)
        slice_contours = self.get_slice_contours(sliceid, out_format='numpy',
                                                 lbda_range=lbda_range, units=units)
        if len(sliceid)==1:
            # generic format.
            slice_contours = np.atleast_3d(slice_contours)

        return np.percentile(slice_contours, [0, 100], axis=1).T.squeeze()

    def inpaint_slice_onto_image(self, sliceid, image, fillvalue=1, oversampling=3, **kwargs):
        """Paint the footprint of a slice onto a detector image.

        The slice contour is turned into a polygon and every pixel whose
        centre falls inside it is set to ``fillvalue``. The test is run on an
        oversampled grid and rebinned, so that the pixels straddling the edge
        of the slice end up with the fraction of their area it covers.

        Parameters
        ----------
        sliceid : int
            Identifier of the slice to paint.
        image : numpy.ndarray
            2D detector image, modified in place.
        fillvalue : float, optional
            Value given to the pixels fully covered by the slice.
            Default is 1.
        oversampling : int, optional
            Oversampling factor per axis, used to estimate the partial
            coverage of the edge pixels. Default is 3.
        **kwargs
            Goes to `get_slice_contours` (with ``units="pixels"`` and
            ``out_format="numpy"``, the defaults).

        Returns
        -------
        numpy.ndarray
            The input ``image``, with the slice footprint painted in.

        See Also
        --------
        get_slice_contours : The slice contour this paints.

        Notes
        -----
        ``image`` is modified in place, the returned array is the input object
        and not a copy. Only the bounding box of the slice is touched, but it
        is overwritten rather than added to, so painting slices whose bounding
        boxes overlap will erase the earlier ones.
        """
        from shapely import contains_xy, geometry

        from slicersim.utils import bin_array
        # get the vertices of the slice...
        verts = self.get_slice_contours(sliceid, **kwargs)
        # ...and create a shapely Polygon of it.
        poly = geometry.Polygon(verts)

        # get the boundaries of where the polygon is defined to ignore the rest.
        xmin, ymin, xmax, ymax = np.asarray(poly.envelope.buffer(1).bounds, dtype="int")
        xy = np.mgrid[xmin:xmax:1/oversampling, ymin:ymax:1/oversampling]
        # so these are the image pixels that could be overlapping with the polygon
        pixels_ = xy.reshape(2, -1)

        # flag which pixel is contain within the polygon
        flagin = contains_xy(poly, *pixels_)
        img_in = (flagin.reshape(xy.shape[1:]).T).astype(float) * fillvalue

        # and rebin to the actual non-oversampled shape
        if oversampling>1:
            img_in = bin_array(img_in, (oversampling, oversampling), np.mean)

        image[ymin:ymax, xmin:xmax] = img_in
        return image


    def get_pixel_positions(self, coordinates):
        """Get the detector pixel positions of slice-field-wavelength coordinates.

        Parameters
        ----------
        coordinates : pandas.DataFrame or array_like
            Coordinates ``(sliceid, fieldpos, wavelength)`` (wavelength in
            Angstrom), as a DataFrame with these 3 columns (in that order) or
            an array of shape ``(N, 3)``.

        Returns
        -------
        numpy.ndarray
            Detector pixel coordinates ``(x, y)``, of shape ``(N, 2)``. NaN
            outside of the calibration domain.
        """
        xy = self.interp_3d_to_2d(coordinates)
        return self.physical_to_pixels(xy)

    def physical_to_pixels(self, a_physical, from_center=True):
        """Convert physical coordinates (mm) to detector pixel coordinates.

        Parameters
        ----------
        a_physical : numpy.ndarray
            Physical coordinates ``(x, y)`` in mm, of shape ``(..., 2)``.
        from_center : bool, optional
            If True, the physical coordinates are relative to the detector
            center, and ``detector_shape / 2`` is added to the output pixel
            coordinates. Default is True.

        Returns
        -------
        numpy.ndarray
            Pixel coordinates, with the same shape as the input.
        """
        a_pixels = a_physical * getattr(u, self._XY_UNITS).to("micrometer") / self._pixel_size
        if from_center:
            a_pixels += self._detector_shape / 2 # set back the centroid to (0,0)

        return a_pixels

    def pixels_to_physical(self, a_pixels, from_center=True):
        """Convert detector pixel coordinates to physical coordinates (mm).

        Parameters
        ----------
        a_pixels : numpy.ndarray
            Pixel coordinates ``(x, y)``, of shape ``(..., 2)``.
        from_center : bool, optional
            If True, ``detector_shape / 2`` is subtracted from the pixel
            coordinates, such that the physical coordinates are relative to
            the detector center. Default is True.

        Returns
        -------
        numpy.ndarray
            Physical coordinates in mm, with the same shape as the input.

        Notes
        -----
        If ``a_pixels`` is a float `numpy.ndarray` and ``from_center`` is
        True, it is modified in place.
        """
        a_pixels = np.atleast_1d(a_pixels) # copies

        if from_center:
            a_pixels -= self._detector_shape / 2 # set back the centroid to (0,0)

        a_physical = a_pixels / ( getattr(u, self._XY_UNITS).to("micrometer") / self._pixel_size)
        return a_physical

    # ------------ #
    # convertions  #
    # ------------ #
    def slice_pos_wave_to_xy(self, sliceid, fieldpos, lbda):
        """Convert slice, field position and wavelength into detector pixels.

        Parameters
        ----------
        sliceid, fieldpos, lbda : float or array_like
            Coordinates, broadcast to the shape of ``sliceid`` (wavelength in
            Angstrom). For instance, ``sliceid=[1, 10, 80], fieldpos=-1,
            lbda=8000`` is understood as field position -1 at 8000 A for each
            of the 3 slices, while ``sliceid=[10, 80], fieldpos=[0, -1],
            lbda=8000`` is understood as field position 0 for slice 10 and -1
            for slice 80, each at 8000 A.

        Returns
        -------
        numpy.ndarray
            Detector pixel coordinates ``(x, y)``, of shape ``(2, N)`` (or
            ``(2,)`` for a single point).

        See Also
        --------
        xy_to_slice_pos_wave : The inverse conversion.
        """
        # structure broadcasting
        sliceid = np.atleast_1d(sliceid)
        fieldpos = np.broadcast_to(fieldpos, sliceid.shape)
        lbda = np.broadcast_to(lbda, sliceid.shape)

        # stack to match the interpolation's expectations
        coordinates = np.stack([sliceid, fieldpos, lbda]).T
        xy = self.interp_3d_to_2d(coordinates)

        # make sure we have that in pixels
        return self.physical_to_pixels(xy).squeeze().T

    def xy_to_slice_pos_wave(self, x, y):
        """Convert detector pixels into slice, field position and wavelength.

        Parameters
        ----------
        x, y : float or array_like
            Detector pixel coordinates, ``y`` being broadcast to the shape of
            ``x``. For instance, ``x=[1, 10], y=90`` is understood as
            ``y=90`` for both ``x=1`` and ``x=10``, while
            ``x=[1, 10], y=[80, 90]`` is understood as the two points
            ``(1, 80)`` and ``(10, 90)``.

        Returns
        -------
        numpy.ndarray
            ``(sliceid, fieldpos, lbda)`` coordinates (wavelength in
            Angstrom), of shape ``(3, N)``. NaN outside of the calibration
            domain.

        See Also
        --------
        slice_pos_wave_to_xy : The inverse conversion.
        """
        x = np.atleast_1d(x)
        y = np.broadcast_to(y, x.shape)

        # coordinates are as expected by interpolate (in physical units, not pixel.)
        xy = np.stack([x, y]).T
        xy = self.pixels_to_physical(xy)
        return self.interp_2d_to_3d(xy).T


    # ======== #
    #   plot   #
    # ======== #
    def show_slices(self, sliceid, which="lbda", ax=None, **kwargs): # pragma: no cover
        """Display slices on the detector.

        Parameters
        ----------
        sliceid : int or array_like
            Slice id(s) to display.
        which : str, optional
            What to color by: ``"lbda"`` for wavelength, ``"id"`` or
            ``"sliceid"`` for slice id. Default is ``"lbda"``.
        ax : matplotlib.axes.Axes or tuple of matplotlib.axes.Axes, optional
            Axes to plot on. If a tuple of 2 axes, the second is used for the
            colorbar. If None, a new figure is created. Default is None.
        **kwargs
            Goes to the internal plotting method: ``cmap``, ``lbda_min``,
            ``lbda_max`` (Angstrom), ``nlbda`` (for ``which="lbda"``), and
            otherwise to `matplotlib.patches.Polygon`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.

        Raises
        ------
        ValueError
            If ``which`` is not recognized.
        """
        if which == "lbda":
            fig = self._show_slice_lbda_(sliceid, ax=ax, **kwargs)
        elif which in ["id", "sliceid"]:
            fig = self._show_sliceid_(sliceid, ax=ax, **kwargs)
        else:
            raise ValueError(f"cannot parse {which=}. 'lbda' or 'id' expected")

        return fig

    def _show_sliceid_(self, sliceid, ax=None,
                       cmap="viridis", lbda_min=3500, lbda_max=17_000, **kwargs): # pragma: no cover
        """Internal method to plot slices colored by slice ID.

        Parameters
        ----------
        sliceid : int or array_like
            Slice ID(s) to display.
        ax : matplotlib.axes.Axes or tuple, optional
            Axes to plot on. If tuple of 2, second is colorbar axis. Default is None.
        cmap : str, optional
            Colormap name. Default is "viridis".
        lbda_min : float, optional
            Minimum wavelength in Angstrom. Default is 3500.
        lbda_max : float, optional
            Maximum wavelength in Angstrom. Default is 17000.
        **kwargs
            Additional keyword arguments for Polygon patches.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib.patches import Polygon

        sliceid = np.atleast_1d(sliceid)

        if ax is not None:
            if len(np.atleast_1d(ax)) == 1:
                ax, axsc = np.atleast_1d(ax)[0], None
            else:
                ax, axsc = ax
            fig = ax.figure

        else:
            fig, (ax, axsc) = plt.subplots(ncols=2, width_ratios=(20,1))

        norm = colors.Normalize(vmin=1, vmax=self.nslices+1)
        cmap = plt.get_cmap(cmap)

        # get slice vertices
        xys = self.get_slice_contours(sliceid, lbda_range=[lbda_min, lbda_max],
                                     units="pixels",
                                     out_format="numpy")
        # generalize 1 slice or n-slices
        if len(sliceid) == 1:
            xys = np.atleast_3d(xys)

        # color each slice by its id.
        for ids, xy_ in zip(sliceid, xys):
                prop = dict(facecolor=cmap(norm(ids)), edgecolor="None")
                poly = Polygon(xy_, **(prop | kwargs) )
                ax.add_patch(poly)

        # make sure this axe aligns with the detector size
        ymax, xmax = self._detector_shape
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)

        # add the colorbar
        if axsc is not None:
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axsc)
            cbar.set_label("slice id")
            axsc.tick_params(labelsize="small")

        return fig

    def _show_slice_lbda_(self, sliceid, ax=None,
                          nlbda=10, cmap="coolwarm",
                         lbda_min=3500, lbda_max=17_000,
                         **kwargs): # pragma: no cover
        """Internal method to plot slices colored by wavelength.

        Parameters
        ----------
        sliceid : int or array_like
            Slice ID(s) to display.
        ax : matplotlib.axes.Axes or tuple, optional
            Axes to plot on. If tuple of 2, second is colorbar axis. Default is None.
        nlbda : int, optional
            Number of wavelength bins for coloring. Default is 10.
        cmap : str, optional
            Colormap name. Default is "coolwarm".
        lbda_min : float, optional
            Minimum wavelength in Angstrom. Default is 3500.
        lbda_max : float, optional
            Maximum wavelength in Angstrom. Default is 17000.
        **kwargs
            Additional keyword arguments for Polygon patches.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib.patches import Polygon

        sliceid = np.atleast_1d(sliceid)

        if ax is not None:
            if len(np.atleast_1d(ax)) == 1:
                ax, axsc = np.atleast_1d(ax)[0], None
            else:
                ax, axsc = ax

            fig = ax.figure

        else:
            fig, (ax, axsc) = plt.subplots(ncols=2, width_ratios=(20,1))

        norm = colors.Normalize(vmin=lbda_min, vmax=lbda_max)
        cmap = plt.get_cmap(cmap)
        lbda_bins = np.linspace(norm.vmin, norm.vmax, nlbda)

        for lbda_min_, lbda_max_ in pairwise(lbda_bins):
            lbda_mean = np.mean([lbda_min_, lbda_max_])
            xys = self.get_slice_contours(sliceid, lbda_range=[lbda_min_, lbda_max_],
                                                units="pixels",
                                                out_format="numpy")
            for xy_ in xys:
                prop = dict(facecolor=cmap(norm(lbda_mean)), edgecolor="None")
                poly = Polygon(xy_, **(prop | kwargs))
                ax.add_patch(poly)

        # make sure this axe aligns with the detector size
        ymax, xmax = self._detector_shape
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)

        # add the colorbar
        if axsc is not None:
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axsc)
            cbar.set_label(r"Wavelength [$\AA$]")
            axsc.tick_params(labelsize="small")

        return fig

    # =============== #
    #   Properties    #
    # =============== #
    @property
    def keys(self):
        """Column names and units of the calibration data, as given at initialization."""
        return self._metakey

    @property
    def nslices(self):
        """Number of unique slice ids in the calibration data."""
        return self.data[ self.keys["sliceid"] ].nunique()

    @property
    def data(self):
        """Calibration data (`pandas.DataFrame`)."""
        return self._data

    @property
    def interp_3d_to_2d(self):
        """Interpolator from (slice id, field position, wavelength [A]) to detector (x, y) [mm].

        A `scipy.interpolate.LinearNDInterpolator`, built at first access.
        """
        if not hasattr(self, "_interp_3d_to_2d") or self._interp_3d_to_2d is None:
            self._interp_3d_to_2d = LinearNDInterpolator(self._sliceposwave, self._xy)

        return self._interp_3d_to_2d

    @property
    def interp_2d_to_3d(self):
        """Interpolator from detector (x, y) [mm] to (slice id, field position, wavelength [A]).

        A `scipy.interpolate.LinearNDInterpolator`, built at first access.
        """
        if not hasattr(self, "_interp_2d_to_3d") or self._interp_2d_to_3d is None:
            self._interp_2d_to_3d = LinearNDInterpolator(self._xy, self._sliceposwave)

        return self._interp_2d_to_3d

    @property
    def detector(self):
        """Detector properties, as a dict with ``"shape"`` (``(ny, nx)`` in pixels) and ``"pixel_size"`` (micrometers)."""
        return {"shape": self._detector_shape,
                "pixel_size": self._pixel_size}

    @property
    def _deprojected_df(self):
        """Get locally stored deprojected data.

        Returns
        -------
        pandas.DataFrame or None
            Deprojected data if previously stored, else None.
        """
        if not hasattr(self, "_hdeprojected_df"):
            self._hdeprojected_df = None

        return self._hdeprojected_df
