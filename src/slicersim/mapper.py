import numpy as np

from .utils import mesh_kwargs, unbin_array


class SlicerMapper():
    
    def __init__(self, spotdata,
                 detector_shape=(4096, 4096), 
                 pixel_size=10, # in micrometer
                 spot_units="mm"
                ):
        """ """
        self._spotdata = spotdata
        self._interp_map = self._build_interp_map_(spotdata)
        self._detector_shape = np.asarray(detector_shape, dtype="int")
        self._pixel_size = float(pixel_size) # in micro
        self._units_in = spot_units
        
    @classmethod
    def from_spotdata(cls, spotdata, spot_units="mm"):
        """ """
        if type(spotdata) in [str, np.str_]:
            spotdata = pandas.read_csv(spotdata, sep=" ")
            
        return cls(spotdata, spot_units=spot_units)

    @staticmethod
    def _build_interp_map_(spotdata, lbda_in="micrometer", lbda_out="angstrom"):
        """ """
        from scipy.interpolate import LinearNDInterpolator
        from astropy import units as u
        units_convert = getattr(u, lbda_in).to(lbda_out)
        spotdata["lbda"] = spotdata["wavel_nm"].astype(float)*units_convert
        sliceposwave = spotdata[["slice", "fieldpos", "lbda"]].astype(float).values
        xy = spotdata[["x_mm", "y_mm"]].astype(float).values # in milimiter
        return LinearNDInterpolator(sliceposwave, xy)
        
    # =============== #
    #   method        #
    # =============== #
    def project_lazulitarget(self, lazulitarget, psf_profile="airy", switch_off=[],
                                 **kwargs):
        """ """
        
        # generate the cubes (one per channel), you can remove contributions.
        (cube_fine, _), (cube_wide, _) = lazulitarget.get_cube(which='both',
                                                               psf_profile=psf_profile,
                                                               switch_off=switch_off)

        # get the wavelength array
        lbda = lazulitarget.simulation.spectrograph.lbda

        # fine grid are slices between 1 and 59 | wide between 59 and 117
        # [::-1] as top <-> bottom definition inversion
        # project them one at the time.
        img_fine = self.project_slice(np.arange(1, 59)[::-1], cube_fine, lbda)
        img_med = self.project_slice(np.arange(59, 117)[::-1], cube_wide, lbda)
        # combine them to get the full image.
        return np.sum([img_med, img_fine], axis=0)

 
    def project_slice(self, sliceid, sliceimg, lbda, oversample=(5, 2), fill_value=0):
        """ project the slice(s) into the detector
        
        Parameters
        ----------
        sliceid: int, list, array
            index of the slice(s).
    
        sliceimg: ndarray
            image associated with the slices 
            [nlbda, (nslices,)  nspatial_spaxels]
    
        lbda: 1d array
            wavelength (in Angstrom) corresponding to 
            the slice image along the spectral direction.
    
        oversample: int, list
            oversampling to be applied to the image. 
            If int, both wavelength and spatial directions 
            are oversampled similarly, otherwise 
            (spectral_oversampling, spatial_oversampling)
    
        fill_value: float
            value in the image prior projecting the image.
    
        Returns
        -------
        image: 2darray
            2d image correspinding to the detector size.
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
        ## 1. image full of nan
        image = np.full(self.detector["shape"], fill_value=fill_value, dtype="float")
        
        # .T[::-1] as numpy vs. mpl conventions: pixels[:,1], pixels[:,0]
        # flag_nan remove entries outsize of interpolation boundaries.
        x_, y_ = pixels[~flag_nan].astype(int).T[::-1]
        image[x_, y_] += value_pixels_t[~flag_nan]

        # the detector image
        return image
    
    def get_slice_contours(self, sliceid, lbda_range=[3_500, 17_000], 
                          out_format="numpy", units="pixels",
                          slice_edge=[-1, 1],
                          combined=False):
        """
        Parameters
        ----------
        out_format: string
            output format of the contours:
            - shapely (or geometry): a shapely polygon
            - numpy (or array): vertices of the edge.

        units: string
            units of the output 

        combined: bool
            for several sliceid are given, should this return the edge of all
            combined (True) or a list of edges (False)
        """
        from shapely import geometry # new dependecy

        if not combined and len(np.atleast_1d(sliceid))>1:
            return [self.get_slice_contours(sliceid_, 
                                            lbda_range=lbda_range,
                                            out_format=out_format, 
                                            units=units,
                                            slice_edge=slice_edge,
                                            combined=False)
                             for sliceid_ in sliceid]
        
        # slicer position are in units of slice (-1, 0, 1)
        slicepos = np.linspace(*slice_edge, 5)
        wavelength = np.linspace(*lbda_range, 10)
        
        # get the mesh of all this information: lbda & slicepos
        df_in = mesh_kwargs(sliceid=sliceid, slicepos=slicepos, 
                            wavelength=wavelength)
        
        # xy are the "position" in the detector
        xy = self.interp_map(df_in)
            
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
        """ get the xmin, xmax, ymin, ymax of the rectangle in boxing the slice(s)

        Parameters
        ----------

        Returns
        -------
        extrema: array
            [[(xmin, xmax), (ymin, ymax)], ] for each slices.
        """ 
        sliceid = np.atleast_1d(sliceid)
        slice_contours = self.get_slice_contours(sliceid, out_format='numpy', 
                                                 lbda_range=lbda_range, units=units)
        if len(sliceid)==1:
            # generic format.
            slice_contours = slice_contours[None,:]
    
        return np.percentile(slice_contours, [0, 100], axis=1).T.squeeze()

    def get_pixel_positions(self, coordinates):
        """ get the position positions corresponding to the input paramaters. 

        coordinates: dataframe
            slice coordinates: [sliceid, slicepos, wavelength]
            
        Returns
        -------
        pixels
        """
        xy = self.interp_map(coordinates)
        return self.physical_to_pixels(xy)
        
    def physical_to_pixels(self, a, from_center=True):
        """ converts physical coordinates into pixel coordinates """
        from astropy import units

        
        a_pixels = a * getattr(units, self._units_in).to("micrometer") / self._pixel_size
        if from_center:
            a_pixels += self._detector_shape / 2 # set back the centroid to (0,0)
            
        return a_pixels

    # ======== #
    #   plot   #
    # ======== #
    def show_slices(self, sliceid, which="lbda", ax=None, **kwargs): # pragma: no cover
        """ """
        if which == "lbda":
            fig = self._show_slice_lbda_(sliceid, ax=ax, **kwargs)
        elif which in ["id", "sliceid"]:
            fig = self._show_sliceid_(sliceid, ax=ax, **kwargs)
        else:
            raise ValueError(f"cannot parse {which=}. 'lbda' or 'id' expected")
            
        return fig
        
    def _show_sliceid_(self, sliceid, ax=None, 
                       cmap="viridis", lbda_min=3500, lbda_max=17_000, **kwargs): # pragma: no cover
        """ """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib import colors

        sliceid = np.atleast_1d(sliceid)

        if ax is not None:
            if len(np.atleast_1d(ax)) == 1:
                ax, axsc = ax, None
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
            xys = xys[None,:]
                
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
        """ """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib import colors

        sliceid = np.atleast_1d(sliceid)
        
        if ax is not None:
            if len(np.atleast_1d(ax)) == 1:
                ax, axsc = ax, None
            else:
                ax, axsc = ax
                
            fig = ax.figure
            
        else:        
            fig, (ax, axsc) = plt.subplots(ncols=2, width_ratios=(20,1))

        norm = colors.Normalize(vmin=lbda_min, vmax=lbda_max)
        cmap = plt.get_cmap(cmap)
        lbda_bins = np.linspace(norm.vmin, norm.vmax, nlbda)

        
        for lbda_min_, lbda_max_ in zip(lbda_bins[:-1], lbda_bins[1:]):
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
    def nslices(self):
        """ """
        return self.spotdata["slice"].nunique()
    @property
    def spotdata(self):
        """ """
        return self._spotdata
    
    @property
    def interp_map(self):
        """ """
        return self._interp_map
        
    @property
    def detector(self):
        """ """
        return {"shape": self._detector_shape,
                "pixel_size": self._pixel_size}
