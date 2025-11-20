import os
import numpy as np
import pandas
    

class CalSpecSource():
    """
    A class to handle the retrieval and processing of CalSpec data.

    This class provides methods to download file lists, get URLs of specific sources,
    and retrieve spectral data.
    """
    
    _ARCHIVE_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
    def __init__(self):
        """ Initialize the CalSpecSource instance.

        This method initializes the instance by downloading the file list of CalSpec sources.
        """
        self._df_source = self.download_filelist()
        
    @classmethod
    def download_filelist(cls):
        """ Download the list of all the CalSpec star files.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the list of files and associated star names.
        """
        import requests
        calspeclit = requests.get(cls._ARCHIVE_URL)
        list_of_files= [entry.split("</a>")[0].split(">")[-1]
                        for entry in calspeclit.text.splitlines() if ".fits" in entry]
        df = pandas.DataFrame(list_of_files, columns=["basename"])
        df["name"] = df["basename"].str.split("_").apply(lambda x: "_".join(x[:-2]) )
        return df.set_index("name")

    def get_url_of(self, name):
        """ get the URL of a CalSpec source

        Parameters
        ----------
        name : str
            The name of the source. 
            You can simply provide the start of the source name.
            for instance: name='bd_17' for 'bd_17d4708'. However
            name='bd' will break as several bd_* star exist.

        Returns
        -------
        str
            The URL of the specified source.
        """
        if name in self.source.index:
            basename = self.source.loc[name]["basename"]
            if type(basename) is pandas.Series: #multiple entries
                basename = basename.iloc[-1] # take the last
        else:
            fetched = self.source[self.source.index.str.startswith(name)]
            if len(fetched) >= 2:
                if fetched.index.nunique() > 1: # crash if different objects
                    raise ValueError(f"multiple matches for {name=}: {list(fetched.index)}")
            basename = fetched.iloc[-1]["basename"]

        return os.path.join(self._ARCHIVE_URL, basename)

    def get_data(self, name):
        """ retrieve fits data a specific CalSpec source.

        Parameters
        ----------
        name : str
            The name of the source. 
            You can simply provide the start of the source name.
            for instance: name='bd_17' for 'bd_17d4708'. However
            name='bd' will break as several bd_* star exist.

        Returns
        -------
        astropy.io.fits
        """
        from astropy.io import fits
        # get the data url
        url_target = self.get_url_of(name)
        return fits.open(url_target)
    
    def get_spectrum(self, name, lbda_range=[3_000, 20_000], var_source=["sys", "stat"]):
        """ retrieve the spectrum of a specific CalSpec source.

        Parameters
        ----------
        name : str
            The name of the source. 
            You can simply provide the start of the source name.
            for instance: name='bd_17' for 'bd_17d4708'. However
            name='bd' will break as several bd_* star exist.

        lbda_range : list
            The wavelength range to filter the spectrum data. 

        var_source : list
            The sources of variance to include. 
            If both sys and stat are given, the sum of variance is given.

        Returns
        -------
        tuple
            (wavelength, flux, variance)
        """
        # grab data from calspec url
        fits_data = self.get_data(name)
        
        # open the spectrum from the fits file and convert it into dataframe.
        df_data = pandas.DataFrame( fits_data[1].data )
        df_data = df_data.set_index(df_data.index.values) # clean
            
        lbda = df_data["WAVELENGTH"].astype(float).values
        flux = df_data["FLUX"].astype(float).values
        err_stat = df_data["STATERROR"].astype(float).values
        err_sys = df_data["SYSERROR"].astype(float).values
        
        # must do flag on np array because of fits format.
        if lbda_range is not None:
            flag = df_data["WAVELENGTH"].between(*lbda_range).astype(bool).values
            lbda = lbda[flag]
            flux = flux[flag]

        # variance
        if var_source is None:
            variance = None
        else:
            if "stat" in var_source:
                variance = err_stat[flag]**2
            else:
                variance = np.zeros(flux.shape)
                
            if "sys" in var_source:
                variance += err_sys[flag]**2
        
        return lbda, flux, variance

    # ============ #
    #  Properties  #
    # ============ #
    @property
    def source(self):
        """ dataframe containing the list of CalSpec sources. """
        return self._df_source




# ================ #
try:
    calspecsource = CalSpecSource() # loads it.
except: # noqa: E722
    calspecsource = None
    import warnings
    warnings.warn("Cannot run calspecsource = CalSpecSource()")

    
