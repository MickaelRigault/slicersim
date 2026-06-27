import os
import numpy as np
import pandas


class PicklesSource():
    """
    A class to handle the retrieval of Pickles stellar atlas spectra.

    This class reads the spectral templates of the Pickles (1998) stellar
    atlas, bundled with the package, and provides them by spectral type.

    Source: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas.html
    """

    _DATA_DIR = os.path.join(os.path.dirname(__file__), "pickles_models")

    def __init__(self):
        """ Initialize the PicklesSource instance.

        This method loads the mapping between spectral types and the bundled
        spectrum files.
        """
        self._df_source = self.load_filelist()

    @classmethod
    def load_filelist(cls):
        """ Load the mapping of all the Pickles spectrum files.

        Returns
        -------
        pandas.DataFrame
            A DataFrame mapping the spectral type ("spt") to its file
            ("filename") and effective temperature ("teff").
        """
        mapping = os.path.join(cls._DATA_DIR, "pickles_mapping.csv")
        df = pandas.read_csv(mapping, sep=r'\s+')
        return df.set_index("spt")

    def get_filename(self, name):
        """ get the filepath of a Pickles source.

        Parameters
        ----------
        name : str
            The spectral type of the source (e.g. 'G2V').

        Returns
        -------
        str
            The full path of the spectrum file.
        """
        if name not in self.source.index:
            raise ValueError(f"No spectrum found for {name=}. "
                             f"Available spectral types are {list(self.source.index)}.")

        basename = self.source.loc[name]["filename"]
        if type(basename) is pandas.Series:  # multiple entries
            basename = basename.iloc[-1]  # take the last

        return os.path.join(self._DATA_DIR, "dat_uvk", basename + ".fits")

    def get_data(self, name):
        """ retrieve fits data of a specific Pickles source.

        Parameters
        ----------
        name : str
            The spectral type of the source (e.g. 'G2V').

        Returns
        -------
        astropy.io.fits
        """
        from astropy.io import fits
        return fits.open(self.get_filename(name))

    def get_spectrum(self, name, lbda_range=None):
        """ retrieve the spectrum of a specific Pickles source.

        Parameters
        ----------
        name : str
            The spectral type of the source (e.g. 'G2V').

        lbda_range : list, optional
            The wavelength range to filter the spectrum data.
            Default is None (no filtering).

        Returns
        -------
        tuple
            (wavelength, flux, variance)
            The Pickles templates are relative, so variance is always None.
        """
        # grab data from the bundled fits file
        fits_data = self.get_data(name)

        # open the spectrum from the fits file and convert it into dataframe.
        df_data = pandas.DataFrame(fits_data[1].data)

        lbda = df_data["WAVELENGTH"].astype(float).values
        flux = df_data["FLUX"].astype(float).values

        if lbda_range is not None:
            flag = df_data["WAVELENGTH"].between(*lbda_range).astype(bool).values
            lbda = lbda[flag]
            flux = flux[flag]

        return lbda, flux, None

    # ============ #
    #  Properties  #
    # ============ #
    @property
    def source(self):
        """ dataframe containing the list of Pickles sources. """
        return self._df_source




# ================ #
try:
    picklessource = PicklesSource()  # loads it.
except:  # noqa: E722
    picklessource = None
    import warnings
    warnings.warn("Cannot run picklessource = PicklesSource()")
