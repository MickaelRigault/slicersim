""" This module handles CalSpec spectrophotometric standard stars.

`CalSpecSource` fetches the HST CalSpec reference atlas from the STScI
archive and exposes the tabulated spectra of the standard stars, which are
used as flux calibration references for the simulated observations.
"""

import os

import numpy as np
import pandas


class CalSpecSource:
    """Access to the HST CalSpec spectrophotometric standard stars.

    This class downloads the list of CalSpec files from the STScI archive and
    provides methods to get the URL, the fits data and the spectrum of a
    given star. A module-level instance, ``calspecsource``, is created at
    import; the file list is only downloaded at first access of `source`.

    References
    ----------
    - Bohlin, R. C., Gordon, K. D., & Tremblay, P.-E. 2014, PASP, 126,
      711.
    """

    _ARCHIVE_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
    def __init__(self):
        """Initialize the CalSpecSource instance.

        The list of CalSpec files is downloaded from the STScI archive
        at first access of `source` (not at import, to keep it fast).
        """
        self._df_source = None

    @classmethod
    def download_filelist(cls):
        """Download the list of all the CalSpec star files.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by star name, with the fits file names in the
            ``"basename"`` column.
        """
        import requests
        calspeclit = requests.get(cls._ARCHIVE_URL, timeout=30)
        calspeclit.raise_for_status()
        list_of_files= [entry.split("</a>")[0].split(">")[-1]
                        for entry in calspeclit.text.splitlines() if ".fits" in entry]
        df = pandas.DataFrame(list_of_files, columns=["basename"])
        df["name"] = df["basename"].str.split("_").apply(lambda x: "_".join(x[:-2]) )
        return df.set_index("name")

    def get_url_of(self, name):
        """Get the URL of a CalSpec source.

        If several files exist for the star, the last one (latest version)
        is used.

        Parameters
        ----------
        name : str
            The name of the source. The start of the name is enough, e.g.
            ``name="bd_17"`` for ``"bd_17d4708"``. However ``name="bd"``
            fails as several ``bd_*`` stars exist.

        Returns
        -------
        str
            The URL of the fits file of the source.

        Raises
        ------
        ValueError
            If ``name`` matches several stars.
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
        """Retrieve the fits data of a CalSpec source.

        Parameters
        ----------
        name : str
            The name of the source, see `get_url_of`.

        Returns
        -------
        astropy.io.fits.HDUList
            The opened fits file.
        """
        from astropy.io import fits
        # get the data url
        url_target = self.get_url_of(name)
        return fits.open(url_target)

    def get_spectrum(self, name, lbda_range=[3_000, 20_000], var_source=["sys", "stat"]):
        """Retrieve the spectrum of a CalSpec source.

        Parameters
        ----------
        name : str
            The name of the source, see `get_url_of`.
        lbda_range : list of float, optional
            Wavelength range ``[min, max]``, in Angstrom, to which the
            spectrum is restricted. Default is ``[3000, 20000]``.
        var_source : list of str or None, optional
            The sources of variance to include, among ``"stat"`` and
            ``"sys"``. If both are given, the variances are summed. If None,
            no variance is returned. Default is ``["sys", "stat"]``.

        Returns
        -------
        lbda : numpy.ndarray
            Wavelength in Angstrom.
        flux : numpy.ndarray
            Flux in erg/s/cm^2/A.
        variance : numpy.ndarray or None
            Flux variance in (erg/s/cm^2/A)^2, None if ``var_source`` is
            None.
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
        """DataFrame of the CalSpec files, indexed by star name."""
        if self._df_source is None:
            self._df_source = self.download_filelist()
        return self._df_source

# ================ #
calspecsource = CalSpecSource() # file list downloaded lazily.
