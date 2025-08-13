Welcome to slicersim
======================

`slicersim` is a Python package for simulating integral field spectrograph (IFS) data.
It provides tools to create realistic datacubes, including various noise sources and instrument effects.

*Top level functionalities of this package is currently built for the
Lazuli Space Telescope. But the code is generic and can be extended to
any Integral Field Units (slicers or micro-lense arrays).*


Sharp start
============

Create the target of interest. Setup the configuration you
want. Obtained simulated spectrum.

Create the target of interest
----------------------

.. tab-set::
    :sync-group: category

    .. tab-item:: Supernovae
        :sync: sn
	       
        .. code-block:: python

	    import slicersim

	    # load a Type Ia Supernovae
	    target = slicersim.LazuliSN(redshift=1.0, c=0.2, phase=1.5)

    .. tab-item:: CalSpec
        :sync: star
	       
        .. code-block:: python

	    import slicersim
	  
	    # provide the name (or short-name) of any calspec star
	    target = slicersim.LazuliCalSpec("bd_17")
	  
    .. tab-item:: Anything
        :sync: flat		  

        .. code-block:: python

	    import slicersim
	    import numpy as np

	    # provide your spectrum | here a flat spectrum
	    lbda = np.arange(3_000, 20_000, 0.5) # every 0.5A
	    flux = np.ones( lbda.shape )

	    # build it forcing it to have a lsst-g of 20 mag.
	    target = slicersim.LazuliTarget(lbda, flux, mag=20, band="lsstg")

	    
Set observing conditions
----------------------

.. tab-set::

    .. tab-item:: to a mean SNR
        :sync: snr
	       
        You can setup the read-out properties such that a requested
        signal-to-noise is reached. 
      
        .. code-block:: python

	    # say you want an average SNR of 20 between [5000, 6000] rest-frame
	    _ = target.setup_to_snr(20, lbda_range=[5000, 6000], frame="rest")

	    # get corresponding total exposure time, 
	    exposure_time = target.get_exposure_time() # see options

	  
    .. tab-item:: to a read-out mode

        You can directly specify how the detector should be red.
       
        .. code-block:: python

	    # Specify the detector read-out mode and number of ramps
	    target.change_detector_mode(nmd=(40, 10, 0), nramp=2)

	    # get corresponding total exposure time, 
	    exposure_time = target.get_exposure_time() # see options

	  
Get realistic spectra
----------------------

.. code-block:: python

    # get the flux and variance in erg/s/cm2/A ; see unit options
    lbda, flux, variance = target.get_spectrum(unit="flambda")

.. tab-set::
    :sync-group: category

    .. tab-item:: Supernovae
        :sync: sn
	       
        Showing the SN Ia
	  
    .. tab-item:: CalSpec
        :sync: star		  

        Showing the CalSpec

    .. tab-item:: Anything
        :sync: flat		  

        Showing Anything
