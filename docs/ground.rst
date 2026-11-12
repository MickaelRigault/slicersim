Ground-based instruments
========================

``slicersim`` also simulates ground-based slicer integral field
spectrographs, through the :mod:`slicersim.ground` module. A ground-based
observation differs from a space one in three ways:

- **the atmosphere** (:class:`~slicersim.atmosphere.Atmosphere`): a
  seeing-limited spatial PSF, the atmospheric transmission (applied to the
  astrophysical sources only) and the sky emission (the scene background).
  The observing conditions, ``seeing``, ``airmass`` and ``moon``, are mutable
  parameters;
- **the spectrograph**
  (:class:`~slicersim.ground_spectrograph.GroundSlicerSpectrograph`): a
  chromatic Moffat PSF whose FWHM is the quadratic sum of the
  **seeing-induced** FWHM (from the atmosphere) and of the **optical** FWHM
  (``optical_fwhm``, telescope + instrument). It uses 1 detector pixel per
  voxel (see :doc:`anamorphosis`) and an LSF given as a FWHM;
- **the detector** (:class:`~slicersim.detector.CCDDetector`): a CCD read
  once per exposure, with NDIT exposures of DIT seconds (``ndit`` is an alias
  of ``nramps``).

:class:`~slicersim.ground.GroundSimulation` combines them. It also adds a
``"box"`` extraction (square aperture) and an exposure time calculator for
CCDs, which finds the NDIT for a fixed DIT, or the DIT for a fixed NDIT.

Using an instrument
-------------------

``slicersim`` ships no ground-based instrument itself: instruments are
described by configuration files, usually provided by an add-on package.
For example, the `slicersim-wst <https://gitlab.in2p3.fr/ycopin/slicersim-wst>`_ add-on provides the blue and red channels
of the Wide-field Spectroscopic Telescope (WST) integral field spectrograph:

.. code-block:: python

   from slicersim.ground import GroundTarget, ground_etc

   target = GroundTarget(lbda, flux, instrument="path/to/my_ifs.toml",
                         seeing=0.8, airmass=1.0, moon="darksky")
   lbda, signal, variance = target.simulation.get_spectrum()

   # NDIT x 1800 s exposures needed for SNR = 10 per pixel in 4800-5200 Å
   exptime, target = ground_etc(target, 10, lbda_range=[4800, 5200], dit=1800.)

   # observing conditions, exposure and PSF parameters can be changed
   target.change_properties(seeing=0.6, moon="greysky", ndit=3, optical_fwhm=0.2)

Writing an instrument configuration
-----------------------------------

The instrument configuration is a TOML file with an ``[atmosphere]`` section
in addition to the usual ``[telescope]``, ``[spectrograph]``, ``[detector]``
and ``[extraction]`` sections. Data files named without a directory
(throughput, dispersion law, sky table) are looked for **next to the
configuration file** first, then in the ``slicersim`` configuration
directory (see :func:`~slicersim.iotools.resolve_config_paths`). An
instrument package can therefore ship its configuration and data files
together, and pass the path of the TOML file as ``instrument``.

The ground-specific entries are:

.. code-block:: toml

   [atmosphere]
   site = "Paranal"
   sky_table = "my_ifs_sky.ecsv"   # transmission and sky emission, see below
   seeing = 0.8                    # seeing FWHM [arcsec] at 5000 Å and zenith
   airmass = 1.0                   # must be tabulated in sky_table
   moon = "darksky"                # must be tabulated in sky_table

   [spectrograph]
   dispersion_resolution = 2.5     # LSF FWHM [px]

   [spectrograph.psf.spatial]
   optical_fwhm = 0.1              # optical (telescope + instrument) FWHM [arcsec]
   moffat_beta = 2.5               # Moffat shape parameter
   moffat_truncation = 0           # 0: exact Moffat; k: truncated at ~k FWHM (pyetc)

   [detector]
   ron = 3.0                       # read-out noise per exposure [e-]
   dark = 8.3e-4                   # dark current [e-/s/px]
   dit = 1800.0                    # exposure time of a single exposure [s]

   [extraction]
   nramps = 1                      # number of exposures (NDIT)
   box_hsize = 2                   # half size of the "box" aperture [spaxels]

The sky table is an ECSV file with a ``wavelength`` column [Å], one
``transmission_<airmass>`` column per airmass (e.g. ``transmission_1.2``)
and one ``emission_<moon>_<airmass>`` column per sky condition and airmass
(e.g. ``emission_darksky_1.2``) in erg/s/cm²/Å/arcsec², both at the
resolution of the instrument (see :func:`~slicersim.atmosphere.read_sky_table`).
