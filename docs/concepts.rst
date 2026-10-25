Key concepts
============

``slicersim`` is organised in layers. Each layer wraps the one below and
exposes fewer, higher-level choices. Start at the top and go down only when
you need more control.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Layer
     - Main objects
     - Use it to...
   * - **Exposure time calculators**
     - `~slicersim.lazuli.lazuli_etc`, `~slicersim.lazuli.lazuli_sn_etc`
     - get an exposure time in one line.
   * - **Targets**
     - `~slicersim.lazuli.LazuliSupernova`, `~slicersim.lazuli.LazuliKilonova`,
       `~slicersim.lazuli.LazuliTarget`, `~slicersim.lazuli.LazuliGalaxy`, ...
       and `~slicersim.sample.Sample`
     - simulate a given astrophysical source with Lazuli: reach a SNR,
       get spectra, compare read-out modes or populations.
   * - **Simulation**
     - `~slicersim.simulation.Simulation`
     - control and inspect every parameter of the scene and the instrument.
   * - **Components**
     - `~slicersim.scene.scene.Scene`, `~slicersim.telescope.Telescope`,
       `~slicersim.spectrograph.SlicerSpectrograph`,
       `~slicersim.detector.Detector`
     - model a new instrument or a new kind of scene.

Targets
-------

A target is the recommended entry point. ``Lazuli*`` classes combine a
source with the Lazuli instrument configuration; they share the same
methods:

.. list-table::
   :widths: 40 60

   * - `~slicersim.target.VirtualTarget.change_properties`
     - change the source (redshift, phase, magnitude, ...).
   * - `~slicersim.target.VirtualTarget.change_detector`
     - set the read-out mode (``nmd``, ``max_group``, ``nramps``).
   * - `~slicersim.lazuli.VirtualLazuliTarget.change_spectrograph`
     - switch between the ``"narrow"`` (40 mas) and ``"wide"`` (80 mas) fields;
       the field also sets the throughput curve used.
   * - `~slicersim.lazuli.VirtualLazuliTarget.change_configuration`
     - switch between the current best estimate (``"cbe"``, default) and
       end-of-life (``"eol"``) instrument performance.
   * - `~slicersim.target.VirtualTarget.setup_to_snr`
     - find the read-out mode that reaches a requested signal-to-noise.
   * - `~slicersim.target.VirtualTarget.get_exposure_time`
     - total exposure time of the current configuration.
   * - `~slicersim.target.VirtualTarget.get_spectrum`
     - a realistic (noisy) spectrum and its variance.
   * - `~slicersim.target.VirtualTarget.get_variance_contribution`
     - the variance broken down by noise source.
   * - `~slicersim.lazuli.VirtualLazuliTarget.get_cube`
     - the simulated 3D data cubes of both fields.

Every target holds its `~slicersim.simulation.Simulation` in
``target.simulation``, so you can always drop to the next layer.

Simulation
----------

A `~slicersim.simulation.Simulation` chains four components:

.. code-block:: text

   Scene  ──►  Telescope  ──►  Spectrograph  ──►  Detector  ──►  spectrum / cube

- **Scene**: point source, background (zodiacal light) and host galaxy;
- **Telescope**: aperture, PSF and thermal emission;
- **Spectrograph**: spaxel grid, dispersion, throughput and line spread;
- **Detector**: QE, dark current, read-out noise and MACC read-out mode.

It is built from a configuration dictionary, read from TOML files shipped
with the package (see :doc:`instrument` for the default values):

.. code-block:: python

   import slicersim

   config = slicersim.get_config(scene="supernova.toml", instrument="lazuli_cbe.toml")
   sim = slicersim.Simulation.from_config(config)

Any of its parameters can then be changed with
`~slicersim.simulation.Simulation.update`, using either the full
``component__parameter`` name or an unambiguous short name:

.. code-block:: python

   sim.mutable_parameters                       # everything you can change
   sim.update(pointsource__redshift=1.0, c=0.1)
   sim.update(detector__nmd=(40, 10, 0), nramps=2)

   lbda, flux, variance = sim.get_spectrum()    # [ADU]
   sim.show_variance_sources()                  # what dominates the noise?

.. image:: gallery/spectra_variance.png
   :alt: Simulated spectrum with the variance broken down by noise source.

Host galaxy
-----------

Any target accepts a ``host``, a `~slicersim.scene.host.Host`: a spectrum
times an image, rendered on the spaxel grid and convolved by the PSF. It is
given by a model name ``"spatial:spectral"``:

- **spatial**: ``sersic`` (analytic profile, size ``r_kpc``) or an NGC
  galaxy, e.g. ``ngc0628`` (its image);
- **spectral**: ``bb`` / ``blackbody``, optionally with a temperature
  (``bb-5000``), or an NGC galaxy, e.g. ``ngc0628`` (its spectrum).

A bare NGC name, ``"ngc0628"``, means ``"ngc0628:ngc0628"``. NGC data come
from the ``lazuli_astrosim_data`` package.

.. code-block:: python

   sn = slicersim.LazuliSupernova(redshift=0.9, host="ngc0628")
   sn = slicersim.LazuliSupernova(redshift=0.9, host="sersic:bb-5000")
   sn = slicersim.LazuliSupernova(redshift=0.9,
                                  host={"refmodel": "sersic:bb-5000", "r_kpc": 3})

   sn.change_properties(host__position=(3, 2))   # host only
   sn.change_properties(redshift=0.5)            # point source and host

The host shares the redshift of the point source. Its brightness
``magabs`` (default -21) is a rest-frame absolute magnitude in ``band``
(K-corrected, as for the supernova), and its physical size is kept fixed, so
changing the redshift moves the whole system consistently. See
`~slicersim.scene.host.get_host_config` for the default parameters and the
:doc:`notebooks/beginner_host_galaxy` tutorial.

A galaxy can also be observed alone, without point source, with
`~slicersim.lazuli.LazuliGalaxy` and the same model names:

.. code-block:: python

   galaxy = slicersim.LazuliGalaxy(redshift=0.5, model="ngc0628")
   galaxy.change_properties(redshift=1.0, position=(5, 0))
   (cube_narrow, _), (cube_wide, _) = galaxy.get_cube(which="both")

Its spectrum (``get_spectrum``, and so ``setup_to_snr``...) is extracted
within an ``aperture``: ``"core"`` (the core, extracted as a point source
would be), ``"<n>-arcsec"`` or ``"<n>-kpc"`` (aperture photometry within n
arcsec or n kpc), centered on the galaxy or on ``aperture__position``:

.. code-block:: python

   galaxy = slicersim.LazuliGalaxy(redshift=0.5, model="ngc0628", aperture="2-kpc")
   config, snr = galaxy.setup_to_snr(10)
   galaxy.change_properties(aperture="1-arcsec", aperture__position=(-20, 12))
   lbda, flux, variance = galaxy.get_spectrum()

The default aperture is ``"1-arcsec"``, centered on the galaxy. See
`~slicersim.simulation.Simulation.get_aperture` and the
:doc:`notebooks/beginner_galaxy` tutorial.

Extraction method
-----------------

The extraction ``method`` of `~slicersim.simulation.Simulation.get_spectrum`
is either ``"optimal"`` (PSF-weighted, the default for a point source) or
``"aperture"`` (aperture photometry: plain summation within the
``aperture``, the default for a galaxy but ``aperture="core"``). It applies to
point sources too, the aperture being then centered on them, and to the host
of a point source, with ``of_host=True``:

.. code-block:: python

   sn = slicersim.LazuliSupernova(redshift=0.9, host="ngc0628")
   lbda, flux, variance = sn.get_spectrum(method="aperture")            # the SN, within 1 arcsec
   lbda, flux, variance = sn.get_spectrum(of_host=True)                 # its host, within 1 arcsec

   # stored, hence used by setup_to_snr, get_variance_contribution...
   sn.change_properties(method="aperture", aperture="0.2-arcsec")
   config, snr = sn.setup_to_snr(10)

See the :doc:`notebooks/beginner_spectrum_extraction` tutorial.

Noise sources
-------------

The variance of a simulated spectrum is the sum of the following
contributions, each of which can be switched off (``switch_off=[...]``) to
study its impact:

- **scene**: ``pointsource``, ``background`` (zodiacal light), ``host``;
- **instrument**: ``thermal`` (thermal emission of the optics);
- **detector**: ``dark``, ``thermal_dark``, ``roic_glow`` and ``ron``
  (read-out noise, which depends on the MACC mode).

See the :doc:`notebooks/beginner_variancesource` tutorial.

Units
-----

- wavelengths are in Angstrom;
- input fluxes are in erg s\ :sup:`-1` cm\ :sup:`-2` Å\ :sup:`-1`
  (or rescaled to a magnitude with ``mag`` and ``band``);
- simulated data are in ADU unless a ``unit`` is requested;
- times are in seconds, angles on the sky in arcsec.
