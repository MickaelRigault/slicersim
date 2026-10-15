The Lazuli instrument model
===========================

The ``Lazuli*`` targets are configured with a model of the Lazuli image
slicer, shipped with the package as TOML and data files in
``slicersim/config/``. This page summarises what is assumed, so you know
what your exposure times and spectra rely on, and how to change it.

.. note::

   The values below are those of the configuration files distributed with
   this version of ``slicersim``. They follow the current state of the
   Lazuli design and may evolve between releases.

Two fields on one detector
--------------------------

Lazuli observes the same region of the sky with two spaxel samplings,
imaged side by side on a single detector:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Field
     - Spaxel grid
     - Spaxel size
     - Use it for...
   * - ``"narrow"``
     - 58 × 58
     - 0.04 arcsec
     - bright or nearby sources, finely sampled PSF (default).
   * - ``"wide"``
     - 58 × 58
     - 0.08 arcsec
     - faint, distant sources: fewer spaxels per PSF, hence less
       detector noise (`~slicersim.sample.Sample` uses it above z = 0.8).

Each field has its own throughput curve. Switch between them with
`~slicersim.lazuli.VirtualLazuliTarget.change_spectrograph`:

.. code-block:: python

   target.change_spectrograph("wide")
   target.get_spectrograph_field()   # ('wide', {...})

`~slicersim.lazuli.VirtualLazuliTarget.get_cube` returns the cubes of both
fields, and `~slicersim.lazuli.VirtualLazuliTarget.to_image` projects them
on the detector.

Default parameters
------------------

Two instrument configurations are provided. The **current best estimate**
(CBE, ``lazuli_cbe.toml``) is the default. The **end-of-life** configuration
(EOL, ``lazuli_eol.toml``) applies degraded throughput and detector
performance on top of it.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Parameter
     - CBE (default)
     - EOL
   * - Primary mirror diameter
     - 3.0 m
     - same
   * - Wavelength range
     - 4 000 – 17 000 Å
     - same
   * - Resolving power :math:`R = \lambda / \Delta\lambda`
     - ≈ 100 (near 1 µm) to ≈ 370 (4 000 Å)
     - same law
   * - Pixels per resolution element
     - 2.3
     - 2.5
   * - Throughput
     - ``throughput_bol.csv``
     - ``throughput_eol.csv``
   * - Detector
     - 4096 × 4096 H4RG, 10 µm pixels
     - same
   * - Read-out noise (per frame)
     - 12 e\ :sup:`-`
     - 20 e\ :sup:`-`
   * - Dark current
     - 2.8 × 10\ :sup:`-3` e\ :sup:`-` s\ :sup:`-1` px\ :sup:`-1`
     - 10\ :sup:`-2` e\ :sup:`-` s\ :sup:`-1` px\ :sup:`-1`
   * - ROIC glow
     - 3 × 10\ :sup:`-3` e\ :sup:`-` px\ :sup:`-1` frame\ :sup:`-1`
     - 10\ :sup:`-2` e\ :sup:`-` px\ :sup:`-1` frame\ :sup:`-1`
   * - Frame time
     - 2.83 s
     - same
   * - Gain / saturation
     - 1 ADU/e\ :sup:`-` / 65 535 ADU
     - same
   * - Maximum number of groups per ramp
     - 64
     - same
   * - Up-the-ramp variance model
     - Rauscher et al. (2007)
     - same
   * - Background
     - zodiacal light (Aldering 2001), ×2 the North Ecliptic Pole level
     - same

Switch a target to the end-of-life configuration with
`~slicersim.lazuli.VirtualLazuliTarget.change_configuration`:

.. code-block:: python

   import slicersim

   target = slicersim.LazuliSupernova(redshift=1.0)
   target.change_configuration("eol")   # back with "cbe"
   target.setup_to_snr(20)

Detector read-out
-----------------

The detector is read up-the-ramp in MACC mode, described by
``nmd = (n, m, d)``:

- ``n``: number of groups in the ramp (at most ``max_group``);
- ``m``: number of frames averaged per group;
- ``d``: number of frames dropped between two groups.

A ramp lasts :math:`[n\,(m + d) - d] \times t_\mathrm{frame}`, and an
exposure is made of ``nramps`` identical ramps.
`~slicersim.target.VirtualTarget.setup_to_snr` searches for the read-out
configuration (``nmd`` and ``nramps``) that reaches the requested
signal-to-noise; you can also set
it yourself with `~slicersim.target.VirtualTarget.change_detector`. See the
:doc:`notebooks/advanced_macc_readout_modes` tutorial.

Using your own configuration
----------------------------

Every parameter above can be changed on a live simulation with
`~slicersim.simulation.Simulation.update` (see
``simulation.mutable_parameters``), for instance:

.. code-block:: python

   target.simulation.update(ron=8, dark=1e-3)

To build a whole new instrument, write a TOML file with the same structure
as ``lazuli_cbe.toml`` and load it with `~slicersim.iotools.get_config`:

.. code-block:: python

   config = slicersim.get_config(scene="supernova.toml",
                                 instrument="/path/to/my_instrument.toml")
   sim = slicersim.Simulation.from_config(config)
