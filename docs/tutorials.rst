Tutorials
=========

Each tutorial is a Jupyter notebook. Use the download button at the top of a
page to run it yourself.

Beginner
--------

Simulate a source, get its exposure time and spectrum — no knowledge of the
instrument model needed.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Exposure time calculator
      :link: notebooks/beginner_ETC_lazulitarget
      :link-type: doc

      ``lazuli_etc`` and ``lazuli_sn_etc``: exposure time and spectrum
      in one call.

   .. grid-item-card:: Lazuli targets
      :link: notebooks/beginner_LazuliTargets
      :link-type: doc

      Supernovae, CalSpec stars or any spectrum: reach a SNR and get the
      read-out mode.

   .. grid-item-card:: Changing target properties
      :link: notebooks/beginner_change_properties
      :link-type: doc

      Change redshift, phase or magnitude of a loaded target.

   .. grid-item-card:: Origin of the variance
      :link: notebooks/beginner_variancesource
      :link-type: doc

      Which noise source dominates, and what happens when you switch it off.

   .. grid-item-card:: Kilonovae
      :link: notebooks/beginner_kilonova
      :link-type: doc

      POSSIS models: viewing angle, phase and exposure time.

   .. grid-item-card:: Populations with ``Sample``
      :link: notebooks/beginner_sample_multiple_targets
      :link-type: doc

      Exposure time and data volume across many targets.

Advanced
--------

Control the instrument and the scene.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Detector and spectrograph
      :link: notebooks/advanced_change_detector_and_spectrograph
      :link-type: doc

      Read-out modes and spatial sampling of the spectrograph.

   .. grid-item-card:: MACC read-out modes
      :link: notebooks/advanced_macc_readout_modes
      :link-type: doc

      How ``nmd`` and the number of ramps set noise and exposure time.

   .. grid-item-card:: Instrument properties
      :link: notebooks/advanced_access_qe_spectral_resolution_etc
      :link-type: doc

      Access QE, throughput, spectral resolution and more.

   .. grid-item-card:: Dispersion resolution
      :link: notebooks/advanced_change_resolution_dispersion
      :link-type: doc

      Change the number of pixels per resolution element.

   .. grid-item-card:: Resolution vs. spot size
      :link: notebooks/advanced_dispersion_resolution_and_spotsize
      :link-type: doc

      Trade-off between dispersion resolution and spot size.

   .. grid-item-card:: Standard stars
      :link: notebooks/advanced_calspec_and_stdstars
      :link-type: doc

      CalSpec observations for flux calibration planning.

   .. grid-item-card:: Building a scene
      :link: notebooks/advanced_scene_tutorial
      :link-type: doc

      What a ``Scene`` is and how to build your own.

   .. grid-item-card:: Host galaxy
      :link: notebooks/advanced_host_galaxy
      :link-type: doc

      Add a structured host background and change its shape.

.. toctree::
   :hidden:
   :caption: Beginner

   notebooks/beginner_ETC_lazulitarget
   notebooks/beginner_LazuliTargets
   notebooks/beginner_change_properties
   notebooks/beginner_variancesource
   notebooks/beginner_kilonova
   notebooks/beginner_sample_multiple_targets

.. toctree::
   :hidden:
   :caption: Advanced

   notebooks/advanced_change_detector_and_spectrograph
   notebooks/advanced_macc_readout_modes
   notebooks/advanced_access_qe_spectral_resolution_etc
   notebooks/advanced_change_resolution_dispersion
   notebooks/advanced_dispersion_resolution_and_spotsize
   notebooks/advanced_calspec_and_stdstars
   notebooks/advanced_scene_tutorial
   notebooks/advanced_host_galaxy
