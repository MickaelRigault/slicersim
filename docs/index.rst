slicersim
=========

**Realistic simulations of integral field spectrograph observations.**

``slicersim`` turns a spectrum on the sky into the spectrum you would actually
measure: it propagates a scene through the telescope, the spectrograph and the
detector, adds every noise source, and tells you how long you need to
expose to reach a given signal-to-noise.

It is the reference simulator of the `Lazuli Space Observatory
<https://scixplorer.org/abs/2026arXiv260706391R/abstract>`_ image slicer,
but its core is generic and applies to any slicer or micro-lens array IFU.

.. code-block:: python

   import slicersim

   sn = slicersim.LazuliSupernova(redshift=1.0, x1=0, c=0.2, phase=1.5)
   sn.setup_to_snr(20)                                 # choose the detector read-out
   exptime = sn.get_exposure_time()                    # [s]
   lbda, flux, variance = sn.get_spectrum(unit="flambda")

.. image:: gallery/spectra_examples.png
   :alt: Two simulated Lazuli spectra of a Type Ia supernova, at z = 0.8 and
         z = 1.2, with their 1-sigma uncertainties.
   :align: center

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: sd-mt-4

   .. grid-item-card:: :octicon:`zap;1.2em` Exposure time calculator
      :class-card: sd-border-0 sd-shadow-sm

      One call gives the exposure time and read-out mode needed to reach a
      requested signal-to-noise, for any target.

   .. grid-item-card:: :octicon:`graph;1.2em` Realistic spectra and cubes
      :class-card: sd-border-0 sd-shadow-sm

      Noisy spectra, 3D cubes and detector images, with photon, read-out,
      dark, thermal and background noise all accounted for.

   .. grid-item-card:: :octicon:`star;1.2em` Built-in astrophysical sources
      :class-card: sd-border-0 sd-shadow-sm

      Type Ia supernovae (SALT, Twins-Embedding), kilonovae (POSSIS),
      CalSpec standards, black bodies, power laws, or your own spectrum.

   .. grid-item-card:: :octicon:`gear;1.2em` Every knob is accessible
      :class-card: sd-border-0 sd-shadow-sm

      Change the detector, the spectrograph sampling, the throughput or the
      scene, and see at once how each noise source contributes.

Installation
------------

.. code-block:: bash

   pip install slicersim

See :doc:`installation` for development installs.

Where to go next
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.5em` Getting started
      :link: quickstart
      :link-type: doc
      :class-card: sd-shadow-sm

      A five-minute tour: build a target, reach a signal-to-noise and get
      a realistic spectrum.

   .. grid-item-card:: :octicon:`book;1.5em` Tutorials
      :link: tutorials
      :link-type: doc
      :class-card: sd-shadow-sm

      Notebooks, from first exposure time calculations to custom scenes and
      detector read-out modes.

   .. grid-item-card:: :octicon:`telescope;1.5em` The Lazuli instrument model
      :link: instrument
      :link-type: doc
      :class-card: sd-shadow-sm

      Fields, detector, throughput and noise: what the simulations assume,
      and how to change it.

   .. grid-item-card:: :octicon:`code-square;1.5em` API reference
      :link: api/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Every public class and function, grouped by what it does.

Citing slicersim
----------------

If ``slicersim`` is useful for your work, please cite `Rigault et al. (2026)
<https://scixplorer.org/abs/2026arXiv260706391R/abstract>`_
(arXiv:2607.06391); the BibTeX entry can be exported from the linked page.

``slicersim`` is developed by M. Rigault, building on the original
``MLAPerf`` simulator by Y. Copin and M. Rigault. It is open source
(Apache 2.0) and `hosted on GitHub <https://github.com/MickaelRigault/slicersim>`_:
bug reports and contributions are welcome.

.. toctree::
   :hidden:
   :caption: User guide

   installation
   quickstart
   concepts
   galaxy_models
   instrument
   tutorials

.. toctree::
   :hidden:
   :caption: Reference

   api/index
