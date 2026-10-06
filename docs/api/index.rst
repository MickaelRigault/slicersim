API reference
=============

Objects are grouped by what they do, from the highest to the lowest level.
Everything listed under :ref:`api-toplevel` is importable directly from
``slicersim`` (e.g. ``slicersim.LazuliSupernova``). The :doc:`modules`
page lists every module with all its content.

.. currentmodule:: slicersim

.. _api-toplevel:

Top-level API
-------------

Exposure time calculators
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   ~lazuli.lazuli_etc
   ~lazuli.lazuli_sn_etc

Lazuli targets
~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   ~lazuli.LazuliSupernova
   ~lazuli.LazuliKilonova
   ~lazuli.LazuliCalSpec
   ~lazuli.LazuliBlackBody
   ~lazuli.LazuliPowerLaw
   ~lazuli.LazuliTarget

Simulation and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   ~simulation.Simulation
   ~iotools.get_config

Populations and calibrations
----------------------------

.. autosummary::
   :nosignatures:

   ~sample.Sample
   ~lazuli.LazuliFlat
   ~lazuli.Lazuli3DFlat

Instrument-agnostic targets
---------------------------

Base classes of the ``Lazuli*`` targets, to use with another instrument
configuration.

.. autosummary::
   :nosignatures:

   ~target.VirtualTarget
   ~target.Target
   ~target.Supernova
   ~target.Kilonova
   ~target.CalSpec
   ~target.BlackBody
   ~target.PowerLaw
   ~lazuli.VirtualLazuliTarget

Scene
-----

.. autosummary::
   :nosignatures:

   ~scene.get_scene
   ~scene.scene.Scene
   ~scene.base.SceneElement
   ~scene.pointsource.PointSource
   ~scene.background.Background
   ~scene.host.Host

Source models (``slicersim.scene.sources``):

.. autosummary::
   :nosignatures:

   ~scene.sources.supernovae.get_snia_pointsource
   ~scene.sources.kilonova.get_kilonova_pointsource
   ~scene.sources.stars.get_calspec_pointsource
   ~scene.sources.blackbody.get_blackbody_flux
   ~scene.sources.generic.get_powerlaw_flux
   ~scene.sources.calspec.CalSpecSource

Instrument
----------

.. autosummary::
   :nosignatures:

   ~telescope.Telescope
   ~spectrograph.SlicerSpectrograph
   ~spectrograph.MLASpectrograph
   ~spectrograph.OpticsThroughput
   ~detector.Detector
   ~thermal.ThermalOptics
   ~thermal.ThermalRadiation
   ~mapper.SlicerMapper

.. toctree::
   :hidden:

   modules
