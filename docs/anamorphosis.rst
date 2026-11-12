Slicer anamorphosis
===================

This page compares how `pyetc <https://github.com/RolandBacon/pyetc>`_ (the WST
exposure time calculator) and ``slicersim`` count detector pixels for an image
slicer, explains the difference, and says which one is right.

Summary
-------

- **pyetc** charges **1 detector pixel per voxel**, where a voxel is one
  spaxel × one spectel. This is the right count for a slicer whose
  anamorphosis acts across the slices.
- **slicersim** (:class:`~slicersim.spectrograph.SlicerSpectrograph`, used for
  Lazuli) uses ``_ANAMORPHOSE = 2`` in two places at once:

  #. it doubles the spaxel grid along x but keeps the spaxel scale, which
     doubles the field of view;
  #. it charges 2 detector pixels per grid element, which doubles the
     detector variance (RON, dark, glow) of every voxel.

.. important::

   **Who's right:** pyetc. For the usual slicer design, the anamorphosis is
   already counted in the width of the spectral resolution element. Counting
   it again as extra pixels per voxel counts it twice.

   The WST implementation in ``slicersim``
   (:class:`~slicersim.ground_spectrograph.GroundSlicerSpectrograph`) therefore
   uses ``_ANAMORPHOSE = 1``, as pyetc does. Lazuli keeps the current,
   conservative behaviour, which the code itself already marks as unconfirmed
   (see below). Changing it is a decision about the Lazuli design, not about
   WST.

What an image slicer's anamorphosis does
----------------------------------------

A slicer cuts the field into slices. Each slice is re-imaged as a pseudo-slit
at the spectrograph entrance.

- **Along a slice** (the spatial axis on the detector), 1 pixel samples
  1 spaxel.
- **Across a slice**, the slice width becomes the slit width. With no
  anamorphosis, a square spaxel would be imaged onto about 1 pixel in the
  dispersion direction, and the spectrum would be under-sampled. The slicer
  optics therefore stretch the image across the slices by a factor *A*
  (typically 2, as in MUSE-like designs). The slit is then *A* pixels wide in
  the dispersion direction.

A monochromatic voxel image therefore covers **1 pixel along the slice ×**
***A* pixels along the dispersion**. Those *A* pixels are the spectral
resolution element (the LSF), and neighbouring wavelengths share them.

- With a spectral sampling of 1 spectel = 1 pixel, each voxel
  ``(spaxel, spectel)`` maps to **exactly 1 detector pixel**.
- A full resolution element of one spaxel covers *A* pixels.
- The source and sky photons are conserved either way.

So the anamorphosis shows up in three places only:

- the LSF width in pixels: ``lsfpix = 2.5`` in pyetc,
  ``dispersion_resolution`` ≈ 2.3–2.5 in ``slicersim``;
- the detector footprint;
- the noise per resolution element (*A* pixels, through the LSF).

It does **not** show up as extra pixels per spectel.

``slicersim``'s own PSF model makes the same assumption.
:meth:`~slicersim.spectrograph.Spectrograph.get_instrumental_psf` takes the
"expected geometric footprint" of the slit to be ``expected_size = 2``
pixels, and divides the extra blur by ``_ANAMORPHOSE`` to turn detector
pixels into slice widths.

pyetc
-----

The detector noise is per voxel, and ``nb_voxels`` counts spaxels × spectels.
See `etc.py#L918-L919
<https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L918-L919>`_,
which is repeated at L973, L1028 and L1085:

.. code-block:: python

   ron_noise = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
   dark_noise = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)

The anamorphosis enters only through ``lsfpix = 2.5`` (`wst.py#L48
<https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/wst.py#L48>`_).
This matches the analysis above.

slicersim
---------

1. Doubled grid, same scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`SlicerSpectrograph.set_spaxels
<slicersim.spectrograph.SlicerSpectrograph.set_spaxels>` does:

.. code-block:: python

   shape = np.asarray(shape) * (1, self._ANAMORPHOSE)
   spx_scale = np.asarray(spx_scale)

For Lazuli, ``spatial_shape = [58, 58]`` and ``spatial_scale = 0.040″`` give
a grid of 58 × 116 elements of 0.040″ × 0.040″. That is a field of
**2.32″ × 4.64″** instead of 2.32″ × 2.32″. Each grid element is still a full
square spaxel: it collects the full-spaxel sky, and the PSF covers as many
grid elements as without anamorphosis. The extra columns are simply more sky.

Before commit ``07ff099`` ("major refactoring of the way anamorphose is taking
into account"), the code was ``shape * (2, 1)`` together with
``spx_scale / (2, 1)``, which kept the field of view. The refactoring dropped
the scale change.

2. Two pixels per grid element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Simulation._get_slicer_cube_`` passes
``npix_per_spaxel=self.spectrograph._ANAMORPHOSE`` to
:meth:`~slicersim.detector.Detector.estimate_pixel_signal`. That method
spreads the voxel flux over that many pixels and multiplies the variance
back. The result: the detector variance per voxel is ×2, while the photon
noise is unchanged. The code comment there already says so:

    It is possible this should be 1 and not 2 (anamorphose) as this
    effectively is the number of pixels per voxel (wavelength bin x spaxel)
    which is 1 for a slicer. This is to be confirmed. Using the anamorphose
    factor here is conservative as I am still unsure.

The analysis above confirms the comment's doubt: the right value is 1.

Impact on Lazuli
^^^^^^^^^^^^^^^^

Setup: the default ``LazuliTarget``, a flat :math:`F_\lambda`, optimal
extraction, and the current ``npix_per_spaxel = 2`` compared with
``npix_per_spaxel = 1``. Everything else is unchanged.

.. list-table::
   :header-rows: 1

   * - :math:`F_\lambda` [erg/s/cm²/Å]
     - npix/voxel
     - SNR 5000 Å
     - SNR 10000 Å
     - SNR 15000 Å
   * - 3e-18
     - 2 (current)
     - 2.72
     - 19.91
     - 24.69
   * - 3e-18
     - 1
     - 3.43 (+26 %)
     - 21.42 (+8 %)
     - 26.77 (+8 %)
   * - 3e-17
     - 2 (current)
     - 15.06
     - 73.78
     - 93.07
   * - 3e-17
     - 1
     - 16.05 (+7 %)
     - 74.83 (+1 %)
     - 94.61 (+2 %)

- **Detector-limited cases** (faint sources, the blue end where RON
  dominates) are the most affected: the SNR is underestimated by up to about
  25 %, and exposure times are overestimated by up to about 1.6×.
- **Photon-limited cases** change by only a few percent.
- **The doubled field** does not matter for a centred point source with
  optimal extraction. It does matter for:

  - large apertures: with ``aperture_radius = 25`` spaxels, the circle no
    longer gets cut by the 2.32″ field edge, so it holds more pixels;
  - field-integrated quantities, such as the total sky or thermal background;
  - extended hosts.

.. warning::

   Restoring the old field-conserving grid (scale/2 along x) while keeping
   ``npix_per_spaxel = 2`` would charge **4** pixels per spaxel. The two
   corrections must not be combined.

Recommendation
--------------

For a slicer whose anamorphosis acts across the slices (the usual design, and
what ``slicersim``'s ``expected_size = 2`` PSF model and
``dispersion_resolution`` ≈ 2.3 already assume):

#. Use **npix_per_spaxel = 1** for slicers (1 pixel per voxel), as pyetc
   does.
#. **Do not enlarge the spaxel grid.** The anamorphosis is already in the LSF
   (``dispersion_resolution``, in pixels) and in the across-slice blur
   conversion of
   :meth:`~slicersim.spectrograph.Spectrograph.get_instrumental_psf`.

If an instrument really did stretch the image *along* the slices (2 pixels per
spaxel on the spatial axis), the consistent model would be the grid
``shape × (1, A)`` with ``scale / (1, A)`` and 1 pixel per grid element. That
gives *A* pixels per spaxel and the right field of view. It would never mean
doubling the grid at constant scale *and* charging *A* pixels per grid
element.

Which design Lazuli has is for the Lazuli optical design to confirm. Changing
it would shift the Lazuli reference numbers, so it is not done here.

WST in slicersim
----------------

:class:`~slicersim.ground_spectrograph.GroundSlicerSpectrograph` sets
``_ANAMORPHOSE = 1``. With that setting:

- the grid is ``spatial_shape = [25, 25]`` of 0.25″, i.e. a 6.25″ × 6.25″
  field;
- ``npix_per_spaxel`` is 1;
- the LSF is 2.5 pixels FWHM.

This is the pyetc convention, and it reproduces pyetc's resolved-source SNR
to 1e-4 (``tests/test_wst.py::test_pyetc_regression``).
:class:`~slicersim.spectrograph.SlicerSpectrograph` itself, and so Lazuli, is
unchanged.
