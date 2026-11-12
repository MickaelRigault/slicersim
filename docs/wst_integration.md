# WST IFS modes

`slicersim` can simulate the two channels (blue and red) of the integral-field spectrograph
(IFS) of the Wide-field Spectroscopic Telescope (WST). It uses the instrument description and
site approximations of the WST exposure time calculator
[pyetc](https://github.com/RolandBacon/pyetc): atmospheric transmission, seeing PSF and the
Paranal sky. The results are validated against pyetc.

```{note}
Reference: pyetc commit
[`e6e351e`](https://github.com/RolandBacon/pyetc/tree/e6e351ebf928d98adad34e8d17e5825b15595bc9)
(`pyetc/wst.py`, `pyetc/etc.py`). The figures and comparisons are in the notebook
{doc}`notebooks/wst_ifs_modes`, which is stored with its outputs.
```

## Quick start

```python
import slicersim

target = slicersim.WSTTarget(lbda, flux, channel="blue",       # or "red"
                             seeing=0.8, airmass=1.0, moon="darksky")
lbda, signal, variance = target.simulation.get_spectrum()

# exposure time (NDIT x 1800 s) to reach SNR = 10 per pixel in 4800-5200 Å
exptime, target = slicersim.wst_etc(lbda, flux, 10, channel="blue",
                                    lbda_range=[4800, 5200])
```

You can change these parameters with `target.change_properties(...)`:

- `seeing`, `airmass`, `moon`;
- `optical_fwhm`, `moffat_beta`, `moffat_truncation`;
- `dit`, and `ndit` (an alias of `nramps`).

For example: `target.change_properties(seeing=0.6, moon="greysky", ndit=3)`.

The entry points are {class}`~slicersim.wst.WSTTarget` and {func}`~slicersim.wst.wst_etc`.
The Lazuli code paths, and `spectrograph.py` itself, are unchanged.

## pyetc ↔ slicersim mapping

| | pyetc (WST IFS) | slicersim (Lazuli) | WST in slicersim |
|---|---|---|---|
| Collecting area | 100 m² × (1 − 0.18 IFS obscuration) | π(D²−d²)/4 | `telescope.surface = 82` m² |
| Spatial PSF | Moffat β = 2.5; FWHM(λ) from the ESO seeing model (seeing at 5000 Å, airmass^0.6, λ^−1/5, outer-scale term for D = 12 m) ⊕ 0.10″ for telescope + instrument | Gaussian ∝ λ (diffraction) ⊕ jitter, or Airy | **new**: chromatic Moffat integrated over the spaxels; FWHM = **seeing-induced** (`Atmosphere`) ⊕ **optical** (`optical_fwhm`, spectrograph) |
| Atmospheric transmission | skycalc `trans` (Paranal, per airmass), on the source only | none | **new**: applied to the astrophysical sources, not to the sky |
| Background | skycalc emission (dark/grey/bright; airmass 1.0/1.2/1.5/2.0) | zodiacal light | `Atmosphere` sky emission as the scene `Background` |
| Throughput | `ifs_<chan>_noatm.fits` (TOT, QE included) | per-element CSV + detector QE | CSV resampled on the pyetc grid, QE = 1 |
| Wavelengths | linear λ₁ + k·Δλ (0.64 Å blue, 0.97 Å red) | dispersion law or constant R | linear `dispersion_law` ECSV, `dispersion_scale = 2/2.5` |
| LSF | Gaussian, FWHM 2.5 px (also used by skycalc) | σ = `dispersion_resolution`/2 | σ = FWHM/2.355; the sky table is not convolved again |
| Sampling | 1 voxel (0.25″ spaxel × 1 spectel) = 1 pixel | anamorphosis ×2 (2 px per spaxel) | no anamorphosis, see {doc}`anamorphosis` |
| Detector | CCD: RON 3 e⁻ per read, dark 3 e⁻/h, NDIT × DIT | HxRG, MACC up-the-ramp | **new**: `CCDDetector`, NDIT ≡ `nramps` |
| Extraction | square box (2n+1)² or adaptive circle | optimal or circular aperture | both, plus pyetc's box (`method="box"`) |
| ETC | flux for a given SNR | MACC mode and number of ramps for a given SNR | NDIT for a fixed DIT, or DIT for a fixed NDIT |

## Package components

| Module or file | Content |
|---|---|
| {mod}`slicersim.atmosphere` | {func}`~slicersim.atmosphere.get_seeing_fwhm`: the **seeing-induced** FWHM only (ESO ETC model, as pyetc). {func}`~slicersim.atmosphere.moffat_density`: normalized Moffat profile. {class}`~slicersim.atmosphere.Atmosphere`: seeing, airmass and moon, all mutable and checked against the tabulated conditions; it provides the transmission and the sky emission. |
| {mod}`slicersim.ground_spectrograph` | {class}`~slicersim.ground_spectrograph.GroundSlicerSpectrograph`, a `SlicerSpectrograph` that holds an `Atmosphere`. Moffat PSF integrated over the spaxels (16 wavelength nodes, cached). FWHM = seeing-induced ⊕ **optical** (`optical_fwhm`: telescope + instrument); see `get_psf_fwhm(which="seeing"/"optical"/"total")` and `get_spatial_psf(incl_atmosphere=, incl_telescope=)`. `moffat_truncation=2` reproduces pyetc. The transmission is in `flambda2photon` but not in `generate_background`. LSF given as a FWHM. No anamorphosis. `get_box_aperture`. |
| {mod}`slicersim.detector` | {class}`~slicersim.detector.CCDDetector`: `dit`; variance (f + d)·DIT + RON² per exposure |
| {mod}`slicersim.wst` | {class}`~slicersim.wst.WSTSimulation`: atmosphere wiring, sky as background, LSF applied before the transmission, `"box"` extraction, CCD `fetch_snr`, `reset`. Also {class}`~slicersim.wst.WSTTarget`, {func}`~slicersim.wst.wst_etc`, `WST_INSTRUMENTS`, and {func}`~slicersim.wst.make_wst_config_files`, which regenerates the configuration files from pyetc data. |
| `slicersim/config/` | `wst_ifs_{blue,red}.toml`, `wst_ifs_{blue,red}_throughput.csv`, `_dispersion.ecsv` and `_sky.ecsv`. The sky tables (~680 kB each) hold the Paranal transmission for airmass 1.0, 1.2, 1.5 and 2.0, and the emission for each airmass and for 3 moon conditions. |
| `tests/test_wst.py` | 11 tests, including the pyetc regression value (23.0663) |

## Validation against pyetc

Setup: a point source with flat F_λ = 10⁻¹⁷ erg/s/cm²/Å, observed for 2 × 1800 s, with pyetc's
`square_fixed` aperture. The values are medians over the channel, and the SNR is per spectral
pixel.

| channel | box | seeing | sky | airmass | source / pyetc | sky (1 spx) / pyetc | SNR pyetc | pyetc formula, 1-spaxel sky | slicersim box, pyetc PSF | slicersim box, exact Moffat | slicersim optimal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blue | 5² | 0.8 | dark | 1.0 | 1.0000 | 1.0000 | 29.22 | 29.22 | 22.06 | 19.44 | 21.47 |
| red | 5² | 1.0 | dark | 1.0 | 1.0000 | 1.0000 | 42.69 | 42.69 | 29.14 | 24.37 | 26.91 |
| red | 7² | 1.0 | grey | 1.2 | 1.0000 | 1.0000 | 43.43 | 43.43 | 19.63 | 16.11 | 19.18 |
| blue | 9² | 1.0 | bright | 1.5 | 1.0000 | 1.0000 | 16.48 | 16.48 | 2.44 | 1.90 | 2.42 |

- **Photon budgets.** With the same PSF approximation, the source and sky photon counts agree
  with pyetc to within 10⁻⁴ (median). The wavelength grids are identical.
- **Resolved-source path.** At 5000 Å (blue, 5 × 5 box, 0.8″ seeing, dark sky), pyetc's
  resolved-source path (`snr_from_cube`, which uses the PSF image as the source) gives
  **23.0663**, and slicersim gives **23.0662**. pyetc's point-source path gives 29.97 for the
  same case.

## Issues found in pyetc

Issues 1, 2 and 4 make pyetc's SNR too optimistic.

1. **Sky noise counted for one spaxel only.** In `ETC.snr_from_ps_spec`, `nph_sky` is
   computed for `area = spaxel_area`, while RON and dark are multiplied by
   `nb_voxels = nspaxels`.
   - pyetc's output is reproduced exactly by `S / √(S + sky₁ + n_spx·(RON²·NDIT + dark))`.
   - Summing the sky over the aperture lowers the SNR by ~25 % in dark time.
   - When the sky dominates, the error grows to ~√n_spx: the SNR is ×7 too high for a 9 × 9
     box in bright time.
2. **Truncated PSF.** `etc.moffat(kfwhm=2)` cuts the Moffat at ~2 FWHM and renormalizes it.
   At 0.8″ seeing, ~85 % of the flux falls in a 5 × 5 box instead of ~72 %, and 100 % for
   boxes of 7 × 7 or more. This overestimates the SNR by another 10–20 %.
3. **Crash for small PSFs.** In the red channel at 0.8″ seeing, pyetc crashes in
   `square_aperture` when the PSF FWHM is below 0.625″. Its truncated PSF image is then only
   5 × 5 spaxels, and mpdaf's `Image.peak` returns NaN.
4. **MOS surface brightness.** The fibre holds 25 spaxels instead of 13 (`2*np.pi*d**2/4`),
   so the SNR is overestimated by ~√2. This is unrelated to the IFS; it turned up while
   reading the code.

GitHub-ready reports, with code links and reproduction scripts, are in `wst_pyetc_issues.md`
at the root of the repository.

## Example results

These use slicersim's defaults: exact Moffat PSF and optimal extraction.

- AB = 22 (flat F_ν), 2 × 1800 s, 0.8″ seeing, dark sky, airmass 1:

  | Median SNR | blue | red |
  |---|---|---|
  | per pixel | 16.1 | 10.5 |
  | per resolution element | 23.9 | 14.5 |

- Number of 1800 s exposures (NDIT) needed for SNR = 10 per pixel, in 4800–5200 Å (blue) and
  7300–7700 Å (red):

  | AB mag | ≤ 21.5 | 22.0 | 22.5 | 23.0 | 23.5 | 24.0 |
  |---|---|---|---|---|---|---|
  | NDIT blue | 1 | 1 | 2 | 4 | 9 | 20 |
  | NDIT red | 1 | 2 | 4 | 10 | 23 | 55 |

- With NDIT fixed at 4 (blue, AB = 22), the DIT solver gives 471 s.
- A single very long DIT saturates the bright red sky lines (65535 e⁻). slicersim then
  returns SNR = inf, so the DIT solver stops at the saturation limit and issues a warning.

## Approximations and limitations

- Sky tables exist only for airmass 1.0, 1.2, 1.5 and 2.0 and three moon conditions
  (Paranal, PWV 3.5 mm). More can be produced with `pyetc.etc.update_skytables`.
- The edge pixels of the skycalc tables are distorted by their LSF convolution, in pyetc as
  well.
- The PSF is a single Moffat whose FWHM is the quadratic sum of the seeing-induced and
  optical FWHMs. The two components are not convolved separately, and `optical_fwhm` is
  achromatic, as in pyetc.
- There is no sky-subtraction noise and no differential atmospheric refraction, as in pyetc.
- The simulated sub-field is 25 × 25 spaxels (6.25″): enough for point sources, but not the
  full IFS field.
- The seeing follows the ESO-ETC model, with the VLT outer-scale term and D = 12 m.
- Saturation is checked at 65535 e⁻; pyetc does not check it.
- Each voxel is 1 pixel (no anamorphosis), as in pyetc; Lazuli's convention is 2. See
  {doc}`anamorphosis`.
- The WST MOS modes (LR/HR, fibre-fed) are out of scope, since they are not slicer IFS.

## Running the notebook

The notebook {doc}`notebooks/wst_ifs_modes` (`docs/notebooks/wst_ifs_modes.ipynb`) needs slicersim (branch `wst`). The pyetc
repository is only needed for two sections, which are skipped otherwise:

- §1.3 checks that the packaged configuration files can be reproduced from pyetc data. They
  currently are identical.
- §3 compares slicersim with pyetc. This also needs pyetc to be importable, with `mpdaf`,
  `skycalc_cli` and `spextra`.

The pyetc location is `PYETC_DIR`; the default is `../../../../WST/pyetc`, relative to
`docs/notebooks/`.
