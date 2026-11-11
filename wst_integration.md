# WST IFS modes in slicersim — integration report

*2026-11-11, branch `wst`. Notebook: [`notebooks/wst_ifs_modes.ipynb`](notebooks/wst_ifs_modes.ipynb)
(executed, outputs stored). Reference: pyetc `e6e351e` (`pyetc/wst.py`, `pyetc/etc.py`).*

## Goal

Simulate the WST integral-field spectrograph (blue and red channels) with slicersim,
using pyetc's instrument description and its site approximations (atmospheric
transmission, seeing PSF, Paranal sky), and validate the result against pyetc.

The WST modes are integrated in the package (branch `wst`, not committed yet):
`slicersim.WSTTarget(lbda, flux, channel="blue"|"red", seeing=..., airmass=..., moon=...)`
and `slicersim.wst_etc(...)`. `spectrograph.py` is unmodified, and the Lazuli code paths are unchanged (full test suite:
126 passed, 115 pre-existing + 11 new).

## pyetc ↔ slicersim mapping

| | pyetc (WST IFS) | slicersim (Lazuli) | WST in slicersim |
|---|---|---|---|
| Collecting area | 100 m² × (1 − 0.18 IFS obscuration) | π(D²−d²)/4 | `telescope.surface = 82` m² |
| Spatial PSF | Moffat β=2.5, FWHM(λ) from the ESO seeing model (seeing at 5000 Å, airmass^0.6, λ^−1/5, outer-scale term for D=12 m) ⊕ 0.10″ tel+ins | Gaussian ∝ λ (diffraction) ⊕ jitter, or Airy | **new**: chromatic Moffat integrated over spaxels, FWHM = **seeing-induced** (`Atmosphere`) ⊕ **optical** (`optical_fwhm`, spectrograph) |
| Atmospheric transmission | skycalc `trans` (Paranal, per airmass), on the source only | none | **new**: applied to astrophysical sources, not to the sky |
| Background | skycalc emission (dark/grey/bright, airmass 1.0/1.2/1.5/2.0) | zodiacal light | `Atmosphere` sky emission as the scene `Background` |
| Throughput | `ifs_<chan>_noatm.fits` (TOT, includes QE) | per-element CSV + detector QE | CSV resampled on the pyetc grid, QE = 1 |
| Wavelengths | linear λ₁ + k·Δλ (0.64 Å blue, 0.97 Å red) | dispersion law or constant R | linear `dispersion_law` ECSV, `dispersion_scale = 2/2.5` |
| LSF | Gaussian FWHM 2.5 px (also used by skycalc) | σ = `dispersion_resolution`/2 | σ = FWHM/2.355; sky table not re-convolved |
| Sampling | 1 voxel (0.25″ spaxel × 1 spectel) = 1 pixel | anamorphosis ×2 (2 px/spaxel) | no anamorphosis |
| Detector | CCD: RON 3 e⁻/read, dark 3 e⁻/h, NDIT × DIT | HxRG, MACC up-the-ramp | **new**: `CCDDetector`, NDIT ≡ `nramps` |
| Extraction | square box (2n+1)² or adaptive circle | optimal or circular aperture | both, plus pyetc box (`method="box"`) |
| ETC | flux for a given SNR | MACC mode & ramps for a SNR | NDIT for a fixed DIT, or DIT for a fixed NDIT |

## Package changes

| File | Change |
|---|---|
| `src/slicersim/atmosphere.py` (new) | `get_seeing_fwhm`: **seeing-induced** FWHM only (ESO ETC model, pyetc); `moffat_density`: normalized Moffat profile; `read_sky_table`; `Atmosphere`: seeing, airmass, moon (mutable, validated against the tabulated conditions), `get_seeing_fwhm(lbda, diameter)`, `get_transmission`, `get_emission` |
| `src/slicersim/ground_spectrograph.py` (new) | `GroundSlicerSpectrograph(SlicerSpectrograph)`, holding an `atmosphere`: Moffat PSF integrated over spaxels (16 wavelength nodes, cached), FWHM = seeing-induced ⊕ **optical** (`optical_fwhm`, telescope + instrument), see `get_psf_fwhm(which="seeing"/"optical"/"total")` and `get_spatial_psf(incl_atmosphere=, incl_telescope=)`; `moffat_truncation=2` reproduces pyetc; transmission in `flambda2photon`, not in `generate_background`; LSF FWHM convention; no anamorphosis; `get_box_aperture`; `atmosphere.*` mutable parameters |
| `src/slicersim/detector.py` | `CCDDetector(Detector)`: `dit`, variance (f + d)·DIT + RON² per exposure |
| `src/slicersim/wst.py` (new) | `WSTSimulation` (atmosphere wiring, sky as background, LSF before transmission, `"box"` extraction, CCD `fetch_snr`, `reset`), `WSTTarget`, `wst_etc`, `WST_INSTRUMENTS`, `make_wst_config_files` (maintenance: regenerates the config files from pyetc data) |
| `src/slicersim/thermal.py` | Planck function: no divide-by-zero warning for T = 0 K ("no thermal emission"); results unchanged |
| `src/slicersim/__init__.py` | exports `WSTTarget`, `wst_etc` |
| `src/slicersim/config/` | `wst_ifs_{blue,red}.toml` (the blue one replaces the draft), `wst_ifs_{blue,red}_throughput.csv`, `_dispersion.ecsv`, `_sky.ecsv` (Paranal transmission for airmass 1.0/1.2/1.5/2.0 and emission × 3 moon conditions, ~680 kB each) |
| `tests/test_wst.py` (new) | 11 tests, including the pyetc regression value (23.0663) |
| `docs/api/modules.rst` | `slicersim.wst`, `slicersim.atmosphere`, `slicersim.ground_spectrograph` |

Mutable parameters: `seeing`, `airmass`, `moon`, `optical_fwhm`, `moffat_beta`,
`moffat_truncation`, `dit`, `ndit` (alias of `nramps`), e.g.
`target.change_properties(seeing=0.6, moon="greysky", ndit=3)`.

## Validation against pyetc

Flat F_λ = 10⁻¹⁷ erg/s/cm²/Å point source, 2 × 1800 s, pyetc `square_fixed` aperture.
Median over the channel; SNR per spectral pixel.

| channel | box | seeing | sky | airmass | source / pyetc | sky (1 spx) / pyetc | SNR pyetc | pyetc formula, 1-spaxel sky | slicersim box, pyetc PSF | slicersim box, exact Moffat | slicersim optimal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blue | 5² | 0.8 | dark | 1.0 | 1.0000 | 1.0000 | 29.22 | 29.22 | 22.06 | 19.44 | 21.47 |
| red | 5² | 1.0 | dark | 1.0 | 1.0000 | 1.0000 | 42.69 | 42.69 | 29.14 | 24.37 | 26.91 |
| red | 7² | 1.0 | grey | 1.2 | 1.0000 | 1.0000 | 43.43 | 43.43 | 19.63 | 16.11 | 19.18 |
| blue | 9² | 1.0 | bright | 1.5 | 1.0000 | 1.0000 | 16.48 | 16.48 | 2.44 | 1.90 | 2.42 |

- **Photon budgets**: with the same PSF approximation, the source and sky photon counts
  agree with pyetc to within 10⁻⁴ (median). The wavelength grids are identical.
- **Resolved path**: at 5000 Å (blue, 5×5, 0.8″, dark), pyetc's resolved-source path
  (`snr_from_cube`, PSF image used as the source) gives **23.0663**, slicersim gives
  **23.0662**, while pyetc's point-source path gives 29.97.

## Issues found in pyetc

Issues 1, 2 and 4 make pyetc's SNR too optimistic.

1. **Sky noise counted for one spaxel only.** In `ETC.snr_from_ps_spec`, `nph_sky` is
   computed for `area = spaxel_area`, while RON and dark are multiplied by
   `nb_voxels = nspaxels`. pyetc's output is reproduced exactly by
   `S / √(S + sky₁ + n_spx·(RON²·NDIT + dark))`. Summing the sky over the aperture lowers the
   SNR by ~25 % in dark time, and by ~√n_spx when sky-dominated (×7 too high for a 9×9 box in
   bright time).
2. **Truncated PSF.** `etc.moffat(kfwhm=2)` cuts the Moffat at ~2 FWHM and renormalises it:
   ~85 % of the flux falls in a 5×5 box instead of ~72 % (0.8″ seeing), and 100 % for ≥ 7×7.
   That's another 10–20 % SNR overestimate.

3. **Crash for small PSFs**: in the red channel (seeing 0.8″), pyetc crashes in
   `square_aperture` (`mpdaf Image.peak` returns NaN) when the PSF FWHM is < 0.625″, because its
   truncated PSF image is then only 5×5 spaxels.
4. **MOS surface brightness**: the fibre holds 25 spaxels instead of 13 (`2*np.pi*d**2/4`),
   so the SNR is overestimated by ~√2. Found while reading the code; not related to the IFS.

GitHub-ready reports with code links and repros: [`wst_pyetc_issues.md`](wst_pyetc_issues.md).

## Example results (slicersim defaults: exact Moffat, optimal extraction)

- AB = 22 (flat F_ν), 2 × 1800 s, 0.8″, dark, airmass 1: median SNR per pixel 16.1 (blue),
  10.5 (red); per resolution element 23.9 and 14.5.
- Exposures of 1800 s needed for SNR = 10 per pixel (blue 4800–5200 Å, red 7300–7700 Å):

| AB mag | ≤ 21.5 | 22.0 | 22.5 | 23.0 | 23.5 | 24.0 |
|---|---|---|---|---|---|---|
| NDIT blue | 1 | 1 | 2 | 4 | 9 | 20 |
| NDIT red | 1 | 2 | 4 | 10 | 23 | 55 |

- With a fixed NDIT = 4 (blue, AB = 22), the solved DIT is 471 s.
- A single very long DIT saturates bright red sky lines (65535 e⁻); slicersim then returns
  SNR = inf, so the DIT solver stops at the saturation limit (now with a warning).

## The former draft `src/slicersim/config/wst_ifs_blue.toml`

It did not load (`diameter_ext = 12.` is invalid TOML; no telescope `emissivity`;
`dispersion = 0.64` is not a slicersim key) and still used Lazuli's throughput, H4RG QE
and MACC detector. It has been replaced by the generated configuration (the emacs backup
`wst_ifs_blue.toml~` is still there).

## Approximations and limitations

- Sky tables only for airmass 1.0, 1.2, 1.5, 2.0 and three moon conditions (Paranal,
  pwv 3.5 mm); more can be produced with `pyetc.etc.update_skytables`.
- The edge pixels of the skycalc tables are distorted by their LSF convolution (in pyetc too).
- Single Moffat whose FWHM is the quadratic sum of the seeing-induced and optical FWHMs
  (no separate convolution of the two components); achromatic `optical_fwhm` (as pyetc).
- No sky-subtraction noise, no differential atmospheric refraction (as pyetc).
- Simulated sub-field of 25×25 spaxels (6.25″): enough for point sources, not the full IFS field.
- ESO-ETC seeing model with the VLT outer-scale term and D = 12 m.
- Saturation tested at 65535 e⁻ (pyetc does not test it).
- 1 pixel per voxel (no anamorphosis), as pyetc; slicersim's Lazuli convention is 2.
- The WST MOS modes (LR/HR, fibre-fed) are out of scope: they are not slicer IFS.

## Running the notebook

- Needs slicersim (branch `wst`). The pyetc repository (`PYETC_DIR`, default
  `../../../WST/pyetc` from `notebooks/`) is only needed to check that the packaged
  configuration files are reproducible from pyetc data (§1.3, currently identical), and for
  the pyetc comparison (§3), which also needs pyetc importable (`mpdaf`, `skycalc_cli`,
  `spextra`). Both are skipped otherwise.
- It was executed in a throwaway Python 3.14 environment, because no environment on this
  machine had numpy/slicersim installed.
