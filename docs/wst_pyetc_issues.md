# pyetc issues found while integrating WST into slicersim

Issues found in [RolandBacon/pyetc](https://github.com/RolandBacon/pyetc) at commit
[`e6e351e`](https://github.com/RolandBacon/pyetc/tree/e6e351ebf928d98adad34e8d17e5825b15595bc9)
while cross-checking slicersim's WST IFS model (see `wst_integration.md`). Each section is
a standalone issue, reported on GitHub as listed below. All repros were run with Python 3.14, mpdaf 3.7,
numpy 2.5.4, astropy 8.0.2.

| # | GitHub | Title | Severity |
|---|---|---|---|
| 1 | [#3](https://github.com/RolandBacon/pyetc/issues/3) | Point source + continuum: sky noise counted for a single spaxel | high: SNR overestimated (×1.3 to ×7) |
| 2 | [#4](https://github.com/RolandBacon/pyetc/issues/4) | Moffat PSF truncated at ~2 FWHM and renormalised | medium: aperture flux overestimated by 15–25 % |
| 3 | [#5](https://github.com/RolandBacon/pyetc/issues/5) | `square_aperture` crashes when the PSF FWHM is below ~2.5 spaxels | medium: crash (red IFS, seeing ≲ 0.8″) |
| 4 | [#6](https://github.com/RolandBacon/pyetc/issues/6) | MOS surface brightness: number of spaxels in the fibre doubled | medium: SNR overestimated by ~√2 |

---

## Issue 1 — Point source + continuum: sky noise counted for a single spaxel instead of the aperture

Reported as [RolandBacon/pyetc#3](https://github.com/RolandBacon/pyetc/issues/3).

(issue from comparison against [slicersim](https://github.com/MickaelRigault/slicersim) using Claude Opus 5.5)

**Summary**

For `ima_type='ps'` and `spec_type='cont'`, `ETC.snr_from_source` calls
[`ETC.snr_from_ps_spec`](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1044-L1093).
There, read-out noise and dark current are summed over the `nspaxels` of the aperture, but the
sky noise is computed for **one spaxel**:

- [L1074](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1074):
  `area = spaxel_area`
- [L1082](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1082):
  `nph_sky = ins_sky * ins_ins * dit * ndit * tel_eff_area * area * (dl/1e4)` (one spaxel)
- [L1083-L1087](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1083-L1087):
  `nb_voxels = nspaxels`, RON and dark scaled by `nb_voxels`
- [L1089](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1089):
  `sky_noise.data = np.sqrt(nph_sky.data)`, **not** scaled by `nb_voxels`
- [L1093](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1093):
  `tot_noise = sqrt(ron² + dark² + sky² + source²)`

The resolved-source path
([`snr_from_cube`](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L881-L935),
then the sum over spaxels at
[L793-L803](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L793-L803))
handles this correctly, so the two paths disagree for the same PSF.

**Minimal repro**

```python
import logging
from pyetc.wst import WST

wst = WST(log=logging.WARNING)
ins = wst.ifs['blue']
spec = wst.get_spec(ins, dict(type='flatcont', wave=[3700, 6100]))
obs = dict(moon='darksky', airmass=1.0, seeing=0.8, ndit=2, dit=1800., spec_type='cont',
           ima_aperture_type='square_fixed', ima_aperture_hsize_spaxels=2)
k = spec.wave.pixel(5000., nearest=True)

wst.set_obs(obs | dict(ima_type='ps'))
ps = wst.snr_from_source(ins, 1e-17, None, spec)['spec']

ima = wst.get_image_psf(ins, 5000.)       # same PSF, used as a "resolved" source
wst.set_obs(obs | dict(ima_type='resolved'))
rs = wst.snr_from_source(ins, 1e-17, ima, spec)['spec']

print(ps['snr'].data[k], rs['snr'].data[k])  # 29.97 vs 23.07, same frac_ima (0.845) and 25 spaxels
```

**Expected**: both paths give the same SNR at 5000 Å (23.07, also obtained independently with slicersim).
**Actual**: the point-source path gives 29.97.

**Impact**: the point-source continuum SNR is overestimated whenever the sky contributes,
up to a factor ~√nspaxels when sky-dominated (median SNR, 2×1800 s):

| channel | box | seeing | sky | airmass | pyetc | sky summed over the box |
|---|---|---|---|---|---|---|
| blue | 5×5 | 0.8″ | dark | 1.0 | 29.2 | 22.1 |
| red | 5×5 | 1.0″ | dark | 1.0 | 42.7 | 29.1 |
| red | 7×7 | 1.0″ | grey | 1.2 | 43.4 | 19.6 |
| blue | 9×9 | 1.0″ | bright | 1.5 | 16.5 | 2.4 |

The same quantity enters `flux_from_source`, so limiting fluxes are too faint as well.

**Suggested fix**: in `snr_from_ps_spec`, use the variance of `nb_voxels` spaxels:
`sky_noise.data = np.sqrt(nph_sky.data * nb_voxels)` (keeping `res['spec']['nph_sky']`
per spaxel, or documenting that it becomes per aperture).

---

## Issue 2 — Moffat PSF truncated at ~2 FWHM and renormalised: aperture flux overestimated

Reported as [RolandBacon/pyetc#4](https://github.com/RolandBacon/pyetc/issues/4).

(issue from comparison against [slicersim](https://github.com/MickaelRigault/slicersim) using Claude Opus 5.5)

**Summary**

[`moffat()`](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1476-L1506)
builds the PSF on a square image of side ~`kfwhm*fwhm` with default `kfwhm=2`
([L1501](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1501)),
then renormalises it to a unit sum
([L1505](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1505)).
A β=2.5 Moffat has heavy wings: ~29 % of its flux lies beyond r = 1 FWHM, and all of it is
put back inside the truncation box. Every point-source PSF uses the default:

- `get_psf_frac_ima`, [L595](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L595)
  and [L612](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L612) (point source + continuum)
- `snr_from_source`, [L719](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L719) (point source + line)
- `get_image_psf`, [L1282](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1282)

(`get_ima` for resolved Moffat sources,
[L252-L255](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L252-L255),
uses `kfwhm=3` by default, which is better but has the same issue.)

**Minimal repro**

```python
import numpy as np
from pyetc.etc import moffat

fwhm, beta, samp = 0.67, 2.5, 0.25                 # blue IFS at 5000 Å, 0.8" seeing
ima = moffat(samp, fwhm, beta, oversamp=10)
print(ima.shape[0] // 10, "spaxels on a side")     # 7 -> 1.75", i.e. ~2.6 FWHM
print(ima.data.sum())                              # 1.0 by construction

# exact fraction of a Moffat inside r = 0.875" (half the box side)
alpha = fwhm / (2 * np.sqrt(2**(1/beta) - 1))
print(1 - (1 + (0.875/alpha)**2)**(1 - beta))      # ~0.82: ~18 % of the flux is missing
```

**Expected**: the flux fraction inside an aperture is that of the full Moffat.
**Actual**: in a 5×5-spaxel box (blue, 0.8″ seeing) pyetc finds 0.85 instead of 0.72, and 1.00
for any box ≥ 7×7 (exact: 0.86 for 7×7, 0.98 for 15×15). The SNR is overestimated by 10–20 %
on top of issue 1 (#3).

**Suggested fix**: normalise the Moffat analytically (amplitude = (β−1)/(π α²)) instead of
dividing by the sum, and/or use a larger default `kfwhm` (≥ 10 for β = 2.5), keeping the
image normalised to the true total flux so that `frac_ima` < 1.

---

## Issue 3 — `square_aperture` crashes when the PSF FWHM is below ~2.5 spaxels

Reported as [RolandBacon/pyetc#5](https://github.com/RolandBacon/pyetc/issues/5).

(issue from comparison against [slicersim](https://github.com/MickaelRigault/slicersim) using Claude Opus 5.5)

**Summary**

With `kfwhm=2`, `moffat()`
([L1501](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L1501))
returns a 5×5-spaxel image as soon as `fwhm < 2.5*spaxel_size` (0.625″ for the 0.25″ IFS
spaxels). On such an image the call to `Image.peak()` in
[`square_aperture`, L544](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L544)
fails. This happens in normal use: the red IFS channel at 0.8″ seeing has FWHM 0.59–0.65″.

**Minimal repro**

```python
import logging
from pyetc.wst import WST

wst = WST(log=logging.WARNING)
ins = wst.ifs['red']
spec = wst.get_spec(ins, dict(type='flatcont', wave=[6000, 9600]))
wst.set_obs(dict(moon='darksky', airmass=1.0, seeing=0.8, ndit=2, dit=1800., spec_type='cont',
                 ima_type='ps', ima_aperture_type='square_fixed', ima_aperture_hsize_spaxels=2))
wst.snr_from_source(ins, 1e-17, None, spec)
# ValueError: cannot convert float NaN to integer   (mpdaf/obj/image.py, Image.peak)
```

The same with `ima_aperture_hsize_spaxels=3`. It works with seeing = 1.0″ in the red (FWHM ≥ 0.74″, 7×7 image).
Calling `square_aperture` directly on `moffat(0.25, 0.62, 2.5, oversamp=10)` raises
`TypeError: 'NoneType' object is not subscriptable`. With `fwhm=0.63` (a 7×7 image) it works.

**Expected**: an SNR for any seeing.
**Actual**: crash for small PSFs.

**Suggested fix**: build the PSF image large enough for the aperture (at least
`2*nsp+1` spaxels plus a margin, which also helps issue 2, #4), and avoid `Image.peak()` for an
analytic PSF whose centre is known (the central spaxel).

---

## Issue 4 — MOS surface brightness: number of spaxels in the fibre doubled

Reported as [RolandBacon/pyetc#6](https://github.com/RolandBacon/pyetc/issues/6).

(issue from comparison against [slicersim](https://github.com/MickaelRigault/slicersim) using Claude Opus 5.5)

**Summary**

For `ima_type='sb'` with a MOS instrument, `snr_from_source` computes the number of spaxels
in the fibre as
[L736](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L736):

```python
nspaxels = int((2*np.pi*ins['aperture']**2/4)/ins['spaxel_size']**2 + 0.5)
```

The fibre area is π d²/4, not 2π d²/4. Two lines below
([L738](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L738))
the aperture area is correctly `np.pi*size_ima**2/4`.

**Minimal repro**

```python
import logging, numpy as np
from pyetc.wst import WST

wst = WST(log=logging.WARNING)
mos = wst.moslr['blue']                                 # 1.0" fibre, 0.25" spaxels
spec = wst.get_spec(mos, dict(type='flatcont', wave=[4000, 5000]))
wst.set_obs(dict(moon='darksky', airmass=1.0, seeing=0.8, ndit=1, dit=1800.,
                 spec_type='cont', ima_type='sb'))
res = wst.snr_from_source(mos, 1e-17, None, spec)
print(res['spec']['nb_spaxels'], round(np.pi * 1.0**2 / 4 / 0.25**2))   # 25 vs 13
```

**Expected**: 13 spaxels.
**Actual**: 25.

**Impact**: signal, sky, RON and dark are all multiplied by `nspaxels`
([L793-L803](https://github.com/RolandBacon/pyetc/blob/e6e351ebf928d98adad34e8d17e5825b15595bc9/pyetc/etc.py#L793-L803)),
so the surface-brightness SNR of the MOS modes is overestimated by √(25/13) ≈ 1.4
(and the limiting surface brightness is too faint).

**Suggested fix**: `nspaxels = int((np.pi*ins['aperture']**2/4)/ins['spaxel_size']**2 + 0.5)`.
