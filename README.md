# slicersim
Simulation of Slicer observations

# Installation
```bash
git clone https://github.com/MickaelRigault/slicersim.git
cd slicersims
pip install .
```

# Quick look

```python
import slicersim
# => a simulation from a config 
config = slicersim.iotools.get_config(scene='supernova.toml')
sim = slicersim.Simulation.from_config(config)

# update the simulation (see sim.mutable_parameters)
sim.update(target__redshift=1.2)
lbda, flux_1, variance_1 = sim.get_spectrum(incl_error=True)

sim.update(target__redshift=0.7)
lbda, flux_2, variance_2 = sim.get_spectrum(incl_error=True)
```

and show your simulated spectra
```python
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=[7,3])

ax.plot(lbda, flux_1)
ax.fill_between(lbda, 
                flux_1-np.sqrt(variance_1),
                flux_1+np.sqrt(variance_1), alpha=0.3,
               label="z=1.2")

ax.plot(lbda, flux_2)
ax.fill_between(lbda, 
                flux_2-np.sqrt(variance_2),
                flux_2+np.sqrt(variance_2), alpha=0.3,
               label="z=0.7")
ax.legend(frameon=False, fontsize="small")
ax.set(xlabel=r"wavelength [$\AA$]", ylabel="flux [ADU]")
```
![readme example](docs/gallery/spectra_examples.png)

***
# Estimation Time Calculator

Load your simulation
```python
config = slicersim.iotools.get_config(scene='supernova.toml')
sim = slicersim.Simulation.from_config(config)
```
Create the SN Ia of interest and fetch the correct configuration that leads to an averave SNR of 10 between rest-frame 4_000 and 7_000 A.
```python
sim.update(target__redshift = 1, target__c=0.3, target__x1=-0.5)
new_config, snr, total_exptime = sim.fetch_snr(10, lbda_range= [4000, 7000], frame="rest")
print(f"{new_config}")
```
```bash
{'nmd': (64, 8, 0), 'nramp': 2}
```
And now let's set this configuration to the simulation
```python
sim.update(**new_config)
sim.get_times()
```
```bash
{'integration_time': 1426.32,
 'exposure_time': 1448.96,
 'tframe': 2.83,
 'tgroup': 22.64,
 'total_exptime': 2897.92}
```
***
# Study origin of variance

Once your simulator is loaded, you have several method to check the variance origin. 
- `estimate_variance_contribution`: that probe the variance origin for a given wavelength rate
- `estimate_variance_contribution_spectra`: similar but for whole spectrum
- `show_variance_sources`: plotting function associated to `estimate_variance_contribution_spectra`

```python
import slicersim
# load the correct simulator
config = slicersim.iotools.get_config(instrument='lazuli.toml')
sim = slicersim.Simulation.from_config(config)

# Set the target you want
sim.update(target__redshift=1., target__x1=0, target__c=0)

# look for the config needed to get an average SNR of 20 in [4000, 6800] rest-frame
new_config, snr, integration_time = sim.fetch_snr(20, lbda_range=[4000, 6800], frame='rest')
sim.update(**new_config) # let's set it

# get the expected spectrum (in ADU)
lbda, flux, variance = sim.get_spectrum()

# Show details
fig = sim.show_variance_sources(flux_calibrated=False)
```
![readme example](docs/gallery/spectra_variance.png)

***
PSF & Noise equivalent area
by default, the code assumes gaussian PSF both for spatial (at the slicer/mla) and cross-dispersion (at pixels) level.

`slicersim.profiles` contains additional PSF model (from (astropy.modeling)[https://docs.astropy.org/en/latest/modeling/predef_models2D.html] 
that can be used to build 2D model or estimate the PSF noise equivalent area (nea).
*(more to come..., see: slicersim.profiles.get_2dpsf_nea())*

***
# Credits
Developped by M. Rigault ; _adapted from the original MLAPerf v:0.18.0 developed by Y. Copin and M. Rigault_
