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
sim.update(redshift=1.2)
lbda, flux_1, variance_1 = sim.get_spectrum(incl_error=True)

sim.update(redshift=0.7)
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


# Credits
_adapted from the original MLAPerf v:0.18.0 developed by Y. Copin (lead) and M. Rigault_
