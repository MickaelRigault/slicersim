"""
Simulator of slicer integral field spectrograph observations.

``slicersim`` simulates observations of the Lazuli Space Observatory
integral field spectrograph (IFS), and more generally of slicer IFS. A
simulation combines a scene (point source such as a supernova, a kilonova or
a standard star, plus background and host), a telescope, a spectrograph and a
detector, and produces realistic spectra, datacubes and detector images with
their noise budget (photon noise, dark current, read-out noise, thermal
emission). The high-level entry points are the Lazuli targets of
`slicersim.lazuli` (e.g. `~slicersim.lazuli.LazuliSupernova`,
`~slicersim.lazuli.LazuliTarget`) and the exposure time calculators
`~slicersim.lazuli.lazuli_sn_etc` and `~slicersim.lazuli.lazuli_etc`; the
underlying engine is `~slicersim.simulation.Simulation`.
"""
__version__ = '1.4.1'

# Top level simulations
# quick access useful for simulation
from .iotools import get_config  # noqa: F401

# Top level target (uses simulation)
from .lazuli import *
from .simulation import *
