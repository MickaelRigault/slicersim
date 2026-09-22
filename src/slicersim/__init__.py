"""
Slicersim is a Python package for simulating integral field spectrograph (IFS) data. 
It serves as Lazuli Space Observatory slicer's simulator.

It provides tools to create realistic datacubes, including various noise sources
and instrument effects.
"""
__version__ = '1.3.0'

# Top level simulations
# quick access useful for simulation
from .iotools import get_config  # noqa: F401

# Top level target (uses simulation)
from .lazuli import *
from .simulation import *
