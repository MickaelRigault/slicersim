"""
Slicersim is a Python package for simulating integral field spectrograph (IFS) data.

It provides tools to create realistic datacubes, including various noise sources
and instrument effects.

Top-level functionalities of this package are currently built for the Lazuli Space
Telescope, but the code is generic and can be extended to any Integral Field
Unit (slicer or micro-lens array).
"""
__version__ = '0.32.0'

# Top level simulations
from .simulation import * # noqa: F403

# Top level target (uses simulation)
from .lazuli import * # noqa: F403

# quick access useful for simulation
from .iotools import get_config # noqa: F401
