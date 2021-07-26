from gtidy3d import materials

from gtidy3d.config import PATH, logger
from gtidy3d.get_simulation import get_simulation
from gtidy3d.get_coupling import get_coupling


__version__ = "0.0.1"
__all__ = [
    "get_simulation",
    "get_coupling",
    "PATH",
    "logger",
    "materials",
    "__version__",
]
