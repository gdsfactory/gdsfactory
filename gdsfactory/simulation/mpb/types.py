import dataclasses
from typing import Callable, Dict, List, Optional, Union

from meep import mpb


@dataclasses.dataclass
class Mode:
    solver: mpb.ModeSolver
    mode_number: int
    wavelength: float
    neff: float
    ng: Optional[float] = None
    fraction_te: Optional[float] = None
    fraction_tm: Optional[float] = None


@dataclasses.dataclass
class WavelengthSweep:
    wavelength: List[float]
    neff: Dict[int, List[float]]
    ng: Dict[int, List[float]]


@dataclasses.dataclass
class WidthSweep:
    width: List[float]
    neff: Dict[int, List[float]]


ModeSolverFactory = Callable[..., mpb.ModeSolver]
ModeSolverOrFactory = Union[mpb.ModeSolver, ModeSolverFactory]
