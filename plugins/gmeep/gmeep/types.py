from typing import List
import dataclasses
from meep import mpb


@dataclasses.dataclass
class Mode:
    neff: float
    ng: float
    solver: mpb.ModeSolver


@dataclasses.dataclass
class SweepWavelength:
    wavelengths: List[float]
    neffs: List[float]
    ngs: List[float]


@dataclasses.dataclass
class SweepWidth:
    widths: List[float]
    neffs: List[float]
    ngs: List[float]
