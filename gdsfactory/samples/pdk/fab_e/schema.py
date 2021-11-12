from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

Layer = Tuple[int, int]
Point = Tuple[float, float]
Points = List[Point]


class SectionModel(BaseModel):
    """
    Args:
        width: of the section
        offset: center to center
        layer:
        ports: optional name of the ports
        name: optional section name
        port_types:
    """

    width: float
    offset: float = 0
    layer: Layer = (1, 0)
    ports: Tuple[Optional[str], Optional[str]] = (None, None)
    name: Optional[str] = None
    port_types: Tuple[str, str] = ("optical", "optical")


class CrossSectionModel(BaseModel):
    width: float = 0.5
    layer: Layer = (1, 0)
    width_wide: Optional[float] = None
    auto_widen: bool = False
    auto_widen_minimum_length: float = 200.0
    taper_length: float = 10.0
    radius: float = 10.0
    cladding_offset: float = 3.0
    layers_cladding: Optional[Tuple[Layer, ...]] = None
    sections: Optional[Tuple[SectionModel, ...]] = None
    port_names: Tuple[str, str] = ("o1", "o2")
    port_types: Tuple[str, str] = ("optical", "optical")
    min_length: float = 10e-3
    start_straight_length: float = 10e-3
    end_straight_length: float = 10e-3
    snap_to_grid: Optional[float] = None


class PathModel(BaseModel):
    points: Points


class PathStraightModel(BaseModel):
    length: float
    npoints: int = 2


class PathEulerModel(BaseModel):
    radius: float = 10
    angle: int = 90
    p: float = 0.5
    use_eff: bool = False
    npoints: int = 720


class PathArcModel(BaseModel):
    radius: float = 10
    angle: int = 90
    npoints: int = 720


class TechModel(BaseModel):
    cross_sections: Dict[str, CrossSectionModel]
    layers: Dict[str, Layer]


class ComponentTwoPortsModel(BaseModel):
    cross_section: CrossSectionModel
    path: PathModel


if __name__ == "__main__":
    x = CrossSectionModel()
    t = TechModel.schema_json()
