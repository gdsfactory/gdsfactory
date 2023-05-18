from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import AnyUrl, BaseModel, Extra, Field

import gdsfactory as gf


class CrossSection(BaseModel):
    __root__: str = Field(
        ..., description="A cross section to use for waveguides or traces."
    )


class RouteSettings(BaseModel):
    cross_section: Optional[CrossSection] = None
    separation: Optional[float] = Field(
        5.0, description="The minimum separation between routes in the bundle [um]."
    )

    class Config:
        """Extra config."""

        extra = Extra.allow


class RoutingStrategy(BaseModel):
    __root__: str = Field(..., description="The type of routing to use")


class Links(BaseModel):
    pass

    class Config:
        """Extra config."""

        extra = Extra.allow


class PortEnum(Enum):
    ne = "ne"
    nc = "nc"
    nw = "nw"
    se = "se"
    sc = "sc"
    sw = "sw"
    ce = "ce"
    cw = "cw"
    cc = "cc"
    center = "center"


class Placement(BaseModel):
    class Config:
        """Extra config."""

        extra = Extra.forbid

    x: Optional[Union[str, float]] = Field(
        None,
        description="The x location at which to place the component.\nThis can either be a number or an other_inst,port definition, meaning it will be placed relative to the port specified on the other instance. \nIf port keyword is defined, this will be relative to the specified port. Otherwise, it will be relative to the cell origin.",
    )
    y: Optional[Union[str, float]] = Field(
        None,
        description="The y location at which to place the component.\nThis can either be a number or an other_inst,port definition, meaning it will be placed relative to the port specified on the other instance. \nIf port keyword is defined, this will be relative to the specified port. Otherwise, it will be relative to the cell origin.",
    )
    port: Optional[Union[str, PortEnum]] = Field(
        None,
        description="The port or keyword used to anchor the component. Either specify any port on the instance or one of these special keywords:\nne, nc, nw, se, sc, sw, ce, cw, cc for the northeast, north-center, northwest, etc. coordinates of the cell",
    )
    rotation: Optional[float] = Field(
        0,
        description="The rotation of the cell about the origin, or port if defined.",
    )
    dx: Optional[float] = Field(
        None,
        description="An additional displacement in the x direction. Useful if x is defined using other_inst,port syntax",
    )
    dy: Optional[float] = Field(
        None,
        description="An additional displacement in the y direction. Useful if y is defined using other_inst,port syntax",
    )
    mirror: Optional[bool] = Field(
        None,
        description="true/false value indicating whether we should flip horizontally.",
    )


class Instance(BaseModel):
    class Config:
        """Extra config."""

        extra = Extra.forbid

    component: str
    settings: Optional[Dict[str, Any]] = Field(
        None, description="Settings for the component"
    )


class Route(BaseModel):
    class Config:
        """Extra config."""

        extra = Extra.forbid

    routing_strategy: Optional[RoutingStrategy] = None
    settings: Optional[RouteSettings] = None
    links: Dict[str, str]


class PicYamlConfiguration(BaseModel):
    _schema: Optional[AnyUrl] = Field(None, alias="$schema")
    instances: Optional[Dict[str, Instance]] = None
    placements: Optional[Dict[str, Placement]] = None
    routes: Optional[Dict[str, Route]] = None
    ports: Optional[Dict[str, str]] = None

    def add_instance(
        self, name: str, component: gf.Component, placement: Optional[Placement] = None
    ) -> None:
        component_name = component.settings.function_name
        component_settings = component.settings.changed
        self.instances[name] = Instance(
            component=component_name, settings=component_settings
        )
        if not placement:
            placement = Placement()
        self.placements[name] = placement

    def move_instance_to(self, name: str, x: float, y: float) -> None:
        self.placements[name].x = x
        self.placements[name].y = y

    def move_instance(self, name: str, dx: float, dy: float) -> None:
        if not self.placements[name].dx:
            self.placements[name].dx = 0
        self.placements[name].dx += dx
        if not self.placements[name].dy:
            self.placements[name].dy = 0
        self.placements[name].dy += dy

    def to_yaml(self, filename) -> None:
        netlist = self.dict()
        with open(filename, mode="w") as f:
            yaml.dump(netlist, f, default_flow_style=None, sort_keys=False)


class SchematicConfiguration(BaseModel):
    _schema: Optional[AnyUrl] = Field(None, alias="$schema")
    instances: Optional[Dict[str, Instance]] = None
    schematic_placements: Optional[Dict[str, Placement]] = None
    nets: Optional[List[List[str]]] = None
    ports: Optional[Dict[str, str]] = None

    @property
    def placements(self):
        return self.schematic_placements

    def add_instance(
        self,
        name: str,
        component: Union[str, gf.Component],
        placement: Optional[Placement] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(component, gf.Component):
            component_name = component.settings.function_name
            component_settings = component.settings.changed
            if settings:
                component_settings = component_settings | settings
            self.instances[name] = Instance(
                component=component_name, settings=component_settings
            )
        else:
            if not settings:
                settings = {}
            self.instances[name] = Instance(component=component, settings=settings)
        if name not in self.placements:
            if not placement:
                placement = Placement()
            self.placements[name] = placement


def create_pic():
    return PicYamlConfiguration(
        instances={},
        placements={},
        routes={},
        ports={},
    )
