from __future__ import annotations

from enum import Enum
from typing import Any

import yaml
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, RootModel

import gdsfactory as gf

SCHEMA_VERSION = 1


class CrossSection(RootModel):
    root: str = Field(
        ..., description="A cross section to use for waveguides or traces."
    )


class RouteSettings(BaseModel):
    cross_section: CrossSection | None = None
    separation: float | None = Field(
        5.0, description="The minimum separation between routes in the bundle [um]."
    )
    model_config = ConfigDict(extra="allow")


class RoutingStrategy(RootModel):
    root: str = Field(..., description="The type of routing to use")


class Links(BaseModel):
    pass
    model_config = ConfigDict(extra="allow")


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
    model_config = ConfigDict(extra="forbid")

    x: str | float | None = Field(
        None,
        description="The x location at which to place the component.\nThis can either be a number or an other_inst,port definition, meaning it will be placed relative to the port specified on the other instance. \nIf port keyword is defined, this will be relative to the specified port. Otherwise, it will be relative to the cell origin.",
    )
    y: str | float | None = Field(
        None,
        description="The y location at which to place the component.\nThis can either be a number or an other_inst,port definition, meaning it will be placed relative to the port specified on the other instance. \nIf port keyword is defined, this will be relative to the specified port. Otherwise, it will be relative to the cell origin.",
    )
    port: str | PortEnum | None = Field(
        None,
        description="The port or keyword used to anchor the component. Either specify any port on the instance or one of these special keywords:\nne, nc, nw, se, sc, sw, ce, cw, cc for the northeast, north-center, northwest, etc. coordinates of the cell",
    )
    rotation: float | None = Field(
        0,
        description="The rotation of the cell about the origin, or port if defined.",
    )
    dx: float | None = Field(
        None,
        description="An additional displacement in the x direction. Useful if x is defined using other_inst,port syntax",
    )
    dy: float | None = Field(
        None,
        description="An additional displacement in the y direction. Useful if y is defined using other_inst,port syntax",
    )
    mirror: bool | None = Field(
        None,
        description="true/false value indicating whether we should flip horizontally.",
    )


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str
    settings: dict[str, Any] | None = Field(
        None, description="Settings for the component"
    )


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing_strategy: RoutingStrategy | None = None
    settings: RouteSettings | None = None
    links: dict[str, str]


class PicYamlConfiguration(BaseModel):
    schema_version: str = Field(
        default=SCHEMA_VERSION, description="The version of the YAML syntax used."
    )
    schema: AnyUrl | None = Field(None, alias="$schema")
    instances: dict[str, Instance] | None = None
    placements: dict[str, Placement] | None = None
    routes: dict[str, Route] | None = None
    ports: dict[str, str] | None = None

    def add_instance(
        self, name: str, component: gf.Component, placement: Placement | None = None
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
        netlist = self.model_dump()
        with open(filename, mode="w") as f:
            yaml.dump(netlist, f, default_flow_style=None, sort_keys=False)


class SchematicConfiguration(BaseModel):
    schema: AnyUrl | None = Field(None, alias="$schema")
    instances: dict[str, Instance] | None = None
    schematic_placements: dict[str, Placement] | None = None
    nets: list[list[str]] | None = None
    ports: dict[str, str] | None = None
    schema_version: int = Field(
        default=SCHEMA_VERSION, description="The version of the YAML syntax used."
    )

    @property
    def placements(self):
        return self.schematic_placements

    def add_instance(
        self,
        name: str,
        component: str | gf.Component,
        placement: Placement | None = None,
        settings: dict[str, Any] | None = None,
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
