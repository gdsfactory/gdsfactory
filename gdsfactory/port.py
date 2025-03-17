"""We use Ports to connect Components with other Components.

we follow start from the bottom left and name the ports counter-clock-wise

.. code::

         3   4
         |___|_
     2 -|      |- 5
        |      |
     1 -|______|- 6
         |   |
         8   7

You can also rename them with W,E,S,N prefix (west, east, south, north).

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

import csv
import functools
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict, Unpack, cast

import kfactory as kf
import numpy as np
from rich.console import Console
from rich.table import Table

from gdsfactory import typings
from gdsfactory.typings import (
    AngleInDegrees,
    ComponentFactory,
    CrossSectionSpec,
    LayerSpec,
    LayerSpecs,
    PathType,
    PortDict,
    Ports,
    PortsDict,
    PortsDictGeneric,
    SelectPorts,
    TPort,
)

if TYPE_CHECKING:
    from gdsfactory.component import Component, ComponentReference

valid_error_types = ["error", "warn", "ignore"]


class PortNotOnGridError(ValueError):
    pass


class PortTypeError(ValueError):
    pass


class PortOrientationError(ValueError):
    pass


def pprint_ports(ports: Ports) -> None:
    """Prints ports in a rich table."""
    console = Console()
    table = Table(show_header=True, header_style="bold")

    keys = ["name", "width", "orientation", "layer", "center", "port_type"]

    for key in keys:
        table.add_column(key)

    for port in ports:
        row = [
            str(i)
            for i in [
                port.name,
                np.round(port.width, 3),
                port.orientation,
                port.layer_info,
                port.center,
                port.port_type,
            ]
        ]
        table.add_row(*row)

    console.print(table)


Port: TypeAlias = kf.DPort


def to_dict(port: kf.port.ProtoPort[Any]) -> dict[str, Any]:
    """Returns dict."""
    return {
        "name": port.name,
        "center": port.center,
        "width": port.width,
        "orientation": port.orientation,
        "layer": port.layer,
        "port_type": port.port_type,
    }


class PortKwargs(TypedDict, total=False):
    layer: int
    port_type: str
    cross_section: CrossSectionSpec
    info: dict[str, int | float | str]


def port_array(
    center: tuple[float, float] = (0.0, 0.0),
    width: float = 0.5,
    orientation: AngleInDegrees = 0,
    pitch: tuple[float, float] = (10.0, 0.0),
    n: int = 2,
    **kwargs: Unpack[PortKwargs],
) -> list[Port]:
    """Returns a list of ports placed in an array.

    Args:
        center: center point of the port.
        width: port width.
        orientation: angle in degrees.
        pitch: period of the port array.
        n: number of ports in the array.
        kwargs: additional arguments.

    """
    from gdsfactory.pdk import get_cross_section, get_layer

    pitch_array = np.array(pitch)
    if "layer" in kwargs:
        kwargs["layer"] = get_layer(kwargs["layer"])
    if "cross_section" in kwargs:
        cross_section = kwargs.pop("cross_section")
        xs = get_cross_section(cross_section)
        if width != xs.width:
            xs = get_cross_section(xs.copy(width=width))
        try:
            sym_xs: kf.SymmetricalCrossSection | None = (
                gf.kcl.get_symmetrical_cross_section(xs.name)
            )
        except KeyError:
            sym_xs = None

        kwargs.pop("cross_section", None)
        info = kwargs.get("info", {})
        info["cross_section"] = xs.name
        kwargs["info"] = info

        return [
            Port(
                name=str(i),
                center=cast(
                    tuple[float, float],
                    tuple(
                        np.array(center) + i * pitch_array - (n - 1) / 2 * pitch_array
                    ),
                ),
                orientation=orientation,
                cross_section=sym_xs,
                **kwargs,
            )  # type: ignore[call-overload]
            for i in range(n)
        ]
    else:
        return [
            Port(
                name=str(i),
                center=cast(
                    tuple[float, float],
                    tuple(
                        np.array(center) + i * pitch_array - (n - 1) / 2 * pitch_array
                    ),
                ),
                orientation=orientation,
                width=width,
                **kwargs,
            )  # type: ignore[call-overload]
            for i in range(n)
        ]


def read_port_markers(
    component: Component, layers: LayerSpecs = ("PORT",)
) -> Component:
    """Returns extracted polygons from component layers.

    Args:
        component: Component to extract markers.
        layers: GDS layer specs.

    """
    from gdsfactory.pdk import get_layer

    layers = [get_layer(layer) for layer in layers]
    return component.extract(layers=layers)


def csv2port(csvpath: PathType) -> dict[str, list[str]]:
    """Reads ports from a CSV file and returns a Dict."""
    ports: dict[str, list[str]] = {}
    with open(csvpath) as csvfile:
        rows = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in rows:
            ports[row[0]] = row[1:]
    return ports


def sort_ports_clockwise(ports: Sequence[TPort]) -> list[TPort]:
    """Sort and return ports in the clockwise direction.

    .. code::

            3   4
            |___|_
        2 -|      |- 5
           |      |
        1 -|______|- 6
            |   |
            8   7

    """
    direction_ports: PortsDictGeneric[TPort] = {x: [] for x in ["E", "N", "W", "S"]}

    for p in ports:
        angle = p.angle * 90
        if angle <= 45 or angle >= 315:
            direction_ports["E"].append(p)
        elif angle <= 135 and angle >= 45:
            direction_ports["N"].append(p)
        elif angle <= 225 and angle >= 135:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: -p.dy)  # sort north to south

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: +p.dx)  # sort west to east

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: +p.dy)  # sort south to north

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: -p.dx)  # sort east to west

    ports = west_ports + north_ports + east_ports + south_ports
    return ports


def sort_ports_counter_clockwise(ports: Sequence[TPort]) -> list[TPort]:
    """Sort and return ports in the counter-clockwise direction.

    .. code::

            4   3
            |___|_
        5 -|      |- 2
           |      |
        6 -|______|- 1
            |   |
            7   8

    """
    direction_ports: PortsDictGeneric[TPort] = {x: [] for x in ["E", "N", "W", "S"]}

    for p in ports:
        angle = p.angle * 90
        if angle <= 45 or angle >= 315:
            direction_ports["E"].append(p)
        elif angle <= 135 and angle >= 45:
            direction_ports["N"].append(p)
        elif angle <= 225 and angle >= 135:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: +p.dy)  # sort south to north

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: -p.dx)  # sort east to west

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: -p.dy)  # sort north to south

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: +p.dx)  # sort west to east

    ports = east_ports + north_ports + west_ports + south_ports
    return list(ports)


def select_ports(
    ports: Ports | ComponentReference,
    layer: LayerSpec | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    orientation: AngleInDegrees | None = None,
    width: float | None = None,
    layers_excluded: Sequence[tuple[int, int]] | None = None,
    port_type: str | None = None,
    names: Sequence[str] | None = None,
    clockwise: bool = True,
    sort_ports: bool = False,
) -> list[typings.Port]:
    """Returns a list of ports from a list of ports.

    Args:
        ports: port list.
        layer: select ports with port GDS layer.
        prefix: select ports with port name prefix.
        suffix: select ports with port name suffix.
        orientation: select ports with orientation in degrees.
        width: select ports with port width.
        layers_excluded: List of layers to exclude.
        port_type: select ports with port type (optical, electrical, vertical_te).
        names: select ports with port names.
        clockwise: if True, sort ports clockwise, False: counter-clockwise.
        sort_ports: if True, sort ports.

    Returns:
        List containing the selected ports.

    """
    from gdsfactory.component import ComponentReference

    if isinstance(ports, ComponentReference):
        ports_ = list(ports.ports)
    else:
        ports_ = list(ports)

    from gdsfactory.pdk import get_layer

    if layer:
        layer = get_layer(layer)
        ports_ = [p for p in ports_ if get_layer(p.layer) == layer]
    else:
        ports_ = list(ports_)

    if prefix:
        ports_ = [p for p in ports_ if p.name and p.name.startswith(prefix)]
    if suffix:
        ports_ = [p for p in ports_ if p.name and p.name.endswith(suffix)]
    if orientation is not None:
        ports_ = [p for p in ports_ if np.isclose(p.orientation, orientation)]

    if layers_excluded:
        ports_ = [p for p in ports_ if p.layer not in map(get_layer, layers_excluded)]
    if width:
        ports_ = [p for p in ports_ if p.width == width]
    if port_type:
        ports_ = [p for p in ports_ if p.port_type == port_type]
    if names:
        ports_ = [p for p in ports_ if p.name in names]

    if sort_ports:
        if clockwise:
            ports_ = list(sort_ports_clockwise(ports_))
        else:
            ports_ = list(sort_ports_counter_clockwise(ports_))
    return ports_


select_ports_optical = partial(select_ports, port_type="optical")
select_ports_electrical = partial(select_ports, port_type="electrical")
select_ports_placement = partial(select_ports, port_type="placement")


def select_ports_list(
    ports: Ports | Ports | ComponentReference, **kwargs: Any
) -> Ports:
    return select_ports(ports=ports, **kwargs)


get_ports_list = select_ports_list


def flipped(port: typings.Port) -> typings.Port:
    p = port.copy()
    p.trans *= kf.kdb.Trans.R180
    return p


def move_copy(port: typings.Port, x: int = 0, y: int = 0) -> typings.Port:
    warnings.warn(
        "Port.move_copy(...) should be used instead of move_copy(Port, ...).",
    )
    _port = port.copy()
    _port.center = (port.center[0] + x, port.center[1] + y)
    return _port


def get_ports_facing(
    ports: Sequence[typings.Port], direction: str = "W"
) -> list[typings.Port]:
    from gdsfactory.component import Component, ComponentReference

    valid_directions = ["E", "N", "W", "S"]

    if direction not in valid_directions:
        raise PortOrientationError(f"{direction} must be in {valid_directions} ")

    if isinstance(ports, dict):
        ports = list(ports)
    elif isinstance(ports, Component | ComponentReference):
        ports = list(ports.ports)

    direction_ports: dict[str, list[typings.Port]] = {
        x: [] for x in ["E", "N", "W", "S"]
    }

    for p in ports:
        angle = p.orientation % 360
        if angle <= 45 or angle >= 315:
            direction_ports["E"].append(p)
        elif angle <= 135 and angle >= 45:
            direction_ports["N"].append(p)
        elif angle <= 225 and angle >= 135:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    return direction_ports[direction]


def deco_rename_ports(component_factory: "ComponentFactory") -> "ComponentFactory":
    @functools.wraps(component_factory)
    def auto_named_component_factory(*args: Any, **kwargs: Any) -> Component:
        component = component_factory(*args, **kwargs)
        auto_rename_ports(component)
        return component

    return auto_named_component_factory


def _rename_ports_facing_side(
    direction_ports: dict[str, list[Port]], prefix: str = ""
) -> None:
    """Renames ports clockwise."""
    for direction, list_ports in list(direction_ports.items()):
        if direction in ["E", "W"]:
            # first sort along x then y
            list_ports.sort(key=lambda p: p.dx)
            list_ports.sort(key=lambda p: p.dy)

        if direction in ["S", "N"]:
            # first sort along y then x
            list_ports.sort(key=lambda p: p.dy)
            list_ports.sort(key=lambda p: p.dx)

        for i, p in enumerate(list_ports):
            p.name = prefix + direction + str(i)


def _rename_ports_facing_side_ccw(
    direction_ports: dict[str, list[Port]], prefix: str = ""
) -> None:
    """Renames ports counter-clockwise."""
    for direction, list_ports in list(direction_ports.items()):
        if direction in ["E", "W"]:
            # first sort along x then y
            list_ports.sort(key=lambda p: -p.dx)
            list_ports.sort(key=lambda p: -p.dy)

        if direction in ["S", "N"]:
            # first sort along y then x
            list_ports.sort(key=lambda p: -p.dy)
            list_ports.sort(key=lambda p: -p.dx)

        for i, p in enumerate(list_ports):
            p.name = prefix + direction + str(i)


def _rename_ports_counter_clockwise(
    direction_ports: dict[Literal["N", "E", "S", "W"], list[Port]],
    prefix: str = "",
) -> None:
    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: +p.dy)  # sort south to north

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: -p.dx)  # sort east to west

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: -p.dy)  # sort north to south

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: +p.dx)  # sort west to east

    ports = east_ports + north_ports + west_ports + south_ports

    for i, p in enumerate(ports):
        p.name = f"{prefix}{i + 1}" if prefix else f"{i + 1}"


def _rename_ports_clockwise(direction_ports: PortsDict, prefix: str = "") -> None:
    """Rename ports in the clockwise directionjstarting from the bottom left corner."""
    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: -p.dy)  # sort north to south

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: +p.dx)  # sort west to east

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: +p.dy)  # sort south to north

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: -p.dx)  # sort east to west
    # south_ports.sort(key=lambda p: p.dy)  #  south first

    ports = west_ports + north_ports + east_ports + south_ports

    for i, p in enumerate(ports):
        p.name = f"{prefix}{i + 1}" if prefix else f"{i + 1}"


def _rename_ports_clockwise_top_right(
    direction_ports: PortsDict, prefix: str = ""
) -> None:
    """Rename ports in clockwise direction starting from the top right corner."""
    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: -p.dy)  # sort north to south

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: +p.dx)  # sort west to east

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: +p.dy)  # sort south to north

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: -p.dx)  # sort east to west

    ports = east_ports + south_ports + west_ports + north_ports

    for i, p in enumerate(ports):
        p.name = f"{prefix}{i + 1}" if prefix else f"{i + 1}"


def rename_ports_by_orientation(
    component: Component,
    layers_excluded: LayerSpecs | None = None,
    select_ports: SelectPorts = select_ports,
    function: Callable[..., None] = _rename_ports_facing_side,
    prefix: str = "o",
    **kwargs: Any,
) -> Component:
    """Returns Component with port names based on port orientation (E, N, W, S).

    Args:
        component: to rename ports.
        layers_excluded: to exclude.
        select_ports: function to select_ports.
        function: to rename ports.
        prefix: to add on each port name.
        kwargs: select_ports settings.

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """
    layers_excluded = layers_excluded or []
    direction_ports: PortsDict = {x: [] for x in ["E", "N", "W", "S"]}

    ports = list(select_ports(component.ports, **kwargs))

    ports_on_layer = [p for p in ports if p.layer not in layers_excluded]

    for p in ports_on_layer:
        angle = p.orientation % 360
        if angle <= 45 or angle >= 315:
            direction_ports["E"].append(p)
        elif angle <= 135 and angle >= 45:
            direction_ports["N"].append(p)
        elif angle <= 225 and angle >= 135:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    function(direction_ports, prefix=prefix)
    return component


def auto_rename_ports(
    component: Component,
    function: Callable[..., None] = _rename_ports_clockwise,
    select_ports_optical: Callable[..., list[typings.Port]]
    | None = select_ports_optical,
    select_ports_electrical: Callable[..., list[typings.Port]]
    | None = select_ports_electrical,
    select_ports_placement: Callable[..., list[typings.Port]]
    | None = select_ports_placement,
    prefix: str = "",
    prefix_optical: str = "o",
    prefix_electrical: str = "e",
    prefix_placement: str = "p",
    port_type: str | None = None,
    **kwargs: Any,
) -> Component:
    """Adds prefix for optical and electrical.

    Args:
        component: to auto_rename_ports.
        function: to rename ports.
        select_ports_optical: to select optical ports.
        select_ports_electrical: to select electrical ports.
        select_ports_placement: to select placement ports.
        prefix_optical: prefix of optical ports.
        prefix_electrical: prefix of electrical ports.
        prefix_placement: prefix of electrical ports.
        port_type: select ports with port type (optical, electrical, vertical_te).
        kwargs: select_ports settings.

    Keyword Args:
        prefix: select ports with port name prefix.
        suffix: select ports with port name suffix.
        orientation: select ports with orientation in degrees.
        width: select ports with port width.
        layers_excluded: List of layers to exclude.
        clockwise: if True, sort ports clockwise, False: counter-clockwise.

    """
    if port_type is None:
        if select_ports_optical:
            rename_ports_by_orientation(
                component=component,
                select_ports=select_ports_optical,
                prefix=prefix_optical,
                function=function,
                **kwargs,
            )
        if select_ports_electrical:
            rename_ports_by_orientation(
                component=component,
                select_ports=select_ports_electrical,
                prefix=prefix_electrical,
                function=function,
                **kwargs,
            )
        if select_ports_placement:
            rename_ports_by_orientation(
                component=component,
                select_ports=select_ports_placement,
                prefix=prefix_placement,
                function=function,
                **kwargs,
            )
    else:
        rename_ports_by_orientation(
            component=component,
            select_ports=select_ports,
            prefix=prefix,
            function=function,
            port_type=port_type,
            **kwargs,
        )
    return component


auto_rename_ports_counter_clockwise = partial(
    auto_rename_ports, function=_rename_ports_counter_clockwise
)
auto_rename_ports_orientation = partial(
    auto_rename_ports, function=_rename_ports_facing_side
)

auto_rename_ports_electrical = partial(auto_rename_ports, select_ports_optical=None)


def map_ports_layer_to_orientation(
    ports: "PortDict",
    function: Callable[..., None] = _rename_ports_facing_side,
    **kwargs: Any,
) -> dict[str, str]:
    """Returns dict of port name to port name original.

    Args:
        ports: dict of ports.
        function: to rename ports.
        kwargs: for the function to rename ports.

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """
    m: dict[str, str] = {}
    selected_ports = select_ports(list(ports.values()), **kwargs)
    layers = {port.layer for port in selected_ports}

    for layer in layers:
        direction_ports: PortsDict = {x: [] for x in ["E", "N", "W", "S"]}
        ports_on_layer = [p.copy() for p in selected_ports if p.layer == layer]

        for p in ports_on_layer:
            p.name_original = p.name  # type: ignore[attr-defined]
            angle = p.orientation % 360
            if angle <= 45 or angle >= 315:
                direction_ports["E"].append(p)
            elif angle <= 135 and angle >= 45:
                direction_ports["N"].append(p)
            elif angle <= 225 and angle >= 135:
                direction_ports["W"].append(p)
            else:
                direction_ports["S"].append(p)
        layer_tuple = layer if isinstance(layer, kf.LayerEnum) else (layer, 0)
        function(direction_ports, prefix=f"{layer_tuple[0]}_{layer_tuple[1]}_")
        m |= {p.name: p.name_original for p in ports_on_layer}  # type: ignore[attr-defined,misc]
    return m


def map_ports_to_orientation_cw(
    ports: PortDict,
    function: Callable[..., None] = _rename_ports_facing_side,
    **kwargs: Any,
) -> dict[str, str]:
    """Returns component or reference port mapping clockwise.

    Args:
        ports: dict of ports.
        function: to rename ports.
        kwargs: for the function to rename ports.


    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """
    direction_ports: PortsDict = {x: [] for x in ["E", "N", "W", "S"]}

    selected_ports = select_ports(list(ports.values()), **kwargs)
    ports_on_layer = [p.copy() for p in selected_ports]

    for p in ports_on_layer:
        p.name_original = p.name  # type: ignore[attr-defined]
        angle = p.orientation % 360
        if angle <= 45 or angle >= 315:
            direction_ports["E"].append(p)
        elif angle <= 135 and angle >= 45:
            direction_ports["N"].append(p)
        elif angle <= 225 and angle >= 135:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)
    function(direction_ports)
    return {p.name: p.name_original for p in ports_on_layer}  # type: ignore[attr-defined,misc]


map_ports_to_orientation_ccw = partial(
    map_ports_to_orientation_cw, function=_rename_ports_facing_side_ccw
)


def auto_rename_ports_layer_orientation(
    component: Component,
    function: Callable[..., None] = _rename_ports_facing_side,
) -> None:
    """Renames port names with layer_orientation  (1_0_W0).

    port orientation (E, N, W, S) numbering is clockwise

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """
    new_ports: dict[str, typings.Port] = {}
    ports = component.ports
    direction_ports: PortsDict = {x: [] for x in ["E", "N", "W", "S"]}
    layers = {port.layer for port in ports}

    for layer in layers:
        ports_on_layer = [p for p in ports if p.layer == layer]

        for p in ports_on_layer:
            p.name_original = p.name  # type: ignore[attr-defined]
            angle = p.orientation % 360
            if angle <= 45 or angle >= 315:
                direction_ports["E"].append(p)
            elif angle <= 135 and angle >= 45:
                direction_ports["N"].append(p)
            elif angle <= 225 and angle >= 135:
                direction_ports["W"].append(p)
            else:
                direction_ports["S"].append(p)

        layer_tuple = layer if isinstance(layer, kf.LayerEnum) else (layer, 0)

        function(direction_ports, prefix=f"{layer_tuple[0]}_{layer_tuple[1]}_")
        new_ports |= {p.name: p for p in ports_on_layer if p.name is not None}


__all__ = [
    "Port",
    "auto_rename_ports",
    "auto_rename_ports_counter_clockwise",
    "auto_rename_ports_orientation",
    "csv2port",
    "deco_rename_ports",
    "flipped",
    "get_ports_facing",
    "map_ports_layer_to_orientation",
    "move_copy",
    "port_array",
    "read_port_markers",
    "rename_ports_by_orientation",
    "select_ports",
    "select_ports_list",
]

if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.components import mzi

    c = mzi()
    p = c.ports["o1"]
    d = gf.port.to_dict(p)
    print(d)
    c.show()
