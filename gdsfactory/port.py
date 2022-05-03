"""
For port naming we follow start from the bottom left and name the ports
counter-clock-wise

.. code::

         3   4
         |___|_
     2 -|      |- 5
        |      |
     1 -|______|- 6
         |   |
         8   7


You can also rename them W,E,S,N prefix (west, east, south, north)

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1


"""
from __future__ import annotations

import csv
import functools
import typing
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import phidl.geometry as pg
from numpy import ndarray
from phidl.device_layout import Port as PortPhidl

from gdsfactory.cross_section import CrossSection
from gdsfactory.serialization import clean_value_json
from gdsfactory.snap import snap_to_grid

if typing.TYPE_CHECKING:
    from gdsfactory.component import Component

Layer = Tuple[int, int]


class PortNotOnGridError(ValueError):
    pass


class PortTypeError(ValueError):
    pass


class PortOrientationError(ValueError):
    pass


class Port(PortPhidl):
    """Ports are useful to connect Components with each other.
    Extends phidl port with layer and cross_section

    Args:
        name: we name ports clock-wise starting from bottom left.
        midpoint: (x, y) port center coordinate.
        width: of the port
        orientation: in degrees (0: east, 90: north, 180: west, 270: south)
        parent: parent component (component to which this port belong to)
        layer: layer tuple.
        port_type: str (optical, electrical, vertical_te, vertical_tm)
        parent: Component that port belongs to.
        cross_section:
        shear_angle: an optional angle to shear port face in degrees.
    """

    _next_uid = 0

    def __init__(
        self,
        name: str,
        midpoint: Tuple[float, float],
        width: float,
        orientation: Optional[float],
        layer: Optional[Tuple[int, int]] = None,
        port_type: str = "optical",
        parent: Optional[Component] = None,
        cross_section: Optional[CrossSection] = None,
        shear_angle: Optional[float] = None,
    ) -> None:
        self.name = name
        self.midpoint = np.array(midpoint, dtype="float64")
        self.width = width
        self.orientation = np.mod(orientation, 360) if orientation else orientation
        self.parent = parent
        self.info: Dict[str, Any] = {}
        self.uid = Port._next_uid
        self.layer = layer
        self.port_type = port_type
        self.cross_section = cross_section
        self.shear_angle = shear_angle

        if cross_section is None and layer is None:
            raise ValueError("You need Port to define cross_section or layer")

        if layer is None:
            layer = cross_section.layer

        if self.width < 0:
            raise ValueError("[PHIDL] Port width must be >=0")
        Port._next_uid += 1

    def to_dict(self) -> Dict[str, Any]:
        d = dict(
            name=self.name,
            width=self.width,
            midpoint=tuple(np.round(self.midpoint, 3)),
            orientation=int(self.orientation) if self.orientation else self.orientation,
            layer=self.layer,
            port_type=self.port_type,
        )
        if self.shear_angle:
            d["shear_angle"] = self.shear_angle
        return clean_value_json(d)

    def __repr__(self) -> str:
        return f"Port (name {self.name}, midpoint {self.midpoint}, width {self.width}, orientation {self.orientation}, layer {self.layer}, port_type {self.port_type})"

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """For pydantic assumes Port is valid if has a name and a valid type

        - has a name
        """
        assert v.name, f"Port has no name, got `{v.name}`"
        # assert v.assert_on_grid(), f"port.midpoint = {v.midpoint} has off-grid points"
        assert isinstance(v, Port), f"TypeError, Got {type(v)}, expecting Port"
        return v

    @property
    def settings(self):
        """TODO! delete this. Use to_dict instead"""
        return dict(
            name=self.name,
            midpoint=self.midpoint,
            width=self.width,
            orientation=self.orientation,
            layer=self.layer,
            port_type=self.port_type,
        )

    @property
    def angle(self):
        """convenient alias"""
        return self.orientation

    @angle.setter
    def angle(self, a) -> None:
        self.orientation = a

    @property
    def position(self) -> Tuple[float, float]:
        return self.midpoint

    @position.setter
    def position(self, p) -> None:
        self.midpoint = np.array(p, dtype="float64")

    def move(self, vector) -> None:
        self.midpoint = self.midpoint + np.array(vector)

    def move_polar_copy(self, d, angle: float) -> Port:
        port = self.copy()
        DEG2RAD = np.pi / 180
        dp = np.array((d * np.cos(DEG2RAD * angle), d * np.sin(DEG2RAD * angle)))
        self.move(dp)
        return port

    def flip(self) -> Port:
        """flips port"""
        port = self.copy()
        port.orientation = (port.orientation + 180) % 360
        return port

    def _copy(self, new_uid: bool = True) -> Port:
        """Keep this case for phidl compatibility"""
        return self.copy(new_uid=new_uid)

    @property
    def endpoints(self):
        """Returns the endpoints of the Port."""
        dxdy = (
            np.array(
                [
                    self.width / 2 * np.cos((self.orientation - 90) * np.pi / 180),
                    self.width / 2 * np.sin((self.orientation - 90) * np.pi / 180),
                ]
            )
            if self.orientation is not None
            else np.array([self.width, self.width])
        )
        left_point = self.midpoint - dxdy
        right_point = self.midpoint + dxdy
        return np.array([left_point, right_point])

    def copy(self, new_uid: bool = True) -> Port:
        new_port = Port(
            name=self.name,
            midpoint=self.midpoint,
            width=self.width,
            orientation=self.orientation,
            parent=self.parent,
            layer=self.layer,
            port_type=self.port_type,
            cross_section=self.cross_section,
            shear_angle=self.shear_angle,
        )
        new_port.info = deepcopy(self.info)
        if not new_uid:
            new_port.uid = self.uid
            Port._next_uid -= 1
        return new_port

    def get_extended_midpoint(self, length: float = 1.0) -> ndarray:
        """Returns an extended midpoint"""
        angle = self.orientation
        c = np.cos(angle)
        s = np.sin(angle)
        return self.midpoint + length * np.array([c, s])

    def snap_to_grid(self, nm: int = 1) -> None:
        self.midpoint = nm * np.round(np.array(self.midpoint) * 1e3 / nm) / 1e3

    def assert_on_grid(self, nm: int = 1) -> None:
        """Ensures ports edges are on grid to avoid snap_to_grid errors."""
        half_width = self.width / 2
        half_width_correct = snap_to_grid(half_width, nm=nm)
        component_name = self.parent.name
        if not np.isclose(half_width, half_width_correct):
            raise PortNotOnGridError(
                f"{component_name}, port = {self.name}, midpoint = {self.midpoint} width = {self.width} will create off-grid points",
                f"you can fix it by changing width to {2*half_width_correct}",
            )

        if self.port_type.startswith("vertical"):
            return

        if self.orientation in [0, 180]:
            x = self.y + self.width / 2
            if not np.isclose(snap_to_grid(x, nm=nm), x):
                raise PortNotOnGridError(
                    f"{self.name} port in {component_name} has an off-grid point {x}",
                    f"you can fix it by moving the point to {snap_to_grid(x, nm=nm)}",
                )
        elif self.orientation in [90, 270]:
            x = self.x + self.width / 2
            if not np.isclose(snap_to_grid(x, nm=nm), x):
                raise PortNotOnGridError(
                    f"{self.name} port in {component_name} has an off-grid point {x}",
                    f"you can fix it by moving the point to {snap_to_grid(x, nm=nm)}",
                )
        else:
            raise PortOrientationError(
                f"{component_name} port {self.name} has invalid orientation"
                f" {self.orientation}"
            )


PortsMap = Dict[str, List[Port]]


def port_array(
    midpoint: Tuple[float, float] = (0.0, 0.0),
    width: float = 0.5,
    orientation: float = 0,
    pitch: Tuple[float, float] = (10.0, 0.0),
    n: int = 2,
    **kwargs,
) -> List[Port]:
    """Returns a list of ports placed in an array

    Args:
        midpoint: center point of the port
        width: port width
        orientation: angle in degrees
        pitch: period of the port array
        n: number of ports in the array

    """
    pitch = np.array(pitch)
    return [
        Port(
            name=str(i),
            width=width,
            midpoint=np.array(midpoint) + i * pitch - (n - 1) / 2 * pitch,
            orientation=orientation,
            **kwargs,
        )
        for i in range(n)
    ]


def read_port_markers(
    component: object, layers: Tuple[Tuple[int, int], ...] = ((1, 10),)
) -> Component:
    """Loads a GDS and returns the extracted ports from layer markers

    Args:
        component: or Component
        layers: Iterable of GDS layers
    """
    return pg.extract(component, layers=layers)


def csv2port(csvpath) -> Dict[str, Port]:
    """Reads ports from a CSV file and returns a Dict"""
    ports = {}
    with open(csvpath, "r") as csvfile:
        rows = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in rows:
            ports[row[0]] = row[1:]

    return ports


def sort_ports_clockwise(ports: Dict[str, Port]) -> Dict[str, Port]:
    """

    .. code::

             3   4
             |___|_
         2 -|      |- 5
            |      |
         1 -|______|- 6
             |   |
             8   7
    """
    port_list = list(ports.values())
    direction_ports: PortsMap = {x: [] for x in ["E", "N", "W", "S"]}

    for p in port_list:
        angle = p.orientation % 360
        if angle <= 45 or angle >= 315:
            direction_ports["E"].append(p)
        elif angle <= 135 and angle >= 45:
            direction_ports["N"].append(p)
        elif angle <= 225 and angle >= 135:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: -p.y)  # sort north to south

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: +p.x)  # sort west to east

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: +p.y)  # sort south to north

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: -p.x)  # sort east to west

    ports = west_ports + north_ports + east_ports + south_ports
    return {port.name: port for port in ports}


def sort_ports_counter_clockwise(ports: Dict[str, Port]) -> Dict[str, Port]:
    """

    .. code::

             4   3
             |___|_
         5 -|      |- 2
            |      |
         6 -|______|- 1
             |   |
             7   8
    """
    port_list = list(ports.values())
    direction_ports: PortsMap = {x: [] for x in ["E", "N", "W", "S"]}

    for p in port_list:
        angle = p.orientation % 360
        if angle <= 45 or angle >= 315:
            direction_ports["E"].append(p)
        elif angle <= 135 and angle >= 45:
            direction_ports["N"].append(p)
        elif angle <= 225 and angle >= 135:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: +p.y)  # sort south to north

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: -p.x)  # sort east to west

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: -p.y)  # sort north to south

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: +p.x)  # sort west to east

    ports = east_ports + north_ports + west_ports + south_ports
    return {port.name: port for port in ports}


def select_ports(
    ports: Dict[str, Port],
    layer: Optional[Tuple[int, int]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    orientation: Optional[int] = None,
    width: Optional[float] = None,
    layers_excluded: Optional[Tuple[Tuple[int, int], ...]] = None,
    port_type: Optional[str] = None,
    clockwise: bool = True,
) -> Dict[str, Port]:
    """Returns a dict of ports from a dict of ports

    Args:
        ports: Dict[str, Port] a port dict {port name: port}
        layer: port GDS layer
        prefix: port name prefix
        suffix: port name suffix
        orientation: in degrees
        width: port width
        layers_excluded: List of layers to exclude
        port_type: optical, electrical, ...
        clockwise: if True, sort ports clockwise, False: counter-clockwise

    Returns:
        Dictionary containing only the ports with the wanted type(s)
        {port name: port}
    """

    from gdsfactory.component import Component, ComponentReference

    # Make it accept Component or ComponentReference
    if isinstance(ports, (Component, ComponentReference)):
        ports = ports.ports

    if layer:
        ports = {p_name: p for p_name, p in ports.items() if p.layer == layer}
    if prefix:
        ports = {
            p_name: p for p_name, p in ports.items() if str(p_name).startswith(prefix)
        }
    if suffix:
        ports = {
            p_name: p for p_name, p in ports.items() if str(p_name).endswith(suffix)
        }
    if orientation is not None:
        ports = {
            p_name: p for p_name, p in ports.items() if p.orientation == orientation
        }

    if layers_excluded:
        ports = {
            p_name: p for p_name, p in ports.items() if p.layer not in layers_excluded
        }
    if width:
        ports = {p_name: p for p_name, p in ports.items() if p.width == width}
    if port_type:
        ports = {p_name: p for p_name, p in ports.items() if p.port_type == port_type}

    if clockwise:
        ports = sort_ports_clockwise(ports)
    else:
        ports = sort_ports_counter_clockwise(ports)
    return ports


select_ports_optical = partial(select_ports, port_type="optical")
select_ports_electrical = partial(select_ports, port_type="electrical")


def select_ports_list(**kwargs) -> List[Port]:
    return list(select_ports(**kwargs).values())


def flipped(port: Port) -> Port:
    _port = port.copy()
    _port.orientation = (_port.orientation + 180) % 360
    return _port


def move_copy(port, x=0, y=0) -> Port:
    _port = port.copy()
    _port.midpoint += (x, y)
    return _port


def get_ports_facing(ports: List[Port], direction: str = "W") -> List[Port]:
    from gdsfactory.component import Component, ComponentReference

    valid_directions = ["E", "N", "W", "S"]

    if direction not in valid_directions:
        raise PortOrientationError(f"{direction} must be in {valid_directions} ")

    if isinstance(ports, dict):
        ports = list(ports.values())
    elif isinstance(ports, (Component, ComponentReference)):
        ports = list(ports.ports.values())

    direction_ports: Dict[str, List[Port]] = {x: [] for x in ["E", "N", "W", "S"]}

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


def deco_rename_ports(component_factory: Callable) -> Callable:
    @functools.wraps(component_factory)
    def auto_named_component_factory(*args, **kwargs):
        component = component_factory(*args, **kwargs)
        auto_rename_ports(component)
        return component

    return auto_named_component_factory


def _rename_ports_facing_side(
    direction_ports: Dict[str, List[Port]], prefix: str = ""
) -> None:
    """Renames ports clockwise"""
    for direction, list_ports in list(direction_ports.items()):

        if direction in ["E", "W"]:
            # first sort along x then y
            list_ports.sort(key=lambda p: p.x)
            list_ports.sort(key=lambda p: p.y)

        if direction in ["S", "N"]:
            # first sort along y then x
            list_ports.sort(key=lambda p: p.y)
            list_ports.sort(key=lambda p: p.x)

        for i, p in enumerate(list_ports):
            p.name = prefix + direction + str(i)


def _rename_ports_facing_side_ccw(
    direction_ports: Dict[str, List[Port]], prefix: str = ""
) -> None:
    """Renames ports counter-clockwise"""
    for direction, list_ports in list(direction_ports.items()):

        if direction in ["E", "W"]:
            # first sort along x then y
            list_ports.sort(key=lambda p: -p.x)
            list_ports.sort(key=lambda p: -p.y)

        if direction in ["S", "N"]:
            # first sort along y then x
            list_ports.sort(key=lambda p: -p.y)
            list_ports.sort(key=lambda p: -p.x)

        for i, p in enumerate(list_ports):
            p.name = prefix + direction + str(i)


def _rename_ports_counter_clockwise(direction_ports, prefix="") -> None:
    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: +p.y)  # sort south to north

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: -p.x)  # sort east to west

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: -p.y)  # sort north to south

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: +p.x)  # sort west to east

    ports = east_ports + north_ports + west_ports + south_ports

    for i, p in enumerate(ports):
        p.name = f"{prefix}{i+1}" if prefix else i + 1


def _rename_ports_clockwise(direction_ports: PortsMap, prefix: str = "") -> None:
    """Rename ports in the clockwise direction starting from the bottom left (west) corner."""
    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: -p.y)  # sort north to south

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: +p.x)  # sort west to east

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: +p.y)  # sort south to north

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: -p.x)  # sort east to west
    # south_ports.sort(key=lambda p: p.y)  #  south first

    ports = west_ports + north_ports + east_ports + south_ports

    for i, p in enumerate(ports):
        p.name = f"{prefix}{i+1}" if prefix else i + 1


def _rename_ports_clockwise_top_right(
    direction_ports: PortsMap, prefix: str = ""
) -> None:
    """Rename ports in the clockwise direction starting from the top right corner."""
    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: -p.y)  # sort north to south

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: +p.x)  # sort west to east

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: +p.y)  # sort south to north

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: -p.x)  # sort east to west

    ports = east_ports + south_ports + west_ports + north_ports

    for i, p in enumerate(ports):
        p.name = f"{prefix}{i+1}" if prefix else i + 1


def rename_ports_by_orientation(
    component: Component,
    layers_excluded: Tuple[Tuple[int, int], ...] = None,
    select_ports: Optional[Callable[..., List[Port]]] = None,
    function=_rename_ports_facing_side,
    prefix: str = "o",
) -> Component:
    """Returns Component with port names based on port orientation (E, N, W, S)

    Args:
        component:
        layers_excluded:
        select_ports:
        function: to rename ports
        prefix: to add on each port name

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
    direction_ports: PortsMap = {x: [] for x in ["E", "N", "W", "S"]}

    ports = component.ports
    ports = select_ports(ports) if select_ports else ports

    ports_on_layer = [p for p in ports.values() if p.layer not in layers_excluded]

    for p in ports_on_layer:
        # Make sure we can backtrack the parent component from the port
        p.parent = component

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
    component.ports = {p.name: p for p in component.ports.values()}
    return component


def auto_rename_ports(
    component: Component,
    function=_rename_ports_clockwise,
    select_ports_optical=select_ports_optical,
    select_ports_electrical=select_ports_electrical,
    prefix_optical: str = "o",
    prefix_electrical: str = "e",
    **kwargs,
) -> Component:
    """Adds prefix for optical and electical.

    Args:
        component:
        function: to rename ports
        select_ports_optical:
        select_ports_electrical:
        prefix_optical:
        prefix_electrical:
    """
    rename_ports_by_orientation(
        component=component,
        select_ports=select_ports_optical,
        prefix=prefix_optical,
        function=function,
        **kwargs,
    )
    rename_ports_by_orientation(
        component=component,
        select_ports=select_ports_electrical,
        prefix=prefix_electrical,
        function=function,
        **kwargs,
    )
    return component


auto_rename_ports_counter_clockwise = partial(
    auto_rename_ports, function=_rename_ports_counter_clockwise
)
auto_rename_ports_orientation = partial(
    auto_rename_ports, function=_rename_ports_facing_side
)


def map_ports_layer_to_orientation(
    ports: Dict[str, Port], function=_rename_ports_facing_side
) -> Dict[str, str]:
    """Returns component or reference port mapping

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """

    m = {}
    direction_ports: PortsMap = {x: [] for x in ["E", "N", "W", "S"]}
    layers = {port.layer for port in ports.values()}

    for layer in layers:
        ports_on_layer = [p.copy() for p in ports.values() if p.layer == layer]

        for p in ports_on_layer:
            p.name_original = p.name
            if p.orientation:
                angle = p.orientation % 360
                if angle <= 45 or angle >= 315:
                    direction_ports["E"].append(p)
                elif angle <= 135 and angle >= 45:
                    direction_ports["N"].append(p)
                elif angle <= 225 and angle >= 135:
                    direction_ports["W"].append(p)
                else:
                    direction_ports["S"].append(p)
        function(direction_ports, prefix=f"{layer[0]}_{layer[1]}_")
        m.update({p.name: p.name_original for p in ports_on_layer})
    return m


def map_ports_to_orientation_cw(
    ports: Dict[str, Port], function=_rename_ports_facing_side, **kwargs
) -> Dict[str, str]:
    """Returns component or reference port mapping clockwise
    **kwargs

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """

    direction_ports: PortsMap = {x: [] for x in ["E", "N", "W", "S"]}

    ports = select_ports(ports, **kwargs)
    ports_on_layer = [p.copy() for p in ports.values()]

    for p in ports_on_layer:
        p.name_original = p.name
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
    return dict({p.name: p.name_original for p in ports_on_layer})


map_ports_to_orientation_ccw = partial(
    map_ports_to_orientation_cw, function=_rename_ports_facing_side_ccw
)


def auto_rename_ports_layer_orientation(
    component: Component,
    function=_rename_ports_facing_side,
    prefix: str = "",
) -> None:
    """Renames port names with layer_orientation  (1_0_W0)
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
    new_ports = {}
    ports = component.ports
    direction_ports: PortsMap = {x: [] for x in ["E", "N", "W", "S"]}
    layers = {port.layer for port in ports.values()}

    for layer in layers:
        ports_on_layer = [p for p in ports.values() if p.layer == layer]

        for p in ports_on_layer:
            p.name_original = p.name
            angle = p.orientation % 360
            if angle <= 45 or angle >= 315:
                direction_ports["E"].append(p)
            elif angle <= 135 and angle >= 45:
                direction_ports["N"].append(p)
            elif angle <= 225 and angle >= 135:
                direction_ports["W"].append(p)
            else:
                direction_ports["S"].append(p)

        function(direction_ports, prefix=f"{layer[0]}_{layer[1]}_")
        new_ports |= {p.name: p for p in ports_on_layer}

    component.ports = new_ports


__all__ = [
    "Port",
    "port_array",
    "read_port_markers",
    "csv2port",
    "select_ports",
    "select_ports_list",
    "flipped",
    "move_copy",
    "get_ports_facing",
    "deco_rename_ports",
    "rename_ports_by_orientation",
    "auto_rename_ports",
    "auto_rename_ports_counter_clockwise",
    "auto_rename_ports_orientation",
    "map_ports_layer_to_orientation",
]

if __name__ == "__main__":

    import gdsfactory as gf

    # c = gf.Component()

    c = gf.components.straight_heater_metal()
    # c.auto_rename_ports()
    # auto_rename_ports_layer_orientation(c)
    # m = map_ports_layer_to_orientation(c.ports)
    # pprint(m)
    # c.show()
    # print(p0)
    p0 = c.get_ports_list(orientation=0, clockwise=False)[0]
    print(p0)
    print(type(p0.to_dict()["midpoint"][0]))
