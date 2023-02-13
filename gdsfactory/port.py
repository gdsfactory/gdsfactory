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
import typing
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from omegaconf import OmegaConf

from gdsfactory.component_layout import _rotate_points
from gdsfactory.cross_section import CrossSection
from gdsfactory.serialization import clean_value_json
from gdsfactory.snap import snap_to_grid

if typing.TYPE_CHECKING:
    from gdsfactory.component import Component

Layer = Tuple[int, int]
Layers = Tuple[Layer, ...]
LayerSpec = Union[Layer, int, str, None]
LayerSpecs = Tuple[LayerSpec, ...]
Float2 = Tuple[float, float]


class PortNotOnGridError(ValueError):
    pass


class PortTypeError(ValueError):
    pass


class PortOrientationError(ValueError):
    pass


class Port:
    """Ports are useful to connect Components with each other.

    Args:
        name: we name ports clock-wise starting from bottom left.
        center: (x, y) port center coordinate.
        width: of the port in um.
        orientation: in degrees (0: east, 90: north, 180: west, 270: south).
        parent: parent component (component to which this port belong to).
        layer: layer tuple.
        port_type: str (optical, electrical, vertical_te, vertical_tm).
        parent: Component that port belongs to.
        cross_section: cross_section spec.
        shear_angle: an optional angle to shear port face in degrees.
    """

    def __init__(
        self,
        name: str,
        orientation: Optional[float],
        center: Tuple[float, float],
        width: Optional[float] = None,
        layer: Optional[Tuple[int, int]] = None,
        port_type: str = "optical",
        parent: Optional[Component] = None,
        cross_section: Optional[CrossSection] = None,
        shear_angle: Optional[float] = None,
    ) -> None:
        """Initializes Port object."""
        self.name = name
        self.center = np.array(center, dtype="float64")
        self.orientation = np.mod(orientation, 360) if orientation else orientation
        self.parent = parent
        self.info: Dict[str, Any] = {}
        self.port_type = port_type
        self.cross_section = cross_section
        self.shear_angle = shear_angle

        if cross_section is None and layer is None:
            raise ValueError("You need Port to define cross_section or layer")

        if cross_section is None and width is None:
            raise ValueError("You need Port to define cross_section or width")

        if cross_section and not isinstance(cross_section, CrossSection):
            raise ValueError(
                f"cross_section = {cross_section} is not a valid CrossSection."
            )

        if layer is None:
            layer = cross_section.layer

        if width is None:
            width = cross_section.width

        self.layer = layer
        self.width = width

        if self.width < 0:
            raise ValueError(f"Port width must be >=0. Got {self.width}")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "width": self.width,
            "center": tuple(np.round(self.center, 3)),
            "orientation": int(self.orientation)
            if self.orientation
            else self.orientation,
            "layer": self.layer,
            "port_type": self.port_type,
        }
        if self.shear_angle:
            d["shear_angle"] = self.shear_angle
        return clean_value_json(d)

    def to_yaml(self) -> str:
        d = {
            "name": self.name,
            "width": float(self.width),
            "center": [float(self.center[0]), float(self.center[1])],
            "orientation": float(self.orientation)
            if self.orientation
            else float(self.orientation),
            "layer": self.layer,
            "port_type": self.port_type,
        }
        d = OmegaConf.create(d)
        return OmegaConf.to_yaml(d)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        s = f"Port (name {self.name}, center {self.center}, width {self.width}, orientation {self.orientation}, layer {self.layer}, port_type {self.port_type})"
        s += f" shear_angle {self.shear_angle}" if self.shear_angle else ""
        return s

    @classmethod
    def __get_validators__(cls):
        """Get validators."""
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """For pydantic assumes Port is valid if has a name and a valid type."""
        assert isinstance(v, Port), f"TypeError, Got {type(v)}, expecting Port"
        assert v.name, f"Port has no name, got `{v.name}`"
        # assert v.assert_on_grid(), f"port.center = {v.center} has off-grid points"
        return v

    @property
    def settings(self):
        """TODO!

        delete this. Use to_dict instead

        """
        return {
            "name": self.name,
            "center": self.center,
            "width": self.width,
            "orientation": self.orientation,
            "layer": self.layer,
            "port_type": self.port_type,
        }

    def move(self, vector) -> None:
        self.center = self.center + np.array(vector)

    def move_polar_copy(self, d: float, angle: float) -> Port:
        """Returns a copy of the port with a distance (d) in um and angle (deg)."""
        port = self.copy()
        DEG2RAD = np.pi / 180
        dp = np.array((d * np.cos(DEG2RAD * angle), d * np.sin(DEG2RAD * angle)))
        port.move(dp)
        return port

    def flip(self, **kwargs) -> Port:
        """Flips port."""
        port = self.copy(**kwargs)
        if port.orientation is None:
            raise ValueError(f"port {self.name!r} has None orientation")
        port.orientation = (port.orientation + 180) % 360
        return port

    def _copy(self) -> Port:
        """Keep this case for phidl compatibility."""
        return self.copy()

    @property
    def endpoints(self) -> None:
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
        left_point = self.center - dxdy
        right_point = self.center + dxdy
        return np.array([left_point, right_point])

    @endpoints.setter
    def endpoints(self, points: Float2) -> None:
        """Sets the endpoints of a Port."""
        p1, p2 = np.array(points[0]), np.array(points[1])
        self.center = (p1 + p2) / 2
        dx, dy = p2 - p1
        self.orientation = np.arctan2(dx, -dy) * 180 / np.pi
        self.width = np.sqrt(dx**2 + dy**2)

    @property
    def normal(self) -> ndarray:
        """Returns a vector normal to the Port."""
        dx = np.cos((self.orientation) * np.pi / 180)
        dy = np.sin((self.orientation) * np.pi / 180)
        return np.array([self.center, self.center + np.array([dx, dy])])

    @property
    def x(self) -> float:
        """Returns the x-coordinate of the Port center."""
        return self.center[0]

    @property
    def y(self) -> float:
        """Returns the y-coordinate of the Port center."""
        return self.center[1]

    def rotate(self, angle: float = 45, center: Optional[Float2] = None) -> Port:
        """Rotates a Port around the specified center point, if no centerpoint \
        specified will rotate around (0,0).

        Args:
            angle: Angle to rotate the Port in degrees.
            center: array-like[2] or None center of the Port.

        """
        self.orientation = np.mod(self.orientation + angle, 360)
        if center is None:
            center = self.center
        self.center = _rotate_points(self.center, angle=angle, center=center)
        return self

    def copy(self, name: Optional[str] = None) -> Port:
        """Returns a copy of the port.

        Args:
            name: optional new name.

        """
        new_port = Port(
            name=name or self.name,
            center=self.center,
            width=self.width,
            orientation=self.orientation,
            parent=self.parent,
            layer=self.layer,
            port_type=self.port_type,
            cross_section=self.cross_section,
            shear_angle=self.shear_angle,
        )
        new_port.info = self.info
        return new_port

    def get_extended_center(self, length: float = 1.0) -> ndarray:
        """Returns an extended port center."""
        angle = self.orientation
        c = np.cos(angle)
        s = np.sin(angle)
        return self.center + length * np.array([c, s])

    def snap_to_grid(self, nm: int = 1) -> None:
        """Snap port center to nm grid."""
        self.center = nm * np.round(np.array(self.center) * 1e3 / nm) / 1e3

    def assert_on_grid(self, nm: int = 1) -> None:
        """Ensures ports edges are on grid to avoid snap_to_grid errors."""
        half_width = self.width / 2
        half_width_correct = snap_to_grid(half_width, nm=nm)
        component_name = self.parent.name
        if not np.isclose(half_width, half_width_correct):
            raise PortNotOnGridError(
                f"{component_name}, port = {self.name!r}, center = {self.center} "
                f"width = {self.width} will create off-grid points",
                f"you can fix it by changing width to {2*half_width_correct}",
            )

        if self.port_type.startswith("vertical"):
            return

        if self.orientation is None:
            return

        elif self.orientation in [0, 180]:
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
    center: Tuple[float, float] = (0.0, 0.0),
    width: float = 0.5,
    orientation: float = 0,
    pitch: Tuple[float, float] = (10.0, 0.0),
    n: int = 2,
    **kwargs,
) -> List[Port]:
    """Returns a list of ports placed in an array.

    Args:
        center: center point of the port.
        width: port width.
        orientation: angle in degrees.
        pitch: period of the port array.
        n: number of ports in the array.

    """
    pitch = np.array(pitch)
    return [
        Port(
            name=str(i),
            width=width,
            center=np.array(center) + i * pitch - (n - 1) / 2 * pitch,
            orientation=orientation,
            **kwargs,
        )
        for i in range(n)
    ]


def read_port_markers(component: object, layers: LayerSpecs = ("PORT",)) -> Component:
    """Returns extracted polygons from component layers.

    Args:
        component: Component to extract markers.
        layers: GDS layer specs.

    """
    from gdsfactory.pdk import get_layer

    layers = [get_layer(layer) for layer in layers]
    return component.extract(layers=layers)


def csv2port(csvpath) -> Dict[str, Port]:
    """Reads ports from a CSV file and returns a Dict."""
    ports = {}
    with open(csvpath) as csvfile:
        rows = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in rows:
            ports[row[0]] = row[1:]

    return ports


def sort_ports_clockwise(ports: Dict[str, Port]) -> Dict[str, Port]:
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
    port_list = list(ports.values())
    direction_ports: PortsMap = {x: [] for x in ["E", "N", "W", "S"]}

    for p in port_list:
        angle = p.orientation % 360 if p.orientation is not None else 0
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
    port_list = list(ports.values())
    direction_ports: PortsMap = {x: [] for x in ["E", "N", "W", "S"]}

    for p in port_list:
        angle = p.orientation % 360 if p.orientation is not None else 0
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
    names: Optional[List[str]] = None,
    clockwise: bool = True,
) -> Dict[str, Port]:
    """Returns a dict of ports from a dict of ports.

    Args:
        ports: Dict[str, Port] a port dict {port name: port}.
        layer: select ports with port GDS layer.
        prefix: select ports with port name prefix.
        suffix: select ports with port name suffix.
        orientation: select ports with orientation in degrees.
        width: select ports with port width.
        layers_excluded: List of layers to exclude.
        port_type: select ports with port type (optical, electrical, vertical_te).
        clockwise: if True, sort ports clockwise, False: counter-clockwise.

    Returns:
        Dict containing the selected ports {port name: port}.

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
    if names:
        ports = {p_name: p for p_name, p in ports.items() if p_name in names}

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
    if port.orientation is None:
        raise ValueError(f"port {port.name!r} has None orientation")
    _port = port.copy()
    _port.orientation = (_port.orientation + 180) % 360
    return _port


def move_copy(port, x=0, y=0) -> Port:
    _port = port.copy()
    _port.center += (x, y)
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
        angle = p.orientation % 360 if p.orientation is not None else 0
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
    """Renames ports clockwise."""
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
    """Renames ports counter-clockwise."""
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
    """Rename ports in the clockwise direction starting from the bottom left \
    (west) corner."""
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
    """Rename ports in the clockwise direction starting from the top right \
    corner."""
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
    layers_excluded: LayerSpec = None,
    select_ports: Optional[Callable[..., List[Port]]] = None,
    function=_rename_ports_facing_side,
    prefix: str = "o",
) -> Component:
    """Returns Component with port names based on port orientation (E, N, W, S).

    Args:
        component: to rename ports.
        layers_excluded: to exclude.
        select_ports: function to select_ports.
        function: to rename ports.
        prefix: to add on each port name.

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

        if p.orientation is not None:
            angle = p.orientation % 360
            if angle <= 45 or angle >= 315:
                direction_ports["E"].append(p)
            elif angle <= 135 and angle >= 45:
                direction_ports["N"].append(p)
            elif angle <= 225 and angle >= 135:
                direction_ports["W"].append(p)
            else:
                direction_ports["S"].append(p)
        else:
            direction_ports["S"].append(p)

    function(direction_ports, prefix=prefix)
    component.ports = {p.name: p for p in component.ports.values()}
    return component


def auto_rename_ports(
    component: Component,
    function=_rename_ports_clockwise,
    select_ports_optical: Optional[Callable] = select_ports_optical,
    select_ports_electrical: Optional[Callable] = select_ports_electrical,
    prefix_optical: str = "o",
    prefix_electrical: str = "e",
    **kwargs,
) -> Component:
    """Adds prefix for optical and electrical.

    Args:
        component: to auto_rename_ports.
        function: to rename ports.
        select_ports_optical: to select optical ports.
        select_ports_electrical: to select electrical ports.
        prefix_optical: prefix of optical ports.
        prefix_electrical: prefix of electrical ports.

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

auto_rename_ports_electrical = partial(auto_rename_ports, select_ports_optical=None)


def map_ports_layer_to_orientation(
    ports: Dict[str, Port], function=_rename_ports_facing_side
) -> Dict[str, str]:
    """Returns component or reference port mapping.

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

    c = gf.Component()
    cross_section = gf.cross_section.strip
    c.add_port(
        "o1",
        center=(0, 0),
        orientation=0,
        port_type="optical",
        cross_section=cross_section,
    )

    # c = gf.components.straight_heater_metal()
    # c.auto_rename_ports()
    # auto_rename_ports_layer_orientation(c)
    # m = map_ports_layer_to_orientation(c.ports)
    # pprint(m)
    # c.show(show_ports=True)
    # print(p0)
    # p0 = c.get_ports_list(orientation=0, clockwise=False)[0]
    # print(p0)
    # print(type(p0.to_dict()["center"][0]))
    # p = Port("o1", orientation=0, center=(9, 0), layer=(1, 0), cross_section=)
