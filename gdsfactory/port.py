"""
For port naming we follow the IPKISS standard

.. code::

         N0  N1
         |___|_
    W1 -|      |- E1
        |      |
    W0 -|______|- E0
         |   |
        S0   S1

"""
import csv
import functools
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import phidl.geometry as pg
from numpy import ndarray
from phidl.device_layout import Device
from phidl.device_layout import Port as PortPhidl

from gdsfactory.snap import snap_to_grid

valid_port_types = ["optical", "rf", "dc", "heater", "vertical_te", "vertical_tm"]


class PortNotOnGridError(ValueError):
    pass


class PortTypeError(ValueError):
    pass


class PortOrientationError(ValueError):
    pass


class Port(PortPhidl):
    """Ports are useful to connect Components with each other.
    Extends phidl port with layer and port_type (optical, dc, rf)

    Args:
        name: we name ports according to orientation (S0, S1, W0, W1, N0 ...)
        midpoint: (0, 0)
        width: of the port
        orientation: in degrees (0: east, 90: north, 180: west, 270: south)
        parent: parent component (component to which this port belong to)
        layer: (1, 0)
        port_type: optical, dc, rf, detector, superconducting, trench


    For port naming we follow the W,E,S,N prefix (west, east, south, north)

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """

    _next_uid = 0

    def __init__(
        self,
        name: Optional[str] = None,
        midpoint: Tuple[float, float] = (0.0, 0.0),
        width: float = 0.5,
        orientation: int = 0,
        parent: Optional[object] = None,
        layer: Tuple[int, int] = (1, 0),
        port_type: str = "optical",
    ) -> None:
        self.name = name
        self.midpoint = np.array(midpoint, dtype="float64")
        self.width = width
        self.orientation = np.mod(orientation, 360)
        self.parent = parent
        self.info = {}
        self.uid = Port._next_uid
        self.layer = layer
        self.port_type = port_type

        if self.width < 0:
            raise ValueError("[PHIDL] Port creation error: width must be >=0")
        Port._next_uid += 1

    def __repr__(self) -> str:
        return (
            "Port (name {}, midpoint {}, width {}, orientation {}, layer {},"
            " port_type {})".format(
                self.name,
                self.midpoint,
                self.width,
                self.orientation,
                self.layer,
                self.port_type,
            )
        )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """For pydantic assumes Port is valid if:
        - has a name
        - has po
        """
        assert v.name, f"Port has no name, got `{v.name}`"
        # assert v.assert_on_grid(), f"port.midpoint = {v.midpoint} has off-grid points"
        return v

    @property
    def settings(self):
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
    def angle(self, a):
        self.orientation = a

    @property
    def position(self):
        return self.midpoint

    @position.setter
    def position(self, p):
        self.midpoint = np.array(p, dtype="float64")

    def move(self, vector):
        self.midpoint = self.midpoint + np.array(vector)

    def move_polar_copy(self, d, angle) -> PortPhidl:
        port = self._copy()
        DEG2RAD = np.pi / 180
        dp = np.array((d * np.cos(DEG2RAD * angle), d * np.sin(DEG2RAD * angle)))
        self.move(dp)
        return port

    def flip(self):
        """flips port"""
        port = self._copy()
        port.angle = (port.angle + 180) % 360
        return port

    def _copy(self, new_uid: bool = True) -> PortPhidl:
        new_port = Port(
            name=self.name,
            midpoint=self.midpoint,
            width=self.width,
            orientation=self.orientation,
            parent=self.parent,
            layer=self.layer,
            port_type=self.port_type,
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
        component_name = self.parent.get_property("name")
        if not np.isclose(half_width, half_width_correct):
            raise PortNotOnGridError(
                f"{component_name}, port = {self.name}, midpoint = {self.midpoint} width = {self.width} will create off-grid points",
                f"you can fix it by changing width to {2*half_width_correct}",
            )

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


def port_array(
    midpoint: Tuple[int, int] = (0, 0),
    width: float = 0.5,
    orientation: int = 0,
    pitch: Tuple[int, int] = (10, 0),
    n: int = 2,
    **kwargs,
) -> List[Port]:
    """returns a list of ports placed in an array

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
            midpoint=np.array(midpoint) + i * pitch - (n - 1) / 2 * pitch,
            orientation=orientation,
            **kwargs,
        )
        for i in range(n)
    ]


def read_port_markers(
    component: object, layers: Iterable[Tuple[int, int]] = ((1, 10),)
) -> Device:
    """loads a GDS and returns the extracted ports from layer markers

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


def is_electrical_port(port: Port) -> bool:
    return port.port_type in ["dc", "rf"]


def select_ports(
    ports: Dict[str, Port],
    port_type: Optional[str] = None,
    layer: Optional[Tuple[int, int]] = None,
    prefix: Optional[str] = None,
    orientation: Optional[int] = None,
) -> Dict[str, Port]:
    """
    Args:
        ports: Dict[str, Port] a port dictionnary {port name: port} (as returned by Component.ports)
        port_type: a port type string
        layer: GDS layer
        prefix: a prefix

    Returns:
        Dictionnary containing only the ports with the wanted type(s)
        {port name: port}
    """

    from gdsfactory.component import Component, ComponentReference

    # Make it accept Component or ComponentReference
    if isinstance(ports, Component) or isinstance(ports, ComponentReference):
        ports = ports.ports

    if port_type:
        if port_type not in valid_port_types:
            raise PortTypeError(
                f"Invalid port_type={port_type} not in {valid_port_types}"
            )
        ports = {p_name: p for p_name, p in ports.items() if p.port_type == port_type}

    if layer:
        ports = {p_name: p for p_name, p in ports.items() if p.layer == layer}
    if prefix:
        ports = {p_name: p for p_name, p in ports.items() if p_name.startswith(prefix)}
    if orientation is not None:
        ports = {
            p_name: p for p_name, p in ports.items() if p.orientation == orientation
        }

    return ports


def select_ports_list(**kwargs) -> List[Port]:
    return list(select_ports(**kwargs).values())


def select_optical_ports(
    ports: Dict[str, Port],
    port_type: str = "optical",
    prefix: Optional[str] = None,
    layer: Optional[Tuple[int, int]] = None,
    orientation: Optional[int] = None,
) -> Dict[str, Port]:
    return select_ports(
        ports, port_type=port_type, prefix=prefix, layer=layer, orientation=orientation
    )


def select_electrical_ports(
    ports: Dict[str, Port],
    port_type: str = "dc",
    prefix: Optional[str] = None,
    layer: Optional[Tuple[int, int]] = None,
    orientation: Optional[int] = None,
) -> Dict[str, Port]:
    return select_ports(
        ports, port_type=port_type, prefix=prefix, layer=layer, orientation=orientation
    )


def flipped(port: Port) -> Port:
    _port = port._copy()
    _port.orientation = (_port.orientation + 180) % 360
    return _port


def move_copy(port, x=0, y=0):
    _port = port._copy()
    _port.midpoint += (x, y)
    return _port


def get_ports_facing(ports: List[Port], direction: str = "W") -> List[Port]:
    from gdsfactory.component import Component, ComponentReference

    valid_directions = ["E", "N", "W", "S"]

    if direction not in valid_directions:
        raise PortOrientationError(f"{direction} must be in {valid_directions} ")

    if isinstance(ports, dict):
        ports = list(ports.values())
    elif isinstance(ports, Component) or isinstance(ports, ComponentReference):
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


def get_non_optical_ports(ports) -> List[Port]:
    from gdsfactory.component import Component, ComponentReference

    if isinstance(ports, dict):
        ports = list(ports.values())
    elif isinstance(ports, Component) or isinstance(ports, ComponentReference):
        ports = list(ports.ports.values())
    res = [p for p in ports if p.port_type not in ["optical"]]
    return res


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
            lbl = prefix + direction + str(i)
            p.name = lbl


def rename_ports_by_orientation(
    component: Device, layers_excluded: Iterable[Tuple[int, int]] = None
) -> Device:
    """Returns Component with port names based on port orientation (E, N, W, S)

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
    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}
    ports_on_process = [
        p for p in component.ports.values() if p.layer not in layers_excluded
    ]

    for p in ports_on_process:
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

    _rename_ports_facing_side(direction_ports)
    component.ports = {p.name: p for p in component.ports.values()}
    return component


def auto_rename_ports(component: Device) -> Device:
    """Returns Component with port names based on port orientation (E, N, W, S)

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """

    def _counter_clockwise(_direction_ports, prefix=""):

        east_ports = _direction_ports["E"]
        east_ports.sort(key=lambda p: p.y)  # sort south to north

        north_ports = _direction_ports["N"]
        north_ports.sort(key=lambda p: -p.x)  # sort east to west

        west_ports = _direction_ports["W"]
        west_ports.sort(key=lambda p: -p.y)  # sort north to south

        south_ports = _direction_ports["S"]
        south_ports.sort(key=lambda p: p.x)  # sort west to east

        ports = east_ports + north_ports + west_ports + south_ports

        for i, p in enumerate(ports):
            p.name = "{}{}".format(prefix, i)

    type_to_ports_naming_functions = {
        "optical": _rename_ports_facing_side,
        "heater": lambda _d: _counter_clockwise(_d, "H_"),
        "dc": lambda _d: _counter_clockwise(_d, "DC_"),
        "superconducting": lambda _d: _counter_clockwise(_d, "SC_"),
        "vertical_te": lambda _d: _counter_clockwise(_d, "vertical_te_"),
        "vertical_tm": lambda _d: _counter_clockwise(_d, "vertical_tm_"),
    }

    type_to_ports = {}

    for p in component.ports.values():
        if p.port_type not in type_to_ports:
            type_to_ports[p.port_type] = []
        type_to_ports[p.port_type] += [p]

    for port_type, port_group in type_to_ports.items():
        if port_type in type_to_ports_naming_functions:
            _func_name_ports = type_to_ports_naming_functions[port_type]
        else:
            raise PortTypeError(
                f"Invalid port_type='{port_type}' in component {component.name}, port {p.name}",
                f"valid types = {list(type_to_ports_naming_functions.keys())}",
            )

        direction_ports = {x: [] for x in ["E", "N", "W", "S"]}
        for p in port_group:
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

        _func_name_ports(direction_ports)

    # Set the port dictionnary with the new names
    component.ports = {p.name: p for p in component.ports.values()}
    return component


__all__ = [
    "Port",
    "port_array",
    "read_port_markers",
    "csv2port",
    "is_electrical_port",
    "select_ports",
    "select_ports_list",
    "select_optical_ports",
    "select_electrical_ports",
    "flipped",
    "move_copy",
    "get_ports_facing",
    "get_non_optical_ports",
    "deco_rename_ports",
    "rename_ports_by_orientation",
    "auto_rename_ports",
]

if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    wg = c << gf.components.straight()
    c.show()
