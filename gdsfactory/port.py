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
import csv
import functools
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import phidl.geometry as pg
from numpy import ndarray
from phidl.device_layout import Device
from phidl.device_layout import Port as PortPhidl

from gdsfactory.snap import snap_to_grid

Layer = Tuple[int, int]


class PortNotOnGridError(ValueError):
    pass


class PortTypeError(ValueError):
    pass


class PortOrientationError(ValueError):
    pass


class Port(PortPhidl):
    """Ports are useful to connect Components with each other.
    Extends phidl port with layer

    Args:
        name: we name ports according to orientation starting from bottom, left
        midpoint: (0, 0)
        width: of the port
        orientation: in degrees (0: east, 90: north, 180: west, 270: south)
        parent: parent component (component to which this port belong to)
        layer: (1, 0)
        port_type: str (optical, electrical, vertical_te, vertical_tm)

    """

    _next_uid = 0

    def __init__(
        self,
        name: Optional[str] = None,
        midpoint: Tuple[float, float] = (0.0, 0.0),
        width: float = 0.5,
        orientation: int = 0,
        layer: Tuple[int, int] = (1, 0),
        port_type: str = "optical",
        parent: Optional[object] = None,
        cross_section: Optional[object] = None,
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
        self.cross_section = cross_section

        if self.width < 0:
            raise ValueError("[PHIDL] Port creation error: width must be >=0")
        Port._next_uid += 1

    def __repr__(self) -> str:
        return f"Port (name {self.name}, midpoint {self.midpoint}, width {self.width}, orientation {self.orientation}, layer {self.layer}, port_type {self.port_type})"

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """For pydantic assumes Port is valid if:

        - has a name
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
            cross_section=self.cross_section,
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


def port_array(
    midpoint: Tuple[int, int] = (0, 0),
    width: float = 0.5,
    orientation: int = 0,
    pitch: Tuple[int, int] = (10, 0),
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
            midpoint=np.array(midpoint) + i * pitch - (n - 1) / 2 * pitch,
            orientation=orientation,
            **kwargs,
        )
        for i in range(n)
    ]


def read_port_markers(
    component: object, layers: Tuple[Tuple[int, int], ...] = ((1, 10),)
) -> Device:
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
    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}

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
    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}

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
    orientation: Optional[int] = None,
    width: Optional[float] = None,
    layers_excluded: Optional[Tuple[Tuple[int, int], ...]] = None,
    port_type: Optional[str] = None,
    clockwise: bool = True,
) -> Dict[str, Port]:
    """Returns a dict of ports from a dict of ports

    Args:
        ports: Dict[str, Port] a port dict {port name: port}
        layer: GDS layer
        prefix: with in port name
        orientation:
        width:
        layers_excluded: List of layers to exclude
        clockwise: if True, sort ports clockwise, False: counter-clockwise

    Returns:
        Dictionary containing only the ports with the wanted type(s)
        {port name: port}
    """

    from gdsfactory.component import Component, ComponentReference

    # Make it accept Component or ComponentReference
    if isinstance(ports, Component) or isinstance(ports, ComponentReference):
        ports = ports.ports

    if layer:
        ports = {p_name: p for p_name, p in ports.items() if p.layer == layer}
    if prefix:
        ports = {
            p_name: p for p_name, p in ports.items() if str(p_name).startswith(prefix)
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


def _rename_ports_counter_clockwise(direction_ports, prefix=""):
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


def _rename_ports_clockwise(direction_ports, prefix: str = ""):
    east_ports = direction_ports["E"]
    east_ports.sort(key=lambda p: -p.y)  # sort north to south

    north_ports = direction_ports["N"]
    north_ports.sort(key=lambda p: +p.x)  # sort west to east

    west_ports = direction_ports["W"]
    west_ports.sort(key=lambda p: +p.y)  # sort south to north

    south_ports = direction_ports["S"]
    south_ports.sort(key=lambda p: -p.x)  # sort east to west

    ports = west_ports + north_ports + east_ports + south_ports

    for i, p in enumerate(ports):
        p.name = f"{prefix}{i+1}" if prefix else i + 1


def rename_ports_by_orientation(
    component: Device,
    layers_excluded: Tuple[Tuple[int, int], ...] = None,
    select_ports: Optional[Callable] = None,
    function=_rename_ports_facing_side,
    prefix: str = "o",
) -> None:
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


def auto_rename_ports(
    component: Device,
    function=_rename_ports_clockwise,
    select_ports_optical=select_ports_optical,
    select_ports_electrical=select_ports_electrical,
    prefix_optical: str = "o",
    prefix_electrical: str = "e",
    **kwargs,
):
    """Adds prefix for optical and electical"""
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
    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}
    layers = {port.layer for port in ports.values()}

    for layer in layers:
        ports_on_layer = [p._copy() for p in ports.values() if p.layer == layer]

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

    m = {}
    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}

    ports = select_ports(ports, **kwargs)
    ports_on_layer = [p._copy() for p in ports.values()]

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
    m.update({p.name: p.name_original for p in ports_on_layer})
    return m


map_ports_to_orientation_ccw = partial(
    map_ports_to_orientation_cw, function=_rename_ports_facing_side_ccw
)


def auto_rename_ports_layer_orientation(
    component: Device,
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
    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}
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
        new_ports.update({p.name: p for p in ports_on_layer})

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
    p0 = c.get_ports_list(orientation=90, clockwise=False)[0]
    print(p0)
