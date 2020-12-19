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
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import phidl.geometry as pg
from phidl.device_layout import Device
from phidl.device_layout import Port as PortPhidl

from pp.drc import snap_to_grid

port_types = ["optical", "rf", "dc", "heater"]


class Port(PortPhidl):
    """Ports are useful to connect Components with each other.
    Extends phidl port with layer and port_type (optical, dc, rf)

    Args:
        name: we name ports according to orientation (S0, S1, W0, W1, N0 ...)
        midpoint: (0, 0)
        width: of the port
        orientation: 0
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

    def move_polar_copy(self, d, angle):
        port = self._copy()
        DEG2RAD = np.pi / 180
        dp = np.array((d * np.cos(DEG2RAD * angle), d * np.sin(DEG2RAD * angle)))
        self.move(dp)
        return port

    def flip(self):
        """ flips port """
        port = self._copy()
        port.angle = (port.angle + 180) % 360
        return port

    def _copy(self, new_uid: bool = True):
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

    def snap_to_grid(self, nm=1):
        self.midpoint = nm * np.round(np.array(self.midpoint) * 1e3 / nm) / 1e3

    def on_grid(self, nm: int = 1) -> None:
        if self.orientation in [0, 180]:
            x = self.y + self.width / 2
            assert np.isclose(
                snap_to_grid(x, nm=nm), x
            ), f"{self.parent} port {self.name} has an off-grid point {x}"
        elif self.orientation in [90, 270]:
            x = self.x + self.width / 2
            assert np.isclose(
                snap_to_grid(x, nm=nm), x
            ), f"{self.parent} port {self.name} has an off-grid point {x}"
        else:
            raise ValueError(
                f"{self.parent} port {self.name} has invalid orientation"
                f" {self.orientation}"
            )


def port_array(midpoint=(0, 0), width=0.5, orientation=0, delta=(10, 0), n=2):
    """ returns list of ports """
    return [
        Port(midpoint=np.array(midpoint) + i * np.array(delta), orientation=orientation)
        for i in range(n)
    ]


def read_port_markers(gdspath, layers=((69, 0))):
    """loads a GDS and returns the extracted device for a particular layer

    Args:
        gdspath: gdspath or Component
        layer: GDS layer
    """
    D = gdspath if isinstance(gdspath, Device) else pg.import_gds(gdspath)
    return pg.extract(D, layers=layers)


def csv2port(csvpath):
    """loads and reads ports from a CSV file
    returns a dict
    """
    ports = {}
    with open(csvpath, "r") as csvfile:
        rows = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in rows:
            ports[row[0]] = row[1:]

    return ports


def is_electrical_port(port):
    return port.port_type in ["dc", "rf"]


def select_ports(
    ports,
    port_type: Union[str, Tuple[int, int]] = "optical",
    prefix: Optional[str] = None,
):
    """
    Args:
        ports: Dict[str, Port] a port dictionnary {port name: port} (as returned by Component.ports)
        layers: a list of port layer or a port type (layer or string)

    Returns:
        Dictionnary containing only the ports with the wanted type(s)
        {port name: port}
    """

    from pp.component import Component, ComponentReference

    # Make it accept Component or ComponentReference
    if isinstance(ports, Component) or isinstance(ports, ComponentReference):
        ports = ports.ports

    ports = {
        p_name: p
        for p_name, p in ports.items()
        if p.port_type == port_type or p.layer == port_type
    }
    if prefix:
        ports = {p_name: p for p_name, p in ports.items() if p_name.startswith(prefix)}
    return ports


def select_optical_ports(ports: Dict[str, Port], prefix=None) -> Dict[str, Port]:
    return select_ports(ports, port_type="optical", prefix=prefix)


def select_electrical_ports(ports, port_type="dc", prefix=None):
    d = select_ports(ports, port_type=port_type, prefix=prefix)
    d.update(select_ports(ports, port_type="electrical"))
    return d


def flipped(port):
    _port = port._copy()
    _port.orientation = (_port.orientation + 180) % 360
    return _port


def move_copy(port, x=0, y=0):
    _port = port._copy()
    _port.midpoint += (x, y)
    return _port


def get_ports_facing(ports: List[Port], direction: str = "W"):
    from pp.component import Component, ComponentReference

    if isinstance(ports, dict):
        ports = list(ports.values())
    elif isinstance(ports, Component) or isinstance(ports, ComponentReference):
        ports = list(ports.ports.values())

    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}

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


def get_non_optical_ports(ports):
    from pp.component import Component, ComponentReference

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
    component: Device, layers_excluded: List[Tuple[int, int]] = None
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
        "dc": lambda _d: _counter_clockwise(_d, "E_"),
        "superconducting": lambda _d: _counter_clockwise(_d, "SC_"),
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
            raise ValueError(
                "Unknown port type <{}> in component {}, port {}".format(
                    port_type, component.name, p
                )
            )

        # Make sure we can backtrack the parent component from the port

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


def test_select_ports_prefix():
    import pp

    c = pp.c.waveguide()
    ports = c.get_ports_list(prefix="W")
    assert len(ports) == 1


def test_select_ports_type():
    import pp

    c = pp.c.mzi2x2(with_elec_connections=True)
    ports = c.get_ports_list(port_type="dc")
    assert len(ports) == 3


if __name__ == "__main__":
    test_select_ports_type()

    import pp

    name = "mmi1x2"
    gdspath = pp.CONFIG["gdslib"] / "gds" / f"{name}.gds"
    csvpath = pp.CONFIG["gdslib"] / "gds" / f"{name}.ports"
    pp.show(gdspath)
    # read_port_markers(gdspath, layer=66)
    p = csv2port(csvpath)
    print(p)
