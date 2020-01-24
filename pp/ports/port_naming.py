import functools


def deco_rename_ports(component_factory):
    @functools.wraps(component_factory)
    def auto_named_component_factory(*args, **kwargs):
        device = component_factory(*args, **kwargs)
        auto_rename_ports(device)
        return device

    return auto_named_component_factory


def _rename_ports_facing_side(direction_ports, prefix=""):
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


def rename_ports_by_orientation(device, layers_excluded=[]):
    """
    Assign standard port names based on the layer of the port
    """

    # Naming functions

    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}

    ports_on_process = [
        p for p in device.ports.values() if p.layer not in layers_excluded
    ]

    for p in ports_on_process:
        # Make sure we can backtrack the parent component from the port
        p.parent = device

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
    device.ports = {p.name: p for p in device.ports.values()}
    return device


def auto_rename_ports(device):
    """
    Assign standard port names based on the layer of the port
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

    for p in device.ports.values():
        if p.port_type not in type_to_ports:
            type_to_ports[p.port_type] = []
        type_to_ports[p.port_type] += [p]

    for port_type, port_group in type_to_ports.items():
        if port_type in type_to_ports_naming_functions:
            _func_name_ports = type_to_ports_naming_functions[port_type]
        else:
            raise ValueError(
                "Unknown port type <{}> in device {}, port {}".format(
                    port_type, device.name, p
                )
            )

        # Make sure we can backtrack the parent component from the port

        direction_ports = {x: [] for x in ["E", "N", "W", "S"]}
        for p in port_group:
            p.parent = device
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
    device.ports = {p.name: p for p in device.ports.values()}
    return device
