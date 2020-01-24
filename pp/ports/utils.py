import pp


def is_electrical_port(port):
    return port.port_type in ["dc", "rf"]


def select_ports(ports, port_type):
    """
    Args:
        ports: a port dictionnary {port name: port} (as returned by Component.ports)
        layers: a list of port layer or a port type (layer or string)

    Returns:
        Dictionnary containing only the ports with the wanted type(s)
        {port name: port}
    """

    # Make it accept Component or ComponentReference
    if isinstance(ports, pp.Component) or isinstance(ports, pp.ComponentReference):
        ports = ports.ports

    return {p_name: p for p_name, p in ports.items() if p.port_type == port_type}


def select_heater_ports(ports):
    return select_ports(ports, port_type="heater")


def select_optical_ports(ports):
    return select_ports(ports, port_type="optical")


def get_optical_ports(ports):
    return select_optical_ports(ports)


def select_electrical_ports(ports):
    d = select_ports(ports, port_type="dc")
    d.update(select_ports(ports, port_type="electrical"))
    return d


def select_dc_ports(ports):
    return select_ports(ports, port_type="dc")


def select_rf_ports(ports):
    return select_ports(ports, port_type="rf")


def select_superconducting_ports(ports):
    d = select_ports(ports, port_type="detector")
    d.update(select_ports(ports, port_type="superconducting"))
    return d


def flipped(port):
    _port = port._copy()
    _port.orientation = (_port.orientation + 180) % 360
    return _port


def move_copy(port, x=0, y=0):
    _port = port._copy()
    _port.midpoint += (x, y)
    return _port


def get_ports_facing(ports, direction="W"):
    if type(ports) == type({}):
        ports = list(ports.values())
    elif isinstance(ports, pp.Component) or isinstance(ports, pp.ComponentReference):
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
    if type(ports) == type({}):
        ports = list(ports.values())
    elif isinstance(ports, pp.Component) or isinstance(ports, pp.ComponentReference):
        ports = list(ports.ports.values())
    res = [p for p in ports if p.port_type not in ["optical"]]
    return res
