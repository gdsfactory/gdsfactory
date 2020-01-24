"""
A port is defined by:

 - position
 - orientation
 - name (which has to be unique within the component being defined)
 - width
 - layer
 - type (optical, dc, rf, detector)

"""


from pp.ports.read_port_markers import read_port_markers
from pp.ports.add_port_markers import add_port_markers
from pp.ports.utils import flipped
from pp.ports.utils import get_ports_facing
from pp.ports.utils import get_non_optical_ports
from pp.ports.utils import get_optical_ports
from pp.ports.utils import select_optical_ports
from pp.ports.utils import select_electrical_ports

from pp.ports.port_naming import deco_rename_ports
from pp.ports.port_naming import auto_rename_ports


__all__ = [
    "read_port_markers",
    "add_port_markers",
    "flipped",
    "deco_rename_ports",
    "auto_rename_ports",
    "get_ports_facing",
    "get_optical_ports",
    "get_non_optical_ports",
    "select_optical_ports",
    "select_electrical_ports",
]
