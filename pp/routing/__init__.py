"""
Route optical and electrical waveguides
"""

from pp.routing.connect_component import add_io_optical
from pp.routing.connect_bundle import connect_bundle
from pp.routing.connect_bundle import link_optical_ports
from pp.routing.connect_bundle import link_electrical_ports
from pp.routing.route_ports_to_side import route_ports_to_side
from pp.routing.route_ports_to_side import route_elec_ports_to_side
from pp.routing.manhattan import round_corners
from pp.routing.manhattan import route_manhattan

__all__ = [
    "add_io_optical",
    "link_optical_ports",
    "link_electrical_ports",
    "route_ports_to_side",
    "route_elec_ports_to_side",
    "connect_bundle",
    "round_corners",
    "route_manhattan",
]

if __name__ == "__main__":
    print(__all__)
