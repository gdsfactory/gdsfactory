"""components for gdsfactory simphony circuit simulation plugin."""
from .bend_circular import bend_circular
from .bend_euler import bend_euler
from .coupler import coupler
from .coupler_ring import coupler_ring
from .gc import gc1550te
from .mmi1x2 import mmi1x2
from .mmi2x2 import mmi2x2
from .mzi import mzi
from .ring_double import ring_double
from .ring_single import ring_single
from .straight import straight

model_factory = dict(
    bend_circular=bend_circular,
    bend_euler=bend_euler,
    coupler_ring=coupler_ring,
    coupler=coupler,
    mmi1x2=mmi1x2,
    mmi2x2=mmi2x2,
    straight=straight,
    gc1550te=gc1550te,
)

circuit_factory = dict(mzi=mzi, ring_double=ring_double, ring_single=ring_single)


component_names = list(model_factory.keys())
circuit_names = list(circuit_factory.keys())
__all__ = component_names + circuit_names
