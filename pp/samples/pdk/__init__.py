from waveguide import waveguide
from bend_circular import bend_circular
from y_splitter import y_splitter
from mzi import mzi


component_type2factory = dict(
    waveguide=waveguide, bend_circular=bend_circular, y_splitter=y_splitter, mzi=mzi
)

_components = list(component_type2factory.keys())
__all__ = _components
