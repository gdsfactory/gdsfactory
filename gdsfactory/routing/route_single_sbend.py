from __future__ import annotations

from functools import lru_cache

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Port


def route_single_sbend(
    component: Component,
    port1: Port,
    port2: Port,
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    allow_layer_mismatch: bool = False,
    allow_width_mismatch: bool = False,
) -> ComponentReference:
    """Returns an Sbend to connect two ports.

    Args:
        component: to add the route to.
        port1: start port.
        port2: end port.
        bend_s: Sbend component.
        cross_section: cross_section.
        allow_layer_mismatch: allow layer mismatch.
        allow_width_mismatch: allow width mismatch.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component()
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.movex(50)
        mmi2.movey(5)
        route = gf.routing.route_single_sbend(c, mmi1.ports['o2'], mmi2.ports['o1'])
        c.plot()
    """
    # Unpack used port values once
    p1x, p1y = port1.center
    p2x, p2y = port2.center
    o1 = port1.orientation
    o2 = port2.orientation

    ysize = p2y - p1y
    xsize = p2x - p1x

    # Compute `size` branch only once using pre-unpacked values.
    if o1 in (0, 180):
        size = (xsize, ysize)
    else:
        size = (ysize, -xsize)

    # Caching optimization: only cache if the specs are hashable
    try:
        # If hashable, cache component creation
        bend_s_hashable = bend_s if isinstance(bend_s, str) else repr(bend_s)
        cross_section_hashable = (
            cross_section if isinstance(cross_section, str) else repr(cross_section)
        )
        bend = _cached_bend_component(bend_s_hashable, size, cross_section_hashable)
    except Exception:
        # Fallback for unhashable cases: original, uncached behavior
        bend = gf.get_component(bend_s, size=size, cross_section=cross_section)

    bend_ref = component << bend
    bend_ref.connect(
        bend_ref.ports[0],
        port1,
        allow_layer_mismatch=allow_layer_mismatch,
        allow_width_mismatch=allow_width_mismatch,
    )

    # Compute orthogonality error using local variables (ints/floats)
    orthogonality_error = abs(abs(o1 - o2) - 180)
    if orthogonality_error > 0.1:
        raise ValueError(
            f"Ports need to have orthogonal orientation {orthogonality_error}\n"
            f"port1 = {o1} deg and port2 = {o2}"
        )
    return bend_ref


def _get_active_pdk_cached():
    global _active_pdk
    if _active_pdk is None:
        _active_pdk = gf.get_active_pdk()
    return _active_pdk


# LRU cache for bend_s, size, cross_section
@lru_cache(maxsize=128)
def _cached_bend_component(bend_s_hash, size, cross_section_hash):
    # Note: bend_s and cross_section can be unhashable, thus pass their hashes
    # bend_s and cross_section must be hashable spec or str or cache won't kick in.
    return gf.get_component(bend_s_hash, size=size, cross_section=cross_section_hash)


if __name__ == "__main__":
    c = gf.Component(name="demo_route_sbend")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.movex(50)
    mmi2.movey(5)
    route_single_sbend(c, mmi1.ports["o2"], mmi2.ports["o1"])
    c.show()

_active_pdk = None
