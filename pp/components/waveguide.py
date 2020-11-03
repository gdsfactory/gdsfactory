from typing import List, Tuple

import hashlib
import pp
from pp.name import autoname
from pp.components.hline import hline

from pp.component import Component


@autoname
def waveguide(
    length: float = 10.0,
    width: float = 0.5,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layers_cladding: List[Tuple[int, int]] = [pp.LAYER.WGCLAD],
    cladding_offset: float = 3.0,
) -> Component:
    """ straight waveguide

    Args:
        length: in X direction
        width: in Y direction

    .. plot::
      :include-source:

      import pp

      c = pp.c.waveguide(length=10, width=0.5)
      pp.plotgds(c)

    """
    c = Component()
    w = width / 2
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)

    wc = w + cladding_offset

    for layer_cladding in layers_cladding:
        c.add_polygon(
            [(0, -wc), (length, -wc), (length, wc), (0, wc)], layer=layer_cladding
        )

    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer)

    c.width = width
    c.length = length
    return c


@autoname
def wg_shallow_rib(width=0.5, layer=pp.LAYER.SLAB150, layers_cladding=[], **kwargs):
    width = pp.bias.width(width)
    return waveguide(
        width=width, layer=layer, layers_cladding=layers_cladding, **kwargs
    )


@autoname
def wg_deep_rib(width=0.5, layer=pp.LAYER.SLAB90, layers_cladding=[], **kwargs):
    width = pp.bias.width(width)
    return waveguide(
        width=width, layer=layer, layers_cladding=layers_cladding, **kwargs
    )


@autoname
def waveguide_biased(width=0.5, **kwargs):
    width = pp.bias.width(width)
    return waveguide(width=width, **kwargs)


def _arbitrary_straight_waveguide(length, windows):
    """
    windows: [(y_start, y_stop, layer), ...]
    """
    md5 = hashlib.md5()
    for e in windows:
        md5.update(str(e).encode())

    component = Component()
    component.name = "ARB_SW_L{}_HASH{}".format(length, md5.hexdigest())
    y_min, y_max, layer0 = windows[0]
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    # Add one port on each side centered at y=0
    for y_start, y_stop, layer in windows:
        w = abs(y_stop - y_start)
        y = (y_stop + y_start) / 2
        _wg = hline(length=length, width=w, layer=layer).ref()
        _wg.movey(y)
        component.add(_wg)
        component.absorb(_wg)
        y_min = min(y_stop, y_start, y_min)
        y_max = max(y_stop, y_start, y_max)
    width = y_max - y_min

    component.add_port(
        name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer0
    )
    component.add_port(
        name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer0
    )

    return component


@autoname
def waveguide_slab(length=10.0, width=0.5, cladding=2.0, slab_layer=pp.LAYER.SLAB90):
    width = pp.bias.width(width)
    ymin = width / 2
    ymax = ymin + cladding
    windows = [(-ymin, ymin, pp.LAYER.WG), (-ymax, ymax, slab_layer)]
    return _arbitrary_straight_waveguide(length=length, windows=windows)


@autoname
def waveguide_trenches(
    length=10.0,
    width=0.5,
    layer=pp.LAYER.WG,
    trench_width=3.0,
    trench_offset=0.2,
    trench_layer=pp.LAYER.SLAB90,
):
    width = pp.bias.width(width)
    w = width / 2
    ww = w + trench_width
    wt = ww + trench_offset
    windows = [(-ww, ww, layer), (-wt, -w, trench_layer), (w, wt, trench_layer)]
    return _arbitrary_straight_waveguide(length=length, windows=windows)


waveguide_ridge = waveguide_slab


@autoname
def waveguide_slot(length=10.0, width=0.5, gap=0.2, layer=pp.LAYER.WG):
    width = pp.bias.width(width)
    gap = pp.bias.gap(gap)
    a = width / 2
    d = a + gap / 2

    windows = [(-a - d, a - d, layer), (-a + d, a + d, layer)]
    return _arbitrary_straight_waveguide(length=length, windows=windows)


def _demo_waveguide():
    c = waveguide()
    pp.write_gds(c)
    return c


if __name__ == "__main__":
    c = waveguide(length=4, pins=True)

    # pp.show(c)
    # print(c.hash_geometry())
    # pp.show(c)

    # print(c.ports)
    # cc = pp.routing.add_fiber_array(c)
    # pp.show(cc)

    # c = waveguide_slab()
    # c = waveguide_trenches()
    # c = waveguide()
    # c = waveguide_slot()
    # c = waveguide_slot(length=11.2, width=0.5)
    # c = waveguide_slot(length=11.2, width=0.5)
    pp.show(c)
