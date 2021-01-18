"""Straight waveguides"""
import hashlib
from typing import Iterable, Optional, Tuple

import pp
from pp.component import Component
from pp.components.hline import hline


@pp.cell
def waveguide(
    length: float = 10.0,
    width: float = 0.5,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layers_cladding: Optional[Iterable[Tuple[int, int]]] = None,
    cladding_offset: float = pp.conf.tech.cladding_offset,
) -> Component:
    """Straight waveguide

    Args:
        length: in X direction
        width: in Y direction
        layer
        layers_cladding
        cladding_offset

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

    if layers_cladding:
        for layer_cladding in layers_cladding:
            c.add_polygon(
                [(0, -wc), (length, -wc), (length, wc), (0, wc)], layer=layer_cladding
            )

    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer)

    c.width = width
    c.length = length
    return c


@pp.cell
def waveguide_biased(width: float = 0.5, **kwargs) -> Component:
    """Waveguide with etch bias"""
    width = pp.bias.width(width)
    return waveguide(width=width, **kwargs)


def _arbitrary_straight_waveguide(length, windows):
    """
    Args:
        length: length
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


@pp.cell
def waveguide_slab(length=10.0, width=0.5, cladding=2.0, slab_layer=pp.LAYER.SLAB150):
    """Waveguide with thinner top Silicon."""
    ymin = width / 2
    ymax = ymin + cladding
    windows = [(-ymin, ymin, pp.LAYER.WG), (-ymax, ymax, slab_layer)]
    return _arbitrary_straight_waveguide(length=length, windows=windows)


@pp.cell
def waveguide_trenches(
    length=10.0,
    width=0.5,
    layer=pp.LAYER.WG,
    trench_width=3.0,
    trench_offset=0.2,
    trench_layer=pp.LAYER.SLAB90,
):
    """Waveguide with trenches on both sides."""
    w = width / 2
    ww = w + trench_width
    wt = ww + trench_offset
    windows = [(-ww, ww, layer), (-wt, -w, trench_layer), (w, wt, trench_layer)]
    return _arbitrary_straight_waveguide(length=length, windows=windows)


@pp.cell
def waveguide_slot(length=10.0, width=0.5, gap=0.2, layer=pp.LAYER.WG):
    """Waveguide with a slot in the middle."""
    gap = pp.bias.gap(gap)
    a = width / 2
    d = a + gap / 2

    windows = [(-a - d, a - d, layer), (-a + d, a + d, layer)]
    return _arbitrary_straight_waveguide(length=length, windows=windows)


if __name__ == "__main__":
    c = waveguide(length=0.3)
    c.pprint()

    # print(c.name)
    # print(c.settings)
    # print(c.settings_changed)
    # pp.write_gds(c)
    # print(c.hash_geometry())

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
