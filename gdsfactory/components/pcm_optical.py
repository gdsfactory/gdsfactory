""" CD SEM structures
"""
from functools import partial
from typing import List, Optional, Tuple

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.manhattan_font import manhattan_text
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import strip
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory, CrossSectionFactory, Layer

LINE_LENGTH = 420.0


@cell
def cdsem_straight_all(
    straight: ComponentFactory = straight,
    cross_section: CrossSectionFactory = strip,
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: List[Tuple[int, int]] = None,
    widths: Tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    pixel_size: float = 1.0,
    length: float = LINE_LENGTH,
    spacing: float = 4.5,
) -> Component:
    c = Component()
    for width in widths:
        cross_section = partial(cross_section, width=width, layer=layer)
        c.add_ref(straight(length=length, cross_section=cross_section))

    c.distribute(direction="y", spacing=spacing)
    return c


@cell
def cdsem_straight_density(
    width: float = 0.372,
    trench_width: float = 0.304,
    x: float = LINE_LENGTH,
    y: float = 50.0,
    margin: float = 2.0,
    label: str = "",
    straight: ComponentFactory = straight,
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: Optional[Tuple[Layer, ...]] = None,
    cross_section: CrossSectionFactory = strip,
    pixel_size: float = 1.0,
) -> Component:
    """Horizontal grating etch lines

    Args:
        width: width
        trench_width: trench_width
    """
    c = Component()
    period = width + trench_width
    n_o_lines = int((y - 2 * margin) / period)
    length = x - 2 * margin

    cross_section = partial(cross_section, width=width, layer=layer)
    tooth = straight(length=length, cross_section=cross_section)

    for i in range(n_o_lines):
        tooth_ref = c.add_ref(tooth)
        tooth_ref.movey((-n_o_lines / 2 + 0.5 + i) * period)
        c.absorb(tooth_ref)

    marker_label = manhattan_text(
        text=f"{label}",
        size=pixel_size,
        layer=layer,
    )
    _marker_label = c.add_ref(marker_label)
    _marker_label.move((length + 3, 10.0))
    c.absorb(_marker_label)
    return c


@cell
def cdsem_uturn(
    width: float = 0.5,
    radius: float = 10.0,
    wg_length: float = LINE_LENGTH,
    straight: ComponentFactory = straight,
    bend90: ComponentFactory = bend_circular,
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: List[Tuple[int, int]] = None,
    cross_section: CrossSectionFactory = strip,
    pixel_size: float = 1.0,
) -> Component:
    """

    Args:
        width: of the line
        cladding_offset:
        radius: bend radius
        wg_length

    """
    c = Component()
    r = radius

    cross_section = partial(cross_section, width=width, layer=layer)
    if wg_length is None:
        wg_length = 2 * r

    bend90 = bend90(cross_section=cross_section, radius=r)
    wg = straight(
        cross_section=cross_section,
        length=wg_length,
    )

    # Add the U-turn on straight layer
    b1 = c.add_ref(bend90)
    b2 = c.add_ref(bend90)
    b2.connect("o2", b1.ports["o1"])

    wg1 = c.add_ref(wg)
    wg1.connect("o1", b1.ports["o2"])

    wg2 = c.add_ref(wg)
    wg2.connect("o1", b2.ports["o1"])

    label = c << manhattan_text(
        text=str(int(width * 1e3)), size=pixel_size, layer=layer
    )
    label.ymax = b2.ymin - 5
    label.x = 0
    b1.rotate(90)
    b2.rotate(90)
    wg1.rotate(90)
    wg2.rotate(90)
    label.rotate(90)
    return c


@cell
def pcm_optical(
    widths: Tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    dense_lines_width: float = 0.3,
    dense_lines_width_difference: float = 20e-3,
    dense_lines_gap: float = 0.3,
    dense_lines_labels: Tuple[str, ...] = ("DL", "DM", "DH"),
    straight: ComponentFactory = straight,
    bend90: ComponentFactory = bend_circular,
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: List[Tuple[int, int]] = None,
    cross_section: CrossSectionFactory = strip,
    pixel_size: float = 1.0,
) -> Component:
    """column with all optical PCMs

    Args:
        widths:  for straight

    """
    c = Component()
    _c1 = cdsem_straight_all(
        straight=straight,
        layer=layer,
        layers_cladding=layers_cladding,
        widths=widths,
        cross_section=cross_section,
        pixel_size=pixel_size,
    )

    all_devices = [_c1]

    all_devices += [
        cdsem_uturn(
            width=width,
            straight=straight,
            bend90=bend90,
            layer=layer,
            layers_cladding=layers_cladding,
            cross_section=cross_section,
            pixel_size=pixel_size,
        )
        for width in widths
    ]

    density_params = [
        (
            dense_lines_width - dense_lines_width_difference,
            dense_lines_gap - dense_lines_width_difference,
            dense_lines_labels[0],
        ),
        (dense_lines_width, dense_lines_gap, dense_lines_labels[1]),
        (
            dense_lines_width + dense_lines_width_difference,
            dense_lines_gap + dense_lines_width_difference,
            dense_lines_labels[2],
        ),
    ]

    all_devices += [
        cdsem_straight_density(
            width=w,
            trench_width=t,
            label=lbl,
            straight=straight,
            layer=layer,
            layers_cladding=layers_cladding,
            cross_section=cross_section,
            pixel_size=pixel_size,
        )
        for w, t, lbl in density_params
    ]

    [c.add_ref(d) for d in all_devices]
    c.align(elements="all", alignment="xmin")
    c.distribute(elements="all", direction="y", spacing=5, separation=True)
    return c


if __name__ == "__main__":
    # c = cdsem_straight()
    # c = cdsem_straight_all()
    # c = cdsem_uturn(width=2)
    # c = cdsem_straight_density()
    c = pcm_optical(layer=(2, 0))
    c.show()
