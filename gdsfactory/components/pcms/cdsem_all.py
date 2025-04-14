"""CD SEM structures."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def cdsem_all(
    widths: tuple[float, ...] = (0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
    dense_lines_width: float | None = 0.3,
    dense_lines_width_difference: float = 20e-3,
    dense_lines_gap: float = 0.3,
    dense_lines_labels: tuple[str, ...] = ("DL", "DM", "DH"),
    straight: ComponentSpec = "straight",
    bend90: ComponentSpec | None = "bend_circular",
    cross_section: CrossSectionSpec = "strip",
    text: ComponentSpec = "text_rectangular",
    spacing: float = 5,
    cdsem_bend180: ComponentSpec = "cdsem_bend180",
    text_size: float = 1,
) -> Component:
    """Column with all optical PCMs.

    Args:
        widths: for straight lines.
        dense_lines_width: in um.
        dense_lines_width_difference: in um.
        dense_lines_gap: in um.
        dense_lines_labels: strings.
        straight: spec.
        bend90: spec.
        cross_section: spec.
        text: spec.
        spacing: from group to group.
        cdsem_bend180: spec.
        text_size: in um.
    """
    c = Component()
    _c1 = gf.get_component(
        "cdsem_straight",
        widths=widths,
        cross_section=cross_section,
    )

    all_devices = [_c1]

    if bend90:
        all_devices += [
            gf.get_component(
                cdsem_bend180,
                width=width,
                straight=straight,
                bend90=bend90,
                cross_section=cross_section,
                text=text,
                text_size=text_size,
            )
            for width in widths
        ]

    if dense_lines_width:
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
            gf.get_component(
                "cdsem_straight_density",
                widths=(w,) * 10,
                gaps=(g,) * 10,
                label=lbl,
                cross_section=cross_section,
                text=text,
                text_size=text_size,
            )
            for w, g, lbl in density_params
        ]

    ymin = 0.0
    for d in all_devices:
        ref = c.add_ref(d)
        ref.xmin = 0
        ref.ymin = ymin
        ymin += ref.ysize + spacing

    return c


if __name__ == "__main__":
    c = cdsem_all()
    c.show()
