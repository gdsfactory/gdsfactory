from typing import Any

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection, cross_section, xsection
from gdsfactory.typings import LayerSpec


@xsection
def strip3(
    width: float = 1.5,
    layer: LayerSpec = (1, 0),
    radius: float = 40.0,
    radius_min: float = 30,
    layer_sides: LayerSpec = (2, 0),
    layer_cover: LayerSpec = (3, 0),
    width_sides: float = 15,
    width_cover: float = 40,
    offset_sides: float = 65,
    **kwargs: Any,
) -> CrossSection:
    sections = (
        gf.Section(layer=layer_sides, width=width_sides, offset=offset_sides),
        gf.Section(layer=layer_sides, width=width_sides, offset=-offset_sides),
        gf.Section(layer=layer_cover, width=width_cover, offset=0),
    )

    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )


if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    c = gf.components.mmi(
        inputs=2,
        outputs=4,
        width=1.5,
        width_taper=2.45,
        length_taper=30,
        length_mmi=550,
        width_mmi=13.9,
        gap_output_tapers=1.1,
        input_positions=[5.325, -1.775],
        cross_section=strip3,
    )
    c.show()
