from __future__ import annotations

__all__ = ["dash"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def dash(
    width: float = 10.0,
    width_end: float = 1.0,
    length: float = 20.0,
    taper_length: float = 5.0,
    tip_length: float = 2.0,
    n_bezier_points: int = 30,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns a dash shape with Bezier-curved tapered tips.

    An elongated shape wider in the middle (width) that tapers via
    Bezier curves to a narrower tip (width_end) at each end. Based on
    the pyNISTtoolbox Dash pattern.

    Args:
        width: width at the center/body of the dash.
        width_end: width at the tips.
        length: total length of the straight body section.
        taper_length: length of each tapered transition.
        tip_length: length of each rounded tip beyond the taper.
        n_bezier_points: points per Bezier curve segment.
        layer: layer spec.
    """
    return cf.dash(
        width=width,
        width_end=width_end,
        length=length,
        taper_length=taper_length,
        tip_length=tip_length,
        n_bezier_points=n_bezier_points,
        layer=layer,
    )


if __name__ == "__main__":
    c = dash()
    c.show()
