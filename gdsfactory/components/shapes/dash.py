from __future__ import annotations

__all__ = ["dash"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


def _bezier3(
    px: list[float], py: list[float], n: int = 30
) -> tuple[list[float], list[float]]:
    """Cubic Bezier from 4 control points."""
    t = np.linspace(0, 1, n)
    x = (
        (1 - t) ** 3 * px[0]
        + 3 * (1 - t) ** 2 * t * px[1]
        + 3 * (1 - t) * t**2 * px[2]
        + t**3 * px[3]
    )
    y = (
        (1 - t) ** 3 * py[0]
        + 3 * (1 - t) ** 2 * t * py[1]
        + 3 * (1 - t) * t**2 * py[2]
        + t**3 * py[3]
    )
    return list(x), list(y)


@gf.cell_with_module_name(tags={"type": "shapes"})
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
    c = Component()
    W = width
    Wend = width_end
    L = length
    Ltap = taper_length
    Lt = tip_length

    alpha = np.pi / 2 - np.arctan((W - Wend) / (2 * Ltap))

    # Top tip
    pxup = Lt * np.cos(alpha)
    pyup = Lt * np.sin(alpha)
    px = [-Wend / 2, -Wend / 2 + pxup, Wend / 2 - pxup, Wend / 2]
    py = [L / 2 + Ltap, L / 2 + Ltap + pyup, L / 2 + Ltap + pyup, L / 2 + Ltap]
    xtip_t, ytip_t = _bezier3(px, py, n_bezier_points)

    # Left upper taper
    px = [-W / 2, -W / 2, -W / 2, -Wend / 2]
    py = [L / 3, L / 2, L / 2, L / 2 + Ltap]
    xangle_lu, yangle_lu = _bezier3(px, py, n_bezier_points)

    # Right upper taper
    px = [Wend / 2, W / 2, W / 2, W / 2]
    py = [L / 2 + Ltap, L / 2, L / 2, L / 3]
    xangle_ru, yangle_ru = _bezier3(px, py, n_bezier_points)

    # Right lower taper
    px = [W / 2, W / 2, W / 2, Wend / 2]
    py = [-L / 3, -L / 2, -L / 2, -L / 2 - Ltap]
    xangle_rb, yangle_rb = _bezier3(px, py, n_bezier_points)

    # Bottom tip
    px = [Wend / 2, Wend / 2 - pxup, -Wend / 2 + pxup, -Wend / 2]
    py = [-L / 2 - Ltap, -L / 2 - Ltap - pyup, -L / 2 - Ltap - pyup, -L / 2 - Ltap]
    xtip_b, ytip_b = _bezier3(px, py, n_bezier_points)

    # Left lower taper
    px = [-Wend / 2, -W / 2, -W / 2, -W / 2]
    py = [-L / 2 - Ltap, -L / 2, -L / 2, -L / 3]
    xangle_lb, yangle_lb = _bezier3(px, py, n_bezier_points)

    # Assemble outline
    xdash = xangle_lu + xtip_t + xangle_ru + xangle_rb + xtip_b + xangle_lb
    ydash = yangle_lu + ytip_t + yangle_ru + yangle_rb + ytip_b + yangle_lb

    points = list(zip(xdash, ydash, strict=False))
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = dash()
    c.show()
