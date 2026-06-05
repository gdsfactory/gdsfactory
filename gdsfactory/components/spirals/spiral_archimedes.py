from __future__ import annotations

__all__ = ["spiral_archimedes"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["spirals"])
def spiral_archimedes(
    width: float = 1.0,
    n_turns: int = 5,
    separation: float = 2.0,
    angle_resolution: float = 2.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns an Archimedes spiral: r = (separation + width) / (2 * pi) * theta.

    Args:
        width: width of the spiral trace.
        n_turns: number of turns.
        separation: gap between adjacent traces.
        angle_resolution: degrees per point.
        layer: layer spec.
    """
    c = Component()
    growth_rate = (separation + width) / (2 * np.pi)

    theta_max = n_turns * 2 * np.pi
    n_points = int(np.ceil(theta_max / np.radians(angle_resolution))) + 1
    theta = np.linspace(0, theta_max, n_points)

    r_center = growth_rate * theta
    hw = width / 2.0

    # Compute tangent direction to find normals.
    # dr/dtheta = growth_rate, so tangent in Cartesian:
    #   tx = dr/dtheta * cos(theta) - r * sin(theta)
    #   ty = dr/dtheta * sin(theta) + r * cos(theta)
    dr = np.full_like(theta, growth_rate)
    tx = dr * np.cos(theta) - r_center * np.sin(theta)
    ty = dr * np.sin(theta) + r_center * np.cos(theta)
    t_len = np.sqrt(tx**2 + ty**2)
    t_len = np.where(t_len == 0, 1.0, t_len)
    # Unit normal (perpendicular to tangent, pointing inward).
    nx = -ty / t_len
    ny = tx / t_len

    x_center = r_center * np.cos(theta)
    y_center = r_center * np.sin(theta)

    outer_x = x_center - nx * hw
    outer_y = y_center - ny * hw
    inner_x = x_center + nx * hw
    inner_y = y_center + ny * hw

    # Find the point of the inner edge which has the minimum distance from the start of the outer edge.
    distances_from_edge_point = np.sqrt(
        (inner_x - outer_x[0]) ** 2 + (inner_y - outer_y[0]) ** 2
    )
    min_to_edge_id = np.argmin(distances_from_edge_point)
    # For a smooth spiral, the point of interest is the start of the inner edge, with distance = width.
    # If that is not the case, then end the spiral at the point which violates its smoothness.
    if min_to_edge_id != 0:
        outer_x = outer_x[min_to_edge_id:]
        outer_y = outer_y[min_to_edge_id:]
        inner_x = inner_x[min_to_edge_id:]
        inner_y = inner_y[min_to_edge_id:]

    # Close the polygon: outer path forward, inner path reversed.
    points_x = np.concatenate([outer_x, inner_x[::-1]])
    points_y = np.concatenate([outer_y, inner_y[::-1]])
    points = np.stack((points_x, points_y), axis=-1)

    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = spiral_archimedes()
    c.show()
