from __future__ import annotations

__all__ = ["coupler_pulley"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


def _cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    n_points: int = 64,
) -> np.ndarray:
    """Compute cubic Bezier curve points."""
    t = np.linspace(0, 1, n_points)[:, None]
    return (
        (1 - t) ** 3 * np.array(p0)
        + 3 * (1 - t) ** 2 * t * np.array(p1)
        + 3 * (1 - t) * t**2 * np.array(p2)
        + t**3 * np.array(p3)
    )


def _offset_curve(pts: np.ndarray, offset: float) -> np.ndarray:
    """Offset a curve by a fixed distance along the local normal."""
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    lengths = np.sqrt(tangents[:, 0] ** 2 + tangents[:, 1] ** 2)
    lengths = np.maximum(lengths, 1e-12)
    tangents /= lengths[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return pts + offset * normals


@gf.cell_with_module_name
def coupler_pulley(
    radius: float = 10.0,
    ring_width: float | None = None,
    gap: float = 0.2,
    coupling_angle: float = 60.0,
    waveguide_width: float = 0.5,
    wg_length: float = 40.0,
    wg_height: float = 10.0,
    n_segments: int = 128,
    layer: LayerSpec = "WG",
    port_type: str = "optical",
) -> Component:
    """Returns a disc or ring with a pulley-coupled waveguide.

    A waveguide wraps symmetrically around the top of a disc/ring
    over coupling_angle degrees. Bezier S-curves (following the CNST
    discPulley construction, eq. 2.22) route the waveguide from the
    coupling arc down to horizontal exits on both sides.

    Args:
        radius: radius of the disc, or outer radius of the ring.
        ring_width: width of the ring annulus. None for a solid disc.
        gap: gap between the waveguide inner edge and the disc/ring.
        coupling_angle: total wrap angle in degrees (symmetric about top).
        waveguide_width: width of the coupling waveguide.
        wg_length: horizontal half-length of the waveguide from the disc
            center to the exit end. Controls S-curve extent. Corresponds
            to CNST parameter L.
        wg_height: vertical drop from disc center to exit waveguide level.
            Corresponds to CNST parameter H.
        n_segments: number of points for each curved section.
        layer: layer spec.
        port_type: port type for optical ports.
    """
    c = Component()

    half_angle = np.radians(coupling_angle / 2)
    wg_r = radius + gap + waveguide_width / 2
    h_exit = -wg_height

    # 1. Disc or ring
    theta_full = np.linspace(0, 2 * np.pi, n_segments * 2, endpoint=False)
    if ring_width is not None:
        inner_r = radius - ring_width
        outer_pts = np.column_stack(
            [radius * np.cos(theta_full), radius * np.sin(theta_full)]
        )
        inner_pts = np.column_stack(
            [inner_r * np.cos(theta_full[::-1]), inner_r * np.sin(theta_full[::-1])]
        )
        c.add_polygon(np.vstack([outer_pts, inner_pts]), layer=layer)
    else:
        c.add_polygon(
            np.column_stack([radius * np.cos(theta_full), radius * np.sin(theta_full)]),
            layer=layer,
        )

    # 2. Coupling arc (wraps around the top of the disc/ring)
    theta_arc = np.linspace(np.pi / 2 + half_angle, np.pi / 2 - half_angle, n_segments)
    arc_center = np.column_stack([wg_r * np.cos(theta_arc), wg_r * np.sin(theta_arc)])

    # 3. Bezier S-curves from coupling arc endpoints to horizontal exits
    # Following CNST eq. 2.22:
    #   P1 = arc endpoint, P2 = (±L, -H)
    #   C1 = P1 + R/4 * tangent_at_P1
    #   C2 = (P2x ∓ L/2, P2y)
    #   R = sqrt(H² + L²)

    L = wg_length
    H_param = wg_height + wg_r * np.sin(
        half_angle
    )  # total vertical drop from arc endpoint to exit
    R_bezier = np.sqrt(H_param**2 + L**2)

    bezier_paths = {}
    for name, arc_angle, x_sign in [
        ("left", np.pi / 2 + half_angle, -1),
        ("right", np.pi / 2 - half_angle, 1),
    ]:
        # P1: coupling arc endpoint
        p1 = np.array([wg_r * np.cos(arc_angle), wg_r * np.sin(arc_angle)])

        # P2: horizontal exit point at (±L, h_exit)
        p2 = np.array([x_sign * L, h_exit])

        # Tangent to circle at arc endpoint, pointing away from arc center
        tangent = np.array([x_sign * np.sin(arc_angle), -x_sign * np.cos(arc_angle)])

        # C1: along tangent from P1 (CNST R/4 distance)
        c1 = p1 + (R_bezier / 4) * tangent

        # C2: near P2, displaced horizontally toward center by L/2
        c2 = np.array([p2[0] - x_sign * L / 2, p2[1]])

        bezier = _cubic_bezier(tuple(p1), tuple(c1), tuple(c2), tuple(p2), n_segments)
        bezier_paths[name] = bezier

    # 4. Assemble full center-line: left_exit → left_bezier → arc → right_bezier → right_exit
    left_exit = np.array([[-L - 5, h_exit], [-L, h_exit]])  # small straight at left end
    right_exit = np.array([[L, h_exit], [L + 5, h_exit]])  # small straight at right end

    center_line = np.vstack(
        [
            left_exit,
            bezier_paths["left"][::-1],  # reversed: from exit toward arc
            arc_center,
            bezier_paths["right"],  # from arc toward exit
            right_exit,
        ]
    )

    # 5. Offset to create waveguide polygon
    hw = waveguide_width / 2
    outer_edge = _offset_curve(center_line, hw)
    inner_edge = _offset_curve(center_line, -hw)
    c.add_polygon(np.vstack([outer_edge, inner_edge[::-1]]), layer=layer)

    # 6. Ports
    c.add_port(
        "o1",
        center=(L + 5, h_exit),
        width=waveguide_width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        "o2",
        center=(-L - 5, h_exit),
        width=waveguide_width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = coupler_pulley()
    c.show()
