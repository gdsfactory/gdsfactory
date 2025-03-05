import math

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import LayerSpec

# Define a type for 2D points
Point = tuple[float, float]


def locToWorld(
    xyp: Point | list[Point], x0: float, y0: float, angle: float
) -> Point | list[Point]:
    """Convert local coordinates to world coordinates using rotation and translation.

    Args:
        xyp: A tuple (x, y) or a list of tuples representing points in local coordinates.
        x0: x-coordinate of the translation (world origin).
        y0: y-coordinate of the translation (world origin).
        angle: Rotation angle in radians.

    Returns:
        The transformed point or list of points in world coordinates.
    """

    def _locToWorld(xp: float, yp: float) -> Point:
        x = x0 - xp * math.cos(angle) + yp * math.sin(angle)
        y = y0 + xp * math.sin(angle) + yp * math.cos(angle)
        return (x, y)

    if not isinstance(xyp, list):
        return _locToWorld(xyp[0], xyp[1])

    xy: list[Point] = []
    for xp, yp in xyp:
        xy.append(_locToWorld(xp, yp))
    return xy


def awg(
    N: int = 40,
    R: float = 100,
    g: float = 100e-3,
    d: float = 1.5,
    w: float = 800e-3,
    m: int = 20,
    dl: float = 10,
    Ni: int = 1,
    di: float = 3,
    L0: float = 0,
    li: float = 0,
    wi: float = 1.5,
    No: int = 1,
    do: float = 3,
    lo: float = 0,
    wo: float = 1.5,
    k_core: float = 1,
    k_bend: float = 1,
    inputs: list[float] | None = None,
    outputs: list[float] | None = None,
    confocal: bool = False,
    lengths: list[float] | None = None,
    defocus: float = 0,
    taper_profile: str = "lin",
    taper_length: float = 40,
    taper_points: list[tuple[float, float]] | None = None,
    core_spacing: float = 2.0,
    bend_radius: float = 5.0,
    bend_width: float = 500e-3,
    bend_taper: float = 15,
    dummy_apertures: int = 0,
    routing_width: float = 500e-3,
    core_layer: LayerSpec = "WG",
    separation: float = 50,
    limit_fpr_size: bool = False,
) -> gf.Component:
    """Arrayed waveguide grating.

    Args:
        N: Number of waveguides.
        R: Radius for curved sections.
        g: Gap parameter.
        d: Spacing parameter.
        w: Waveguide width.
        m: Order of the grating.
        dl: Delta length.
        Ni: Number of input ports.
        di: Input port spacing.
        L0: Initial straight length for routes.
        li: Input taper length.
        wi: Input taper width.
        No: Number of output ports.
        do: Output port spacing.
        lo: Output taper length.
        wo: Output taper width.
        k_core: Core length scaling factor.
        k_bend: Bend length scaling factor.
        inputs: Optional list of input port positions.
        outputs: Optional list of output port positions.
        confocal: Flag to indicate if the design is confocal.
        lengths: List of additional path lengths for waveguides.
        defocus: Defocus amount for IO design.
        taper_profile: Type of taper profile ("lin", "quad", or "custom").
        taper_length: Length of the taper.
        taper_points: Custom taper points if taper_profile is "custom".
        core_spacing: Spacing between waveguides.
        bend_radius: Bend radius for curved waveguides.
        bend_width: Width of the bending section.
        bend_taper: Taper length for bend transitions.
        dummy_apertures: Number of dummy apertures to add.
        routing_width: Width of the routing taper region.
        core_layer: Layer specification for the waveguide.
        separation: Vertical separation between FPR centers.
        limit_fpr_size: Flag to limit the FPR size.

    Returns:
        A Component representing the AWG.
    """
    if taper_points is None:
        taper_points = []
    if lengths is None:
        lengths = []
    if outputs is None:
        outputs = []
    if inputs is None:
        inputs = []
    component = gf.Component("AWG")

    def gen_fpr(x0: float, y0: float) -> None:
        """Generate the Free Propagation Region (FPR) polygon.

        Args:
            x0: x-coordinate offset for the FPR.
            y0: y-coordinate offset for the FPR.
        """
        xy: list[Point] = []
        angle: float = math.pi
        if limit_fpr_size:
            angle = 3 / 2 * (N + 8) * d / R

        # Create the upper FPR curve
        for a in np.linspace(-angle / 2, angle / 2, 100):
            xy.append((R * (1 - math.cos(a)), R * math.sin(a)))

        angle = math.pi / 3
        if limit_fpr_size:
            angle = (N + 8) * d / R

        # Create the lower FPR curve
        for a in np.linspace(angle / 2, -angle / 2, 100):
            xy.append((R * math.cos(a), R * math.sin(a)))

        # Translate polygon to the proper position
        xy_array = np.array(xy) + np.array([x0, y0])
        component.add_polygon(xy_array, layer=core_layer)

    def gen_io(
        x0: float, y0: float, out: bool = False, radius: float = 25, points: int = 20
    ) -> None:
        """Generate the input/output (IO) taper region and add corresponding ports.

        Args:
            x0: x-coordinate offset.
            y0: y-coordinate offset.
            out: If True, generate output ports; otherwise, input ports.
            radius: Taper bend radius.
            points: Number of points for curve approximation.
        """
        b: float = R * 0.02
        N_io: int = Ni if not out else No
        d_io: float = di if not out else do
        l_io: float = li if not out else -lo
        w_io: float = wi if not out else wo

        # Calculate relative positions for ports
        pos: list[float] = [
            (l_io + (k - (N_io - 1) / 2) * max(d_io, w_io)) / R for k in range(N_io)
        ]
        if out and outputs:
            pos = [x / R for x in outputs]
        elif not out and inputs:
            pos = [x / R for x in inputs]

        N_io = len(pos)
        xmax: float = -taper_length - radius / 5
        for k in range(N_io):
            a: float = pos[k]
            xc: float = R - (R + defocus) * math.cos(a)
            yc: float = R * math.sin(a)
            xy: list[Point] = []

            # Build upper taper curve
            xy1: list[Point] = [
                (-b, w_io / 2),
                (0, w_io / 2),
                (taper_length, routing_width / 2),
            ]
            xy.extend(locToWorld(xy1, xc, yc, a / 2))

            h: int = 1 if a > 0 else -1
            xy1 = [
                (
                    taper_length + radius * math.sin(t),
                    h * (radius + h * routing_width / 2 - radius * math.cos(t)),
                )
                for t in np.linspace(0, abs(a / 2), points)
            ]
            xy.extend(locToWorld(xy1, xc, yc, a / 2))
            xy.append((xmax, xy[-1][1]))

            # Add IO port with consistent naming
            name: str = f"in{k}" if not out else f"out{k}"
            component.add_port(
                name=name,
                center=(x0 + xmax, y0 + xy[-1][1] - routing_width / 2),
                layer=core_layer,
                width=routing_width,
                orientation=0,
            )
            xy.append((xmax, xy[-1][1] - routing_width))

            # Build lower taper curve
            xy1 = [
                (
                    taper_length + radius * math.sin(t),
                    h * (radius - h * routing_width / 2 - radius * math.cos(t)),
                )
                for t in np.linspace(abs(a / 2), 0, points)
            ]
            xy.extend(locToWorld(xy1, xc, yc, a / 2))
            xy1 = [(taper_length, -routing_width / 2), (0, -w_io / 2), (-b, -w_io / 2)]
            xy.extend(locToWorld(xy1, xc, yc, a / 2))
            component.add_polygon(xy, layer=core_layer)

    def gen_wgs(
        x0: float, y0: float, out: bool = False, radius: float = 25, points: int = 20
    ) -> None:
        """Generate waveguide segments and add corresponding ports.

        Args:
            x0: x-coordinate offset.
            y0: y-coordinate offset.
            out: If True, generate output side waveguides; otherwise, input side.
            radius: Bend radius parameter.
            points: Number of points for curve approximation.
        """
        b: float = R * 0.01
        w_val: float = d - g
        p: float = bend_width

        tp: float = (N - 1) / 2 * d / R
        lp: float = (
            (N - 1) / 2 * core_spacing
            - radius * (1 - math.cos(tp))
            - (R + taper_length) * math.sin(tp)
        ) / math.sin(tp)
        xp: float = (R + taper_length + lp + p) * math.cos(tp) + (
            radius + p
        ) * math.sin(tp)

        # Define taper profiles for waveguide transitions
        if taper_profile == "lin":
            taper_profile_up: list[Point] = [
                (0, +w_val / 2),
                (b, +w_val / 2),
                (taper_length + b, +p / 2),
            ]
            taper_profile_dn: list[Point] = [
                (taper_length + b, -p / 2),
                (b, -w_val / 2),
                (0, -w_val / 2),
            ]
        elif taper_profile == "quad":
            x_vals, y_vals = (
                np.linspace(0, taper_length, 100),
                np.linspace(w_val, p, 100),
            )
            taper_profile_up = [(0, +w_val / 2)] + [
                (x_vals[i] + b, y_vals[i] / 2) for i in range(len(x_vals))
            ]
            taper_profile_dn = [
                (x_vals[i] + b, -y_vals[i] / 2) for i in range(len(x_vals) - 1, -1, -1)
            ] + [(0, -w_val / 2)]
        elif taper_profile == "custom":
            taper_profile_up = [(0, +w_val / 2)] + [
                (pt[0] + b, pt[1]) for pt in taper_points
            ]
            taper_profile_dn = [
                (pt[0] + b, -pt[1]) for pt in reversed(taper_points)
            ] + [(0, -w_val / 2)]
        else:
            raise ValueError(f"Unknown taper profile type '{taper_profile}'")

        for k in range(-dummy_apertures, N + dummy_apertures):
            t: float = (k - (N - 1) / 2) * d / R
            h: int = 1 if t >= 0 else -1
            yp_val: float = (k - (N - 1) / 2) * core_spacing
            lp_k: float = (
                yp_val
                - h * radius * (1 - math.cos(t))
                - (R + taper_length) * math.sin(t)
            ) / math.sin(t)
            xc: float = (R - b) * math.cos(t)
            yc: float = (R - b) * math.sin(t)
            xy: list[Point] = []

            if k < 0 or k > N - 1:
                # Create a dummy aperture polygon
                xy = [
                    (-b, w_val / 2),
                    (0, w_val / 2),
                    (taper_length + b, 0.3 / 2),
                    (taper_length + b, -0.3 / 2),
                    (0, -w_val / 2),
                    (-b, -w_val / 2),
                ]
                xy = locToWorld(xy, xc, yc, math.pi - t)
            else:
                # Build the top taper and bend for the waveguide
                xy_bend_top: list[Point] = locToWorld(
                    taper_profile_up, xc, yc, math.pi - t
                )
                pts = [
                    (
                        b + taper_length + lp_k + radius * math.sin(a),
                        h * (radius - radius * math.cos(a)),
                    )
                    for a in np.linspace(0, t, points)
                ]
                xy_bend_top.extend(locToWorld(pts, xc, yc, math.pi - t))
                xy_bend_top.append((xp, xy_bend_top[-1][1]))
                xy_bend_top.append((xp + 0.1, xy_bend_top[-1][1]))
                xy.extend(xy_bend_top)

                # Add a port for this waveguide with consistent naming
                name: str = f"awin{k}" if not out else f"awout{k}"
                component.add_port(
                    name=name,
                    center=(x0 + xp, y0 + xy[-1][1] + p / 2),
                    layer=core_layer,
                    width=p,
                    orientation=0,
                )

                # Build the bottom taper and bend for the waveguide
                xy_bend_bot: list[Point] = [
                    (xp + 0.1, xy_bend_top[-1][1] + p),
                    (xp, xy_bend_top[-1][1] + p),
                ]
                pts_bot = [
                    (
                        b + taper_length + lp_k + radius * math.sin(a),
                        h * (radius - radius * math.cos(a)),
                    )
                    for a in np.linspace(t, 0, points)
                ]
                xy_bend_bot.extend(locToWorld(pts_bot, xc, yc, math.pi - t))
                xy_bend_bot.extend(locToWorld(taper_profile_dn, xc, yc, math.pi - t))
                xy.extend(xy_bend_bot)
            component.add_polygon(xy, layer=core_layer)

    def gen_routes() -> None:
        """Generate routing paths connecting the AWG arms.
        Each route connects the input side (port "awinX") to the output side (port "awoutX")
        using a series of straight and Euler bend segments.
        """
        # Adjust taper length based on the bend width and waveguide width.
        tl: float = bend_taper if bend_width != w else 0.1
        # Calculate the minimum vertical length difference between the first input and output ports.
        l2_min: float = abs(
            component.ports["awin0"].dcenter[1] - component.ports["awout0"].dcenter[1]
        ) - 2 * (bend_radius + tl)
        pk: float = 0
        for k in range(N):
            # Use provided lengths if available, otherwise default to 10.
            l0: float = lengths[k] if k < len(lengths) else 10
            l2: float = l2_min + 2 * k * core_spacing
            l1: float = (
                (pk + m * 2 * math.pi) / 2 - k_bend * l0 - k_core * l2 / 2
            ) / k_core
            l1 = max(l1, 2 * tl + bend_radius)
            pk = k_bend * 2 * l0 + k_core * (2 * l1 + l2)

            # Construct the routing path using straight and Euler bend segments.
            wg_path = gf.path.straight(length=L0 + l1, npoints=1000)
            wg_path += gf.path.euler(angle=90, radius=bend_radius, npoints=100)
            wg_path += gf.path.straight(length=l2, npoints=1000)
            wg_path += gf.path.euler(angle=90, radius=bend_radius, npoints=100)
            wg_path += gf.path.straight(length=L0 + l1, npoints=1000)

            wg_cross_section = gf.cross_section.strip(width=w, layer=core_layer)
            wg_component = gf.path.extrude(wg_path, cross_section=wg_cross_section)
            # Offset each route vertically by the channel spacing.
            component.add_ref(wg_component).dmove((0, k * core_spacing))

    # Generate input side (top) components: FPR, waveguides, and IO tapers.
    gen_fpr(0, R / 2 + separation / 2)
    gen_wgs(0, R / 2 + separation / 2, out=False, radius=R / 3)
    gen_io(0, R / 2 + separation / 2, out=False)

    # Generate output side (bottom) components: FPR, waveguides, and IO tapers.
    gen_fpr(0, -R / 2 - separation / 2)
    gen_wgs(0, -R / 2 - separation / 2, out=True, radius=R / 3)
    gen_io(0, -R / 2 - separation / 2, out=True)

    # Generate the routes connecting the arms.
    gen_routes()

    return component


if __name__ == "__main__":
    # Create an AWG component with a quad taper profile and display it.
    c = awg(lengths=[10] * 40, taper_profile="quad")
    c.show()
