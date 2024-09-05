"""Sample AWG."""

import math

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import LayerSpec


def locToWorld(xyp, x0, y0, angle):
    def _locToWorld(xp, yp):
        x = x0 - xp * math.cos(angle) + yp * math.sin(angle)
        y = y0 + xp * math.sin(angle) + yp * math.cos(angle)
        return (x, y)

    if not isinstance(xyp, list):
        return _locToWorld(xyp[0], xyp[1])

    xy = []
    for xp, yp in xyp:
        xy.append(_locToWorld(xp, yp))

    return xy


def create_awg(
    N=40,
    R=100,
    g=100e-3,
    d=1.5,
    w=800e-3,
    m=20,
    dl=10,
    Ni=1,
    di=3,
    L0=0,
    li=0,
    wi=1.5,
    No=1,
    do=3,
    lo=0,
    wo=1.5,
    k_core=1,
    k_bend=1,
    inputs=None,
    outputs=None,
    confocal=False,
    lengths=None,
    defocus=0,
    taper_profile="lin",
    taper_length=40,
    taper_points=None,
    core_spacing=2.0,
    bend_radius=5.0,
    bend_width=500e-3,
    bend_taper=15,
    dummy_apertures=0,
    routing_width=500e-3,
    core_layer: LayerSpec = "WG",
    separation=50,
    limit_fpr_size=False,
):
    if taper_points is None:
        taper_points = []
    if lengths is None:
        lengths = []
    if outputs is None:
        outputs = []
    if inputs is None:
        inputs = []
    component = gf.Component("AWG")

    def gen_fpr(x0, y0):
        xy = []
        angle = math.pi
        if limit_fpr_size:
            angle = 3 / 2 * (N + 8) * d / R

        for a in np.linspace(-angle / 2, angle / 2, 100):
            xy.append((R * (1 - math.cos(a)), R * math.sin(a)))

        angle = math.pi / 3
        if limit_fpr_size:
            angle = (N + 8) * d / R

        for a in np.linspace(angle / 2, -angle / 2, 100):
            xy.append((R * math.cos(a), R * math.sin(a)))

        xy = np.array(xy) + np.array([x0, y0])
        component.add_polygon(xy, layer=core_layer)

    def gen_io(x0, y0, out=False, radius=25, points=20):
        b = R * 0.02
        N_io = Ni if not out else No
        d_io = di if not out else do
        l_io = li if not out else -lo
        w_io = wi if not out else wo

        pos = [(l_io + (k - (N_io - 1) / 2) * max(d_io, w_io)) / R for k in range(N_io)]
        if out and outputs:
            pos = [x / R for x in outputs]
        elif not out and inputs:
            pos = [x / R for x in inputs]

        N_io = len(pos)

        xmax = -taper_length - radius / 5
        for k in range(N_io):
            a = pos[k]
            xc = R - (R + defocus) * math.cos(a)
            yc = R * math.sin(a)
            xy = []

            xy1 = [(-b, w_io / 2), (0, w_io / 2), (taper_length, routing_width / 2)]
            xy.extend(locToWorld(xy1, xc, yc, a / 2))

            h = 1 if a > 0 else -1
            xy1 = [
                (
                    taper_length + radius * math.sin(t),
                    h * (radius + h * routing_width / 2 - radius * math.cos(t)),
                )
                for t in np.linspace(0, abs(a / 2), points)
            ]
            xy.extend(locToWorld(xy1, xc, yc, a / 2))

            xy.append((xmax, xy[-1][1]))
            name = f"in{k}" if not out else f"out{N_io - 1 - k}"
            component.add_port(
                name=name,
                center=(x0 + xmax, y0 + xy[-1][1] - routing_width / 2),
                layer=core_layer,
                width=routing_width,
                orientation=0,
            )
            xy.append((xmax, xy[-1][1] - routing_width))

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

    def gen_wgs(x0, y0, out=False, radius=25, points=20):
        b = R * 0.01
        w = d - g
        p = bend_width

        yp = (N - 1) / 2 * core_spacing
        tp = (N - 1) / 2 * d / R
        lp = (
            yp - radius * (1 - math.cos(tp)) - (R + taper_length) * math.sin(tp)
        ) / math.sin(tp)
        xp = (R + taper_length + lp + p) * math.cos(tp) + (radius + p) * math.sin(tp)

        if taper_profile == "lin":
            taper_profile_up = [(0, +w / 2), (b, +w / 2), (taper_length + b, +p / 2)]
            taper_profile_dn = [(taper_length + b, -p / 2), (b, -w / 2), (0, -w / 2)]
        elif taper_profile == "quad":
            x, y = np.linspace(0, taper_length, 100), np.linspace(w, p, 100)
            taper_profile_up = [(0, +w / 2)] + [
                (x[i] + b, y[i] / 2) for i in range(len(x))
            ]
            taper_profile_dn = [
                (x[i] + b, -y[i] / 2) for i in range(len(x) - 1, -1, -1)
            ] + [(0, -w / 2)]
        elif taper_profile == "custom":
            taper_profile_up = [(0, +w / 2)] + [(p[0] + b, p[1]) for p in taper_points]
            taper_profile_dn = [(p[0] + b, -p[1]) for p in reversed(taper_points)] + [
                (0, -w / 2)
            ]
        else:
            raise ValueError(f"Unknown taper profile type '{taper_profile}'")

        for k in range(-dummy_apertures, N + dummy_apertures):
            t = (k - (N - 1) / 2) * d / R

            h = 1 if t >= 0 else -1
            yp = (k - (N - 1) / 2) * core_spacing
            lp = (
                yp - h * radius * (1 - math.cos(t)) - (R + taper_length) * math.sin(t)
            ) / math.sin(t)

            xc = (R - b) * math.cos(t)
            yc = (R - b) * math.sin(t)

            xy = []

            if k < 0 or k > N - 1:
                xy = [
                    (-b, w / 2),
                    (0, w / 2),
                    (taper_length + b, 0.3 / 2),
                    (taper_length + b, -0.3 / 2),
                    (0, -w / 2),
                    (-b, -w / 2),
                ]
                xy = locToWorld(xy, xc, yc, math.pi - t)
            else:
                xy_bend_top = locToWorld(taper_profile_up, xc, yc, math.pi - t)
                pts = [
                    (
                        b + taper_length + lp + radius * math.sin(a),
                        h * (radius - radius * math.cos(a)),
                    )
                    for a in np.linspace(0, t, points)
                ]
                xy_bend_top.extend(locToWorld(pts, xc, yc, math.pi - t))
                xy_bend_top.append((xp, xy_bend_top[-1][1]))
                xy_bend_top.append((xp + 0.1, xy_bend_top[-1][1]))
                xy.extend(xy_bend_top)

                name = f"awin{k}" if not out else f"awout{N - 1 - k}"
                component.add_port(
                    name=name,
                    center=(x0 + xp, y0 + xy[-1][1] + p / 2),
                    layer=core_layer,
                    width=p,
                    orientation=0,
                )

                xy_bend_bot = [
                    (xp + 0.1, xy_bend_top[-1][1] + p),
                    (xp, xy_bend_top[-1][1] + p),
                ]
                pts = [
                    (
                        b + taper_length + lp + radius * math.sin(a),
                        h * (radius - radius * math.cos(a)),
                    )
                    for a in np.linspace(t, 0, points)
                ]
                xy_bend_bot.extend(locToWorld(pts, xc, yc, math.pi - t))
                xy_bend_bot.extend(locToWorld(taper_profile_dn, xc, yc, math.pi - t))
                xy.extend(xy_bend_bot)

            component.add_polygon(xy, layer=core_layer)

    gen_fpr(0, R / 2 + separation / 2)
    gen_wgs(0, R / 2 + separation / 2, out=False, radius=R / 3)
    gen_io(0, R / 2 + separation / 2, out=False)

    # gen_fpr(0, -R / 2 - separation / 2)
    # gen_wgs(0, -R / 2 - separation / 2, out=True, radius=R / 3)
    # gen_io(0, -R / 2 - separation / 2, out=True)

    # tl = bend_taper if bend_width != w else 0.1
    # l2_min = abs(
    #     component.ports[f"awin{0}"].dcenter[1]
    #     - component.ports[f"awout{0}"].dcenter[1]
    # ) - 2 * (bend_radius + tl)
    # pk = 0
    # for k in range(N):
    #     l0 = lengths[k]
    #     l2 = l2_min + 2 * k * core_spacing
    #     l1 = ((pk + m * 2 * math.pi) / 2 - k_bend * l0 - k_core * l2 / 2) / k_core
    #     l1 = max(l1, 2 * tl + bend_radius)
    #     pk = k_bend * 2 * l0 + k_core * (2 * l1 + l2)

    #     wg_path = gf.path.straight(length=L0 + l1, npoints=1000)
    #     wg_path += gf.path.euler(angle=90, radius=bend_radius, npoints=100)
    #     wg_path += gf.path.straight(length=l2, npoints=1000)
    #     wg_path += gf.path.euler(angle=90, radius=bend_radius, npoints=100)
    #     wg_path += gf.path.straight(length=L0 + l1, npoints=1000)

    #     wg_cross_section = gf.cross_section.strip(width=w, layer=core_layer)
    #     wg_component = gf.path.extrude(wg_path, cross_section=wg_cross_section)
    #     component.add_ref(wg_component).dmove((0, k * core_spacing))

    return component


if __name__ == "__main__":
    c = create_awg(lengths=[10] * 40, taper_profile="quad")
    c.show()
