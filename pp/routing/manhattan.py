import uuid
import numpy as np
from pp.components import waveguide
from pp.name import clean_name
from pp.component import Component, ComponentReference

import pp
from pp.geo_utils import angles_deg
from numpy import bool_, float64, ndarray
from typing import Callable, Dict, List, Optional, Tuple
from pp.port import Port

TOLERANCE = 0.0001
DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD

O2D = {0: "East", 180: "West", 90: "North", 270: "South"}


def _get_ports_facing(ports: Dict[str, Port], orientation: int = 0) -> List[Port]:
    return [p for p in ports.values() if p.orientation == orientation]


def _get_unique_port_facing(ports: Dict[str, Port], orientation: int = 0) -> List[Port]:
    ports = _get_ports_facing(ports, orientation)

    if len(ports) > 1:
        orientation = orientation % 360
        direction = O2D[orientation]
        raise ValueError(
            "_get_unique_port_facing: \n\
            should have only one port facing {}\n\
            Got {} with names {}".format(
                direction, len(ports), ports
            )
        )

    return ports


def _get_bend_ports(bend: Component) -> List[Port]:
    """
    Any standard bend/corner has two ports: one facing west and one facing north
    Returns these two ports in this order
    """

    ports = bend.ports

    p_w = _get_unique_port_facing(ports, 180)
    p_n = _get_unique_port_facing(ports, 90)

    return p_w + p_n


def _get_straight_ports(straight: Component) -> List[Port]:
    """
    Any standard straight wire/waveguide has two ports:
    one facing west and one facing east
    Returns these two ports in this order
    """
    ports = straight.ports

    p_w = _get_unique_port_facing(ports, 180)
    p_e = _get_unique_port_facing(ports, 0)

    return p_w + p_e


def gen_sref(
    structure: Component,
    rotation_angle: int,
    x_reflection: bool,
    port_name: str,
    position: ndarray,
) -> ComponentReference:
    """
    place sref of `port_name` of `structure` at `position`
    # Keep this convention, otherwise phidl port transform won't work
    # 1 ) Mirror
    # 2 ) Rotate
    # 3 ) Move
    """
    position = np.array(position)

    if port_name is None:
        port_position = np.array([0, 0])
    else:
        port_position = structure.ports[port_name].midpoint

    ref = pp.ComponentReference(component=structure, origin=(0, 0))

    if x_reflection:  # Vertical mirror: Reflection across x-axis
        y0 = port_position[1]
        ref.reflect(p1=(0, y0), p2=(1, y0))

    ref.rotate(rotation_angle, center=port_position)

    ref.move(port_position, position)

    return ref


def _is_vertical(p0: ndarray, p1: ndarray) -> bool_:
    return np.abs(p0[0] - p1[0]) < TOLERANCE


def _is_horizontal(p0: ndarray, p1: ndarray) -> bool_:
    return np.abs(p0[1] - p1[1]) < TOLERANCE


def get_straight_distance(p0: ndarray, p1: ndarray) -> float64:
    if _is_vertical(p0, p1):
        return np.abs(p0[1] - p1[1])
    if _is_horizontal(p0, p1):
        return np.abs(p0[0] - p1[0])

    raise ValueError("Waveguide {} {} is not manhattan".format(p0, p1))


def transform(
    points: ndarray,
    translation: ndarray = (0, 0),
    angle_deg: float64 = 0,
    x_reflection: bool = False,
) -> ndarray:
    """
    Args:
        points (np.array of shape (N,2) ): points to be transformed
        translation (2d like array): translation vector
        angle_deg (float): rotation angle
        x_reflection (bool): if True, mirror the shape across the x axis  (y -> -y)
    """
    # Copy
    pts = points[:, :]

    # Translate
    pts = pts + np.array(translation)

    # Rotate
    c = np.cos(DEG2RAD * angle_deg)
    s = np.sin(DEG2RAD * angle_deg)
    rotation_matrix = np.array([[c, s], [-s, c]])
    pts = np.dot(pts, rotation_matrix)

    # Mirror
    if x_reflection:
        pts[:, 1] = -pts[:, 1]

    return pts


def reverse_transform(
    points: ndarray,
    translation: ndarray = (0, 0),
    angle_deg: float64 = 0,
    x_reflection: bool = False,
) -> ndarray:
    """
    Args:
        points (np.array of shape (N,2) ): points to be transformed
        translation (2d like array): translation vector
        angle_deg (float): rotation angle
        x_reflection (bool): if True, mirror the shape across the x axis  (y -> -y)
    """
    angle_deg = -angle_deg

    # Copy
    pts = points[:, :]

    # Mirror
    if x_reflection:
        pts[:, 1] = -pts[:, 1]

    # Rotate
    c = np.cos(DEG2RAD * angle_deg)
    s = np.sin(DEG2RAD * angle_deg)
    rotation_matrix = np.array([[c, s], [-s, c]])
    pts = np.dot(pts, rotation_matrix)

    # Translate
    pts = pts - np.array(translation)

    return pts


def _generate_route_manhattan_points(
    input_port: Port,
    output_port: Port,
    bs1: float64,
    bs2: float64,
    start_straight: float = 0.01,
    end_straight: float = 0.01,
    min_straight: float = 0.01,
) -> ndarray:
    """
    Args:
        input_port:
        output_port:
        bs1, bs2: float, float : the bend size

    """

    threshold = TOLERANCE

    # transform I/O to the case where output is at (0, 0) pointing east (180)
    p_input = input_port.midpoint
    p_output = output_port.midpoint

    pts_io = np.stack([p_input, p_output], axis=0)

    angle = output_port.orientation

    a0 = -angle + 180
    transform_params = (-p_output, a0, False)

    _pts_io = transform(pts_io, *transform_params)
    p = _pts_io[0, :]
    _p_output = _pts_io[1, :]

    a = int(input_port.orientation + a0) % 360
    s = start_straight
    count = 0
    points = [p]

    while 1:
        count += 1
        if count > 40:
            raise AttributeError(
                "Too many iterations for in {} -> out {}".format(
                    input_port, output_port
                )
            )
        # not ready for final bend: go over possibilities
        sigp = np.sign(p[1])
        if not sigp:
            sigp = 1
        if a % 360 == 0:
            # same directions
            if abs(p[1]) < threshold and p[0] <= threshold:
                # Reach the output!
                points += [_p_output]
                break
            elif (
                p[0] + (bs1 + bs2 + end_straight + s) < threshold
                and abs(p[1]) - (bs1 + bs2 + min_straight) > -threshold
            ):
                # sufficient space for S-bend
                p = (-end_straight - bs2, p[1])
                a = -sigp * 90
            elif (
                p[0] + (2 * bs1 + 2 * bs2 + end_straight + s + min_straight) < threshold
            ):
                # sufficient distance to move aside
                p = (p[0] + s + bs1, p[1])
                a = -sigp * 90
            elif abs(p[1]) - (2 * bs1 + 2 * bs2 + 2 * min_straight) > -threshold:
                p = (p[0] + s + bs1, p[1])
                a = -sigp * 90
            else:
                p = (p[0] + s + bs1, p[1])
                a = sigp * 90

        elif a == 180:
            # opposite directions
            if abs(p[1]) - (bs1 + bs2 + min_straight) > -threshold:
                # far enough: U-turn
                p = (min(p[0] - s, -end_straight) - bs2, p[1])
                a = -sigp * 90
            else:
                # more complex turn
                p = (
                    min(p[0] - s - bs1, -end_straight - min_straight - 2 * bs1 - bs2),
                    p[1],
                )
                a = -sigp * 90
        elif a % 180 == 90:
            siga = -np.sign((a % 360) - 180)
            if not siga:
                siga = 1

            if ((-p[1] * siga) - (s + bs2) > -threshold) and (
                -p[0] - (end_straight + bs2)
            ) > -threshold:
                # simple case: one right angle to the end
                p = (p[0], 0)
                a = 0
            elif (p[1] * siga) <= threshold and p[0] + (
                end_straight + bs1
            ) > -threshold:
                # go to the west, and then turn upward
                # this will sometimes result in too sharp bends, but there is no avoiding this!

                _y = min(
                    max(min(min_straight, 0.5 * abs(p[1])), abs(p[1]) - s - bs1),
                    bs1 + bs2 + min_straight,
                )

                p = (p[0], sigp * _y)
                if count == 1:  # take care of the start_straight case
                    p = (p[0], -sigp * max(start_straight, _y))

                a = 180
            elif -p[0] - (end_straight + 2 * bs1 + bs2 + min_straight) > -threshold:
                # go sufficiently up, and then east
                p = (p[0], siga * max(p[1] * siga + s + bs1, bs1 + bs2 + min_straight))
                a = 0

            elif -p[0] - (end_straight + bs2) > -threshold:
                # make vertical S-bend to get sufficient room for movement
                points += [(p[0], p[1] + siga * (bs2 + s))]
                p = (
                    min(
                        p[0] - bs1 + bs2 + min_straight,
                        -2 * bs1 - bs2 - end_straight - min_straight,
                    ),
                    p[1] + siga * (bs2 + s),
                )
                # `a` remains the same
            else:
                # no viable solution for this case. May result in crossed waveguides
                p = (p[0], p[1] + sigp * (s + bs1))
                a = 180
        points += [p]
        s = min_straight + bs1

    points = np.stack([np.array(_p) for _p in points], axis=0)
    points = reverse_transform(points, *transform_params)
    return points


def _get_bend_reference_parameters(
    p0: ndarray, p1: ndarray, p2: ndarray, bend_cell: Component
) -> Tuple[ndarray, int, bool]:
    """
    8 possible configurations
    First mirror , Then rotate

       p2        p2
        |         |
        | C     A |
       p1-<-p0->-p1      dp1 horizontal
        |         |      dp2 vertical
        | D     B |
       p2        p2


       p2-<-p1->-p2
            |
          F |  E         dp1 vertical
            p0           dp2 horizontal
          H |  G
            |
       p2-<-p1->-p2
    """

    # is_horizontal(dp1), s1, s2 : transform (rotation, vertical mirror)
    transforms_map = {
        (True, 1, 1): (0, False),  # A No transform
        (True, 1, -1): (0, True),  # B Vertical mirror
        (True, -1, 1): (180, True),  # C Horizontal mirror
        (True, -1, -1): (180, False),  # D R180
        (False, 1, 1): (90, True),  # E R90 + vertical mirror
        (False, 1, -1): (90, False),  # F R270
        (False, -1, 1): (270, False),  # G R270
        (False, -1, -1): (270, True),  # H R270 + vertical mirror
    }

    b1, b2 = [p.midpoint for p in _get_bend_ports(bend_cell)]

    bsx = b2[0] - b1[0]
    bsy = b2[1] - b1[1]

    dp1 = p1 - p0
    dp2 = p2 - p1
    is_h_dp1 = np.abs(dp1[1]) < TOLERANCE

    if is_h_dp1:
        xd1 = dp1[0]
        yd2 = dp2[1]
        s1 = np.sign(xd1)
        s2 = np.sign(yd2)

        bend_origin = p1 - np.array([s1 * bsx, 0])

    else:
        yd1 = dp1[1]
        xd2 = dp2[0]
        s1 = int(np.sign(yd1))
        s2 = int(np.sign(xd2))

        bend_origin = p1 - np.array([0, s1 * bsy])

    t = transforms_map[(is_h_dp1, s1, s2)]

    return bend_origin, t[0], t[1]


def make_ref(component_factory: Callable) -> Callable:
    def _make_ref(*args, **kwargs):
        return component_factory(*args, **kwargs).ref()

    return _make_ref


def remove_flat_angles(points: ndarray) -> ndarray:
    a = angles_deg(np.vstack(points))
    da = a - np.roll(a, 1)
    da = np.mod(np.round(da, 3), 180)

    # To make sure we do not remove points at the edges
    da[0] = 1
    da[-1] = 1

    to_rm = list(np.where(np.abs(da[:-1]) < 1e-9)[0])
    if type(points) == list:
        while to_rm:
            i = to_rm.pop()
            points.pop(i)

    else:
        points = points[da != 0]

    return points


@make_ref
def round_corners(
    points,
    bend90,
    straight_factory,
    taper=None,
    straight_factory_fall_back_no_taper=None,
    mirror_straight=False,
    straight_ports=None,
):
    """
    returns a rounded waveguide route from a list of manhattan points
    Args:
        points: manhattan route defined by waypoints
        bend90: the bend to use for 90Deg turns
        straight_factory: the straight factory to use to generate straight portions
        taper: taper for straight portions. If None, no tapering
        straight_factory_fall_back_no_taper: factory to use for straights in case there is no space to put a pair of tapers
        straight_ports: port names for straights. If not specified, will use some heuristic to find them
    """
    ## If there is a taper, make sure its length is known
    if taper:
        if "length" not in taper.info:
            _taper_ports = list(taper.ports.values())
            taper.info["length"] = _taper_ports[-1].x - _taper_ports[0].x

    if straight_factory_fall_back_no_taper is None:
        straight_factory_fall_back_no_taper = straight_factory

    ## Remove any flat angle, otherwise the algorithm won't work
    points = remove_flat_angles(points)

    cell_tmp_name = "connector_{}".format(uuid.uuid4())

    cell = pp.Component(name=cell_tmp_name)

    points = np.array(points)

    straight_sections = []  # (p0, angle, length)
    p0_straight = points[0]
    p1 = points[1]

    total_length = 0  # Keep track of the total path length

    if "length" in bend90.info:
        bend_length = bend90.info["length"]
    else:
        bend_length = 0

    dp = p1 - p0_straight
    a0 = None
    if _is_vertical(p0_straight, p1):
        if dp[1] > 0:
            a0 = 90
        elif dp[1] < 0:
            a0 = 270
    elif _is_horizontal(p0_straight, p1):
        if dp[0] > 0:
            a0 = 0
        elif dp[0] < 0:
            a0 = 180

    assert a0 is not None, "Points should be manhattan, got {} {}".format(
        p0_straight, p1
    )

    pname_west, pname_north = [p.name for p in _get_bend_ports(bend90)]

    n_o_bends = points.shape[0] - 2
    total_length += n_o_bends * bend_length
    # Add bend sections and record straight-section information
    for i in range(1, points.shape[0] - 1):
        bend_origin, rotation, x_reflection = _get_bend_reference_parameters(
            points[i - 1], points[i], points[i + 1], bend90
        )

        bend_ref = gen_sref(bend90, rotation, x_reflection, pname_west, bend_origin)
        cell.add(bend_ref)

        straight_sections += [
            (p0_straight, a0, get_straight_distance(p0_straight, bend_origin))
        ]

        p0_straight = bend_ref.ports[pname_north].midpoint
        a0 = bend_ref.ports[pname_north].orientation

    straight_sections += [
        (p0_straight, a0, get_straight_distance(p0_straight, points[-1]))
    ]

    wg_refs = []
    for straight_origin, angle, length in straight_sections:
        with_taper = False
        wg_width = list(bend90.ports.values())[0].width

        total_length += length

        if taper is not None and length > 2 * taper.info["length"] + 1.0:
            length = length - 2 * taper.info["length"]
            with_taper = True

        if with_taper:

            # First taper:
            # Taper starts where straight would have started
            taper_origin = straight_origin

            pname_west, pname_east = [p.name for p in _get_straight_ports(taper)]
            taper_ref = taper.ref(
                position=taper_origin, port_id=pname_west, rotation=angle
            )

            wg_width = taper.ports[pname_east].width

            cell.add(taper_ref)
            wg_refs += [taper_ref]

            # Update start straight position
            straight_origin = taper_ref.ports[pname_east].midpoint

        # Straight waveguide
        if with_taper or taper is None:
            wg = straight_factory(length=length, width=wg_width)
        else:
            wg = straight_factory_fall_back_no_taper(length=length, width=wg_width)

        if straight_ports is None:
            straight_ports = [p.name for p in _get_straight_ports(wg)]
        pname_west, pname_east = straight_ports

        wg.move(wg.ports[pname_west], (0, 0))
        wg_ref = pp.ComponentReference(wg)
        if mirror_straight:
            wg_ref.reflect_v(list(wg_ref.ports.values())[0].name)

        wg_ref.rotate(angle)
        wg_ref.move(straight_origin)
        cell.add(wg_ref)
        wg_refs += [wg_ref]

        port_index_out = 1
        if with_taper:
            # Second taper:
            # Origin at end of straight waveguide, starting from east side of taper

            taper_origin = wg_ref.ports[pname_east]
            pname_west, pname_east = [p.name for p in _get_straight_ports(taper)]
            taper_ref = taper.ref(
                position=taper_origin, port_id=pname_east, rotation=angle + 180
            )

            cell.add(taper_ref)
            wg_refs += [taper_ref]
            port_index_out = 0

    cell.add_port(name="input", port=list(wg_refs[0].ports.values())[0])
    cell.add_port(name="output", port=list(wg_refs[-1].ports.values())[port_index_out])

    """
    # Update name with uuid - too expensive to compute geometrical hash every time
    # Prefix with zz to make connectors appear at end of cell lists

    # The geometrical hash lacks caching right now and ends up taking a
    # lot of time to compute on every single connector
    """

    cell.name = f"zz_conn_{clean_name(str(uuid.uuid4()))[:16]}"
    cell.info["length"] = total_length
    cell.length = total_length
    return cell


def generate_manhattan_waypoints(
    input_port: Port,
    output_port: Port,
    bend90: Optional[Component] = None,
    bend_radius: None = None,
    start_straight: float = 0.01,
    end_straight: float = 0.01,
    min_straight: float = 0.01,
    **kwargs,
) -> ndarray:
    """

    """

    if bend90 is None and bend_radius is None:
        raise ValueError(
            "Either bend90 or bend_radius must be set. \
        Got {} {}".format(
                bend90, bend_radius
            )
        )

    if bend90 is not None and bend_radius is not None:
        raise ValueError(
            "Either bend90 or bend_radius must be set. \
        Got {} {}".format(
                bend90, bend_radius
            )
        )

    if bend90:
        pname_west, pname_north = [p.name for p in _get_bend_ports(bend90)]
        p1 = bend90.ports[pname_west].midpoint
        p2 = bend90.ports[pname_north].midpoint

        bsx = p2[0] - p1[0]
        bsy = p2[1] - p1[1]

    elif bend_radius:
        bsx = bend_radius
        bsy = bend_radius

    points = _generate_route_manhattan_points(
        input_port, output_port, bsx, bsy, start_straight, end_straight, min_straight
    )
    return points


def route_manhattan(
    input_port: Port,
    output_port: Port,
    bend90: Component,
    straight_factory: Callable,
    taper: None = None,
    start_straight: float = 0.01,
    end_straight: float = 0.01,
    min_straight: float = 0.01,
) -> ComponentReference:
    bend90 = pp.call_if_func(bend90)

    points = generate_manhattan_waypoints(
        input_port,
        output_port,
        bend90=bend90,
        start_straight=start_straight,
        end_straight=end_straight,
        min_straight=min_straight,
    )
    return round_corners(points, bend90, straight_factory, taper)


def test_manhattan():
    from pp.components.bend_circular import bend_circular

    top_cell = pp.Component()

    inputs = [
        Port("in1", (10, 5), 0.5, 90),
        Port("in2", (-10, 20), 0.5, 0),
        Port("in3", (10, 30), 0.5, 0),
        Port("in4", (-10, -5), 0.5, 90),
    ]

    outputs = [
        Port("in1", (90, -60), 0.5, 180),
        Port("in2", (-100, 20), 0.5, 0),
        Port("in3", (100, -25), 0.5, 0),
        Port("in4", (-150, -65), 0.5, 270),
    ]

    for input_port, output_port in zip(inputs, outputs):

        # input_port = Port("input_port", (10,5), 0.5, 90)
        # output_port = Port("output_port", (90,-60), 0.5, 180)

        bend = bend_circular(radius=5.0)
        cell = route_manhattan(
            input_port,
            output_port,
            bend,
            waveguide,
            start_straight=5.0,
            end_straight=5.0,
        )

        top_cell.add(cell)
    return top_cell


if __name__ == "__main__":
    top_cell = test_manhattan()
    pp.show(top_cell)
