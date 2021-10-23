from typing import List

import numpy as np
from numpy import ndarray

from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.cross_section import strip
from gdsfactory.geometry.functions import path_length
from gdsfactory.routing.manhattan import (
    _is_horizontal,
    _is_vertical,
    remove_flat_angles,
)
from gdsfactory.types import ComponentFactory, CrossSectionFactory


def path_length_matched_points(
    list_of_waypoints: List[ndarray],
    margin: float = 0.0,
    modify_segment_i: int = -2,
    extra_length: float = 0.0,
    nb_loops: int = 1,
    bend: ComponentFactory = bend_euler,
    cross_section: CrossSectionFactory = strip,
    **kwargs,
) -> List[ndarray]:
    """
    Several types of paths won't match correctly.
    We do not try to handle all the corner cases here.
    You will need to modify the input list of waypoints in some cases.

    Args:
        list_of_waypoints:  [[p1, p2, p3,...], [q1, q2, q3,...], ...]
            the number of turns have to be identical
            (usually means same number of points.
            exception is if there are some flat angles)
        margin: some extra space to budget for in addition to the bend radius
            in most cases, the default is fine
        modify_segment_i: index of the segment which accomodates the new turns
            default is next to last segment (-2)
        extra_length: distance added to all path length compensation.
            Useful is we want to add space for extra taper on all branches
        nb_loops: number of extra loops added in the path
            if nb_loops==0, no extra loop is added, instead, in each route,
            the segment indexed by `modify_segment_i` is elongated to match
            the longuest route in `list_of_waypoints`
        bend: bend function
        cross_section: cross_section factory
        **kwargs

    Returns: another list of waypoints where
        - the path_lenth of each waypoints list are identical
        - the number of turns are identical

    """

    if nb_loops >= 1:
        return path_length_matched_points_add_waypoints(
            list_of_waypoints=list_of_waypoints,
            modify_segment_i=modify_segment_i,
            bend=bend,
            margin=margin,
            extra_length=extra_length,
            nb_loops=nb_loops,
            cross_section=cross_section,
            **kwargs,
        )
    else:
        return path_length_matched_points_modify_segment(
            list_of_waypoints=list_of_waypoints,
            modify_segment_i=modify_segment_i,
            extra_length=extra_length,
        )


def path_length_matched_points_modify_segment(
    list_of_waypoints,
    modify_segment_i,
    extra_length,
):
    if not isinstance(list_of_waypoints, list):
        raise ValueError(
            "list_of_waypoints should be a list, got {}".format(type(list_of_waypoints))
        )
    list_of_waypoints = [
        remove_flat_angles(waypoints) for waypoints in list_of_waypoints
    ]
    lengths = [path_length(waypoints) for waypoints in list_of_waypoints]
    L0 = max(lengths)

    N = len(list_of_waypoints[0])

    # Find how many turns there are per path
    nb_turns = [len(waypoints) - 2 for waypoints in list_of_waypoints]

    # The paths have to have the same number of turns, otherwise this algo
    # cannot path length match
    if min(nb_turns) != max(nb_turns):
        raise ValueError(
            f"Number of turns in paths have to be identical got {nb_turns}"
        )

    if modify_segment_i < 0:
        modify_segment_i = modify_segment_i + N + 1

    list_new_waypoints = []

    # For each list of waypoints, modify one segment in-place
    for i, waypoints in enumerate(list_of_waypoints):
        p_s0, p_s1, p_next = waypoints[modify_segment_i - 1 : modify_segment_i + 2]

        p_s0 = np.array(p_s0)
        p_s1 = np.array(p_s1)

        L = lengths[i]

        # Path length compensation length
        dL = (L0 - L) / 2

        # Additional fixed length
        dL = dL + extra_length

        # Modify the segment to accomodate for path length matching
        # Two cases: vertical or horizontal segment
        if _is_vertical(p_s0, p_s1):
            sx = np.sign(p_next[0] - p_s1[0])

            dx = -sx * dL
            dp = (dx, 0)
            # Sequence of displacements to apply

        elif _is_horizontal(p_s0, p_s1):
            sy = np.sign(p_next[1] - p_s1[1])

            dy = -sy * dL
            dp = (0, dy)

        waypoints[modify_segment_i - 1] = p_s0 + dp
        waypoints[modify_segment_i] = p_s1 + dp

        list_new_waypoints += [waypoints]
    return list_new_waypoints


def path_length_matched_points_add_waypoints(
    list_of_waypoints: List[ndarray],
    modify_segment_i: int = -2,
    bend: ComponentFactory = bend_euler,
    margin: float = 0.0,
    extra_length: float = 0.0,
    nb_loops: int = 1,
    cross_section: CrossSectionFactory = strip,
    **kwargs,
) -> List[ndarray]:
    """
    Args:
        list_of_waypoints: a list of list_of_points:
            [[p1, p2, p3,...], [q1, q2, q3,...], ...]
            - the number of turns have to be identical
                (usually means same number of points. exception is if there are
                some flat angles)
        modify_segment_i: index of the segment which accomodates the new turns
            default is next to last segment
        bend: for bends
        margin: some extra space to budget for in addition to the bend radius
            in most cases, the default is fine
        extra_length: distance added to all path length compensation.
            Useful is we want to add space for extra taper on all branches
        nb_loops: number of extra loops added in the path
        cross_section: factory
        **kwargs: cross_section settings

    returns:
        another list of waypoints where:
            - the path_lenth of each waypoints list are identical
            - the number of turns are identical

    Several types of paths won't match correctly. We do not try to handle
    all the corner cases here. If the paths are not well behaved,
    the input list_of_waypoints needs to be modified.

    To have flexibility in the path length, we need to add 4 bends
    One path has to be converted in this way:

    .. code::

             __
            |  |
            |  |  This length is adjusted to make all path with the same length
            |  |
         ___|  |___

    """

    if not isinstance(list_of_waypoints, list):
        raise ValueError(
            f"list_of_waypoints should be a list, got {type(list_of_waypoints)}"
        )
    list_of_waypoints = [
        remove_flat_angles(waypoints) for waypoints in list_of_waypoints
    ]
    lengths = [path_length(waypoints) for waypoints in list_of_waypoints]
    L0 = max(lengths)
    N = len(list_of_waypoints[0])

    # Find how many turns there are per path
    nb_turns = [len(waypoints) - 2 for waypoints in list_of_waypoints]

    # The paths have to have the same number of turns, otherwise cannot path-length
    # match with this algorithm
    if min(nb_turns) != max(nb_turns):
        raise ValueError(
            f"Number of turns in paths have to be identical got {nb_turns}"
        )

    # Get the points for the segment we need to modify
    bend90 = bend(cross_section=cross_section, **kwargs)

    a = margin + bend90.info.dy
    if modify_segment_i < 0:
        modify_segment_i = modify_segment_i + N + 1
    list_new_waypoints = []

    for i, waypoints in enumerate(list_of_waypoints):
        p_s0, p_s1, p_next = waypoints[modify_segment_i - 2 : modify_segment_i + 1]
        p_s1 = np.array(p_s1)
        L = lengths[i]

        # Path length compensation length
        dL = (L0 - L) / (2 * nb_loops)

        # Additional fixed length
        dL = dL + extra_length

        # Generate a new sequence of points which will replace this segment
        # Two cases: vertical or horizontal segment
        if _is_vertical(p_s0, p_s1):
            sx = np.sign(p_next[0] - p_s1[0])
            sy = np.sign(p_s1[1] - p_s0[1])

            dx = sx * (2 * a + dL)
            dy = sy * 2 * a

            # First new point to insert
            q0 = p_s1 + (0, -2 * nb_loops * dy)

            # Sequence of displacements to apply
            seq = [(dx, 0), (0, dy), (-dx, 0), (0, dy)] * nb_loops
            seq.pop()  # Remove last point to avoid flat angle with next point

        elif _is_horizontal(p_s0, p_s1):
            sy = np.sign(p_next[1] - p_s1[1])
            sx = np.sign(p_s1[0] - p_s0[0])

            dx = sx * 2 * a
            dy = sy * (2 * a + dL)

            # First new point to insert
            q0 = p_s1 + (-2 * dx * nb_loops, 0)

            # Sequence of displacements to apply
            seq = [(0, dy), (dx, 0), (0, -dy), (dx, 0)] * nb_loops
            seq.pop()  # Remove last point to avoid flat angle with next point

        # Generate points to insert
        qs = [q0]
        for dp in seq:
            qs += [qs[-1] + dp]

        inserted_points = np.stack(qs, axis=0)
        waypoints = np.array(waypoints)

        # Insert the points
        new_points = np.vstack(
            [
                waypoints[: modify_segment_i - 1],
                inserted_points,
                waypoints[modify_segment_i - 1 :],
            ]
        )
        list_new_waypoints += [new_points]

    return list_new_waypoints


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    c1 = c << gf.components.straight_array(n=4, spacing=50)
    c2 = c << gf.components.straight_array(n=4, spacing=20)
    c1.y = 0
    c2.y = 0
    c2.x = 300

    routes = gf.routing.get_bundle_path_length_match(
        c1.get_ports_list(prefix="E"),
        c2.get_ports_list(prefix="W"),
        radius=5,
        layer=(2, 0),
    )
    for route in routes:
        c.add(route.references)
    c.show()
