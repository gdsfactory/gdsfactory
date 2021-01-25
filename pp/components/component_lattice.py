import itertools
from typing import Dict, List, Tuple

from numpy import float64

import pp
from pp.component import Component
from pp.components.coupler import coupler
from pp.components.crossing_waveguide import compensation_path, crossing45
from pp.config import GRID_PER_UNIT
from pp.port import get_ports_facing
from pp.routing.repackage import package_optical2x2

COUNTER = itertools.count()


def gen_tmp_port_name() -> str:
    return "{}".format(next(COUNTER))


def swap(waveguides, i, j):
    a = waveguides[i]
    waveguides[i] = waveguides[j]
    waveguides[j] = a
    return waveguides


def dist(i, wgs1, wgs2):
    a = wgs1.index(i)
    b = wgs2.index(i)
    return b - a


def get_sequence_cross(
    waveguides_start, waveguides_end, iter_max=100, symbols=["X", "-"]
):
    """
    Args:
        waveguides_start : list of the input port indices
        waveguides_end : list of the output port indices
        symbols : [`X` , `S`]
        symbols to be used in the returned sequence:
        `X`: represents the crossing symbol: two Xs next
            to each-other means that the two modes have to be swapped
        `S`: Straight waveguide, or compensation path typically

    Returns:
        sequence of crossings to achieve the permutations between two columns of I/O
    """
    wgs = list(waveguides_start)
    waveguides_end = list(waveguides_end)
    N = len(wgs)
    sequence = []
    X, S = symbols  # Cross, Straight symbols
    nb_iters = 0
    while wgs != waveguides_end:
        if nb_iters > iter_max:
            print(
                "Exceeded max number of iterations. The following I/O are mismatched:"
            )
            for i in range(len(waveguides_end)):
                print(wgs[i], "<->", waveguides_end[i])
            return sequence

        if nb_iters > 2 and sequence[-1] == sequence[-2]:
            print("Two consecutive sequences are the same. Got stuck")
            return sequence

        swaps = []
        i = 0
        total_dist = 0
        while i < N - 1:
            a = wgs[i]
            b = wgs[i + 1]
            d1 = dist(a, wgs, waveguides_end)
            d2 = dist(b, wgs, waveguides_end)
            total_dist += abs(d1) + abs(d2)

            """
            # The equality cases are very important:
            # if one waveguide needs to cross, then even if the other one is
            # already at the right place, it must swap to allow the other one
            # to cross
            """

            if d1 >= 0 and d2 <= 0 and not (d1 == 0 and d2 == 0):
                wgs = swap(wgs, i, i + 1)
                swaps += [X, X]
                i += 1
                # We cannot swap twice the same waveguide on the same iteration, so we
                # skip the next waveguide by incrementing

                # Edge case: Cannot swap if only one wg left so it has to be a straight
                if i == N - 2:
                    swaps += [S]
            else:
                # Edge case if only one wg remain it is straight
                swaps += [S]
                if i == N - 2:
                    swaps += [S]
            i += 1

        sequence.append(swaps)
        nb_iters += 1
    return sequence


def component_sequence_to_str(sequence):
    """
    Transform a sequence of components (such as the one obtained from
    get_sequence_cross_str) into an ASCII block which can be used either as
    a cartoon or as an input for component_lattice(lattice = ...)
    """
    component_txt_lattice = ""
    M = len(sequence[0])

    for i in range(M):
        j = M - 1 - i
        line = ""
        for col in sequence:
            line += col[j]
        line += "\n"
        component_txt_lattice += line

    return component_txt_lattice


def get_sequence_cross_str(waveguides_start, waveguides_end, iter_max=100):
    seq = get_sequence_cross(
        waveguides_start, waveguides_end, iter_max=iter_max, symbols=["X", "-"]
    )

    return component_sequence_to_str(seq)


@pp.cell(autoname=False)
def component_lattice(
    lattice: str = """
        C-X
        CXX
        CXX
        C-X
        """,
    components: None = None,
    name: str = "lattice",
) -> Component:
    """
    Returns a lattice Component of N inputs and outputs with components at given locations
    Columns must have components with the same x spacing between ports input/output ports
    Lines must have components with the same y spacing between input and output ports

    Lattice example:

    .. code::

        X-X
        XCX
        XCX
        X-X

    .. plot::
      :include-source:

      import pp
      from pp.routing.repackage import package_optical2x2
      from pp.components.crossing_waveguide import crossing45
      from pp.components.crossing_waveguide import compensation_path

      components =  {
            "C": package_optical2x2(component=pp.c.coupler(), port_spacing=40.0),
            "X": crossing45(port_spacing=40.0),
            "-": compensation_path(crossing45=crossing45(port_spacing=40.0)),
      }
      c = pp.c.component_lattice(components=components)
      pp.plotgds(c)

    """
    components = components or {
        "C": package_optical2x2(component=coupler(), port_spacing=40.0),
        "X": crossing45(port_spacing=40.0),
        "-": compensation_path(crossing45=crossing45(port_spacing=40.0)),
    }

    # Find y spacing and check that all components have same y spacing

    y_spacing = None
    for cmp in components.values():
        # cmp = pp.call_if_func(cmp)

        for direction in ["W", "E"]:
            ports_dir = get_ports_facing(cmp.ports, direction)
            ports_dir.sort(key=lambda p: p.y)
            nb_ports = len(ports_dir)
            if nb_ports > 1:
                _y_spacing = (ports_dir[-1].y - ports_dir[0].y) / (nb_ports - 1)
                if y_spacing is None:
                    y_spacing = _y_spacing
                else:
                    assert abs(y_spacing - _y_spacing) < 0.1 / GRID_PER_UNIT, (
                        "All component must have the same y port spacing. Got"
                        f" {y_spacing}, {_y_spacing} for {cmp.name}"
                    )

    a = y_spacing
    columns, columns_to_length = parse_lattice(lattice, components)
    keys = sorted(columns.keys())

    components_to_nb_input_ports = {}
    for c in components.keys():
        components_to_nb_input_ports[c] = len(get_ports_facing(components[c], "W"))

    component = pp.Component(name)
    x = 0
    for i in keys:
        col = columns[i]
        j = 0
        L = columns_to_length[i]
        skip = 0  # number of lines to skip depending on the number of ports
        for c in col:
            y = -j * a
            if skip == 1:
                j += skip
                skip = 0
                continue

            if c in components.keys():
                # Compute the number of ports to skip: They will already be
                # connected since they belong to this component

                nb_inputs = components_to_nb_input_ports[c]
                skip = nb_inputs - 1
                _cmp = components[c].ref((x, y), port_id="W{}".format(skip))
                component.add(_cmp)

                if i == 0:
                    _ports = get_ports_facing(_cmp, "W")
                    for _p in _ports:
                        component.add_port(gen_tmp_port_name(), port=_p)

                if i == keys[-1]:
                    _ports = get_ports_facing(_cmp, "E")
                    for _p in _ports:
                        component.add_port(gen_tmp_port_name(), port=_p)

            else:
                raise ValueError(
                    "component symbol {} is not part of                 components"
                    " dictionnary".format(c)
                )

            j += 1
        x += L

    component = pp.port.rename_ports_by_orientation(component)
    return component


def parse_lattice(
    lattice: str, components: Dict[str, Component]
) -> Tuple[Dict[int, List[str]], Dict[int, float64]]:
    # extract each column
    lines = lattice.replace(" ", "").split("\n")
    columns = {}
    columns_to_length = {}
    for line in lines:
        if len(line) > 0:
            i = 0
            for c in line:
                if i not in columns.keys():
                    columns[i] = []

                columns[i].append(c)
                if c in components.keys():
                    cmp = components[c]
                    columns_to_length[i] = cmp.ports["E0"].x - cmp.ports["W0"].x

                i += 1

    return columns, columns_to_length


if __name__ == "__main__":
    components = {
        "C": package_optical2x2(component=pp.c.coupler, port_spacing=40.0),
        "X": crossing45(port_spacing=40.0),
        "-": compensation_path(crossing45=crossing45(port_spacing=40.0)),
    }
    c = pp.c.component_lattice(components=components)
    c.pprint()
    pp.show(c)
