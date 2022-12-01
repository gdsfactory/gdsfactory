from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

from numpy import float64

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.coupler import coupler
from gdsfactory.components.crossing_waveguide import compensation_path, crossing45
from gdsfactory.port import get_ports_facing

COUNTER = itertools.count()


def gen_tmp_port_name() -> str:
    return f"{next(COUNTER)}"


def swap(straights, i, j):
    a = straights[i]
    straights[i] = straights[j]
    straights[j] = a
    return straights


def dist(i, wgs1, wgs2):
    a = wgs1.index(i)
    b = wgs2.index(i)
    return b - a


def get_sequence_cross(
    straights_start, straights_end, iter_max: int = 100, symbols=("X", "-")
):
    """Returns sequence of crossings to achieve the permutations between two columns of I/O.

    Args:
        straights_start: list of the input port indices.
        straights_end: list of the output port indices.
        iter_max: maximum iterations.
        symbols: [`X` , `S`].

    Notes:
        symbols to be used in the returned sequence:
        - `X`:represents the crossing symbol: two Xs next to each-other means that the two
            modes have to be swapped
        - `S`:Straight straight, or compensation path typically

    """
    wgs = list(straights_start)
    straights_end = list(straights_end)
    N = len(wgs)
    sequence = []
    X, S = symbols  # Cross, Straight symbols
    nb_iters = 0
    while wgs != straights_end:
        if nb_iters > iter_max:
            print(
                "Exceeded max number of iterations. The following I/O are mismatched:"
            )
            for i in range(len(straights_end)):
                print(wgs[i], "<->", straights_end[i])
            return sequence

        if nb_iters > 2 and sequence[-1] == sequence[-2]:
            print("Two consecutive sequences are the same. Got stuck")
            return sequence

        swaps = []
        i = 0
        # total_dist = 0

        while i < N - 1:
            a = wgs[i]
            b = wgs[i + 1]
            d1 = dist(a, wgs, straights_end)
            d2 = dist(b, wgs, straights_end)
            # total_dist += abs(d1) + abs(d2)

            # The equality cases are very important:
            # if one straight needs to cross, then even if the other one is
            # already at the right place, it must swap to allow the other one
            # to cross

            if d1 >= 0 and d2 <= 0 and (d1 != 0 or d2 != 0):
                wgs = swap(wgs, i, i + 1)
                swaps += [X, X]
                i += 1
            else:
                # Edge case if only one wg remain it is straight
                swaps += [S]
            # We cannot swap twice the same straight on the same iteration, so we
            # skip the next straight by incrementing

            # Edge case: Cannot swap if only one wg left so it has to be a straight
            if i == N - 2:
                swaps += [S]
            i += 1

        sequence.append(swaps)
        nb_iters += 1
    return sequence


def component_sequence_to_str(sequence):
    """Transform a sequence of components (such as the one obtained from.

    get_sequence_cross_str) into an ASCII block which can be used either as
    a cartoon or as an input for component_lattice(lattice = ...)
    """
    component_txt_lattice = ""
    M = len(sequence[0])

    for i in range(M):
        j = M - 1 - i
        line = "".join(col[j] for col in sequence) + "\n"
        component_txt_lattice += line

    return component_txt_lattice


def get_sequence_cross_str(straights_start, straights_end, iter_max: int = 100):
    seq = get_sequence_cross(
        straights_start, straights_end, iter_max=iter_max, symbols=["X", "-"]
    )

    return component_sequence_to_str(seq)


@gf.cell
def component_lattice(
    lattice: str = """
        C-X
        CXX
        CXX
        C-X
        """,
    symbol_to_component: Dict[str, Component] = None,
    grid_per_unit: int = 1000,
) -> Component:
    """Return a lattice Component of N inputs and outputs Columns must have.

    components with the same x spacing between input/output ports Lines must
    have components with the same y spacing between input/output ports.

    Args:
        lattice: ASCII map with character.
        symbol_to_component: dict of ASCII character to component.
        grid_per_unit: int.

    Lattice example:

    .. code::

        X-X
        XCX
        XCX
        X-X

    .. plot::
      :include-source:

      import gdsfactory as gf
      from gdsfactory.components.crossing_waveguide import crossing45
      from gdsfactory.components.crossing_waveguide import compensation_path

      symbol_to_component =  {
            "C": gf.routing.fanout2x2(component=gf.components.coupler(), port_spacing=40.0),
            "X": crossing45(port_spacing=40.0),
            "-": compensation_path(crossing45=crossing45(port_spacing=40.0)),
      }
      c = gf.components.component_lattice(symbol_to_component=symbol_to_component)
      c.plot()
    """
    x = crossing45(port_spacing=40)
    symbol_to_component = symbol_to_component or {
        "C": gf.routing.fanout2x2(component=coupler(), port_spacing=40.0),
        "X": x,
        "-": compensation_path(x),
    }

    # Find y spacing and check that all components have same y spacing
    y_spacing = None
    for component in symbol_to_component.values():
        component = gf.get_component(component)
        # component = component.copy()
        # component.auto_rename_ports_orientation()

        for direction in ["W", "E"]:
            ports_dir = get_ports_facing(component.ports, direction)
            ports_dir.sort(key=lambda p: p.y)
            nb_ports = len(ports_dir)
            if nb_ports > 1:
                _y_spacing = (ports_dir[-1].y - ports_dir[0].y) / (nb_ports - 1)
                if y_spacing is None:
                    y_spacing = _y_spacing
                else:
                    assert abs(y_spacing - _y_spacing) < 0.1 / grid_per_unit, (
                        "All component must have the same y port spacing. Got"
                        f" {y_spacing}, {_y_spacing} for {component.name}"
                    )

    a = y_spacing
    columns, columns_to_length = parse_lattice(lattice, symbol_to_component)
    keys = sorted(columns.keys())

    components_to_nb_input_ports = {
        c: len(get_ports_facing(symbol_to_component[c], "W"))
        for c in symbol_to_component.keys()
    }

    component = gf.Component()
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

            if c in symbol_to_component.keys():
                # Compute the number of ports to skip: They will already be
                # connected since they belong to this component

                nb_inputs = components_to_nb_input_ports[c]
                skip = nb_inputs - 1

                ports_cw = symbol_to_component[c].get_ports_list(clockwise=True)
                _cmp = symbol_to_component[c].ref((x, y), port_id=ports_cw[skip].name)

                # _cmp = symbol_to_component[c].ref((x, y), port_id="oW{}".format(skip))
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
                symbols = list(symbol_to_component.keys())
                raise ValueError(
                    f"symbol {c!r} not in symbol_to_component dict {symbols}"
                )

            j += 1
        x += L

    component.auto_rename_ports()
    return component


def parse_lattice(
    lattice: str, symbol_to_component: Dict[str, Component]
) -> Tuple[Dict[int, List[str]], Dict[int, float64]]:
    """Extract each column.

    Args:
        lattice: string describing lattice.
        symbol_to_component: dict of ASCII character to component.
    """
    lines = lattice.replace(" ", "").split("\n")
    columns = {}
    columns_to_length = {}
    for line in lines:
        if len(line) > 0:
            for i, c in enumerate(line):
                if i not in columns.keys():
                    columns[i] = []

                columns[i].append(c)
                if c in symbol_to_component:
                    cmp = symbol_to_component[c]
                    pcw = cmp.get_ports_list(clockwise=True)
                    pccw = cmp.get_ports_list(clockwise=False)

                    # columns_to_length[i] = cmp.ports["oE0"].x - cmp.ports["oW0"].x
                    columns_to_length[i] = (
                        cmp.ports[pccw[0].name].x - cmp.ports[pcw[0].name].x
                    )

    return columns, columns_to_length


if __name__ == "__main__":
    components_dict = {
        "C": gf.routing.fanout2x2(component=gf.components.coupler(), port_spacing=40.0),
        "X": crossing45(port_spacing=40.0),
        "-": compensation_path(crossing45=crossing45(port_spacing=40.0)),
    }
    c = gf.components.component_lattice(symbol_to_component=components_dict)
    # c= gf.routing.fanout2x2(component=gf.components.coupler(), port_spacing=40.0)
    # c= crossing45(port_spacing=40.0)
    # c = compensation_path(crossing45=crossing45(port_spacing=40.0))
    c.pprint_ports()
    c.show(show_ports=True)
