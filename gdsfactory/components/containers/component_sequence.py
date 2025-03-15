from __future__ import annotations

from collections import Counter

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import AngleInDegrees


class SequenceGenerator:
    def __init__(
        self,
        start_sequence: str = "IL",
        repeated_sequence: str = "ASASBSBS",
        end_sequence: str = "LO",
    ) -> None:
        """Sequence generator.

        Main use case: any type of cascade of components with repeating patterns
        such as serpentine, cutbacks etc...
        Component sequences have two ports by default.
        it adds aliases for the components forming the sequence.
        They use the component symbol with a suffix index starting from 1,
        so you may access the ports from any subcomponent.

        Usually we can break these components in 3 parts:
        - there is a starting pattern with input and possibly some special
        connections
        - then a repeating pattern
        - An ending pattern with an output

        Example of symbol meaning

        A: bend connected with input W0
        B: bend connected with input N0
        I: taper with input '1'
        O: taper with input '2'
        S: short straight waveguide
        L: long straight waveguide

        Args:
            start_sequence: starting sequence.
            end_sequence: ending sequence.
            repeated_sequence: repeating sequence.
        """
        self.start_sequence = start_sequence
        self.end_sequence = end_sequence
        self.repeated_sequence = repeated_sequence

    def get_sequence(self, n: int = 2) -> str:
        return self.start_sequence + n * self.repeated_sequence + self.end_sequence


def parse_component_name(name: str) -> tuple[str, bool]:
    """If the component name has more than one character and starts with "!".

    then we need to flip along the axis given by the input port angle.
    """
    return (name[1:], True) if len(name) != 1 and name[0] == "!" else (name, False)


def _flip_ref(c_ref: ComponentReference, port_name: str) -> ComponentReference:
    if port_name not in c_ref.ports:
        port_names = [port.name for port in c_ref.ports]
        raise ValueError(f"{port_name=} not in {c_ref.cell.name!r} {port_names}")
    a = c_ref.ports[port_name].orientation
    if a in [0, 180]:
        y = c_ref.ports[port_name].center[1]
        c_ref.dmirror_y(y)
    else:
        x = c_ref.ports[port_name].center[0]
        c_ref.dmirror_x(x)
    return c_ref


def component_sequence(
    sequence: str,
    symbol_to_component: dict[str, tuple[Component, str, str]],
    ports_map: dict[str, tuple[str, str]] | None = None,
    port_name1: str = "o1",
    port_name2: str = "o2",
    start_orientation: AngleInDegrees = 0.0,
) -> Component:
    """Returns component from ASCII sequence.

    if you prefix a symbol with ! it mirrors the component

    Args:
        sequence: a string or a list of symbols.
        symbol_to_component: maps symbols to (component, input, output).
        ports_map: (optional) extra port mapping using the convention.
            {port_name: (alias_name, port_name)}
        port_name1: input port_name.
        port_name2: output port_name.
        start_orientation: in degrees.

    Returns:
        component: containing the sequence of sub-components
            instantiated and connected together in the sequence order.

    .. plot::
        :include-source:

        import gdsfactory as gf

        bend180 = gf.components.bend_circular180()
        wg_pin = gf.components.straight_pin(length=40)
        wg = gf.components.straight()

        # Define a map between symbols and (component, input port, output port)
        symbol_to_component = {
            "A": (bend180, 'o1', 'o2'),
            "B": (bend180, 'o2', 'o1'),
            "H": (wg_pin, 'o1', 'o2'),
            "-": (wg, 'o1', 'o2'),
        }

        # Each character in the sequence represents a component
        s = "AB-H-H-H-H-BA"
        c = gf.components.component_sequence(sequence=s, symbol_to_component=symbol_to_component)
        c.plot()
    """
    ports_map = ports_map or {}
    named_references_counter: Counter[str] = Counter()
    component = Component()

    # Add first component reference and input port
    symbol = sequence[0] if "!" not in sequence[0] else sequence[:2]
    index = 2 if "!" in sequence[0] else 1
    name_start_device, do_flip = parse_component_name(symbol)
    component_input, input_port, prev_port = symbol_to_component[name_start_device]
    prev_device = component.add_ref(component_input, name=f"{symbol}{index}")
    named_references_counter.update({name_start_device: 1})

    if do_flip:
        prev_device = _flip_ref(prev_device, input_port)

    prev_device.drotate(angle=start_orientation)

    try:
        component.add_port(name=port_name1, port=prev_device.ports[input_port])
    except KeyError as exc:
        port_names = [port.name for port in prev_device.ports]
        raise KeyError(
            f"{prev_device.parent_cell.name!r} input_port {input_port!r} not in {port_names}"
        ) from exc

    ref: ComponentReference | None = None
    next_port: str | None = None

    while index < len(sequence):
        s = sequence[index]

        if s == "!":
            # if it's the last character skip
            if index + 1 >= len(sequence):
                index += 1
                continue
            s = sequence[index + 1]
            do_flip = True
            index += 1
        else:
            do_flip = False

        index += 1
        component_i, input_port, next_port = symbol_to_component[s]
        component_i = gf.get_component(component_i)
        named_references_counter.update({s: 1})
        alias = f"{s}{named_references_counter[s]}"
        ref = component.add_ref(component_i, name=alias)

        if do_flip:
            ref = _flip_ref(ref, input_port)

        try:
            ref.connect(input_port, prev_device.ports[prev_port])
        except KeyError as exc:
            port_names = [port.name for port in prev_device.ports]
            raise KeyError(
                f"{prev_device.parent_cell.name!r} port {prev_port!r} not in {port_names}"
            ) from exc

        prev_device = ref
        prev_port = next_port

    # Deal with edge case where the sequence contains only one component
    if len(sequence) == 1:
        ref = prev_device
        next_port = prev_port

    assert ref is not None
    assert next_port is not None

    component.add_port(name=port_name2, port=ref.ports[next_port])

    # Add any extra port specified in ports_map
    for name, (ref_name, alias_port_name) in ports_map.items():
        component.add_port(
            name=name, port=component.insts[ref_name].ports[alias_port_name]
        )

    return component


if __name__ == "__main__":
    import gdsfactory as gf

    bend180 = gf.components.bend_circular180()
    wg_pin = gf.components.straight_pin(length=40)
    wg = gf.components.straight()

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component_map = {
        "A": (bend180, "o1", "o2"),
        "B": (bend180, "o2", "o1"),
        "H": (wg_pin, "o1", "o2"),
        "-": (wg, "o1", "o2"),
    }

    # Each character in the sequence represents a component
    sequence = "AB"
    sequence = "HH!"
    sequence = "!HH"
    sequence = "H"
    sequence = "AB-H-H-H-H-BA"
    c = component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component_map
    )
    c.show()
    # n = c.get_netlist()
    # c = gf.read.from_yaml(n)

    # _ = c << gf.c.straight()
    # c.name = "top"
    # lyrdb = c.connectivity_check()
    # gf.show(c, lyrdb=lyrdb)
