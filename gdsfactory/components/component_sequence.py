from __future__ import annotations

from collections import Counter

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


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
        """
        self.start_sequence = start_sequence
        self.end_sequence = end_sequence
        self.repeated_sequence = repeated_sequence

    def get_sequence(self, n=2):
        return self.start_sequence + n * self.repeated_sequence + self.end_sequence


def parse_component_name(name: str) -> tuple[str, bool]:
    """If the component name has more than one character and starts with "!".

    then we need to flip along the axis given by the input port angle.
    """
    return (name[1:], True) if len(name) != 1 and name[0] == "!" else (name, False)


def _flip_ref(c_ref, port_name):
    a = c_ref.ports[port_name].orientation
    if a in [0, 180]:
        c_ref.mirror_y(port_name)
    else:
        c_ref.mirror_x(port_name)
    return c_ref


@gf.cell
def component_sequence(
    sequence: str,
    symbol_to_component: dict[str, tuple[ComponentSpec, str, str]],
    ports_map: dict[str, tuple[str, str]] | None = None,
    port_name1: str = "o1",
    port_name2: str = "o2",
    start_orientation: float = 0.0,
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
    named_references_counter = Counter()
    ports_map = ports_map or {}

    component = Component()

    # Add first component reference and input port
    symbol = sequence[0] if "!" not in sequence[0] else sequence[:2]
    index = 2 if "!" in sequence[0] else 1
    name_start_device, do_flip = parse_component_name(symbol)
    component_input, input_port, prev_port = symbol_to_component[name_start_device]

    named_references_counter.update({name_start_device: 1})
    alias = f"{name_start_device}{named_references_counter[name_start_device]}"
    prev_device = component.add_ref(component_input, alias=alias)

    if do_flip:
        prev_device = _flip_ref(prev_device, input_port)

    prev_device.rotate(angle=start_orientation)

    try:
        component.add_port(name=port_name1, port=prev_device.ports[input_port])
    except KeyError as exc:
        raise KeyError(
            f"{prev_device.parent.name!r} input_port {input_port!r} "
            f"not in {list(prev_device.ports.keys())}"
        ) from exc

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
        ref = component.add_ref(component_i, alias=alias)

        if do_flip:
            ref = _flip_ref(ref, input_port)

        try:
            ref.connect(input_port, prev_device.ports[prev_port])
        except KeyError as exc:
            raise KeyError(
                f"{prev_device.parent.name!r} port {prev_port!r} "
                f"not in {list(prev_device.ports.keys())}"
            ) from exc

        prev_device = ref
        prev_port = next_port

    # Deal with edge case where the sequence contains only one component
    if len(sequence) == 1:
        ref = prev_device
        next_port = prev_port

    try:
        component.add_port(name=port_name2, port=ref.ports[next_port])
    except BaseException:
        print(sequence)
        raise

    # Add any extra port specified in ports_map
    for name, (alias, alias_port_name) in ports_map.items():
        component.add_port(name=name, port=component[alias].ports[alias_port_name])

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
    c = gf.components.component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component_map
    )
    # n = c.get_netlist()
    # c = gf.read.from_yaml(n)
    c.show(show_ports=True)
    # c.pprint()
    # print(c.named_references.keys())
