import pytest

import gdsfactory as gf
from gdsfactory.components.containers.component_sequence import (
    SequenceGenerator,
    _flip_ref,
    component_sequence,
    parse_component_name,
)


def test_sequence_generator_default() -> None:
    sg = SequenceGenerator()
    sequence = sg.get_sequence(2)
    assert sequence == "ILASASBSBSASASBSBSLO"


def test_sequence_generator_custom() -> None:
    sg = SequenceGenerator(start_sequence="A", repeated_sequence="BC", end_sequence="D")
    sequence = sg.get_sequence(3)
    assert sequence == "ABCBCBCD"


def test_parse_component_name() -> None:
    assert parse_component_name("!A") == ("A", True)
    assert parse_component_name("A") == ("A", False)
    assert parse_component_name("!AB") == ("AB", True)
    assert parse_component_name("AB") == ("AB", False)


def test_flip_ref() -> None:
    c = gf.components.straight()
    ref_component = gf.Component()
    ref = ref_component.add_ref(c)
    _flip_ref(ref, "o2")

    with pytest.raises(ValueError):
        _flip_ref(ref, "o3")

    b = gf.components.bend_euler()
    ref_component = gf.Component()
    ref = ref_component.add_ref(b)
    _flip_ref(ref, "o2")


def test_component_sequence() -> None:
    bend180 = gf.components.bend_circular180()
    wg_pin = gf.components.straight_pin(length=40)
    wg = gf.components.straight()

    symbol_to_component = {
        "A": (bend180, "o1", "o2"),
        "B": (bend180, "o2", "o1"),
        "H": (wg_pin, "o1", "o2"),
        "-": (wg, "o1", "o2"),
    }

    sequence = "AB-H-H-H-H-BA"
    c = component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)
    assert c is not None
    assert len(c.ports) > 0


def test_component_sequence_flip() -> None:
    bend180 = gf.components.bend_circular180()
    wg_pin = gf.components.straight_pin(length=40)
    wg = gf.components.straight()

    symbol_to_component = {
        "A": (bend180, "o1", "o2"),
        "B": (bend180, "o2", "o1"),
        "H": (wg_pin, "o1", "o2"),
        "-": (wg, "o1", "o2"),
    }

    sequence = "!H-H-H-H-!"
    c = component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)
    assert c is not None
    assert len(c.ports) > 0


def test_component_sequence_single_component() -> None:
    wg = gf.components.straight()
    symbol_to_component = {
        "-": (wg, "o1", "o2"),
    }
    sequence = "-"
    c = component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)
    assert c is not None
    assert len(c.ports) > 0


def test_component_sequence_with_ports_map() -> None:
    wg = gf.components.straight()
    symbol_to_component = {
        "-": (wg, "o1", "o2"),
    }
    sequence = "-"
    ports_map = {"extra_port": ("-1", "o1")}
    c = component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component, ports_map=ports_map
    )
    assert c is not None
    assert "extra_port" in c.ports


def test_component_sequence_errors() -> None:
    wg = gf.components.straight()
    symbol_to_component = {
        "-": (wg, "o3", "o1"),
    }
    sequence = "-"
    with pytest.raises(KeyError):
        component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)

    symbol_to_component = {
        "-": (wg, "o1", "o3"),
        "A": (wg, "o1", "o2"),
    }
    sequence = "-A"
    with pytest.raises(KeyError):
        component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)


def test_component_sequence_different_sequence() -> None:
    bend180 = gf.components.bend_circular180()
    wg_pin = gf.components.straight_pin(length=40)
    wg = gf.components.straight()
    symbol_to_component = {
        "A": (bend180, "o1", "o2"),
        "B": (bend180, "o2", "o1"),
        "H": (wg_pin, "o1", "o2"),
        "-": (wg, "o1", "o2"),
        "!": (wg, "o1", "o2"),
    }

    sequence = "!A-H-B-!B-!"
    c = component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)
    assert c is not None
    assert len(c.ports) > 0


if __name__ == "__main__":
    wg = gf.components.straight()
    symbol_to_component = {
        "-": (wg, "o1", "o3"),
        "A": (wg, "o1", "o2"),
    }
    sequence = "-A"
    component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)
    pytest.main([__file__, "-s"])
