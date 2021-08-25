import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler180
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.taper_from_csv import taper_0p5_to_3_l36
from gdsfactory.types import ComponentFactory


@gf.cell
def cutback_component(
    component: ComponentFactory = taper_0p5_to_3_l36,
    cols: int = 4,
    rows: int = 5,
    bend_radius: int = 10,
    port1: str = "o1",
    port2: str = "o2",
    middle_couples: int = 2,
) -> Component:
    """Flips the component, good for tapers that end in wide straights

    Args:
        component
        cols
        rows
        bend_radius: for bend

    """
    component = component() if callable(component) else component
    bend180 = bend_euler180(radius=bend_radius)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (component, port1, port2),
        "B": (component, port2, port1),
        "D": (bend180, "o1", "o2"),
        "C": (bend180, "o2", "o1"),
    }

    # Generate the sequence of staircases

    s = ""
    for i in range(rows):
        s += "AB" * cols
        s += "D" if i % 2 == 0 else "C"

    s = s[:-1]
    s += "AB" * middle_couples

    for i in range(rows):
        s += "AB" * cols
        s += "D" if (i + rows) % 2 == 0 else "C"

    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(sequence=s, symbol_to_component=symbol_to_component)
    c.update_settings(n_devices=len(s))
    return c


@gf.cell
def cutback_component_flipped(
    component: ComponentFactory = taper_0p5_to_3_l36,
    cols: int = 4,
    rows: int = 5,
    bend_radius: int = 10,
    port1: str = "o2",
    port2: str = "o1",
    middle_couples: int = 2,
) -> Component:
    component = component() if callable(component) else component
    bend180 = bend_euler180(radius=bend_radius)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (component, port1, port2),
        "B": (component, port2, port1),
        "D": (bend180, "o1", "o2"),
        "C": (bend180, "o2", "o1"),
    }

    # Generate the sequence of staircases

    s = ""
    for i in range(rows):
        s += "AB" * cols
        s += "C" if i % 2 == 0 else "D"

    s = s[:-1]
    s += "AB" * middle_couples

    for i in range(rows):
        s += "AB" * cols
        s += "D" if (i + rows + 1) % 2 == 0 else "C"

    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(sequence=s, symbol_to_component=symbol_to_component)
    c.update_settings(n_devices=len(s))
    return c


if __name__ == "__main__":
    c = cutback_component()
    c = cutback_component_flipped()
    c.show()
