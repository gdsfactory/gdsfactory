import pp
from pp.component import Component
from pp.components import bend_euler180
from pp.components.component_sequence import component_sequence
from pp.components.taper_from_csv import taper_0p5_to_3_l36
from pp.types import ComponentFactory


@pp.cell
def cutback_component(
    component: ComponentFactory = taper_0p5_to_3_l36,
    cols: int = 4,
    rows: int = 5,
    bend_radius: int = 10,
    port1_id: str = "W0",
    port2_id: str = "E0",
    middle_couples: int = 2,
) -> Component:
    """Flips the component, good for tapers that end in wide straights

    Args:
        component
        cols
        rows

    """
    component = component() if callable(component) else component
    bend180 = bend_euler180(radius=bend_radius)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (component, port1_id, port2_id),
        "B": (component, port2_id, port1_id),
        "D": (bend180, "W0", "W1"),
        "C": (bend180, "W1", "W0"),
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


@pp.cell
def cutback_component_flipped(
    component: ComponentFactory = taper_0p5_to_3_l36,
    cols: int = 4,
    rows: int = 5,
    bend_radius: int = 10,
    port1_id: str = "E0",
    port2_id: str = "W0",
    middle_couples: int = 2,
) -> Component:
    component = component() if callable(component) else component
    bend180 = bend_euler180(radius=bend_radius)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (component, port1_id, port2_id),
        "B": (component, port2_id, port1_id),
        "D": (bend180, "W0", "W1"),
        "C": (bend180, "W1", "W0"),
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
    c.show()
