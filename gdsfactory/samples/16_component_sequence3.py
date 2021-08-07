import gdsfactory as gf


@gf.cell
def test_cutback_phase(straight_length=100, bend_radius=10.0, n=2):
    bend180 = gf.components.bend_circular180(radius=bend_radius)
    pm_wg = gf.components.straight_pin(length=straight_length)
    wg_short = gf.components.straight(length=1.0)
    wg_short2 = gf.components.straight(length=2.0)
    wg_heater = gf.components.straight_heater(length=10.0)
    taper = gf.components.taper_strip_to_ridge()

    # Define a map between symbbols (components, input, output)
    symbol_to_component = {
        "I": (taper, "1", "wg_2"),
        "O": (taper, "wg_2", "1"),
        "S": (wg_short, "W0", "E0"),
        "P": (pm_wg, "W0", "E0"),
        "A": (bend180, "W0", "W1"),
        "B": (bend180, "W1", "W0"),
        "H": (wg_heater, "W0", "E0"),
        "-": (wg_short2, "W0", "E0"),
    }

    # Generate a sequence
    # This is simply a chain of characters. Each of them represents a component
    # with a given input and and a given output

    repeated_sequence = "SIPOSASIPOSB"
    heater_seq = "-H-H-H-H-"
    sequence = repeated_sequence * n + "SIPO" + heater_seq
    return gf.components.component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )


if __name__ == "__main__":
    c = test_cutback_phase(n=1)
    c.show()
