"""
Tests that we get a suitable electrical netlist that represents the physical geometry of our circuit.
We use the `get_missing_models` function in `sax` to extract that it is representing our netlist component correctly.
"""
import json  # NOQA : F401

import sax

import gdsfactory as gf


def test_extract_electrical_netlist_straight_heater_metal():
    """
    This component is a good test because it contains electrical and optical ports, and we can see how effective the
    implementation of adding ports is in this case.
    What we want here is to extract all the electrical models required from our netlist. If our netlist is properly
    composed, then it is possible to have a connectivity that actually represents electrical elements in our circuit,
    and hence, eventually convert it to SPICE.
    """
    our_heater = gf.components.straight_heater_metal_simple()
    our_heater.show()
    our_heater_netlist = our_heater.get_netlist(
        exclude_port_types="optical",
        allow_multiple=True,
    )
    # print(our_heater_netlist)
    # print(sax.get_required_circuit_models(our_heater_netlist))
    # print(our_heater_netlist["instances"].keys())
    # print(our_heater_netlist["ports"])
    # print("connections out")
    # print(our_heater_netlist["connections"])
    assert sax.get_required_circuit_models(our_heater_netlist) == [
        "straight",
        "taper",
        "via_stack",
    ]

    our_recursive_heater_netlist = our_heater.get_netlist_recursive(
        exclude_port_types="optical",
        allow_multiple=True,
    )
    # print(json.dumps(our_recursive_heater_netlist, sort_keys=True, indent=4))
    # print(sax.get_required_circuit_models(our_recursive_heater_netlist))
    assert {"straight", "taper", "compass"}.issubset(
        set(sax.get_required_circuit_models(our_recursive_heater_netlist))
    )


def test_netlist_model_extraction():
    """
    Tests that our PIN placement allows the netlister and sax to extract our electrical component model in a composed circuit.
    """

    @gf.cell
    def connected_metal():
        test = gf.Component()
        a = test << gf.components.straight(cross_section="metal1")
        test.add_ports(a.ports)
        return test

    connected_metal().show()
    models_list = sax.get_required_circuit_models(connected_metal().get_netlist())
    # print(models_list)
    assert models_list == ["straight"]

    models_list_electrical_netlist = sax.get_required_circuit_models(
        connected_metal().get_netlist(exclude_port_types="optical")
    )
    # print(models_list_electrical_netlist)
    assert models_list == models_list_electrical_netlist


def test_via_array_netlist_connectivity():
    """
    We want to test how the pins generated in a via array allow us to extract connectivity for its component in between metal layers.
    """
    our_via_stack = gf.components.via_stack()
    our_via_stack.show()
    pass


def test_mzi2x2_2x2_phase_shifter():
    """
    We want to know how we extract the electrical netlist in a more hierarchical composed design.
    """
    our_mzi_phase_shifter = gf.components.mzi2x2_2x2_phase_shifter()
    our_mzi_phase_shifter_netlist_electrical = our_mzi_phase_shifter.get_netlist(
        exclude_port_types="optical",
    )
    our_mzi_phase_shifter.show()
    assert sax.get_required_circuit_models(
        our_mzi_phase_shifter_netlist_electrical
    ) == ["straight_heater_metal_simple"]

    # We need to extract only electrical models.
    our_recursive_mzi_phase_shifter_netlist_electrical = (
        our_mzi_phase_shifter.get_netlist_recursive(
            exclude_port_types="optical",
            allow_multiple=True,
        )
    )
    assert {"straight", "taper", "compass"}.issubset(
        set(
            sax.get_required_circuit_models(
                our_recursive_mzi_phase_shifter_netlist_electrical
            )
        )
    )

    # print(json.dumps(our_recursive_mzi_phase_shifter_netlist_electrical, sort_keys=True, indent=4))
    # print(sax.get_required_circuit_models(our_recursive_mzi_phase_shifter_netlist_electrical))


def test_no_effect_on_original_components():
    passive_mzi = gf.components.mzi2x2_2x2()
    passive_mzi.show()
    passive_mzi_phase_shifter_netlist_electrical = passive_mzi.get_netlist_recursive(
        exclude_port_types="optical",
    )
    assert passive_mzi_phase_shifter_netlist_electrical is not None
    # print(passive_mzi_phase_shifter_netlist_electrical[list(passive_mzi_phase_shifter_netlist_electrical.keys())[0]]["instances"].keys())


if __name__ == "__main__":
    # TODO turn all back on.
    test_extract_electrical_netlist_straight_heater_metal()
    test_mzi2x2_2x2_phase_shifter()
    test_netlist_model_extraction()
    test_no_effect_on_original_components()
    test_via_array_netlist_connectivity()
