from typing import Any, Dict, List, Union

from gdsfactory.components import cells

skip_test = {
    "version_stamp",
    "extend_ports_list",
    "extend_port",
    "grating_coupler_tree",
    "compensation_path",
    "spiral_inner_io_with_gratings",
    "component_sequence",
    "straight_heater_metal_90_90",
    "straight_heater_metal_undercut_90_90",
    "mzi_phase_shifter_top_heater_metal",
}

components_to_test = set(cells.keys()) - skip_test


def tuplify(iterable: Union[List, Dict]) -> Any:
    """From a list or tuple returns a tuple."""
    if isinstance(iterable, list):
        return tuple(map(tuplify, iterable))
    if isinstance(iterable, dict):
        return {k: tuplify(v) for k, v in iterable.items()}
    return iterable


def sort_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: d[k] for k in sorted(d)}


# @pytest.mark.parametrize("component_type", components_to_test)
# def test_properties_components(component_type: str) -> Component:
#     """Write component to GDS with settings written on a label.
#     Then import the GDS and check that the settings imported match the original settings.
#     """
#     cnew = gf.Component()
#     c1 = factory[component_type]()
#     c1ref = cnew << c1

#     add_settings_label(cnew, reference=c1ref)
#     gdspath = cnew.write_gds_with_metadata()

#     c2 = import_gds(gdspath)
#     add_settings_from_label(c2)

#     c1s = sort_dict(tuplify(OmegaConf.to_container(c1.settings.full)))
#     c2s = sort_dict(tuplify(OmegaConf.to_container(c2.settings.full)))

#     # c1s.pop("info")
#     # c2s.pop("info")
#     # c1s.pop("changed")
#     # c2s.pop("changed")

#     d = diff(c1s, c2s)
#     # print(c1s)
#     print(c2s)
#     print(d)
#     assert len(d) == 0, f"imported settings are different from original {d}"
#     return c2


pass
# c = test_properties_components(component_type=list(component_names)[0])
# c = test_properties_components(component_type="ring_single")
# c = test_properties_components(component_type="mzit")
# c = test_properties_components(component_type="bezier")
# c = test_properties_components(component_type="wire_straight")
# c = test_properties_components(component_type="straight")
# c = test_properties_components(component_type="grating_coupler_tree")
# c = test_properties_components(component_type="wire")
# c = test_properties_components(component_type="bend_circular")
# c = test_properties_components(component_type="mzi_arm")
# c = test_properties_components(component_type="straight_pin")
# c.show(show_ports=True)
