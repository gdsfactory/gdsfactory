from typing import Any, Dict, List, Union

import pytest
from jsondiff import diff

import gdsfactory as gf
from gdsfactory.add_pins import add_settings_label
from gdsfactory.component import Component
from gdsfactory.components import component_factory, component_names
from gdsfactory.import_gds import add_settings_from_label, import_gds


def tuplify(iterable: Union[List, Dict]) -> Any:
    """From a list or tuple returns a tuple."""
    if isinstance(iterable, list):
        return tuple(map(tuplify, iterable))
    if isinstance(iterable, dict):
        return {k: tuplify(v) for k, v in iterable.items()}
    return iterable


def sort_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: d[k] for k in sorted(d)}


@pytest.mark.parametrize(
    "component_type",
    component_names
    - set(
        ["grating_coupler_tree", "compensation_path", "spiral_inner_io_with_gratings"]
    ),
)
def test_properties_components(component_type: str) -> Component:
    """Write component to GDS with setttings written on a label.
    Then import the GDS and check that the settings imported match the original.
    """
    cnew = gf.Component()
    c1 = component_factory[component_type]()
    c1ref = cnew << c1

    ignore = ("sequence", "symbol_to_component", "ports_map")
    add_settings_label(cnew, reference=c1ref, ignore=ignore)
    gdspath = cnew.write_gds_with_metadata()

    c2 = import_gds(gdspath)
    add_settings_from_label(c2)

    c1s = sort_dict(tuplify(c1.get_settings(ignore=ignore)))
    c2s = sort_dict(tuplify(c2.get_settings(ignore=ignore)))

    c1s.pop("info")
    c2s.pop("info")
    d = diff(c1s, c2s)
    # print(c1s)
    # print(c2s)
    # print(d)
    assert len(d) == 0, f"imported settings are different from original {d}"
    return c2


if __name__ == "__main__":
    # c = test_properties_components(component_type=list(component_names)[0])
    # c = test_properties_components(component_type="ring_single")
    # c = test_properties_components(component_type="mzit")
    # c = test_properties_components(component_type="bezier")
    # c = test_properties_components(component_type="bend_s")
    # c = test_properties_components(component_type="straight")
    # c = test_properties_components(component_type="grating_coupler_tree")
    # c = test_properties_components(component_type="wire")
    # c = test_properties_components(component_type="bend_circular")
    # c = test_properties_components(component_type="mzi_arm")
    c = test_properties_components(component_type="mzi")
    c.show()
