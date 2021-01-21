from typing import Dict, List, Union

import pytest
from jsondiff import diff

import pp
from pp.add_pins import _add_settings_label
from pp.components import component_factory, component_names
from pp.import_gds import add_settings_from_label, import_gds


def tuplify(iterable: Union[List, Dict]):
    """From a list or tuple returns a tuple."""
    if isinstance(iterable, list):
        return tuple(map(tuplify, iterable))
    if isinstance(iterable, dict):
        return {k: tuplify(v) for k, v in iterable.items()}
    return iterable


def sort_dict(d):
    return {k: d[k] for k in sorted(d)}


@pytest.mark.parametrize(
    "component_type",
    component_names - set(["grating_coupler_tree", "compensation_path"]),
)
def test_properties_components(component_type):
    """Write component to GDS with setttings written on a label.
    Then import the GDS and check that the settings imported match the original.
    """
    cnew = pp.Component()
    c1 = component_factory[component_type]()
    c1ref = cnew << c1

    _add_settings_label(cnew, reference=c1ref)
    gdspath = pp.write_component(cnew)

    c2 = import_gds(gdspath)
    add_settings_from_label(c2)
    c1s = sort_dict(tuplify(c1.get_settings()))
    c2s = sort_dict(tuplify(c2.get_settings()))
    d = diff(c1s, c2s)
    print(c1s)
    print(c2s)
    assert len(d) == 0
    return c2


if __name__ == "__main__":
    # c = test_properties_components(component_type=list(component_names)[0])
    # c = test_properties_components(component_type="ring_single")
    # c = test_properties_components(component_type="mzit")
    # c = test_properties_components(component_type="bezier")
    # c = test_properties_components(component_type="bend_s")
    # c = test_properties_components(component_type="waveguide")
    c = test_properties_components(component_type="grating_coupler_tree")
    pp.show(c)
