from typing import Union, List, Dict
import pytest
from jsondiff import diff
import pp
from pp.components import (
    component_factory,
    _components,
)
from pp.import_gds import add_settings_from_label
from pp.import_gds import import_gds


def tuplify(iterable: Union[List, Dict]):
    """From a list or tuple returns a tuple."""
    if isinstance(iterable, list):
        return tuple(map(tuplify, iterable))
    if isinstance(iterable, dict):
        return {k: tuplify(v) for k, v in iterable.items()}
    return iterable


@pytest.mark.parametrize("component_type", _components)
def test_properties_components(component_type):
    """Write component to GDS with setttings_label
    """
    c1 = component_factory[component_type]()
    gdspath = pp.write_component(c1, with_settings_label=True)
    c2 = import_gds(gdspath)
    add_settings_from_label(c2)
    c1s = tuplify(c1.get_settings())
    c2s = tuplify(c2.get_settings())
    d = diff(c1s, c2s)
    print(c1s)
    print(c2s)
    assert len(d) == 0


if __name__ == "__main__":
    # test_properties_components(component_type=list(_components)[0])
    # test_properties_components(component_type="ring")
    # test_properties_components(component_type="bezier")
    # test_properties_components(component_type="bend_s")
    test_properties_components(component_type="waveguide")
