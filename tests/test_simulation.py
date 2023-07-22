from copy import deepcopy
import gdsfactory as gf
from gdsfactory.simulation.get_sparameters_path import (
    get_component_hash,
    get_sparameters_path_lumerical,
)
from gdsfactory.generic_tech import LAYER_STACK


def test_simulation_get_component_hash() -> None:
    c = gf.components.rectangle()
    h = get_component_hash(c)
    assert isinstance(h, str)


def test_simulation_get_sparameters_path() -> None:
    nm = 1e-3
    layer_stack2 = deepcopy(LAYER_STACK)
    layer_stack2.layers["core"].thickness = 230 * nm

    c = gf.components.straight()

    p1 = get_sparameters_path_lumerical(component=c)
    p2 = get_sparameters_path_lumerical(component=c, layer_stack=layer_stack2)
    p3 = get_sparameters_path_lumerical(c, material_name_to_lumerical=dict(si=3.6))

    assert p1
    assert p2
    assert p3


if __name__ == "__main__":
    # test_simulation_get_component_hash()
    test_simulation_get_sparameters_path()
