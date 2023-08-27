import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory.samples.all_angle_routing as aar_samples
from gdsfactory.difftest import difftest
from gdsfactory.pdk import get_active_pdk

AAR_YAML_PICS = aar_samples.get_yaml_pics()

cells_to_test = [
    cell_name
    for cell_name in AAR_YAML_PICS
    if "error" not in cell_name and "wrong" not in cell_name
]
error_cells = [cell_name for cell_name in AAR_YAML_PICS if "error" in cell_name]


@pytest.fixture(params=cells_to_test, scope="function")
def component_name(request) -> str:
    return request.param


@pytest.fixture(params=error_cells, scope="function")
def bad_component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    # make sure we are flattening invalid refs
    flatten_invalid_refs_default = (
        get_active_pdk().gds_write_settings.flatten_invalid_refs
    )
    get_active_pdk().gds_write_settings.flatten_invalid_refs = True

    try:
        component = AAR_YAML_PICS[component_name]()
        difftest(component, test_name=component_name)
    finally:
        # reset back to what it was, so we don't mess up other tests
        get_active_pdk().gds_write_settings.flatten_invalid_refs = (
            flatten_invalid_refs_default
        )


def test_bad_cells_throw_errors(bad_component_name):
    bad_func = AAR_YAML_PICS[bad_component_name]
    with pytest.raises(ValueError):
        bad_func()


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = AAR_YAML_PICS[component_name]()
    data_regression.check(component.to_dict())


if __name__ == "__main__":
    # name = cells_to_test[0]
    # name = "aar_bundles"
    name = "aar_gone_wrong"
    # name = "aar_error_intermediate_180"
    # name = "aar_error_overconstrained"
    c = AAR_YAML_PICS[name]()
    print(sorted([i.name for i in c.get_dependencies()]))
    c.show()
