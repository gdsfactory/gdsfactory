import pydantic
import pytest

import gdsfactory


@gdsfactory.cell
def component_with_straight(component: gdsfactory.Component) -> gdsfactory.Component:
    c = gdsfactory.Component()
    c.add_ref(component)
    c.add_ref(gdsfactory.components.straight())
    return c


def test_validator_pass():
    component = gdsfactory.components.straight(length=10)
    component_with_straight(component=component)


def test_validator_fail_empty():
    component = gdsfactory.Component()
    with pytest.raises(pydantic.ValidationError):
        component_with_straight(component=component)


def test_validator_fail_name_too_long():
    component = gdsfactory.Component(name="a" * 33)

    # component_with_straight(component=component)
    with pytest.raises(pydantic.ValidationError):
        component_with_straight(component=component)


if __name__ == "__main__":
    # test_validator_pass()
    # test_validator_fail_empty()
    test_validator_fail_name_too_long()
