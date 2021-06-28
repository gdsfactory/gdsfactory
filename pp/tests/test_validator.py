import pydantic
import pytest

import pp


@pp.cell_with_validator
def component_with_straight(component: pp.Component) -> pp.Component:
    c = pp.Component()
    c.add_ref(component)
    c.add_ref(pp.c.straight())
    return c


def test_validator_pass():
    component = pp.c.straight(length=10)
    component_with_straight(component=component)


def test_validator_fail_empty():
    component = pp.Component()
    with pytest.raises(pydantic.ValidationError):
        component_with_straight(component=component)


def test_validator_fail_name_too_long():
    component = pp.Component(name="a" * 33)

    # component_with_straight(component=component)
    with pytest.raises(pydantic.ValidationError):
        component_with_straight(component=component)


if __name__ == "__main__":
    # test_validator_pass()
    # test_validator_fail_empty()
    test_validator_fail_name_too_long()
