import pydantic
import pytest

import gdsfactory as gf


@gf.cell
def component_with_straight(component: gf.Component) -> gf.Component:
    c = gf.Component()
    c.add_ref(component)
    c.add_ref(gf.components.straight())
    return c


def test_validator_pass() -> None:
    component = gf.components.straight(length=10)
    component_with_straight(component=component)


# def test_validator_fail_empty():
#     component = gf.Component()
#     with pytest.raises(pydantic.ValidationError):
#         component_with_straight(component=component)


def test_validator_fail_name_too_long() -> None:
    component = gf.Component(name="a" * 200)

    # component_with_straight(component=component)
    with pytest.raises(pydantic.ValidationError):
        component_with_straight(component=component)


if __name__ == "__main__":
    # test_validator_pass()
    # test_validator_fail_empty()
    test_validator_fail_name_too_long()
