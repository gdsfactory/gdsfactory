from functools import partial

import gdsfactory as gf
from gdsfactory.components.containers.add_trenches import add_trenches, add_trenches90
from gdsfactory.get_factories import is_cell


def test_is_cell() -> None:
    assert is_cell(add_trenches90)
    assert is_cell(add_trenches)

    def random_function(_: int) -> int:
        return 1

    assert not is_cell(random_function)
    assert not is_cell(partial(random_function, 1))


def test_is_gf_cell() -> None:
    @gf.cell
    def with_no_paren() -> gf.Component:
        return gf.Component()

    @gf.cell()
    def with_paren() -> gf.Component:
        return gf.Component()

    @gf.cell(basename="my_component")
    def with_args() -> gf.Component:
        return gf.Component()

    assert getattr(with_no_paren, "is_gf_cell", None) is True
    assert getattr(with_paren, "is_gf_cell", None) is True
    assert getattr(with_args, "is_gf_cell", None) is True


def test_is_gf_vcell() -> None:
    @gf.vcell
    def with_no_paren() -> gf.ComponentAllAngle:
        return gf.ComponentAllAngle()

    @gf.vcell()
    def with_paren() -> gf.ComponentAllAngle:
        return gf.ComponentAllAngle()

    @gf.vcell(basename="my_component")
    def with_args() -> gf.ComponentAllAngle:
        return gf.ComponentAllAngle()

    assert getattr(with_no_paren, "is_gf_vcell", None) is True
    assert getattr(with_paren, "is_gf_vcell", None) is True
    assert getattr(with_args, "is_gf_vcell", None) is True
