from __future__ import annotations

import pytest

import gdsfactory as gf


def test_named_references() -> None:
    c = gf.Component()
    c.add_ref(gf.components.straight(), alias="straight_1")
    c.add_ref(gf.components.straight())
    assert len(c.named_references) == 2


def test_named_references_with_alias() -> None:
    c = gf.Component()
    c.add_ref(gf.components.straight(), alias="straight_1")
    with pytest.raises(ValueError):
        c.add_ref(gf.components.straight(), alias="straight_1")


def test_fail_when_alias_exists() -> None:
    c = gf.Component()
    ref1 = c.add_ref(gf.components.straight())
    ref1.name = "straight_1"

    with pytest.raises(ValueError):
        ref2 = c.add_ref(gf.components.straight())
        ref2.name = "straight_1"


if __name__ == "__main__":
    # test_fail_when_alias_exists()
    test_named_references_with_alias()
    test_named_references()

    # import gdsfactory as gf

    # c = gf.Component("component_with_fill")
    # ref1 = c.add_ref(gf.components.straight())
    # ref1.name = "straight_1"

    # ref2 = c.add_ref(gf.components.straight())
    # ref2.name = "straight_1"
    # print(c.named_references)
    # c.show()
