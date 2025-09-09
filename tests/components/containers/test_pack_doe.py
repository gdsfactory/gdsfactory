from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.components.containers.pack_doe import (
    generate_doe,
    pack_doe,
    pack_doe_grid,
)


def test_generate_doe() -> None:
    doe = "mmi1x2"
    settings = dict(length_mmi=(2.5, 100), width_mmi=(4, 10))
    component_list, settings_list = generate_doe(
        doe=doe, settings=settings, do_permutations=True
    )
    assert len(component_list) == 4
    assert len(settings_list) == 4


def test_generate_doe_with_function() -> None:
    doe = "mmi1x2"
    settings = dict(length_mmi=(2.5, 100), width_mmi=(4, 10))

    def i(c: gf.Component) -> gf.Component:
        return c

    component_list, settings_list = generate_doe(
        doe=doe, settings=settings, do_permutations=True, function=i
    )
    assert len(component_list) == 4
    assert len(settings_list) == 4


def test_pack_doe_grid_with_function() -> None:
    doe = "mmi1x2"
    settings = dict(length_mmi=(2.5, 100), width_mmi=(4, 10))

    def i(c: gf.Component) -> gf.Component:
        return c

    pack_doe_grid(doe=doe, settings=settings, do_permutations=True, function=i)


def test_pack_doe() -> None:
    doe = "mmi1x2"
    settings = dict(length_mmi=(2, 100), width_mmi=(4, 10))
    component = pack_doe(doe=doe, settings=settings)
    assert component is not None
    assert len(component.info["doe_names"]) == 2
    assert len(component.info["doe_settings"]) == 2


def test_pack_doe_grid() -> None:
    doe = "mmi1x2"
    settings = dict(length_mmi=(2.5, 100), width_mmi=(4, 10))
    component = pack_doe_grid(
        doe=doe,
        settings=settings,
        with_text=True,
        spacing=(100, 100),
        shape=(2, 2),
        do_permutations=True,
    )
    assert component is not None
    assert len(component.info["doe_names"]) == 4
    assert len(component.info["doe_settings"]) == 4


def test_pack_doe_grid_without_text() -> None:
    doe = "mmi1x2"
    settings = dict(length_mmi=(2.5, 100), width_mmi=(4, 10))
    component = pack_doe_grid(
        doe=doe,
        settings=settings,
        with_text=False,
        spacing=(100, 100),
        shape=(2, 2),
        do_permutations=True,
    )
    assert component is not None
    assert len(component.info["doe_names"]) == 4
    assert len(component.info["doe_settings"]) == 4


def test_pack_doe_error() -> None:
    doe = "mmi1x2"
    settings = dict(length_mmi=(2.5, 100), width_mmi=(4, 10))
    with pytest.raises(ValueError):
        pack_doe(
            doe=doe,
            settings=settings,
            do_permutations=True,
            max_size=(135, 135),
            precision=1,
        )
