"""
This is a sample on how to define custom components.
You can make a repo out of this file, having one custom component per file
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Union

import pytest

import pp
from pp.add_termination import add_gratings_and_loop_back
from pp.component import Component, ComponentReference
from pp.components.spiral_inner_io import spiral_inner_io_euler
from pp.config import CONFIG
from pp.mask.merge_metadata import merge_metadata
from pp.port import Port
from pp.routing.get_route import get_route_from_waypoints


def _route_filter(
    *args, **kwargs
) -> Union[
    Dict[str, Union[List[ComponentReference], Dict[str, Port], float]],
    ComponentReference,
]:
    return get_route_from_waypoints(
        *args, taper_factory=None, start_straight=5.0, end_straight=5.0, **kwargs
    )


def add_te(component: Component, **kwargs) -> Component:
    c = pp.routing.add_fiber_array(
        component=component,
        grating_coupler=pp.components.grating_coupler_elliptical_te,
        route_filter=_route_filter,
        **kwargs,
    )
    c.test = "passive_optical_te"
    return c


def add_tm(component, **kwargs):
    c = pp.routing.add_fiber_array(
        component=component,
        grating_coupler=pp.components.grating_coupler_elliptical_tm,
        route_filter=_route_filter,
        bend_radius=20,
        **kwargs,
    )
    return c


@pp.cell
def coupler_te(
    gap: float,
    length: int,
) -> Component:
    """ sample of component cutback """
    c = pp.components.coupler(gap=gap, length=length)
    cc = add_te(c)
    return cc


@pp.cell
def spiral_te(wg_width: float = 0.5, length: int = 2) -> Component:
    """sample of component cutback

    Args:
        wg_width: um
        lenght: mm
    """
    c = spiral_inner_io_euler(wg_width=wg_width, length=length)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.components.grating_coupler_elliptical_te,
        bend_factory=pp.components.bend_euler,
    )
    return cc


@pp.cell
def spiral_tm(wg_width=0.5, length=2):
    """ sample of component cutback """
    c = spiral_inner_io_euler(wg_width=wg_width, length=length, dx=10, dy=10, N=5)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.components.grating_coupler_elliptical_tm,
        bend_factory=pp.components.bend_euler,
    )
    return cc


@pytest.fixture
def cleandir():
    build_folder = CONFIG["samples_path"] / "mask_custom" / "build"
    if build_folder.exists():
        shutil.rmtree(build_folder)


@pytest.fixture
def chdir():
    workspace_folder = CONFIG["samples_path"] / "mask_custom"
    os.chdir(workspace_folder)


@pytest.mark.usefixtures("cleandir")
def test_mask(precision: float = 1e-9) -> Path:
    workspace_folder = CONFIG["samples_path"] / "mask_pack"
    build_path = workspace_folder / "build"
    mask_path = build_path / "mask"

    mask_path.mkdir(parents=True, exist_ok=True)

    gdspath = mask_path / "sample_mask.gds"
    markdown_path = gdspath.with_suffix(".md")
    json_path = gdspath.with_suffix(".json")
    test_metadata_path = gdspath.with_suffix(".tp.json")

    components = [spiral_te(length=length) for length in [2, 4, 6]]
    components += [coupler_te(length=length, gap=0.2) for length in [10, 20, 30, 40]]
    c = pp.pack(components)
    m = c[0]
    m.name = "sample_mask"
    m.write_gds(gdspath)

    merge_metadata(gdspath=gdspath)

    assert gdspath.exists()
    assert markdown_path.exists()
    assert json_path.exists()
    assert test_metadata_path.exists()
    return gdspath


if __name__ == "__main__":
    c = test_mask()
    pp.klive.show(c)
