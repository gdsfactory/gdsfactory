"""
This is a sample on how to define custom components.
You can make a repo out of this file, having one custom component per file
"""
import shutil
from pathlib import Path

import numpy as np

import gdsfactory as gf
from gdsfactory.add_termination import add_gratings_and_loopback
from gdsfactory.component import Component
from gdsfactory.components.spiral_inner_io import spiral_inner_io_euler
from gdsfactory.config import CONFIG
from gdsfactory.mask.merge_metadata import merge_metadata


def add_te(component: Component, **kwargs) -> Component:
    c = gf.routing.add_fiber_array(
        component=component,
        grating_coupler=gf.components.grating_coupler_elliptical_te,
        **kwargs,
    )
    c.test = "passive_optical_te"
    return c


def add_tm(component, **kwargs):
    c = gf.routing.add_fiber_array(
        component=component,
        grating_coupler=gf.components.grating_coupler_elliptical_tm,
        bend_radius=20,
        **kwargs,
    )
    return c


@gf.cell
def coupler_te(
    gap: float,
    length: int,
) -> Component:
    """Evanescent coupler with TE grating coupler."""
    c = gf.components.coupler(gap=gap, length=length)
    cc = add_te(c)
    return cc


@gf.cell
def spiral_te(width: float = 0.5, length: int = 2) -> Component:
    """Spiral with TE grating_coupler

    Args:
        width: waveguide width um
        lenght: cm
    """
    c = spiral_inner_io_euler(width=width, length=length)
    cc = add_gratings_and_loopback(
        component=c,
        grating_coupler=gf.components.grating_coupler_elliptical_te,
        bend_factory=gf.components.bend_euler,
    )
    return cc


@gf.cell
def spiral_tm(width=0.5, length=20e3):
    """Spiral with TM grating_coupler

    Args:
        width: waveguide width um
        lenght: um
    """
    c = spiral_inner_io_euler(width=width, length=length, dx=10, dy=10, N=5)
    cc = add_gratings_and_loopback(
        component=c,
        grating_coupler=gf.components.grating_coupler_elliptical_tm,
        bend_factory=gf.components.bend_euler,
    )
    return cc


def test_mask(precision: float = 1e-9) -> Path:
    workspace_folder = CONFIG["samples_path"] / "mask_pack"
    build_path = workspace_folder / "build"
    mask_path = build_path / "mask"

    shutil.rmtree(build_path, ignore_errors=True)
    mask_path.mkdir(parents=True, exist_ok=True)

    gdspath = mask_path / "sample_mask.gds"
    markdown_path = gdspath.with_suffix(".md")
    json_path = gdspath.with_suffix(".json")
    test_metadata_path = gdspath.with_suffix(".tp.json")

    components = [spiral_te(length=length) for length in np.array([2, 4, 6]) * 1e4]
    components += [coupler_te(length=length, gap=0.2) for length in [10, 20, 30, 40]]
    c = gf.pack(components)
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
    gf.klive.show(c)
