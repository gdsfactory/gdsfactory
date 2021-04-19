"""This is a sample on how to define custom components.
"""
import os
import shutil
from pathlib import Path

import pytest

import pp
from pp.add_termination import add_gratings_and_loop_back
from pp.autoplacer.yaml_placer import place_from_yaml
from pp.components.spiral_inner_io import spiral_inner_io_euler
from pp.config import CONFIG
from pp.generate_does import generate_does
from pp.mask.merge_metadata import merge_metadata
from pp.routing.get_route import get_route_from_waypoints


def _route_filter(*args, **kwargs):
    return get_route_from_waypoints(
        *args, taper_factory=None, start_straight=5.0, end_straight=5.0, **kwargs
    )


def add_te(component, **kwargs):
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
        **kwargs,
    )
    return c


@pp.cell
def coupler_te(gap, length):
    """Sample of component cutback."""
    c = pp.components.coupler(gap=gap, length=length)
    cc = add_te(c)
    return cc


@pp.cell
def spiral_te(wg_width=0.5, length_cm=2):
    """Waveguide Spiral for straight loss.

    Args:
        wg_width: um
        lenght: mm
    """
    c = spiral_inner_io_euler(wg_width=wg_width, length=length_cm)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.components.grating_coupler_elliptical_te,
        bend_factory=pp.components.bend_circular,
    )
    return cc


@pp.cell
def spiral_tm(wg_width=0.5, length_cm=2):
    """Waveguide Spiral for straight loss."""
    c = spiral_inner_io_euler(wg_width=wg_width, length=length_cm, dx=10, dy=10, N=5)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.components.grating_coupler_elliptical_tm,
        bend_factory=pp.components.bend_circular,
    )
    return cc


component_factory = dict(
    spiral_te=spiral_te, spiral_tm=spiral_tm, coupler_te=coupler_te
)


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
def test_mask(precision: float = 2e-9) -> Path:
    workspace_folder = CONFIG["samples_path"] / "mask_custom"
    build_path = workspace_folder / "build"
    doe_root_path = build_path / "cache_doe"
    doe_metadata_path = build_path / "doe"
    mask_path = build_path / "mask"
    does_yml = workspace_folder / "does.yml"

    mask_path.mkdir(parents=True, exist_ok=True)

    gdspath = mask_path / "sample_mask.gds"
    markdown_path = gdspath.with_suffix(".md")
    json_path = gdspath.with_suffix(".json")
    test_metadata_path = gdspath.with_suffix(".tp.json")

    generate_does(
        str(does_yml),
        component_factory=component_factory,
        precision=precision,
        doe_root_path=doe_root_path,
        doe_metadata_path=doe_metadata_path,
    )

    top_level = place_from_yaml(does_yml, precision=precision, root_does=doe_root_path)
    top_level.write(str(gdspath))

    merge_metadata(gdspath=gdspath)

    assert gdspath.exists()
    assert markdown_path.exists()
    assert json_path.exists()
    assert test_metadata_path.exists()

    report = open(markdown_path).read()
    assert report.count("#") == 2, f" only {report.count('#')} DOEs in {markdown_path}"

    return gdspath


if __name__ == "__main__":
    # gdspath_mask = test_mask()
    # pp.show(gdspath_mask)
    c = coupler_te(gap=0.3, length=2.0)
    # c = spiral_te(length_cm=6.)
    c.show()

    # lengths = [18.24, 36.48, 54.72, 72.96, 91.2]
    # for length in lengths:
    #     c = coupler_te(gap=0.3, length=length)
