"""
This is a sample on how to define custom components.
You can make a repo out of this file, having one custom component per file
"""
import os
import shutil
import pytest
import pp
from pp.config import CONFIG
from pp.autoplacer.yaml_placer import place_from_yaml
from pp.components.spiral_inner_io import spiral_inner_io_euler
from pp.add_termination import add_gratings_and_loop_back
from pp.routing.connect import connect_strip_way_points
from pp.add_padding import add_padding_to_grid
from pp.generate_does import generate_does

from pp.mask.merge_metadata import merge_metadata


def _route_filter(*args, **kwargs):
    return connect_strip_way_points(
        *args, taper_factory=None, start_straight=5.0, end_straight=5.0, **kwargs
    )


def add_te(component, **kwargs):
    c = pp.routing.add_fiber_array(
        component,
        grating_coupler=pp.c.grating_coupler_elliptical_te,
        route_filter=_route_filter,
        **kwargs,
    )
    c.test = "passive_optical_te"
    c = add_padding_to_grid(c)
    return c


def add_tm(component, **kwargs):
    c = pp.routing.add_fiber_array(
        component,
        grating_coupler=pp.c.grating_coupler_elliptical_tm,
        route_filter=_route_filter,
        bend_radius=20,
        **kwargs,
    )
    c = add_padding_to_grid(c)
    return c


@pp.autoname
def coupler_te(gap, length, wg_width=0.5, nominal_wg_width=0.5):
    """ sample of component cutback """
    c = pp.c.coupler(wg_width=wg_width, gap=gap, length=length)
    cc = add_te(c)
    return cc


@pp.autoname
def spiral_te(wg_width=0.5, length=2):
    """ sample of component cutback

    Args:
        wg_width: um
        lenght: mm
    """
    c = spiral_inner_io_euler(wg_width=wg_width, length=length)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.c.grating_coupler_elliptical_te,
        bend_factory=pp.c.bend_circular,
    )
    return cc


@pp.autoname
def spiral_tm(wg_width=0.5, length=2):
    """ sample of component cutback """
    c = spiral_inner_io_euler(wg_width=wg_width, length=length, dx=10, dy=10, N=5)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.c.grating_coupler_elliptical_tm,
        bend_factory=pp.c.bend_circular,
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
def test_mask(precision=2e-9):
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
    # from pprint import pprint
    # pprint(component_factory)
    c = test_mask()
    pp.klive.show(c)
    # c = coupler_te(gap=0.3, length=20)
    # pp.show(c)
