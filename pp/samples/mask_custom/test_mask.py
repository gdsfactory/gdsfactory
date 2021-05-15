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


def add_te(component, **kwargs):
    c = pp.routing.add_fiber_array(
        component=component,
        grating_coupler=pp.components.grating_coupler_elliptical_te,
        **kwargs,
    )
    c.test = "passive_optical_te"
    return c


def add_tm(component, **kwargs):
    c = pp.routing.add_fiber_array(
        component=component,
        grating_coupler=pp.components.grating_coupler_elliptical_tm,
        **kwargs,
    )
    return c


@pp.cell
def coupler_te(gap, length):
    """Directional coupler with TE grating couplers."""
    c = pp.components.coupler(gap=gap, length=length)
    cc = add_te(c)
    return cc


@pp.cell
def spiral_te(width=0.5, length=20e3):
    """Waveguide Spiral with TE grating_coupler

    Args:
        width: um
        lenght: um
    """
    c = spiral_inner_io_euler(width=width, length=length)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.components.grating_coupler_elliptical_te,
        bend_factory=pp.components.bend_circular,
    )
    return cc


@pp.cell
def spiral_tm(width=0.5, length=20e3):
    """Waveguide Spiral with TM grating_coupler.

    Args:
        width: um
        lenght: um
    """
    c = spiral_inner_io_euler(width=width, length=length, dx=10, dy=10, N=5)
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
    # c = coupler_te(gap=0.3, length=2.0)
    # c = spiral_te(length=60e3)
    # c.show()

    # lengths = [18.24, 36.48, 54.72, 72.96, 91.2]
    # for length in lengths:
    #     c = coupler_te(gap=0.3, length=length)

    gdsp = test_mask()
