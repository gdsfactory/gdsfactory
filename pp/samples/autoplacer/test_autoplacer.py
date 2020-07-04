import pathlib
import shutil

import pytest
import pp

from pp.samples.autoplacer.spiral import SPIRAL
from pp.autoplacer.yaml_placer import place_from_yaml
from pp.generate_does import generate_does

from pp.mask.merge_metadata import merge_metadata


workspace_folder = pathlib.Path(__file__).parent
build_path = workspace_folder / "build"
config_yml = workspace_folder / "config.yml"
doe_root_path = build_path / "cache_doe"
mask_path = build_path / "mask"
doe_metadata_path = build_path / "doe"

gdspath = mask_path / "sample_mask.gds"
markdown_path = gdspath.with_suffix(".md")
json_path = gdspath.with_suffix(".json")
test_metadata_path = gdspath.with_suffix(".tp.json")


@pytest.fixture
def cleandir():
    if build_path.exists():
        shutil.rmtree(build_path)


@pytest.mark.usefixtures("cleandir")
def test_autoplacer():
    mask_path.mkdir(parents=True, exist_ok=True)

    # Map the component factory names in the YAML file to the component factory
    name2factory = {"SPIRAL": SPIRAL}

    print(doe_root_path)
    generate_does(
        str(config_yml),
        component_type2factory=name2factory,
        doe_root_path=doe_root_path,
        doe_metadata_path=doe_metadata_path,
    )
    top_level = place_from_yaml(config_yml, doe_root_path)
    top_level.write(str(gdspath))

    merge_metadata(gdspath=gdspath)

    assert gdspath.exists()
    assert markdown_path.exists()
    assert json_path.exists()
    assert test_metadata_path.exists()

    report = open(markdown_path).read()
    assert report.count("#") >= 1, f" only {report.count('#')} DOEs in {markdown_path}"
    return gdspath


if __name__ == "__main__":
    c = test_autoplacer()
    pp.show(c)
