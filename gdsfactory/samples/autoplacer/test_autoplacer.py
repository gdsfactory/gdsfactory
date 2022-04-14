import pathlib
import shutil

from gdsfactory.config import logger
from gdsfactory.pdk import ACTIVE_PDK
from gdsfactory.samples.autoplacer.spiral import spiral
from gdsfactory.sweep.write_sweeps import write_sweeps

workspace_folder = pathlib.Path(__file__).parent
build_path = workspace_folder / "build"
config_yml = workspace_folder / "config.yml"
doe_root_path = build_path / "cache_doe"
mask_path = build_path / "mask"
doe_metadata_path = build_path / "sweep"

gdspath = mask_path / "sample_mask.gds"
markdown_path = gdspath.with_suffix(".md")
config_path = gdspath.with_suffix(".yml")
json_path = gdspath.with_suffix(".json")
test_metadata_path = gdspath.with_suffix(".tp.yml")
logpath = gdspath.with_suffix(".log")


def test_autoplacer():
    from gdsfactory.autoplacer.yaml_placer import place_from_yaml

    shutil.rmtree(build_path, ignore_errors=True)
    mask_path.mkdir(parents=True, exist_ok=True)

    # Add custom component to pdk cells
    ACTIVE_PDK.register_cell("spiral", spiral)

    logger.add(sink=logpath)
    logger.info("writring does to", doe_root_path)
    write_sweeps(
        str(config_yml),
        doe_root_path=doe_root_path,
        doe_metadata_path=doe_metadata_path,
    )
    top_level = place_from_yaml(config_yml, doe_root_path)
    top_level.write(str(gdspath))

    assert gdspath.exists()
    return gdspath


if __name__ == "__main__":
    import gdsfactory as gf

    c = test_autoplacer()
    gf.show(c)
