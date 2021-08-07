import pathlib
import shutil

from gdsfactory.autoplacer.yaml_placer import place_from_yaml
from gdsfactory.config import logger
from gdsfactory.generate_does import generate_does
from gdsfactory.mask.merge_metadata import merge_metadata


def test_mask():
    """Returns gdspath for a Mask

    - Write GDS files defined in does.yml (with JSON metadata)
    - place them into a mask following placer information in does.yml
    - merge mask JSON metadata into a combined JSON file

    """

    cwd = pathlib.Path(__file__).absolute().parent
    does_path = cwd / "does.yml"

    build_path = cwd / "build"
    doe_root_path = cwd / "build" / "cache_doe_directory"
    mask_path = cwd / "build" / "mask"
    gdspath = mask_path / "mask.gds"
    logpath = gdspath.with_suffix(".log")
    mask_path.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(build_path, ignore_errors=True)
    logger.add(sink=logpath)
    generate_does(
        str(does_path),
        doe_root_path=doe_root_path,
    )
    top_level = place_from_yaml(does_path, root_does=doe_root_path)
    top_level.write(str(gdspath))
    merge_metadata(gdspath)
    assert gdspath.exists()
    return gdspath


if __name__ == "__main__":
    import gdsfactory as gf

    c = test_mask()
    gf.show(c)

    cwd = pathlib.Path(__file__).absolute().parent
    does_path = cwd / "does.yml"

    build_path = cwd / "build"
