import pathlib
import pytest
import pp

from pp.autoplacer.yaml_placer import place_from_yaml
from pp.generate_does import generate_does
from pp.mask.merge_metadata import merge_metadata


@pytest.mark.usefixtures("cleandir")
def test_mask():
    """

    """
    cwd = pathlib.Path(__file__).absolute().parent
    does_path = cwd / "does.yml"

    doe_root_path = cwd / "build" / "cache_doe_directory"
    mask_path = cwd / "build" / "mask"
    gdspath = mask_path / "mask.gds"
    mask_path.mkdir(parents=True, exist_ok=True)

    generate_does(
        str(does_path), doe_root_path=doe_root_path,
    )
    top_level = place_from_yaml(does_path, root_does=doe_root_path)
    top_level.write(str(gdspath))
    merge_metadata(gdspath)
    assert gdspath.exists()
    return gdspath


if __name__ == "__main__":
    c = test_mask()
    pp.show(c)
