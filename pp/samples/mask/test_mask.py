import pathlib
import pytest
import pp
from pp.config import load_config

from pp.autoplacer.yaml_placer import place_from_yaml
from pp.generate_does import generate_does


@pytest.mark.usefixtures("cleandir")
def test_mask():
    """

    """
    cwd = pathlib.Path(__file__).absolute().parent
    config_path = cwd / "config.yml"
    does_path = cwd / "does.yml"

    config = load_config(config_path)
    doe_root_path = config["cache_doe_directory"]
    gdspath = config["mask"]["gds"]

    generate_does(
        str(does_path), doe_root_path=doe_root_path,
    )
    top_level = place_from_yaml(does_path, root_does=doe_root_path)
    top_level.write(str(gdspath))
    assert gdspath.exists()
    return gdspath


if __name__ == "__main__":
    c = test_mask()
    pp.show(c)
