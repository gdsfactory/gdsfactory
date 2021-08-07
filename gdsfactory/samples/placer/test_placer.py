import pathlib

import gdsfactory
from gdsfactory.placer import component_grid_from_yaml


def test_placer() -> gdsfactory.Component:
    cwd = pathlib.Path(__file__).parent
    filepath = cwd / "config.yml"
    dirpath = cwd / "build" / "mask"
    dirpath.mkdir(exist_ok=True, parents=True)
    gdspath = dirpath / "test_placer.gds"

    top_level = component_grid_from_yaml(filepath)
    top_level.write_gds(gdspath=gdspath)
    assert gdspath.exists()
    return top_level


if __name__ == "__main__":

    c = test_placer()
    c.show()
