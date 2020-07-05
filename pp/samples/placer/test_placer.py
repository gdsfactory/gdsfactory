import pathlib
import pp
from pp.placer import component_grid_from_yaml


def test_placer():
    cwd = pathlib.Path(__file__).parent
    filepath = cwd / "config.yml"
    gdspath = cwd / "build" / "mask" / "test_placer.gds"

    top_level = component_grid_from_yaml(filepath)
    pp.write_gds(top_level, gdspath)
    assert gdspath.exists()
    return gdspath


if __name__ == "__main__":
    c = test_placer()
    pp.show(c)
