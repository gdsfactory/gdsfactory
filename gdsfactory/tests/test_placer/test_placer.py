import pathlib

import gdsfactory as gf
from gdsfactory.placer import component_grid_from_yaml


def test_placer() -> gf.Component:
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

    # c = test_placer()
    # c.show()

    yaml_str = """

mask:
    width: 10000
    height: 10000
    name: placer_example_method2

mmi2x2:
    component: mmi2x2
    settings:
        length_mmi: [11, 12]
        width_mmi: [3.6, 7.8]
    do_permutation: False

    spacing: [50., 100.]
    origin: [0., 0.]
    shape: [2, 1]

mzi2x2:
    doe_name: doe2
    component: mzi
    settings:
        length_x: [60, 80, 100]
        length_y: [60, 80, 100]
    do_permutation: True

    spacing: [200., 300.]
    origin: [100., 100.]
    shape: [3, 3]

"""

    c = component_grid_from_yaml(yaml_str)
    c.show()
