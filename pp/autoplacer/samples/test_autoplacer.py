import pathlib
import pp

from pp.autoplacer.samples.spiral import SPIRAL
from pp.autoplacer.yaml_placer import place_from_yaml
from pp.generate_does import generate_does


def test_autoplacer():
    workspace_folder = pathlib.Path(__file__).parent
    filepath_yml = workspace_folder / "config.yml"

    filepath_gds = str(workspace_folder / "build" / "mask" / "top_level.gds")

    # Map the component factory names in the YAML file to the component factory
    name2factory = {"SPIRAL": SPIRAL}

    generate_does(filepath_yml, component_type2factory=name2factory)
    top_level = place_from_yaml(filepath_yml)
    top_level.write(filepath_gds)
    assert filepath_gds
    return filepath_gds


if __name__ == "__main__":
    c = test_autoplacer()
    pp.show(c)
