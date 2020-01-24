import pathlib
import pp
from pp.config import load_config

from pp.autoplacer.samples.spiral import SPIRAL
from pp.autoplacer.yaml_placer import place_from_yaml
from pp.placer import generate_does


def main():
    workspace_folder = pathlib.Path(__file__).parent
    filepath_yml = workspace_folder / "config.yml"
    filepath_gds = str(workspace_folder / "build" / "mask" / "top_level.gds")
    config = load_config(filepath_yml)

    # Map the component factory names in the YAML file to the component factory
    name2factory = {"SPIRAL": SPIRAL}

    generate_does(config, component_type2factory=name2factory)
    top_level = place_from_yaml(config)
    top_level.write(filepath_gds)
    return filepath_gds


if __name__ == "__main__":
    c = main()
    pp.show(c)
