import pathlib
import pp
from pp.config import load_config

from pp.autoplacer.samples.spiral import SPIRAL
from pp.autoplacer.yaml_placer import place_from_yaml
from pp.placer import generate_does
from pp.components import component_type2factory


def main():
    """

    """
    workspace_folder = pathlib.Path(__file__).absolute().parent
    filepath_yml = workspace_folder / "config.yml"
    filepath_does_yml = workspace_folder / "does.yml"

    CONFIG = load_config(filepath_yml)

    filepath_gds = str(
        workspace_folder / "build" / "mask" / f"{CONFIG['mask']['name']}.gds"
    )

    # Map the component factory names in the YAML file to the component factory
    component_type2factory.update({"SPIRAL": SPIRAL})

    generate_does(CONFIG, component_type2factory=component_type2factory)
    top_level = place_from_yaml(filepath_does_yml, CONFIG["cache_doe_directory"])
    top_level.write(filepath_gds)
    return filepath_gds


if __name__ == "__main__":
    c = main()
    pp.show(c)
