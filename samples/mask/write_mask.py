import pathlib
import pp
from pp.config import load_config

from pp.autoplacer.samples.spiral import SPIRAL
from pp.autoplacer.yaml_placer import place_from_yaml
from pp.components import component_type2factory
from pp.generate_does import generate_does


def main():
    """

    """
    cwd = pathlib.Path(__file__).absolute().parent
    config_path = cwd / "config.yml"
    does_path = cwd / "does.yml"

    config = load_config(config_path)
    doe_directory = config["cache_doe_directory"]
    gdspath = config["mask"]["gds"]

    # Map the component factory names in the YAML file to the component factory
    component_type2factory.update({"SPIRAL": SPIRAL})

    generate_does(
        str(does_path),
        component_type2factory=component_type2factory,
        doe_root_path=doe_directory,
    )
    top_level = place_from_yaml(does_path, doe_directory)
    top_level.write(str(gdspath))
    return gdspath


if __name__ == "__main__":
    c = main()
    pp.show(c)
