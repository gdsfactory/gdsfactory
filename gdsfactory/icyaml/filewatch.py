"""filewatcher"""

import json
import time
from pathlib import Path

import jsonschema
import yaml

from gdsfactory.config import CONFIG, logger
from gdsfactory.read.from_yaml import from_yaml

schema_path = CONFIG["schema_netlist"]
schema_dict = json.loads(schema_path.read_text())

logger.info(f"Loaded netlist schema from {str(schema_path)!r}")


def filewatch(filepath: str):
    filepath = Path(filepath)
    logger.info(f"Watching {str(filepath)!r}")

    try:
        while True:
            yaml_text = filepath.read_text()
            yaml_dict = yaml.safe_load(yaml_text)

            if yaml_dict is not None:
                try:
                    jsonschema.validate(yaml_dict, schema_dict)
                    c = from_yaml(yaml_text)
                    c.show()
                    time.sleep(1)
                except (
                    ValueError,
                    ModuleNotFoundError,
                    KeyError,
                    Exception,
                    jsonschema.exceptions.ValidationError,
                ) as e:
                    logger.error(e)

    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    filewatch("/home/jmatres/ubc/ubcpdk/circuits/mask.ic.yaml")
