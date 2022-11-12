import json

import jsonschema
import pytest
import yaml

from gdsfactory.config import CONFIG

schema_path = CONFIG["schema_netlist"]
schema_dict = json.loads(schema_path.read_text())
yaml_text_invalid = """
name: demo

wrong: hi

instances:
    y:
        component: coupler
"""

yaml_text_valid = """

name: mzi

pdk: ubcpdk

settings:
   dy: -90

info:
    polarization: te
    wavelength: 1.55
    description: mzi for ubcpdk

instances:
    yr:
      component: y_splitter
    yl:
      component: y_splitter

placements:
    yr:
        rotation: 180
        x: 100
        y: 0

routes:
    route_top:
        links:
            yl,opt2: yr,opt3
        settings:
            cross_section: strip
    route_bot:
        links:
            yl,opt3: yr,opt2
        routing_strategy: get_bundle_from_steps
        settings:
          steps: [dx: 30, dy: '${settings.dy}', dx: 20]
          cross_section: strip

ports:
    o1: yl,opt1
    o2: yr,opt1
"""


def test_schema_valid() -> None:
    yaml_dict = yaml.safe_load(yaml_text_valid)
    jsonschema.validate(yaml_dict, schema_dict)


def test_schema_invalid() -> None:
    yaml_dict = yaml.safe_load(yaml_text_invalid)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(yaml_dict, schema_dict)


if __name__ == "__main__":
    # test_schema_valid()
    test_schema_invalid()
