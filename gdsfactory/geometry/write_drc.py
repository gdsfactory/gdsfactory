"""Write DRC rule decks in klayout.

TODO:

- add min area
- define derived layers (composed rules)
"""

import pathlib
from dataclasses import asdict, is_dataclass
from typing import List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from gdsfactory.config import logger
from gdsfactory.install import get_klayout_path
from gdsfactory.types import Dict, Layer, PathType

layer_name_to_min_width: Dict[str, float]

RuleType = Literal[
    "width",
    "space",
    "enclosing",
]


def rule_width(value: float, layer: str, angle_limit: float = 90) -> str:
    """Min feature size"""
    category = "width"
    error = f"{layer} {category} {value}um"
    return (
        f"{layer}.{category}({value}, angle_limit({angle_limit}))"
        f".output('{error}', '{error}')"
    )


def rule_space(value: float, layer: str, angle_limit: float = 90) -> str:
    """Min Space between shapes of layer"""
    category = "space"
    error = f"{layer} {category} {value}um"
    return (
        f"{layer}.{category}({value}, angle_limit({angle_limit}))"
        f".output('{error}', '{error}')"
    )


def rule_separation(value: float, layer1: str, layer2: str):
    """Min space between different layers"""
    error = f"min {layer1} {layer2} separation {value}um"
    return f"{layer1}.separation({layer2}, {value})" f".output('{error}', '{error}')"


def rule_enclosing(
    value: float, layer1: str, layer2: str, angle_limit: float = 90
) -> str:
    """Layer1 must be enclosed by layer2 by value.
    checks if layer1 encloses (is bigger than) layer2 by value
    """
    error = f"{layer1} enclosing {layer2} by {value}um"
    return (
        f"{layer1}.enclosing({layer2}, angle_limit({angle_limit}), {value})"
        f".output('{error}', '{error}')"
    )


def write_layer_definition(layer_map: Dict[str, Layer]) -> str:
    """Returns layer_map definition script for klayout

    Args:
        layer_map: can be dict or dataclass

    """
    layer_map = asdict(layer_map) if is_dataclass(layer_map) else layer_map
    return [
        f"{key} = input({value[0]}, {value[1]})" for key, value in layer_map.items()
    ]


def write_drc_deck(rules: List[str], layer_map: Dict[str, Layer]) -> str:
    """Returns drc_rule_deck for klayou

    Args:
        rules: list of rules
        layer_map: layer definitions can be dict or dataclass

    """
    script = []
    script += write_layer_definition(layer_map=layer_map)
    script += ["\n"]
    script += rules
    return "\n".join(script)


def write_drc_deck_macro(
    name="generic",
    filepath: Optional[PathType] = None,
    shortcut: str = "Ctrl+Shift+D",
    **kwargs,
) -> str:
    """Write script for klayout rule deck

    Args:
        name: drc rule deck name
        filepath: Optional macro path (defaults to .klayout/drc/name.lydrc)

    Keyword Args:
        rules: list of rules
        layer_map: layer definitions can be dict or dataclass

    Keyword Args:
        rules: list of rules
        layer_map: layer definitions can be dict or dataclass

    """
    script = f"""<?xml version="1.0" encoding="utf-8"?>
<klayout-macro>
 <description>{name} DRC</description>
 <version/>
 <category>drc</category>
 <prolog/>
 <epilog/>
 <doc/>
 <autorun>false</autorun>
 <autorun-early>false</autorun-early>
 <shortcut>{shortcut}</shortcut>
 <show-in-menu>true</show-in-menu>
 <group-name>drc_scripts</group-name>
 <menu-path>tools_menu.drc.end</menu-path>
 <interpreter>dsl</interpreter>
 <dsl-interpreter-name>drc-dsl-xml</dsl-interpreter-name>
 <text># {name} DRC

# Read about DRC scripts in the User Manual under "Design Rule Check (DRC)"
# Based on SOEN pdk https://github.com/usnistgov/SOEN-PDK/tree/master/tech/OLMAC
# http://klayout.de/doc/manual/drc_basic.html

report("generic DRC")
tiles(100)
tile_borders(2)
threads(3)
"""
    script += write_drc_deck(**kwargs)

    script += """
</text>
</klayout-macro>
"""
    filepath = filepath or get_klayout_path() / "drc" / f"{name}.lydrc"
    filepath = pathlib.Path(filepath)
    filepath.write_text(script)
    logger.info(f"Wrote DRC deck to {filepath}")
    return script


if __name__ == "__main__":
    import gdsfactory as gf

    rules = [
        rule_width(layer="WG", value=0.2),
        rule_space(layer="WG", value=0.2),
        rule_width(layer="M1", value=1),
        rule_width(layer="M2", value=2),
        rule_space(layer="M2", value=2),
        rule_separation(layer1="HEATER", layer2="M1", value=1.0),
        rule_enclosing(layer1="M1", layer2="VIAC", value=0.2),
    ]

    drc_rule_deck = write_drc_deck_macro(rules=rules, layer_map=gf.LAYER)
    print(drc_rule_deck)
