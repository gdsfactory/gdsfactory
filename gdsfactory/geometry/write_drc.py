"""Write DRC rule decks in klayout."""

import pathlib
from dataclasses import asdict, is_dataclass
from typing import List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import BaseModel

from gdsfactory.config import logger
from gdsfactory.install import get_klayout_path
from gdsfactory.types import Dict, Layer, PathType

layer_name_to_min_width: Dict[str, float]

RuleType = Literal[
    "width",
    "space",
    "enclosing",
]


class DrcRule(BaseModel):
    category: RuleType
    value: float
    layer1: str
    layer2: Optional[str] = None
    angle_limit: float = 90

    def get_string(self):
        if self.layer2:
            error = f"{self.layer1} {self.category} {self.value}"
            return (
                f"{self.layer1}.{self.category}({self.layer2}, angle_limit({self.angle_limit}), {self.value})"
                f".output('{error}', '{self.layer2} minimum {self.category} {self.value}')"
            )

        else:
            error = f"{self.layer1} {self.category} {self.value}"
            return (
                f"{self.layer1}.{self.category}({self.value}, angle_limit({self.angle_limit}))"
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


def write_drc_deck(rules: List[DrcRule], layer_map: Dict[str, Layer]) -> str:
    """Returns drc_rule_deck for klayou

    Args:
        rules: list of rules
        layer_map: layer definitions can be dict or dataclass

    """
    script = []
    script += write_layer_definition(layer_map=layer_map)
    script += ["\n"]
    script += [rule.get_string() for rule in rules]
    return "\n".join(script)


def write_drc_deck_macro(
    name="generic", filepath: Optional[PathType] = None, **kwargs
) -> str:
    """Write script for klayout rule deck

    Args:
        name: drc rule deck name

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
 <shortcut>Ctrl+Shift+D</shortcut>
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
        DrcRule(category="width", layer1="WG", value=0.2),
        DrcRule(category="space", layer1="WG", value=0.2),
        DrcRule(category="width", layer1="M1", value=1),
        DrcRule(category="width", layer1="M2", value=2),
        DrcRule(category="space", layer1="M2", value=2),
        DrcRule(category="enclosing", layer1="M2", layer2="M1", value=2),
    ]

    drc_rule_deck = write_drc_deck_macro(rules=rules, layer_map=gf.LAYER)
    print(drc_rule_deck)
