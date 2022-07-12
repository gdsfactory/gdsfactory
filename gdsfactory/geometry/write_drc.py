"""Write DRC rule decks in klayout.

TODO:

- add min area
- define derived layers (composed rules)
"""

import pathlib
from dataclasses import asdict, is_dataclass
from typing import List, Optional

from gdsfactory.config import logger
from gdsfactory.install import get_klayout_path
from gdsfactory.types import Dict, Layer, PathType

layer_name_to_min_width: Dict[str, float]


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


def rule_separation(value: float, layer1: str, layer2: str) -> str:
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


def write_layer_definition(layer_map: Dict[str, Layer]) -> List[str]:
    """Returns layer_map definition script for klayout

    Args:
        layer_map: can be dict or dataclass

    """
    layer_map = asdict(layer_map) if is_dataclass(layer_map) else layer_map
    layer_map = dict(layer_map)
    return [
        f"{key} = input({value[0]}, {value[1]})" for key, value in layer_map.items()
    ]


def write_drc_deck(rules: List[str], layer_map: Dict[str, Layer]) -> str:
    """Returns drc_rule_deck for klayout.

    Args:
        rules: list of rules
        layer_map: layer definitions can be dict or dataclass
    """
    script = []
    script += write_layer_definition(layer_map=layer_map)
    script += ["\n"]
    script += rules
    return "\n".join(script)


modes = ["tiled", "default", "deep"]


def write_drc_deck_macro(
    name: str = "generic",
    filepath: Optional[PathType] = None,
    shortcut: str = "Ctrl+Shift+D",
    mode: str = "tiled",
    threads: int = 4,
    tile_size: int = 500,
    tile_borders: Optional[int] = None,
    **kwargs,
) -> str:
    """Write klayout DRC macro.

    You can customize the shortcut to run the DRC macro from the Klayout GUI.

    Args:
        name: drc rule deck name.
        filepath: Optional macro path (defaults to .klayout/drc/name.lydrc).
        shortcut: to run macro from klayout GUI.
        mode: tiled, default or deep (hiearchical).
        threads: number of threads.
        tile_size: in um for tile mode.
        tile_borders: sides for each. Defaults None to automatic.

    Keyword Args:
        rules: list of rules.
        layer_map: layer definitions can be dict or dataclass.

    modes:

    - default
        - flat polygon handling
        - single thread
        - no overhead
        - use for small layout
        - no side effects
    - tiled
        - need to optimize tile size (maybe 500x500um). Works of each tile individually.
        - finite lookup range
        - output is flat
        - multithreading enable
    - deep
        - hierarchical mode
        - experimental
        - preserves hierarchical

    .. code::

        import gdsfactory as gf
        from gdsfactory.geometry.write_drc import (
            write_drc_deck_macro,
            rule_enclosing,
            rule_width,
            rule_space,
            rule_separation,
        )
        rules = [
            rule_width(layer="WG", value=0.2),
            rule_space(layer="WG", value=0.2),
            rule_width(layer="M1", value=1),
            rule_width(layer="M2", value=2),
            rule_space(layer="M2", value=2),
            rule_separation(layer1="HEATER", layer2="M1", value=1.0),
            rule_enclosing(layer1="VIAC", layer2="M1", value=0.2),
        ]

        drc_rule_deck = write_drc_deck_macro(rules=rules, layer_map=gf.LAYER)
        print(drc_rule_deck)

    """

    if mode not in modes:
        raise ValueError(f"{mode!r} not in {modes}")

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

report("{name} DRC")
"""

    if mode == "tiled":
        script += f"""
threads({threads})
tiles({tile_size})
"""
        if tile_borders:
            script += f"""
tile_borders({tile_borders})
"""
    elif mode == "deep":
        script += """
deep
"""

    script += write_drc_deck(**kwargs)

    script += """
</text>
</klayout-macro>
"""
    filepath = filepath or get_klayout_path() / "drc" / f"{name}.lydrc"
    dirpath = filepath.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = pathlib.Path(filepath)
    filepath.write_text(script)
    logger.info(f"Wrote DRC deck to {str(filepath)!r} with shortcut {shortcut!r}")
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
        rule_enclosing(layer1="VIAC", layer2="M1", value=0.2),
    ]

    drc_rule_deck = write_drc_deck_macro(rules=rules, layer_map=gf.LAYER)
    print(drc_rule_deck)
