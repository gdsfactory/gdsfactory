"""Classes and utils for working with KLayout technology files (.lyp, .lyt).

This module enables conversion between gdsfactory settings and KLayout technology.
"""

import pathlib
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import Any

import aenum
import klayout.db as db
from pydantic import BaseModel, ConfigDict, field_validator

from gdsfactory.config import PATH
from gdsfactory.technology import LayerStack, LayerViews
from gdsfactory.technology.xml_utils import make_pretty_xml
from gdsfactory.typings import ConnectivitySpec, PathType

prefix_d25 = """<?xml version="1.0" encoding="utf-8"?>
<klayout-macro>
 <description/>
 <version/>
 <category>d25</category>
 <prolog/>
 <epilog/>
 <doc/>
 <autorun>false</autorun>
 <autorun-early>false</autorun-early>
 <priority>0</priority>
 <shortcut/>
 <show-in-menu>true</show-in-menu>
 <group-name>d25_scripts</group-name>
 <menu-path>tools_menu.d25.end</menu-path>
 <interpreter>dsl</interpreter>
 <dsl-interpreter-name>d25-dsl-xml</dsl-interpreter-name>
 <text>

"""
suffix_d25 = """
</text>
</klayout-macro>
"""


class KLayoutTechnology(BaseModel):
    """A container for working with KLayout technologies (requires KLayout Python package).

    Useful to import/export Layer Properties (.lyp) and Technology (.lyt) files.

    Properties:
        name: technology name.
        layer_map: Maps names to GDS layer numbers.
        layer_views: Defines all the layer display properties needed for a .lyp file from LayerView objects.
        technology: KLayout Technology object from the KLayout API. Set name, dbu, etc.
        connectivity: List of layer names connectivity for netlist tracing.
    """

    name: str
    layer_map: dict[str, tuple[int, int]]
    layer_views: LayerViews | None = None
    layer_stack: LayerStack | None = None
    connectivity: Sequence[ConnectivitySpec] | None = None

    @field_validator("layer_map", mode="before")
    @classmethod
    def check_layer_map(cls, layer_map: Any) -> Any:
        if isinstance(layer_map, aenum.EnumType):
            _layer_map: dict[str, tuple[int, int]] = {
                name: (layer_enum.layer, layer_enum.datatype)
                for name, layer_enum in layer_map.__members__.items()
            }
            return _layer_map
        return layer_map

    def write_tech(
        self,
        tech_dir: PathType,
        lyp_filename: str = "layers.lyp",
        lyt_filename: str = "tech.lyt",
        d25_filename: str | None = None,
        mebes_config: dict[str, Any] | None = None,
    ) -> None:
        """Write technology files into 'tech_dir'.

        Args:
            tech_dir: Where to write the technology files to.
            lyp_filename: Name of the layer properties file.
            lyt_filename: Name of the layer technology file.
            d25_filename: Name of the 2.5D stack file (only works on KLayout >= 0.28). Defaults to self.name.
            mebes_config: A dictionary specifying the KLayout mebes reader config.

        """
        technology = db.Technology()
        d25_filename = d25_filename or f"{self.name}.lyd25"
        tech_path = pathlib.Path(tech_dir)
        lyp_path = tech_path / lyp_filename
        lyt_path = tech_path / lyt_filename
        d25_path = tech_path / "d25" / d25_filename
        d25_dir = tech_path / "d25"
        d25_dir.mkdir(exist_ok=True, parents=True)

        if not technology.name:
            technology.name = self.name

        technology.layer_properties_file = lyp_path.name

        if self.layer_views:
            self.layer_views.to_lyp(lyp_path)
            print(f"Wrote {str(lyp_path)!r}")

        root = ET.XML(technology.to_xml().encode("utf-8"))

        # KLayout tech doesn't include mebes config, so add it after lefdef config:
        if not mebes_config:
            mebes_config = {
                "invert": False,
                "subresolution": True,
                "produce-boundary": True,
                "num-stripes-per-cell": 64,
                "num-shapes-per-cell": 0,
                "data-layer": 1,
                "data-datatype": 0,
                "data-name": "DATA",
                "boundary-layer": 0,
                "boundary-datatype": 0,
                "boundary-name": "BORDER",
                "layer-map": "layer_map()",
                "create-other-layers": True,
            }
        mebes = ET.Element("mebes")
        for k, v in mebes_config.items():
            v = str(v).lower() if isinstance(v, bool) else str(v)
            ET.SubElement(mebes, k).text = v

        reader_opts = root.find("reader-options")
        if reader_opts is not None:
            lefdef_idx = list(reader_opts).index(reader_opts.find("lefdef"))  # type: ignore[arg-type]
            reader_opts.insert(lefdef_idx + 1, mebes)

        # FIXME
        if self.layer_stack:
            print(d25_path)
        #     dbu = len(str(technology.dbu).split(".")[-1])
        #     d25_script = (
        #         prefix_d25
        #         + self.layer_stack.get_klayout_3d_script(
        #             layer_views=self.layer_views,
        #             dbu=dbu,
        #         )
        #         + suffix_d25
        #     )
        #     d25_path.write_bytes(d25_script.encode("utf-8"))
        #     print(f"Wrote {str(d25_path)!r}")

        self._define_connections(root)

        lyt_path.write_bytes(make_pretty_xml(root))
        print(f"Wrote {str(lyt_path)!r}")

    def _define_connections(self, root: ET.Element) -> None:
        if not self.connectivity:
            return
        src_elements = [e for e in list(root) if e.tag == "connectivity"]
        if len(src_elements) != 1:
            raise KeyError("Could not get a single index for the src element.")
        src_element = src_elements[0]
        layers: set[str] = set()
        for first_layer_name, *layer_names in self.connectivity:
            connection = ",".join(
                [first_layer_name]
                + (layer_names if len(layer_names) == 2 else ["", *layer_names])
            )

            for layer_name in layer_names:
                layers.add(layer_name)

            ET.SubElement(src_element, "connection").text = connection

        if self.layer_map:
            for layer in layers:
                ET.SubElement(
                    src_element, "symbols"
                ).text = (
                    f"{layer}='{self.layer_map[layer][0]}/{self.layer_map[layer][1]}'"
                )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


if __name__ == "__main__":
    import klayout.db as kdb

    from gdsfactory.config import PATH
    from gdsfactory.generic_tech import LAYER, LAYER_STACK

    lyp = LayerViews(PATH.klayout_yaml)
    # lyp = LayerViews.from_lyp(str(PATH.klayout_yaml))

    # str_xml = open(PATH.klayout_tech / "tech.lyt").read()
    # new_tech = db.Technology.technology_from_xml(str_xml)
    # generic_tech = KLayoutTechnology(layer_views=lyp)
    connectivity = [
        ("NPP", "VIAC", "M1"),
        ("PPP", "VIAC", "M1"),
        ("M1", "VIA1", "M2"),
        ("M2", "VIA2", "M3"),
    ]

    c = generic_tech = KLayoutTechnology(
        name="generic_tech",
        layer_views=lyp,
        connectivity=connectivity,
        layer_map=LAYER,  # type: ignore[arg-type]
        layer_stack=LAYER_STACK,
    )
    tech_dir = PATH.klayout_tech
    tech_dir.mkdir(exist_ok=True, parents=True)
    generic_tech.write_tech(tech_dir=tech_dir)

    Tech = kdb.Technology()
    technology = kdb.Technology.technology_from_xml(str(PATH.klayout_lyt))
