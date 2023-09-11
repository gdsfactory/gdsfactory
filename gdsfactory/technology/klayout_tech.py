"""Classes and utils for working with KLayout technology files (.lyp, .lyt).

This module enables conversion between gdsfactory settings and KLayout technology.
"""

import pathlib
import xml.etree.ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

from gdsfactory.config import PATH
from gdsfactory.technology import LayerStack, LayerViews
from gdsfactory.technology.xml_utils import make_pretty_xml
from gdsfactory.typings import PathType

try:
    import klayout.db as db
except ImportError as e:
    print("You can install `pip install klayout.")
    raise e

Layer = tuple[int, int]
ConductorViaConductorName = tuple[str, str, str]

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

    # TODO: Add import method
    # TODO: Also interop with xs scripts?

    name: str
    layer_map: dict[str, Layer]
    layer_views: LayerViews | None = None
    layer_stack: LayerStack | None = None
    connectivity: list[ConductorViaConductorName] | None = None
    technology: db.Technology = Field(default_factory=db.Technology)

    def write_tech(
        self,
        tech_dir: PathType,
        lyp_filename: str = "layers.lyp",
        lyt_filename: str = "tech.lyt",
        d25_filename: str | None = None,
        mebes_config: dict | None = None,
    ) -> None:
        """Write technology files into 'tech_dir'.

        Args:
            tech_dir: Where to write the technology files to.
            lyp_filename: Name of the layer properties file.
            lyt_filename: Name of the layer technology file.
            d25_filename: Name of the 2.5D stack file (only works on KLayout >= 0.28). Defaults to self.name.
            mebes_config: A dictionary specifying the KLayout mebes reader config.

        """

        d25_filename = d25_filename or f"{self.name}.lyd25"

        tech_path = pathlib.Path(tech_dir)
        lyp_path = tech_path / lyp_filename
        lyt_path = tech_path / lyt_filename
        d25_path = tech_path / "d25" / d25_filename
        d25_dir = tech_path / "d25"
        d25_dir.mkdir(exist_ok=True, parents=True)

        if not self.technology.name:
            self.technology.name = self.name

        self.technology.layer_properties_file = lyp_path.name

        if self.layer_views:
            self.layer_views.to_lyp(lyp_path)
            print(f"Wrote {str(lyp_path)!r}")

        root = ET.XML(self.technology.to_xml().encode("utf-8"))

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
        lefdef_idx = list(reader_opts).index(reader_opts.find("lefdef"))
        reader_opts.insert(lefdef_idx + 1, mebes)

        if self.layer_stack:
            dbu = len(str(self.technology.dbu).split(".")[-1])
            d25_script = (
                prefix_d25
                + self.layer_stack.get_klayout_3d_script(
                    layer_views=self.layer_views,
                    dbu=dbu,
                )
                + suffix_d25
            )
            d25_path.write_bytes(d25_script.encode("utf-8"))
            print(f"Wrote {str(d25_path)!r}")

        self._define_connections(root)

        lyt_path.write_bytes(make_pretty_xml(root))
        print(f"Wrote {str(lyt_path)!r}")

    def _define_connections(self, root) -> None:
        if not self.connectivity:
            return
        src_element = [e for e in list(root) if e.tag == "connectivity"]
        if len(src_element) != 1:
            raise KeyError("Could not get a single index for the src element.")
        src_element = src_element[0]
        layers = set()
        for layer_name_c1, layer_name_via, layer_name_c2 in self.connectivity:
            connection = ",".join([layer_name_c1, layer_name_via, layer_name_c2])

            layers.add(layer_name_c1)
            layers.add(layer_name_via)
            layers.add(layer_name_c2)

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
        ignore_extra=True,
        extra="ignore",
    )


layer_views = LayerViews.from_lyp(str(PATH.klayout_lyp))


def yaml_test() -> None:
    tech_dir = PATH.repo / "extra" / "test_tech"

    # Load from existing layer properties file
    lyp = LayerViews.from_lyp(str(PATH.klayout_lyp))
    print("Loaded from .lyp", lyp)

    # Export layer properties to yaml files
    layer_yaml = str(tech_dir / "layers.yml")
    lyp.to_yaml(layer_yaml)

    # Load layer properties from yaml files and check that they're the same
    lyp_loaded = LayerViews.from_yaml(layer_yaml)
    print("Loaded from .yaml", lyp_loaded)

    assert lyp_loaded == lyp


if __name__ == "__main__":
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
        layer_map=dict(LAYER),
        layer_stack=LAYER_STACK,
    )
    tech_dir = PATH.klayout_tech
    # tech_dir = pathlib.Path("/home/jmatres/.klayout/salt/gdsfactory/tech/")
    tech_dir.mkdir(exist_ok=True, parents=True)
    generic_tech.write_tech(tech_dir=tech_dir)

    # yaml_test()
