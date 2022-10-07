"""Classes and utilities for working with KLayout technology files (.lyp, .lyt).

This module will enable conversion between gdsfactory settings and KLayout technology.
"""

import os
import pathlib
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from lxml import etree
from pydantic import BaseModel, Field, validator
from typing_extensions import Literal

from gdsfactory.config import PATH
from gdsfactory.tech import LayerStack

Layer = Tuple[int, int]
ConductorViaConductorName = Tuple[str, str, str]


def append_file_extension(filename: Union[str, pathlib.Path], extension: str) -> str:
    """Try appending extension to file."""
    # Handle whether given with '.'
    if "." not in extension:
        extension = f".{extension}"

    if isinstance(filename, str) and not filename.endswith(extension):
        filename += extension

    if isinstance(filename, pathlib.Path) and not str(filename).endswith(extension):
        filename = filename.with_suffix(extension)
    return filename


class LayerView(BaseModel):
    """KLayout layer properties.

    Docstrings copied from KLayout documentation (with some modifications):
    https://www.klayout.de/lyp_format.html

    Parameters:
        layer: GDSII layer.
        name: Name of the Layer.
        layer_in_name: Whether to display the name as 'name layer/datatype' rather than just the layer.
        width: This is the line width of the frame in pixels (or empty for the default which is 1).
        line_style: This is the number of the line style used to draw the shape boundaries.
            An empty string is "solid line". The values are "Ix" for one of the built-in styles
            where "I0" is "solid", "I1" is "dotted" etc.
        dither_pattern: This is the number of the dither pattern used to fill the shapes.
            The values are "Ix" for one of the built-in pattern where "I0" is "solid" and "I1" is "clear".
        frame_color: The color of the frames.
        fill_color: The color of the fill pattern inside the shapes.
        animation: This is a value indicating the animation mode.
            0 is "none", 1 is "scrolling", 2 is "blinking" and 3 is "inverse blinking".
        fill_brightness: This value modifies the brightness of the fill color. See "frame-brightness".
        frame_brightness: This value modifies the brightness of the frame color.
            0 is unmodified, -100 roughly adds 50% black to the color which +100 roughly adds 50% white.
        xfill: Whether boxes are drawn with a diagonal cross.
        marked: Whether the entry is marked (drawn with small crosses).
        transparent: Whether the entry is transparent.
        visible: Whether the entry is visible.
        valid: Whether the entry is valid. Invalid layers are drawn but you can't select shapes on those layers.
    """

    layer: Optional[Layer] = None
    name: str = "unnamed"
    layer_in_name: bool = False
    frame_color: Optional[str] = None
    fill_color: Optional[str] = None
    frame_brightness: Optional[int] = 0
    fill_brightness: Optional[int] = 0
    dither_pattern: Optional[str] = None
    line_style: Optional[str] = None
    valid: bool = True
    visible: bool = True
    transparent: bool = False
    width: Optional[int] = None
    marked: bool = False
    xfill: bool = False
    animation: int = 0
    group_members: Optional[Dict[str, "LayerView"]] = None

    def __init__(self, **data):
        """Initialize LayerView object."""
        super().__init__(**data)
        for key, val in self.dict().items():
            if key not in LayerView.__fields__.keys():
                if self.group_members is None:
                    self.group_members = {}
                if val["name"] != key:
                    val["name"] = key
                self.group_members[key] = LayerView(**val)

    @validator("frame_color", "fill_color")
    def color_is_valid(cls, color):
        import numpy as np
        from matplotlib.colors import CSS4_COLORS

        try:
            if color is None:  # not specified
                color = None
            elif np.size(color) == 3:  # in format (0.5, 0.5, 0.5)
                color = np.array(color)
                if np.any(color > 1) or np.any(color < 0):
                    raise ValueError
                color = np.array(np.round(color * 255), dtype=int)
                color = "#{:02x}{:02x}{:02x}".format(*color)
            elif color[0] == "#":  # in format #1d2e3f
                if len(color) != 7:
                    raise ValueError
                int(color[1:], 16)  # Will throw error if not hex format
            else:  # in named format 'gold'
                color = CSS4_COLORS[color.lower()]
        except Exception as error:
            raise ValueError(
                "LayerView color must be specified as a "
                "0-1 RGB triplet, (e.g. [0.5, 0.1, 0.9]), an HTML hex color string "
                "(e.g. '#a31df4'), or a CSS3 color name (e.g. 'gold' or "
                "see http://www.w3schools.com/colors/colors_names.asp )"
            ) from error
        return color

    def __str__(self):
        """Returns a formatted view of properties and their values."""
        return "LayerView:\n\t" + "\n\t".join(
            [f"{k}: {v}" for k, v in self.dict().items()]
        )

    def __repr__(self):
        """Returns a formatted view of properties and their values."""
        return self.__str__()

    def _get_xml_element(self, tag: str) -> etree.Element:
        """Get XML Element from attributes."""
        prop_keys = [
            "frame-color",
            "fill-color",
            "frame-brightness",
            "fill-brightness",
            "dither-pattern",
            "line-style",
            "valid",
            "visible",
            "transparent",
            "width",
            "marked",
            "xfill",
            "animation",
            "name",
            "source",
        ]
        el = etree.Element(tag)
        for prop_name in prop_keys:
            if prop_name == "source":
                layer = self.layer
                prop_val = f"{layer[0]}/{layer[1]}@1" if layer else "*/*@*"
            elif prop_name == "name" and self.layer_in_name:
                prop_val = f"{self.name} {self.layer[0]}/{self.layer[1]}"
            else:
                prop_val = getattr(self, "_".join(prop_name.split("-")), None)
                if isinstance(prop_val, bool):
                    prop_val = f"{prop_val}".lower()
            subel = etree.SubElement(el, prop_name)
            if prop_val is not None:
                subel.text = str(prop_val)
        return el

    def to_xml(self) -> etree.Element:
        """Return an XML representation of the LayerView."""
        props = self._get_xml_element("properties")

        if self.group_members is not None:
            for gm in self.group_members.values():
                props.append(gm._get_xml_element("group-members"))
        return props


class CustomPattern(BaseModel):
    """Custom pattern."""

    name: str
    order: int
    pattern: str
    pattern_type: Literal["dither", "line"]

    def to_xml(self) -> etree.Element:
        el = etree.Element(f"custom-{self.pattern_type}-pattern")

        subel = etree.SubElement(el, "pattern")
        lines = self.pattern.split("\n")
        if len(lines) == 1:
            subel.text = lines[0]
        else:
            for line in lines:
                etree.SubElement(subel, "line").text = line

        etree.SubElement(el, "order").text = str(self.order)
        etree.SubElement(el, "name").text = self.name
        return el


def _process_name(
    name: str, layer_pattern: Union[str, re.Pattern]
) -> Optional[Tuple[str, bool]]:
    r"""Strip layer info from name if it exists.

    Args:
        name: XML-formatted name entry.
        layer_pattern: Regex pattern to match layers with. Defaults to r'(\d+|\*)/(\d+|\*)'.
    """
    if not name:
        return None
    layer_in_name = False
    match = re.search(layer_pattern, name)
    if match:
        name = name[: match.start()].strip()
        layer_in_name = True
    return name, layer_in_name


def _process_layer(
    layer: str, layer_pattern: Union[str, re.Pattern]
) -> Optional[Layer]:
    r"""Convert .lyp XML layer entry to a Layer.

    Args:
        layer: XML-formatted layer entry.
        layer_pattern: Regex pattern to match layers with. Defaults to r'(\d+|\*)/(\d+|\*)'.
    """
    match = re.search(layer_pattern, layer)
    if not match:
        raise OSError(f"Could not read layer {layer}!")
    v = match.group().split("/")
    return None if v == ["*", "*"] else (int(v[0]), int(v[1]))


def _properties_to_layerview(
    element, layer_pattern: Union[str, re.Pattern]
) -> Optional[LayerView]:
    r"""Read properties from .lyp XML and generate LayerViews from them.

    Args:
        element: XML Element to iterate over.
        tag: Optional tag to iterate over.
        layer_pattern: Regex pattern to match layers with. Defaults to r'(\d+|\*)/(\d+|\*)'.
    """
    prop_dict = {"layer_in_name": False}
    for prop in element.iterchildren():
        prop_tag = prop.tag
        if prop_tag == "name":
            val = _process_name(prop.text, layer_pattern)
            if val is None:
                return None
            if isinstance(val, tuple):
                val, layer_in_name = val
                prop_dict["layer_in_name"] = layer_in_name
        elif prop_tag == "source":
            val = _process_layer(prop.text, layer_pattern)
            prop_tag = "layer"
        elif prop_tag == "group-members":
            props = [
                _properties_to_layerview(e, layer_pattern)
                for e in element.iterchildren("group-members")
            ]
            val = {p.name: p for p in props}
        else:
            val = prop.text
        prop_tag = "_".join(prop_tag.split("-"))
        prop_dict[prop_tag] = val
    return LayerView(**prop_dict)


LayerViews = Dict[str, LayerView]


class LayerDisplayProperties(BaseModel):
    """A container for layer properties for KLayout layer property (.lyp) files."""

    layer_views: LayerViews = Field(default_factory=LayerViews)
    custom_dither_patterns: Optional[Dict[str, CustomPattern]] = None
    custom_line_styles: Optional[Dict[str, CustomPattern]] = None

    def __init__(self, **data):
        """Initialize LayerDisplayProperties object."""
        super().__init__(**data)

        for key, val in self.dict().items():

            if key not in self.__fields__:
                if val["name"] != key:
                    val["name"] = key
                    setattr(self, key, LayerView(**val))
                self.add_layer(**val)

    def add_layer(
        self,
        layer_display_props: Optional[LayerView] = None,
        layer: Optional[Layer] = None,
        name: str = "unnamed",
        valid: bool = True,
        visible: bool = True,
        transparent: bool = False,
        marked: bool = False,
        xfill: bool = False,
        frame_brightness: int = 0,
        fill_brightness: int = 0,
        animation: int = 0,
        frame_color: Optional[str] = None,
        fill_color: Optional[str] = None,
        dither_pattern: Optional[str] = None,
        line_style: Optional[str] = None,
        width: Optional[int] = None,
        group_members: Optional[Dict[str, LayerView]] = None,
    ) -> None:
        """Adds a layer to LayerDisplayProperties.

        Docstrings copied from KLayout documentation:
        https://www.klayout.de/lyp_format.html

        Args:
            layer_display_props: Add layer from existing LayerDisplayProperties, overrides all other args.
            layer: GDSII layer.
            name: Name of the Layer.
            width: This is the line width of the frame in pixels (or empty for the default which is 1).
            line_style: This is the number of the line style used to draw the shape boundaries.
                An empty string is "solid line". The values are "Ix" for one of the built-in styles
                where "I0" is "solid", "I1" is "dotted" etc.
            dither_pattern: This is the number of the dither pattern used to fill the shapes.
                The values are "Ix" for one of the built-in pattern where "I0" is "solid" and "I1" is "clear".
            frame_color: The color of the frames.
            fill_color: The color of the fill pattern inside the shapes.
            animation: This is a value indicating the animation mode.
                0 is "none", 1 is "scrolling", 2 is "blinking" and 3 is "inverse blinking".
            fill_brightness: This value modifies the brightness of the fill color. See "frame-brightness".
            frame_brightness: This value modifies the brightness of the frame color.
                0 is unmodified, -100 roughly adds 50% black to the color which +100 roughly adds 50% white.
            xfill: Whether boxes are drawn with a diagonal cross.
            marked: Whether the entry is marked (drawn with small crosses).
            transparent: Whether the entry is transparent.
            visible: Whether the entry is visible.
            valid: Whether the entry is valid. Invalid layers are drawn but shapes on those layers can't be selected.
            group_members: Optional dict of LayerViews to group beneath this LayerView.
        """
        new_layer = layer_display_props or LayerView(
            layer=layer,
            name=name,
            valid=valid,
            visible=visible,
            transparent=transparent,
            marked=marked,
            xfill=xfill,
            frame_brightness=frame_brightness,
            fill_brightness=fill_brightness,
            animation=animation,
            frame_color=frame_color,
            fill_color=fill_color,
            dither_pattern=dither_pattern,
            line_style=line_style,
            width=width,
            group_members=group_members,
        )
        if name in self.layer_views:
            raise ValueError(
                f"Adding {name!r} already defined {list(self.layer_views.keys())}"
            )
        else:
            self.layer_views[name] = new_layer

    def get_layer_views(self, exclude_groups: bool = False) -> Dict[str, LayerView]:
        """Return all LayerViews.

        Args:
            exclude_groups: Whether to exclude LayerViews that contain other LayerViews.
        """
        layers = {}
        for name, view in self.layer_views.items():
            if view.group_members is None:
                layers[name] = view
            elif not exclude_groups:
                for member_name, member_view in view.group_members.items():
                    layers[member_name] = member_view
        return layers

    def get_layer_view_groups(self) -> Dict[str, LayerView]:
        """Return the LayerViews that contain other LayerViews."""
        return {
            name: view
            for name, view in self.layer_views.items()
            if view.group_members is not None
        }

    def __str__(self) -> str:
        """Prints the number of LayerView objects in the LayerDisplayProperties object."""
        return (
            f"LayerDisplayProperties ({len(self.get_layer_views())} layers total, {len(self.get_layer_view_groups())} groups) \n"
            f"{self.get_layer_views()}"
        )

    def get(self, name: str) -> LayerView:
        """Returns Layer from name.

        Args:
            name: Name of layer.
        """
        if name not in self.layer_views:
            raise ValueError(f"Layer {name!r} not in {list(self.layer_views.keys())}")
        else:
            return self.layer_views[name]

    def __getitem__(self, val):
        """Allows accessing to the layer names like ls['gold2'].

        Args:
            val: Layer name to access within the LayerDisplayProperties.

        Returns:
            self.layers[val]: LayerView in the LayerDisplayProperties.

        """
        try:
            return self.layer_views[val]
        except Exception as error:
            raise ValueError(
                f"LayerView {val!r} not in LayerDisplayProperties {list(self.layer_views.keys())}"
            ) from error

    def get_from_tuple(self, layer_tuple: Layer) -> LayerView:
        """Returns LayerView from layer tuple.

        Args:
            layer_tuple: Tuple of (gds_layer, gds_datatype).

        Returns:
            LayerView
        """
        tuple_to_name = {v.layer: k for k, v in self.layer_views.items()}
        if layer_tuple not in tuple_to_name:
            raise ValueError(
                f"LayerView {layer_tuple} not in {list(tuple_to_name.keys())}"
            )

        name = tuple_to_name[layer_tuple]
        return self.layer_views[name]

    def get_layer_tuples(self) -> Set[Layer]:
        """Returns a tuple for each layer."""
        return {layer.layer for layer in self.get_layer_views().values()}

    def to_lyp(self, filepath: str, overwrite: bool = True) -> None:
        """Write all layer properties to a KLayout .lyp file.

        Args:
            filepath: to write the .lyp file to (appends .lyp extension if not present).
            overwrite: Whether to overwrite an existing file located at the filepath.
        """
        filepath = pathlib.Path(filepath)
        filepath = append_file_extension(filepath, ".lyp")

        if os.path.exists(filepath) and not overwrite:
            raise OSError("File exists, cannot write.")

        root = etree.Element("layer-properties")

        for lv in self.layer_views.values():
            root.append(lv.to_xml())

        for dp in self.custom_dither_patterns.values():
            root.append(dp.to_xml())

        for ls in self.custom_line_styles.values():
            root.append(ls.to_xml())

        with open(filepath, "wb") as file:
            file.write(
                etree.tostring(
                    root, encoding="utf-8", pretty_print=True, xml_declaration=True
                )
            )

    @staticmethod
    def from_lyp(
        filepath: str, layer_pattern: Optional[Union[str, re.Pattern]] = None
    ) -> "LayerDisplayProperties":
        r"""Write all layer properties to a KLayout .lyp file.

        Args:
            filepath: to write the .lyp file to (appends .lyp extension if not present).
            layer_pattern: Regex pattern to match layers with. Defaults to r'(\d+|\*)/(\d+|\*)'.
        """
        layer_pattern = re.compile(layer_pattern or r"(\d+|\*)/(\d+|\*)")

        filepath = append_file_extension(filepath, ".lyp")

        if not os.path.exists(filepath):
            raise OSError("File not found!")

        tree = etree.parse(filepath)
        root = tree.getroot()
        if root.tag != "layer-properties":
            raise OSError("Layer properties file incorrectly formatted, cannot read.")

        layer_views = {}
        for layer_block in root.iter("properties"):
            lv = _properties_to_layerview(layer_block, layer_pattern=layer_pattern)
            if lv is not None:
                layer_views[lv.name] = lv

        custom_dither_patterns = {}
        for dither_block in root.iterchildren("custom-dither-pattern"):
            name = dither_block.find("name").text
            if name is None:
                continue
            custom_dither_patterns[name] = CustomPattern(
                pattern_type="dither",
                name=name,
                order=dither_block.find("order").text,
                pattern="\n".join(
                    [line.text for line in dither_block.find("pattern").iterchildren()]
                ),
            )
        custom_line_styles = {}
        for line_block in root.iterchildren("custom-line-style"):
            name = line_block.find("name").text
            if name is None:
                continue
            custom_line_styles[name] = CustomPattern(
                pattern_type="line",
                name=name,
                order=line_block.find("order").text,
                pattern="\n".join(
                    [line.text for line in line_block.find("pattern").iterchildren()]
                ),
            )
        return LayerDisplayProperties(
            layer_views=layer_views,
            custom_dither_patterns=custom_dither_patterns,
            custom_line_styles=custom_line_styles,
        )


class KLayoutTechnology(BaseModel):
    """A container for working with KLayout technologies (requires klayout Python package).

    Useful for importing/exporting Layer Properties (.lyp) and Technology (.lyt) files.

    Properties:
        layer_properties: Defines all the layer display properties needed for a .lyp file from LayerView objects.
        technology: KLayout Technology object from the KLayout API. Set name, dbu, etc.
        connectivity: List of layer names connectivity for netlist tracing.
    """

    import klayout.db as db

    name: str
    layer_properties: Optional[LayerDisplayProperties] = None
    technology: db.Technology = Field(default_factory=db.Technology)
    connectivity: Optional[List[ConductorViaConductorName]] = None

    def export_technology_files(
        self,
        tech_dir: str,
        lyp_filename: str = "layers",
        lyt_filename: str = "tech",
        layer_stack: Optional[LayerStack] = None,
    ) -> None:
        """Write technology files into 'tech_dir'.

        Args:
            tech_dir: Where to write the technology files to.
            lyp_filename: Name of the layer properties file.
            lyt_filename: Name of the layer technology file.
            layer_stack: If specified, write a 2.5D section in the technology file based on the LayerStack.
        """
        # Format file names if necessary
        lyp_filename = append_file_extension(lyp_filename, ".lyp")
        lyt_filename = append_file_extension(lyt_filename, ".lyt")

        tech_path = pathlib.Path(tech_dir)
        lyp_path = tech_path / lyp_filename
        lyt_path = tech_path / lyt_filename

        # Specify relative file name for layer properties file
        self.technology.layer_properties_file = lyp_filename

        # TODO: Also interop with xs scripts?

        # Write lyp to file
        self.layer_properties.to_lyp(lyp_path)

        root = etree.XML(self.technology.to_xml().encode("utf-8"))
        subelement = etree.SubElement(root, "name")
        subelement.text = self.name

        if layer_stack is not None:
            # KLayout 0.27.x won't have a way to read/write the 2.5D info for technologies, so add manually
            # Should be easier in 0.28.x
            d25_element = [e for e in list(root) if e.tag == "d25"]
            if len(d25_element) != 1:
                raise KeyError("Could not get a single index for the d25 element.")
            d25_element = d25_element[0]

            src_element = [e for e in list(d25_element) if e.tag == "src"]
            if len(src_element) != 1:
                raise KeyError("Could not get a single index for the src element.")
            src_element = src_element[0]

            for layer_level in layer_stack.layers.values():
                src_element.text += f"{layer_level.layer[0]}/{layer_level.layer[1]}: {layer_level.zmin} {layer_level.thickness}\n"

        # root['connectivity']['connection']= '41/0,44/0,45/0'
        if connectivity is not None:
            src_element = [e for e in list(root) if e.tag == "connectivity"]
            if len(src_element) != 1:
                raise KeyError("Could not get a single index for the src element.")
            src_element = src_element[0]
            for layer_name_c1, layer_name_via, layer_name_c2 in self.connectivity:
                layer_c1 = self.layer_properties.layer_views[layer_name_c1].layer
                layer_via = self.layer_properties.layer_views[layer_name_via].layer
                layer_c2 = self.layer_properties.layer_views[layer_name_c2].layer
                connection = (
                    ",".join(
                        [
                            f"{layer[0]}/{layer[1]}"
                            for layer in [layer_c1, layer_via, layer_c2]
                        ]
                    )
                    + "\n"
                )

                subelement = etree.SubElement(src_element, "connection")
                subelement.text = connection

        script = etree.tostring(
            root,
            encoding="utf-8",
            pretty_print=True,
            xml_declaration=True,
        ).decode("utf8")

        # Write lyt to file
        lyt_path.write_text(script)

    class Config:
        """Allow db.Technology type."""

        arbitrary_types_allowed = True


LAYER_PROPERTIES = LayerDisplayProperties.from_lyp(str(PATH.klayout_lyp))

if __name__ == "__main__":
    from gdsfactory.tech import LAYER_STACK

    lyp = LayerDisplayProperties.from_lyp(str(PATH.klayout_lyp))

    # str_xml = open(PATH.klayout_tech / "tech.lyt").read()
    # new_tech = db.Technology.technology_from_xml(str_xml)

    connectivity = [("M1", "VIA1", "M2"), ("M2", "VIA2", "M3")]

    c = generic_tech = KLayoutTechnology(
        name="generic", layer_properties=lyp, connectivity=connectivity
    )
    tech_dir = PATH.repo / "extra" / "test_tech"
    # tech_dir = pathlib.Path("/home/jmatres/.klayout/salt/gdsfactory/tech/")
    tech_dir.mkdir(exist_ok=True, parents=True)

    generic_tech.export_technology_files(tech_dir=tech_dir, layer_stack=LAYER_STACK)
