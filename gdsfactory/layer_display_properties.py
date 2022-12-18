"""Classes and utilities for working with KLayout technology files (.lyp, .lyt).

This module enables conversion between gdsfactory settings and KLayout technology.
"""

import os
import pathlib
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Set, Tuple, Union
from xml.dom.minidom import Node

from pydantic import BaseModel, Field, validator

Layer = Tuple[int, int]


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


def _strip_xml(node):
    """Strip XML of excess whitespace.

    Source: https://stackoverflow.com/a/16919069
    """
    for x in node.childNodes:
        if x.nodeType == Node.TEXT_NODE:
            if x.nodeValue:
                x.nodeValue = x.nodeValue.strip()
        elif x.nodeType == Node.ELEMENT_NODE:
            _strip_xml(x)


def make_pretty_xml(root: ET.Element) -> bytes:
    xml_doc = xml.dom.minidom.parseString(ET.tostring(root))

    _strip_xml(xml_doc)
    xml_doc.normalize()

    return xml_doc.toprettyxml(indent=" ", newl="\n", encoding="utf-8")


class LayerColor(BaseModel):
    """Fill and frame color.

    Attributes:
        frame: The color of the frames.
        fill: The color of the fill pattern inside the shapes.
    """

    fill: Optional[str] = None
    frame: Optional[str] = None

    @validator("fill", "frame")
    def check_color(cls, color):
        if color is None:
            return None
        from matplotlib.colors import CSS4_COLORS, is_color_like, to_hex

        if not is_color_like(color):
            raise ValueError(
                f"LayerView 'color' must be a valid CSS4 color (was {color}). "
                "For more info, see: https://www.w3schools.com/colors/colors_names.asp"
            )

        # Color given by name
        if isinstance(color, str) and color[0] != "#":
            return CSS4_COLORS[color.lower()]

        # Color either an RGBA tuple or a hex string
        else:
            return to_hex(color)


class LayerBrightness(BaseModel):
    """Fill and frame brightness.

    0 is unmodified, -100 roughly adds 50% black to the color which +100 roughly adds 50% white.

    Attributes:
        fill: This value modifies the brightness of the fill color.
        frame: This value modifies the brightness of the frame color.
    """

    fill: Optional[int] = 0
    frame: Optional[int] = 0


class CustomDitherPattern(BaseModel):
    """Custom dither pattern. See KLayout documentation for more info.

    Attributes:
        order: Order of pattern.
        pattern: Pattern to use.
    """

    order: int
    pattern: str

    def to_xml(self, name: str) -> ET.Element:
        el = ET.Element("custom-dither-pattern")

        subel = ET.SubElement(el, "pattern")
        lines = self.pattern.split("\n")
        if len(lines) == 1:
            subel.text = lines[0]
        else:
            for line in lines:
                ET.SubElement(subel, "line").text = line

        ET.SubElement(el, "order").text = str(self.order)
        ET.SubElement(el, "name").text = name
        return el


class CustomLineStyle(BaseModel):
    """Custom line style. See KLayout documentation for more info.

    Attributes:
        order: Order of pattern.
        pattern: Pattern to use.
    """

    order: int
    pattern: str

    def to_xml(self, name: str) -> ET.Element:
        el = ET.Element("custom-line-pattern")

        ET.SubElement(el, "pattern").text = self.pattern
        ET.SubElement(el, "order").text = str(self.order)
        ET.SubElement(el, "name").text = name
        return el


class CustomPatterns(BaseModel):
    dither_patterns: Dict[str, CustomDitherPattern] = Field(default_factory=dict)
    line_styles: Dict[str, CustomLineStyle] = Field(default_factory=dict)

    def to_yaml(self, filename: Union[str, pathlib.Path]) -> None:
        """Export custom patterns to a yaml file.

        Args:
            filename: Name of output file.
        """
        import yaml

        def _str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
            if "\n" in data:  # check for multiline string
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        filepath = pathlib.Path(append_file_extension(filename, ".yml"))

        yaml.add_representer(str, _str_presenter)
        yaml.representer.SafeRepresenter.add_representer(str, _str_presenter)

        filepath.write_bytes(
            yaml.safe_dump_all(
                [self.dict()], indent=2, sort_keys=False, encoding="utf-8"
            )
        )

    @staticmethod
    def from_yaml(filename: str) -> "CustomPatterns":
        """Import a CustomPatterns object from a yaml file.

        Args:
            filename: Name of output file.
        """
        from omegaconf import OmegaConf

        with open(filename) as file:
            loaded_yml = OmegaConf.to_container(OmegaConf.load(file))
        if not (
            "dither_patterns" in loaded_yml.keys() or "line_styles" in loaded_yml.keys()
        ):
            raise KeyError("The yaml file is not properly formatted.")
        return CustomPatterns(**loaded_yml)

    def __str__(self):
        """Prints the number of each type of custom pattern."""
        return f"CustomPatterns: {len(self.dither_patterns)} dither patterns, {len(self.line_styles)} line styles"


class LayerView(BaseModel):
    """KLayout layer properties.

    Docstrings copied from KLayout documentation (with some modifications):
    https://www.klayout.de/lyp_format.html

    Attributes:
        layer: GDSII layer.
        layer_in_name: Whether to display the name as 'name layer/datatype' rather than just the layer.
        width: This is the line width of the frame in pixels (or empty for the default which is 1).
        line_style: This is the number of the line style used to draw the shape boundaries.
            An empty string is "solid line". The values are "Ix" for one of the built-in styles
            where "I0" is "solid", "I1" is "dotted" etc.
        dither_pattern: This is the number of the dither pattern used to fill the shapes.
            The values are "Ix" for one of the built-in pattern where "I0" is "solid" and "I1" is "clear".
        animation: This is a value indicating the animation mode.
            0 is "none", 1 is "scrolling", 2 is "blinking" and 3 is "inverse blinking".
        color: Display color of the layer.
        brightness: Brightness of the fill and frame.
        xfill: Whether boxes are drawn with a diagonal cross.
        marked: Whether the entry is marked (drawn with small crosses).
        transparent: Whether the entry is transparent.
        visible: Whether the entry is visible.
        valid: Whether the entry is valid. Invalid layers are drawn but you can't select shapes on those layers.
        group_members: Add a list of group members to the LayerView.
    """

    layer: Optional[Layer] = None
    layer_in_name: bool = False
    color: Optional[LayerColor] = Field(default_factory=LayerColor)
    brightness: Optional[LayerBrightness] = Field(default_factory=LayerBrightness)
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

        # Iterate through all items, adding group members as needed
        for name, field in self.__fields__.items():
            default = field.get_default()
            if isinstance(default, LayerView):
                self.group_members[name] = default

    @validator("color")
    def parse_color(cls, color):
        if isinstance(color, dict):
            color = LayerColor(**color)
        elif not isinstance(color, LayerColor):
            color = LayerColor(fill=color, frame=color)
        return color

    @validator("brightness")
    def parse_brightness(cls, brightness):
        if isinstance(brightness, dict):
            brightness = LayerBrightness(**brightness)
        elif not isinstance(brightness, LayerBrightness):
            brightness = LayerBrightness(fill=brightness, frame=brightness)
        return brightness

    def __str__(self):
        """Returns a formatted view of properties and their values."""
        return "LayerView:\n\t" + "\n\t".join(
            [f"{k}: {v}" for k, v in self.dict().items()]
        )

    def __repr__(self):
        """Returns a formatted view of properties and their values."""
        return self.__str__()

    def _build_xml_element(self, tag: str, name: str) -> ET.Element:
        """Get XML Element from attributes."""
        prop_dict = {
            "frame-color": self.color.frame,
            "fill-color": self.color.fill,
            "frame-brightness": self.brightness.frame,
            "fill-brightness": self.brightness.fill,
            "dither-pattern": self.dither_pattern,
            "line-style": self.line_style,
            "valid": str(self.valid).lower(),
            "visible": str(self.visible).lower(),
            "transparent": str(self.transparent).lower(),
            "width": self.width,
            "marked": str(self.marked).lower(),
            "xfill": str(self.xfill).lower(),
            "animation": self.animation,
            "name": name
            if not self.layer_in_name
            else f"{name} {self.layer[0]}/{self.layer[1]}",
            "source": f"{self.layer[0]}/{self.layer[1]}@1"
            if self.layer is not None
            else "*/*@*",
        }
        el = ET.Element(tag)
        for key, value in prop_dict.items():
            subel = ET.SubElement(el, key)

            if value is None:
                continue

            if isinstance(value, bool):
                value = str(value).lower()

            subel.text = str(value)
        return el

    def to_xml(self, name: str) -> ET.Element:
        """Return an XML representation of the LayerView."""
        props = self._build_xml_element("properties", name=name)
        for member_name, member in self.group_members.items():
            props.append(member._build_xml_element("group-members", name=member_name))
        return props

    @staticmethod
    def _process_name(
        name: str, layer_pattern: Union[str, re.Pattern]
    ) -> Tuple[Optional[str], Optional[bool]]:
        """Strip layer info from name if it exists.

        Args:
            name: XML-formatted name entry.
            layer_pattern: Regex pattern to match layers with.
        """
        if not name:
            return None, None
        layer_in_name = False
        match = re.search(layer_pattern, name)
        if match:
            name = name[: match.start()].strip()
            layer_in_name = True
        return name, layer_in_name

    @staticmethod
    def _process_layer(
        layer: str, layer_pattern: Union[str, re.Pattern]
    ) -> Optional[Layer]:
        """Convert .lyp XML layer entry to a Layer.

        Args:
            layer: XML-formatted layer entry.
            layer_pattern: Regex pattern to match layers with.
        """
        match = re.search(layer_pattern, layer)
        if not match:
            raise OSError(f"Could not read layer {layer}!")
        v = match.group().split("/")
        return None if v == ["*", "*"] else (int(v[0]), int(v[1]))

    @classmethod
    def from_xml_element(
        cls, element: ET.Element, layer_pattern: Union[str, re.Pattern]
    ) -> Optional[Tuple[str, "LayerView"]]:
        """Read properties from .lyp XML and generate LayerViews from them.

        Args:
            element: XML Element to iterate over.
            layer_pattern: Regex pattern to match layers with.
        """
        name, layer_in_name = cls._process_name(
            element.find("name").text, layer_pattern
        )
        if name is None:
            return None

        group_members = {}
        for member in element.iterfind("group-members"):
            member_name, member_lv = cls.from_xml_element(member, layer_pattern)
            group_members[member_name] = member_lv

        return (
            name,
            LayerView(
                layer=cls._process_layer(element.find("source").text, layer_pattern),
                color=LayerColor(
                    fill=element.find("fill-color").text,
                    frame=element.find("frame-color").text,
                ),
                brightness=LayerBrightness(
                    fill=element.find("fill-brightness").text,
                    frame=element.find("frame-brightness").text,
                ),
                dither_pattern=element.find("dither-pattern").text,
                line_style=element.find("line-style").text,
                valid=element.find("valid").text,
                visible=element.find("visible").text,
                transparent=element.find("transparent").text,
                width=element.find("width").text,
                marked=element.find("marked").text,
                xfill=element.find("xfill").text,
                animation=element.find("animation").text,
                layer_in_name=layer_in_name,
                name=name,
                group_members=group_members,
            ),
        )


class LayerDisplayProperties(BaseModel):
    """A container for layer properties for KLayout layer property (.lyp) files.

    Attributes:
        layer_views: Dictionary of LayerViews describing how to display gds layers.
        custom_patterns: CustomPatterns object containing custom dither patterns and line styles.
    """

    layer_views: Dict[str, LayerView] = Field(default_factory=dict)
    custom_patterns: Optional[CustomPatterns] = None

    def __init__(self, **data):
        """Initialize LayerDisplayProperties object."""
        super().__init__(**data)

        for field in self.dict():
            val = getattr(self, field)
            if isinstance(val, LayerView):
                self.add_layer_view(name=field, layer_view=val)

    def add_layer_view(self, name: str, layer_view: Optional[LayerView]) -> None:
        """Adds a layer to LayerDisplayProperties.

        Args:
            name: Name of the LayerView.
            layer_view: LayerView to add.
        """
        if name in self.layer_views:
            raise ValueError(
                f"Adding {name!r} already defined {list(self.layer_views.keys())}"
            )
        else:
            self.layer_views[name] = layer_view

    def get_layer_views(self, exclude_groups: bool = False) -> Dict[str, LayerView]:
        """Return all LayerViews.

        Args:
            exclude_groups: Whether to exclude LayerViews that contain other LayerViews.
        """
        layers = {}
        for name, view in self.layer_views.items():
            if view.group_members and not exclude_groups:
                for member_name, member in view.group_members.items():
                    layers[member_name] = member
                continue
            layers[name] = view
        return layers

    def get_layer_view_groups(self) -> Dict[str, LayerView]:
        """Return the LayerViews that contain other LayerViews."""
        return {name: lv for name, lv in self.layer_views.items() if lv.group_members}

    def __str__(self) -> str:
        """Prints the number of LayerView objects in the LayerDisplayProperties object."""
        lvs = self.get_layer_views()
        groups = self.get_layer_view_groups()
        return (
            f"LayerDisplayProperties: {len(lvs)} layers ({len(groups)} groups)\n"
            f"\t{self.custom_patterns}"
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

    def __getitem__(self, val: str):
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

    def to_lyp(
        self, filepath: Union[str, pathlib.Path], overwrite: bool = True
    ) -> None:
        """Write all layer properties to a KLayout .lyp file.

        Args:
            filepath: to write the .lyp file to (appends .lyp extension if not present).
            overwrite: Whether to overwrite an existing file located at the filepath.

        """
        filepath = pathlib.Path(append_file_extension(filepath, ".lyp"))

        if os.path.exists(filepath) and not overwrite:
            raise OSError("File exists, cannot write.")

        root = ET.Element("layer-properties")

        for name, lv in self.layer_views.items():
            root.append(lv.to_xml(name))

        if self.custom_patterns is not None:
            for name, dp in self.custom_patterns.dither_patterns.items():
                root.append(dp.to_xml(name))

            for name, ls in self.custom_patterns.line_styles.items():
                root.append(ls.to_xml(name))

        filepath.write_bytes(make_pretty_xml(root))

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

        tree = ET.parse(filepath)
        root = tree.getroot()
        if root.tag != "layer-properties":
            raise OSError("Layer properties file incorrectly formatted, cannot read.")

        layer_views = {}
        for properties_element in root.iter("properties"):
            name, lv = LayerView.from_xml_element(
                properties_element, layer_pattern=layer_pattern
            )
            if lv:
                layer_views[name] = lv

        dither_patterns = {}
        for dither_block in root.iter("custom-dither-pattern"):
            name = dither_block.find("name").text
            order = dither_block.find("order").text

            if name is None or order is None:
                continue
            pattern = "\n".join(
                [line.text for line in dither_block.find("pattern").iter()]
            )

            dither_patterns[name] = CustomDitherPattern(
                order=int(order),
                pattern=pattern.lstrip(),
            )
        line_styles = {}
        for line_block in root.iter("custom-line-style"):
            name = line_block.find("name").text
            order = line_block.find("order").text

            if name is None or order is None:
                continue

            line_styles[name] = CustomLineStyle(
                order=int(order),
                pattern=line_block.find("pattern").text,
            )
        custom_patterns = CustomPatterns(
            dither_patterns=dither_patterns, line_styles=line_styles
        )

        return LayerDisplayProperties(
            layer_views=layer_views, custom_patterns=custom_patterns
        )

    def to_yaml(
        self,
        layer_file: Union[str, pathlib.Path],
        pattern_file: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """Export layer properties to two yaml files.

        Args:
            layer_file: Name of the file to write LayerViews to.
            pattern_file: Name of the file to write custom dither patterns and line styles to.
        """
        import yaml

        lf_path = pathlib.Path(append_file_extension(layer_file, ".yml"))
        lvs = {name: lv.dict() for name, lv in self.layer_views.items()}

        lf_path.write_bytes(
            yaml.safe_dump_all([lvs], indent=2, sort_keys=False, encoding="utf-8")
        )

        if pattern_file:
            append_file_extension(pattern_file, ".yml")
            self.custom_patterns.to_yaml(pattern_file)

    @staticmethod
    def from_yaml(
        layer_file: str = None, pattern_file: Optional[str] = None
    ) -> "LayerDisplayProperties":
        """Import layer properties from two yaml files.

        Args:
            layer_file: Name of the file to read LayerViews from.
            pattern_file: Name of the file to read custom dither patterns and line styles from.
        """
        from omegaconf import OmegaConf

        append_file_extension(layer_file, ".yml")

        with open(layer_file) as lf:
            layers = OmegaConf.to_container(OmegaConf.load(lf))
        props = LayerDisplayProperties(
            layer_views={name: LayerView(**lv) for name, lv in layers.items()}
        )
        if pattern_file:
            append_file_extension(pattern_file, ".yml")
            props.custom_patterns = CustomPatterns.from_yaml(pattern_file)
        return props
