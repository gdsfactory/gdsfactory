"""A GDS layer is a tuple of two integers.

You can:

- Load LayerViews from Klayout XML file (.lyp) (recommended)
- Define your layers in a Pydantic BaseModel


"""
from __future__ import annotations

import os
import pathlib
import re
import xml.etree.ElementTree as ET
from typing import Dict, Literal, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from pydantic.color import Color

from gdsfactory.config import logger

PathLike = Union[pathlib.Path, str]

Layer = Tuple[int, int]

FrameAndFill = Literal["frame", "fill"]
FrameAndFillColor = Dict[FrameAndFill, Color]
FrameAndFillBrightness = Dict[FrameAndFill, int]


class CustomDitherPattern(BaseModel):
    """Custom dither pattern. See KLayout documentation for more info.

    Attributes:
        order: Order of pattern.
        pattern: Pattern to use.
    """

    name: str
    order: int
    pattern: str

    class Config:
        """YAML output uses name as the key."""

        fields = {"name": {"exclude": True}}

    def to_xml(self) -> ET.Element:
        el = ET.Element("custom-dither-pattern")

        subel = ET.SubElement(el, "pattern")
        lines = self.pattern.split("\n")
        if len(lines) == 1:
            subel.text = lines[0]
        else:
            for line in lines:
                ET.SubElement(subel, "line").text = line

        ET.SubElement(el, "order").text = str(self.order)
        ET.SubElement(el, "name").text = self.name
        return el


class CustomLineStyle(BaseModel):
    """Custom line style. See KLayout documentation for more info.

    Attributes:
        order: Order of pattern.
        pattern: Pattern to use.
    """

    name: str
    order: int
    pattern: str

    class Config:
        """YAML output uses name as the key."""

        fields = {"name": {"exclude": True}}

    def to_xml(self) -> ET.Element:
        el = ET.Element("custom-line-pattern")

        ET.SubElement(el, "pattern").text = str(self.pattern)
        ET.SubElement(el, "order").text = str(self.order)
        ET.SubElement(el, "name").text = self.name
        return el


class LayerView(BaseModel):
    """KLayout layer properties.

    Docstrings adapted from KLayout documentation:
    https://www.klayout.de/lyp_format.html

    Attributes:
        name: Layer name.
        info: Extra info to include in the LayerView.
        alpha:
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
        color: Display color(s) of the layer frame and fill. Pass either a single Color or a dictionary whose keys are
            "frame" and "fill" (i.e. {"frame": "#000000", "fill": [10, 10, 10]})
            Accepts Pydantic Color types. See: https://docs.pydantic.dev/usage/types/#color-type for more info.
        brightness: Brightness of the fill and frame.
        xfill: Whether boxes are drawn with a diagonal cross.
        marked: Whether the entry is marked (drawn with small crosses).
        transparent: Whether the entry is transparent.
        visible: Whether the entry is visible.
        valid: Whether the entry is valid. Invalid layers are drawn but you can't select shapes on those layers.
        group_members: Add a list of group members to the LayerView.
    """

    name: str
    info: Optional[str] = None
    alpha: Optional[float] = None
    layer: Optional[Layer] = None
    layer_in_name: bool = False
    color: Optional[Union[Color, FrameAndFillColor]] = None
    brightness: Optional[Union[int, FrameAndFillBrightness]] = None
    dither_pattern: Optional[Union[str, CustomDitherPattern]] = None
    line_style: Optional[Union[str, CustomLineStyle]] = None
    valid: bool = True
    visible: bool = True
    transparent: bool = False
    width: Optional[int] = None
    marked: bool = False
    xfill: bool = False
    animation: int = 0
    group_members: Optional[Dict[str, LayerView]] = Field(default_factory=dict)

    class Config:
        """YAML output uses name as the key."""

        fields = {"name": {"exclude": True}}

    def _alpha_from_lyp(self):
        if not self.visible:
            return 0.0
        elif not self.transparent:
            dither_name = getattr(self.dither_pattern, "name", self.dither_pattern)
            return 0.1 if dither_name == "I1" else 1.0
        else:
            return 0.5

    def __init__(
        self,
        gds_layer: Optional[int] = None,
        gds_datatype: Optional[int] = None,
        **data,
    ):
        """Initialize LayerView object."""
        if (gds_layer is not None) and (gds_datatype is not None):
            if data["layer"] is not None:
                raise KeyError(
                    "Specify either 'layer' or both 'gds_layer' and 'gds_datatype'."
                )
            data["layer"] = (gds_layer, gds_datatype)

        super().__init__(**data)

        if self.alpha is None:
            self.alpha = self._alpha_from_lyp()

        # Iterate through all items, adding group members as needed
        for name, field in self.__fields__.items():
            default = field.get_default()
            if isinstance(default, LayerView):
                self.group_members[name] = default

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
        from gdsfactory.utils.color_utils import ensure_six_digit_hex_color

        colors = self.color

        colors = {
            "frame-color": colors["frame"] if isinstance(colors, dict) else colors,
            "fill-color": colors["fill"] if isinstance(colors, dict) else colors,
        }
        for key, color in colors.items():
            if color is not None:
                color = ensure_six_digit_hex_color(color.as_hex())
            colors[key] = color

        brightnesses = self.brightness
        brightnesses = {
            "frame-brightness": brightnesses["frame"]
            if isinstance(brightnesses, dict)
            else brightnesses,
            "fill-brightness": brightnesses["fill"]
            if isinstance(brightnesses, dict)
            else brightnesses,
        }
        prop_dict = {
            **colors,
            **brightnesses,
            "dither-pattern": self.dither_pattern.name
            if isinstance(self.dither_pattern, CustomDitherPattern)
            else self.dither_pattern,
            "line-style": self.line_style.name
            if isinstance(self.line_style, CustomLineStyle)
            else self.line_style,
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

    def to_xml(self) -> ET.Element:
        """Return an XML representation of the LayerView."""
        props = self._build_xml_element("properties", name=self.name)
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
    ) -> Optional[LayerView]:
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

        color = {
            "fill": element.find("fill-color").text,
            "frame": element.find("frame-color").text,
        }
        if color["fill"] == color["frame"]:
            color = color["fill"]

        brightness = {
            "fill": element.find("fill-brightness").text,
            "frame": element.find("frame-brightness").text,
        }
        if brightness["fill"] == brightness["frame"]:
            brightness = brightness["fill"]

        lv = LayerView(
            name=name,
            layer=cls._process_layer(element.find("source").text, layer_pattern),
            color=color,
            brightness=brightness,
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
        )

        # Add only if needed, so we can filter by defaults when dumping to yaml
        group_members = {}
        for member in element.iterfind("group-members"):
            member_lv = cls.from_xml_element(member, layer_pattern)
            group_members[member_lv.name] = member_lv

        if group_members != {}:
            lv.group_members = group_members

        return lv


class LayerViews(BaseModel):
    """A container for layer properties for KLayout layer property (.lyp) files.

    Attributes:
        layer_views: Dictionary of LayerViews describing how to display gds layers.
        custom_dither_patterns: Custom dither patterns.
        custom_line_styles: Custom line styles.
    """

    layer_views: Dict[str, LayerView] = Field(default_factory=dict)
    custom_dither_patterns: Dict[str, CustomDitherPattern] = Field(default_factory=dict)
    custom_line_styles: Dict[str, CustomLineStyle] = Field(default_factory=dict)

    def __init__(self, filepath: Optional[PathLike] = None, **data):
        """Initialize LayerViews object."""
        if filepath is not None:
            filepath = pathlib.Path(filepath)
            if filepath.suffix == ".lyp":
                lvs = LayerViews.from_lyp(filepath=filepath)
                logger.info(
                    f"Importing LayerViews from KLayout layer properties file: {filepath}."
                )
            elif (filepath.suffix == ".yaml") or (filepath.suffix == ".yml"):
                lvs = LayerViews.from_yaml(layer_file=filepath)
                logger.info(f"Importing LayerViews from YAML file: {filepath}.")
            else:
                raise ValueError(f"Unable to load LayerViews from {filepath}.")

            data["layer_views"] = lvs.layer_views
            data["custom_line_styles"] = lvs.custom_line_styles
            data["custom_dither_patterns"] = lvs.custom_dither_patterns

        super().__init__(**data)

        for field in self.dict():
            val = getattr(self, field)
            if isinstance(val, LayerView):
                self.add_layer_view(name=field, layer_view=val)

    def add_layer_view(self, name: str, layer_view: Optional[LayerView]) -> None:
        """Adds a layer to LayerViews.

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

        # If the dither pattern is a CustomDitherPattern, add it to custom_patterns
        dither_pattern = layer_view.dither_pattern
        if (
            isinstance(dither_pattern, CustomDitherPattern)
            and dither_pattern not in self.custom_dither_patterns.keys()
        ):
            self.custom_dither_patterns[dither_pattern.name] = dither_pattern

        # If dither_pattern is the name of a custom pattern, replace string with the CustomDitherPattern
        elif (
            isinstance(dither_pattern, str)
            and dither_pattern in self.custom_dither_patterns.keys()
        ):
            layer_view.dither_pattern = self.custom_dither_patterns[dither_pattern]

        line_style = layer_view.line_style
        if (
            isinstance(line_style, CustomLineStyle)
            and line_style not in self.custom_line_styles.keys()
        ):
            self.custom_line_styles[line_style.name] = line_style
        elif (
            isinstance(line_style, str) and line_style in self.custom_line_styles.keys()
        ):
            layer_view.line_style = self.custom_line_styles[line_style]

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
        """Prints the number of LayerView objects in the LayerViews object."""
        lvs = self.get_layer_views()
        groups = self.get_layer_view_groups()
        return (
            f"LayerViews: {len(lvs)} layers ({len(groups)} groups)\n"
            f"\tCustomDitherPatterns: {list(self.custom_dither_patterns.keys())}\n"
            f"\tCustomLineStyles: {list(self.custom_line_styles.keys())}\n"
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
            val: Layer name to access within the LayerViews.

        Returns:
            self.layers[val]: LayerView in the LayerViews.

        """
        try:
            return self.layer_views[val]
        except Exception as error:
            raise ValueError(
                f"LayerView {val!r} not in LayerViews {list(self.layer_views.keys())}"
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

    def clear(self) -> None:
        """Deletes all layers in the LayerViews."""
        self.layer_views = {}

    def preview_layerset(self, size: float = 100.0, spacing: float = 100.0) -> object:
        """Generates a Component with all the layers.

        Args:
            size: square size.
            spacing: spacing between each square.

        """
        import gdsfactory as gf

        D = gf.Component(name="layerset", with_uuid=True)
        scale = size / 100
        num_layers = len(self.get_layer_views())
        matrix_size = int(np.ceil(np.sqrt(num_layers)))
        sorted_layers = sorted(
            self.get_layer_views().values(), key=lambda x: (x.layer[0], x.layer[1])
        )
        for n, layer in enumerate(sorted_layers):
            layer_tuple = layer.layer
            R = gf.components.rectangle(
                size=(100 * scale, 100 * scale), layer=layer_tuple
            )
            T = gf.components.text(
                text=f"{layer.name}\n{layer_tuple[0]} / {layer_tuple[1]}",
                size=20 * scale,
                position=(50 * scale, -20 * scale),
                justify="center",
                layer=layer_tuple,
            )

            xloc = n % matrix_size
            yloc = int(n // matrix_size)
            D.add_ref(R).movex((100 + spacing) * xloc * scale).movey(
                -(100 + spacing) * yloc * scale
            )
            D.add_ref(T).movex((100 + spacing) * xloc * scale).movey(
                -(100 + spacing) * yloc * scale
            )
        return D

    def to_lyp(
        self, filepath: Union[str, pathlib.Path], overwrite: bool = True
    ) -> None:
        """Write all layer properties to a KLayout .lyp file.

        Args:
            filepath: to write the .lyp file to (appends .lyp extension if not present).
            overwrite: Whether to overwrite an existing file located at the filepath.

        """
        from gdsfactory.utils.xml_utils import make_pretty_xml

        filepath = pathlib.Path(filepath)

        if os.path.exists(filepath) and not overwrite:
            raise OSError("File exists, cannot write.")

        root = ET.Element("layer-properties")

        for name, lv in self.layer_views.items():
            root.append(lv.to_xml())

        for name, dp in self.custom_dither_patterns.items():
            root.append(dp.to_xml())

        for name, ls in self.custom_line_styles.items():
            root.append(ls.to_xml())

        filepath.write_bytes(make_pretty_xml(root))

    @staticmethod
    def from_lyp(
        filepath: Union[str, pathlib.Path],
        layer_pattern: Optional[Union[str, re.Pattern]] = None,
    ) -> LayerViews:
        r"""Write all layer properties to a KLayout .lyp file.

        Args:
            filepath: to write the .lyp file to (appends .lyp extension if not present).
            layer_pattern: Regex pattern to match layers with. Defaults to r'(\d+|\*)/(\d+|\*)'.
        """
        layer_pattern = re.compile(layer_pattern or r"(\d+|\*)/(\d+|\*)")

        filepath = pathlib.Path(filepath)

        if not os.path.exists(filepath):
            raise OSError("File not found!")

        tree = ET.parse(filepath)
        root = tree.getroot()
        if root.tag != "layer-properties":
            raise OSError("Layer properties file incorrectly formatted, cannot read.")

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
                name=name,
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
                name=name,
                order=int(order),
                pattern=line_block.find("pattern").text,
            )

        layer_views = {}
        for properties_element in root.iter("properties"):
            lv = LayerView.from_xml_element(
                properties_element, layer_pattern=layer_pattern
            )
            if lv:
                layer_views[lv.name] = lv

        return LayerViews(
            layer_views=layer_views,
            custom_dither_patterns=dither_patterns,
            custom_line_styles=line_styles,
        )

    def to_yaml(
        self, layer_file: Union[str, pathlib.Path], prefer_named_color: bool = True
    ) -> None:
        """Export layer properties to two yaml files.

        Args:
            layer_file: Name of the file to write LayerViews to.
            prefer_named_color: Write the name of a color instead of its hex representation when possible.
        """
        import yaml

        from gdsfactory.utils.yaml_utils import (
            add_color_yaml_presenter,
            add_multiline_str_yaml_presenter,
            add_tuple_yaml_presenter,
        )

        lf_path = pathlib.Path(layer_file)

        add_tuple_yaml_presenter()
        add_multiline_str_yaml_presenter()
        add_color_yaml_presenter(prefer_named_color=prefer_named_color)

        lvs = {
            name: lv.dict(exclude_none=True, exclude_defaults=True, exclude_unset=True)
            for name, lv in self.layer_views.items()
        }

        out_dict = {
            "LayerViews": lvs,
            "CustomDitherPatterns": {
                name: dp.dict() for name, dp in self.custom_dither_patterns.items()
            },
            "CustomLineStyles": {
                name: ls.dict() for name, ls in self.custom_line_styles.items()
            },
        }

        lf_path.write_bytes(
            yaml.dump(
                out_dict,
                indent=2,
                sort_keys=False,
                default_flow_style=False,
                encoding="utf-8",
            )
        )

    @staticmethod
    def from_yaml(layer_file: Union[str, pathlib.Path]) -> LayerViews:
        """Import layer properties from two yaml files.

        Args:
            layer_file: Name of the file to read LayerViews, CustomDitherPatterns, and CustomLineStyles from.
        """
        from omegaconf import OmegaConf

        layer_file = pathlib.Path(layer_file)

        properties = OmegaConf.to_container(OmegaConf.load(layer_file.open()))
        lvs = {}
        for name, lv in properties["LayerViews"].items():
            if "group_members" in lv:
                lv["group_members"] = {
                    member_name: LayerView(name=member_name, **member_view)
                    for member_name, member_view in lv["group_members"].items()
                }
            lvs[name] = LayerView(name=name, **lv)

        return LayerViews(
            layer_views=lvs,
            custom_dither_patterns={
                name: CustomDitherPattern(name=name, **dp)
                for name, dp in properties["CustomDitherPatterns"].items()
            },
            custom_line_styles={
                name: CustomLineStyle(name=name, **ls)
                for name, ls in properties["CustomLineStyles"].items()
            },
        )


def _name_to_short_name(name_str: str) -> str:
    """Maps the name entry of the lyp element to a name of the layer.

    i.e. the dictionary key used to access it.
    Default format of the lyp name are:

        - key - layer/datatype - description
        - key - description

    """
    from gdsfactory.name import clean_name

    if name_str is None:
        raise OSError(f"layer {name_str} has no name")
    fields = name_str.split("-")
    name = fields[0].split()[0].strip()
    return clean_name(name, remove_dots=True)


def _name_to_description(name_str) -> str:
    """Gets the description of the layer contained in the lyp name field.

    It is not strictly necessary to have a description. If none there, it returns ''.

    Default format of the lyp name are:

        - key - layer/datatype - description
        - key - description

    """
    if name_str is None:
        raise OSError(f"layer {name_str!r} has no name")
    fields = name_str.split()
    return " ".join(fields[1:]) if len(fields) > 1 else ""


def test_load_lyp():
    from gdsfactory.config import layer_path

    lys = LayerViews.from_lyp(layer_path)
    assert len(lys.layer_views) > 10, len(lys.layer_views)
    return lys


if __name__ == "__main__":
    from gdsfactory.generic_tech import load_lyp_generic

    LAYER_VIEWS = load_lyp_generic()
    # import gdsfactory as gf

    # c = gf.components.rectangle(layer=(123, 0))
    # c.plot()

    # print(LAYER_VIEWS)
    # print(LAYER_STACK.get_from_tuple((1, 0)))
    # print(LAYER_STACK.get_layer_to_material())
    # layer = LayerColor(color="gold")
    # print(layer)

    # lys = test_load_lyp()
    c = LAYER_VIEWS.preview_layerset()
    c.show(show_ports=True)
    # print(LAYERS_OPTICAL)
    # print(layer("wgcore"))
    # print(layer("wgclad"))
    # print(layer("padding"))
    # print(layer("TEXT"))
    # print(type(layer("wgcore")))
