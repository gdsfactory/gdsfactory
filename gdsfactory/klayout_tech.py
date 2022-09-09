"""Classes for KLayout-specific layer settings.

This module will enable conversion between gdsfactory settings and KLayout technology.
"""

import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
from lxml import etree
from matplotlib.colors import CSS4_COLORS
from pydantic import BaseModel, Field, validator
from typing_extensions import Literal

from gdsfactory.types import Layer


class LayerView(BaseModel):
    """KLayout layer properties.

    Docstrings copied from KLayout documentation:
    https://www.klayout.de/lyp_format.html

    Parameters:
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
        valid: Whether the entry is valid. Invalid layers are drawn but you can't select shapes on those layers.
    """

    layer: Optional[Layer] = None
    name: str = "unnamed"
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
                "KLayoutLayerProperty color must be specified as a "
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


_layer_re = re.compile("([0-9]+|\\*)/([0-9]+|\\*)")


def _process_name(name: str) -> Optional[str]:
    """Strip layer info from name if it exists.

    Args:
        name: XML-formatted name entry
    """
    if not name:
        return None
    match = re.search(_layer_re, name)
    if match:
        return name[: match.start()].strip()
    return name


def _process_layer(layer: str) -> Optional[Layer]:
    """Convert .lyp XML layer entry to a Layer.

    Args:
        layer: XML-formatted layer entry
    """
    match = re.search(_layer_re, layer)
    if not match:
        raise OSError(f"Could not read layer {layer}!")
    v = match.group().split("/")
    if v == ["*", "*"]:
        return None
    return int(v[0]), int(v[1])


def _properties_to_layerview(element, tag: Optional[str] = None) -> Optional[LayerView]:
    """Read properties from .lyp XML and generate LayerViews from them.

    Args:
        tag: Optional tag to iterate over.
    """
    prop_dict = {}
    for prop in element.iterchildren(tag=tag):
        prop_tag = prop.tag
        if prop_tag == "name":
            val = _process_name(prop.text)
            if val is None:
                return None
        elif prop_tag == "source":
            val = _process_layer(prop.text)
            prop_tag = "layer"
        elif prop_tag == "group-members":
            props = [
                _properties_to_layerview(e)
                for e in element.iterchildren("group-members")
            ]
            val = {p.name: p for p in props}
        else:
            val = prop.text
        prop_tag = "_".join(prop_tag.split("-"))
        prop_dict[prop_tag] = val
    return LayerView(**prop_dict)


class KLayoutLayerProperties(BaseModel):
    """A container for layer properties for KLayout layer property (.lyp) files."""

    layer_views: Dict[str, LayerView] = Field(default_factory=dict)
    custom_dither_patterns: Optional[Dict[str, CustomPattern]] = None
    custom_line_styles: Optional[Dict[str, CustomPattern]] = None

    def __init__(self, **data):
        """Initialize KLayoutLayerProperties object."""
        super().__init__(**data)

        for key, val in self.dict().items():

            if key not in self.__fields__:
                if val["name"] != key:
                    val["name"] = key
                    setattr(self, key, LayerView(**val))
                self.add_layer(**val)

    def add_layer(
        self,
        klayout_layer_props: Optional[LayerView] = None,
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
        """Adds a layer to KLayoutLayerProperties.

        Docstrings copied from KLayout documentation:
        https://www.klayout.de/lyp_format.html

        Args:
            klayout_layer_props: Add layer from existing KLayoutLayerProperty, overrides all other args.
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
        new_layer = klayout_layer_props or LayerView(
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
        layers = {}
        for name, view in self.layer_views.items():
            if view.group_members is not None:
                layers[name] = view
        return layers

    def __str__(self):
        """Prints the number of KLayoutLayerProperty objects in the KLayoutLayerProperties object."""
        # if len(self.groups) == 0:
        #     return (
        #         f"KLayoutLayerProperties ({len(self.layers)} layers total) \n"
        #         f"{list(self.layers)}"
        #     )
        return (
            f"KLayoutLayerProperties ({len(self.get_layer_views())} layers total, {len(self.get_layer_view_groups())} groups) \n"
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
            val: Layer name to access within the KLayoutLayerProperties.

        Returns:
            self.layers[val]: KLayoutLayerProperty in the KLayoutLayerProperties.

        """
        try:
            return self.layer_views[val]
        except Exception as error:
            raise ValueError(
                f"Layer {val!r} not in LayerColors {list(self.layer_views.keys())}"
            ) from error

    def get_from_tuple(self, layer_tuple: Tuple[int, int]) -> LayerView:
        """Returns KLayoutLayerProperty from layer tuple.

        Args:
            layer_tuple: Tuple of (gds_layer, gds_datatype).

        Returns:
            KLayoutLayerProperty
        """
        tuple_to_name = {v.layer: k for k, v in self.layer_views.items()}
        if layer_tuple not in tuple_to_name:
            raise ValueError(
                f"Layer color {layer_tuple} not in {list(tuple_to_name.keys())}"
            )

        name = tuple_to_name[layer_tuple]
        return self.layer_views[name]

    def get_layer_tuples(self):
        """Returns a tuple for each layer."""
        return {layer.layer for layer in self.get_layer_views().values()}

    def to_lyp(self, filepath: str, overwrite: bool = True) -> None:
        """Write all layer properties to a KLayout .lyp file.

        Args:
            filepath: to write the .lyp file to (appends .lyp extension if not present)
            overwrite: Whether to overwrite an existing file located at the filepath.
        """
        if not filepath.endswith(".lyp"):
            filepath += ".lyp"

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
    def from_lyp(filepath: str) -> "KLayoutLayerProperties":
        """Write all layer properties to a KLayout .lyp file.

        Args:
            filepath: to write the .lyp file to (appends .lyp extension if not present)
        """
        if not filepath.endswith(".lyp"):
            filepath += ".lyp"

        if not os.path.exists(filepath):
            raise OSError("File not found!")

        tree = etree.parse(filepath)
        root = tree.getroot()
        if not root.tag == "layer-properties":
            raise OSError("Layer properties file incorrectly formatted, cannot read.")

        layer_views = {}
        for layer_block in root.iter("properties"):
            lv = _properties_to_layerview(layer_block)
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
        return KLayoutLayerProperties(
            layer_views=layer_views,
            custom_dither_patterns=custom_dither_patterns,
            custom_line_styles=custom_line_styles,
        )


# TODO: Write to .lyt technology file


if __name__ == "__main__":

    # lc: LayerColors = LAYER_COLORS
    # # class LayerViewGroup(LayerView):
    #
    # class DefaultProperties(KLayoutLayerProperties):
    #     WG = LayerView(layer=(1, 0))
    #     WGCLAD = LayerView(layer=(111, 0))
    #     SLAB150 = LayerView(layer=(2, 0))
    #     SLAB90 = LayerView(layer=(3, 0))
    #     DEEPTRENCH = LayerView(layer=(4, 0))
    #     GE = LayerView(layer=(5, 0))
    #     WGN = LayerView(layer=(34, 0))
    #     WGN_CLAD = LayerView(layer=(36, 0))
    #
    #     class DopingGroup(LayerView):
    #         N = LayerView(layer=(20, 0))
    #         NP = LayerView(layer=(22, 0))
    #         NPP = LayerView(layer=(24, 0))
    #         P = LayerView(layer=(21, 0))
    #         PP = LayerView(layer=(23, 0))
    #         PPP = LayerView(layer=(25, 0))
    #         GEN = LayerView(layer=(26, 0))
    #         GEP = LayerView(layer=(27, 0))
    #     Doping = DopingGroup()
    #
    #     HEATER = LayerView(layer=(47, 0))
    #     M1 = LayerView(layer=(41, 0))
    #     M2 = LayerView(layer=(45, 0))
    #     M3 = LayerView(layer=(49, 0))
    #     VIAC = LayerView(layer=(40, 0))
    #     VIA1 = LayerView(layer=(44, 0))
    #     VIA2 = LayerView(layer=(43, 0))
    #     PADOPEN = LayerView(layer=(46, 0))
    #
    #     DICING = LayerView(layer=(100, 0))
    #     NO_TILE_SI = LayerView(layer=(71, 0))
    #     PADDING = LayerView(layer=(67, 0))
    #     DEVREC = LayerView(layer=(68, 0))
    #     FLOORPLAN = LayerView(layer=(64, 0))
    #     TEXT = LayerView(layer=(66, 0))
    #     PORT = LayerView(layer=(1, 10))
    #     PORTE = LayerView(layer=(1, 11))
    #     PORTH = LayerView(layer=(70, 0))
    #     SHOW_PORTS = LayerView(layer=(1, 12))
    #     LABEL = LayerView(layer=(201, 0))
    #     LABEL_SETTINGS = LayerView(layer=(202, 0))
    #     TE = LayerView(layer=(203, 0))
    #     TM = LayerView(layer=(204, 0))
    #     DRC_MARKER = LayerView(layer=(205, 0))
    #     LABEL_INSTANCE = LayerView(layer=(206, 0))
    #     ERROR_MARKER = LayerView(layer=(207, 0))
    #     ERROR_PATH = LayerView(layer=(208, 0))
    #
    #     class SimulationGroup(LayerView):
    #         SOURCE = LayerView(layer=(110, 0))
    #         MONITOR = LayerView(layer=(101, 0))
    #     Simulation = SimulationGroup()
    #
    # lyp = DefaultProperties()

    # print(lyp)
    # lyp.to_lyp("test_lyp")

    filepath = "/home/thomas/layout/gdsfactory/gdsfactory/klayout/tech/layers.lyp"
    lyp = KLayoutLayerProperties.from_lyp(filepath)
    lyp.to_lyp("test_lyp.lyp")
