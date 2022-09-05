"""Classes for KLayout-specific layer settings.

This module will enable conversion between gdsfactory settings and KLayout technology.
"""

import os
from typing import Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
from matplotlib.colors import CSS4_COLORS
from pydantic import BaseModel, Field, validator

from gdsfactory.types import Layer


class KLayoutLayerProperty(BaseModel):
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
        return "<KLayoutLayerProperty>:\n\t" + "\n\t".join(
            [f"{k}: {v}" for k, v in self.dict().items()]
        )

    def __repr__(self):
        """Returns a formatted view of properties and their values."""
        return self.__str__()


class KLayoutGroupProperty(KLayoutLayerProperty):
    members: List[str]


class KLayoutLayerProperties(BaseModel):
    """A container for layer properties for KLayout layer property (.lyp) files.

    Experimental, only accepts a single level of grouping.
    """

    layers: Dict[str, KLayoutLayerProperty] = Field(default_factory=dict)
    groups: Dict[str, KLayoutGroupProperty] = Field(default_factory=dict)

    def add_layer(
        self,
        klayout_layer_props: Optional[KLayoutLayerProperty] = None,
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

        """
        new_layer = klayout_layer_props or KLayoutLayerProperty(
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
        )
        if name in self.layers:
            raise ValueError(
                f"Adding {name!r} already defined {list(self.layers.keys())}"
            )
        else:
            self.layers[name] = new_layer

    def create_group(
        self,
        members: Optional[Union[List[str], Dict[str, KLayoutLayerProperty]]] = None,
        klayout_group_props: Optional[KLayoutGroupProperty] = None,
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
    ) -> None:
        """Adds a group to KLayoutLayerProperties.

        Similar to adding a layer, but with no Layer and must include "members".

        Docstrings copied from KLayout documentation:
        https://www.klayout.de/lyp_format.html

        Args:
            members: Layers to add to the group.
                Can either be a list of names of layers added previously or a dict of name, KLayoutLayerProperty pairs
            klayout_group_props: Add group from existing KLayoutGroupProperty, overrides all other args.
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

        """
        if klayout_group_props is None and members is None:
            raise ValueError(
                "Either a KLayoutGroupProperty or a list/dict of members must be specified!"
            )
        new_group = klayout_group_props or KLayoutGroupProperty(
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
            members=members if isinstance(members, list) else list(members.keys()),
        )
        if name in self.groups:
            raise ValueError(
                f"Adding {name!r} already defined {list(self.groups.keys())}"
            )
        self.groups[name] = new_group
        if isinstance(members, dict):
            for member_props in members.values():
                self.add_layer(klayout_layer_props=member_props)

    def __str__(self):
        """Prints the number of KLayoutLayerProperty objects in the KLayoutLayerProperties object."""
        if len(self.groups) == 0:
            return (
                f"KLayoutLayerProperties ({len(self.layers)} layers total) \n"
                f"{list(self.layers)}"
            )
        return (
            f"KLayoutLayerProperties ({len(self.layers)} layers total, {len(self.groups)} groups) \n"
            f"{list(self.layers)}"
        )

    def get(self, name: str) -> KLayoutLayerProperty:
        """Returns Layer from name.

        Args:
            name: Name of layer.
        """
        if name not in self.layers:
            raise ValueError(f"Layer {name!r} not in {list(self.layers.keys())}")
        else:
            return self.layers[name]

    def __getitem__(self, val):
        """Allows accessing to the layer names like ls['gold2'].

        Args:
            val: Layer name to access within the KLayoutLayerProperties.

        Returns:
            self.layers[val]: KLayoutLayerProperty in the KLayoutLayerProperties.

        """
        try:
            return self.layers[val]
        except Exception as error:
            raise ValueError(
                f"Layer {val!r} not in LayerColors {list(self.layers.keys())}"
            ) from error

    def get_from_tuple(self, layer_tuple: Tuple[int, int]) -> KLayoutLayerProperty:
        """Returns KLayoutLayerProperty from layer tuple.

        Args:
            layer_tuple: Tuple of (gds_layer, gds_datatype).

        Returns:
            KLayoutLayerProperty
        """
        tuple_to_name = {v.layer: k for k, v in self.layers.items()}
        if layer_tuple not in tuple_to_name:
            raise ValueError(
                f"Layer color {layer_tuple} not in {list(tuple_to_name.keys())}"
            )

        name = tuple_to_name[layer_tuple]
        return self.layers[name]

    def get_layer_tuples(self):
        """Returns a tuple for each layer."""
        return {layer.layer for layer in self.layers.values()}

    def clear(self) -> None:
        """Deletes all layers in the LayerColors."""
        self.layers = {}

    @staticmethod
    def _write_props(
        file: TextIO,
        props: Union[KLayoutLayerProperty, KLayoutGroupProperty],
        level: int = 0,
    ) -> None:
        """Write properties to xml file.

        Args:
            file: Open file to write to.
            props: KLayoutLayerProperty or KLayoutGroupProperty object to write to file.
            level: Indentation level.
        """
        props_dict = props.dict()

        name = props_dict.pop("name")
        layer = props_dict.pop("layer")
        source = f"{layer[0]}/{layer[1]}@1" if layer else "*/*@*"
        props_dict.update({"name": name, "source": source})

        for prop, val in props_dict.items():
            if prop == "members":
                continue
            prop_str = "-".join(prop.split("_"))
            if isinstance(val, bool):
                val = f"{val}".lower()
            sp = level * " "
            file.write(
                f"{sp}<{prop_str}>{val}</{prop_str}>\n"
                if val is not None
                else f"{sp}<{prop_str}/>\n"
            )

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
        # TODO: Sort layers and groups beforehand
        with open(filepath, "w+") as file:
            file.write('<?xml version="1.0" encoding="utf-8"?>\n')
            file.write("<layer-properties>\n")

            grouped_layers = []
            for group_name, group in self.groups.items():
                if group_name != group.name:
                    group.name = group_name
                file.write(" <properties>\n")
                members = group.members

                self._write_props(file, group, level=2)

                for member in members:
                    file.write("  <group-members>\n")
                    self._write_props(file, self.layers[member], level=3)
                    file.write("  </group-members>\n")
                    grouped_layers.append(member)
                file.write(" </properties>\n")

            for layer_name, layer_props in self.layers.items():
                if layer_name in grouped_layers:
                    continue

                file.write(" <properties>\n")
                self._write_props(file, layer_props, level=2)
                file.write(" </properties>\n")
            file.write("</layer-properties>\n")

    def from_lyp(self):
        raise NotImplementedError()


# TODO: Write to .lyt technology file


if __name__ == "__main__":
    from gdsfactory.layers import LAYER_COLORS, LayerColors

    lc: LayerColors = LAYER_COLORS

    lyp = KLayoutLayerProperties(
        layers={
            layer.name: KLayoutLayerProperty(
                layer=(layer.gds_layer, layer.gds_datatype),
                name=layer.name,
                fill_color=layer.color,
                frame_color=layer.color,
                dither_pattern=layer.dither,
            )
            for layer in lc.layers.values()
        },
        groups={
            "Doping": KLayoutGroupProperty(
                members=["N", "NP", "NPP", "P", "PP", "PPP", "PDPP", "GENPP", "GEPPP"]
            ),
            "Simulation": KLayoutGroupProperty(
                members=["SIM_REGION", "MONITOR", "SOURCE"]
            ),
        },
    )

    #    lyp.create_group(members=["wg", "wg-2"], frame_color="blue")
    print(lyp)
    lyp.to_lyp("test_lyp")
