from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import kfactory as kf
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from gdsfactory.component import Component

if TYPE_CHECKING:
    from gdsfactory.technology import LayerViews


class LogicalLayer(BaseModel):
    """GDS design layer."""

    layer: tuple[int, int] | kf.LayerEnum | int

    def __eq__(self, other):
        """Check if two LogicalLayer instances are equal.

        This method compares the 'layer' attribute of the two LogicalLayer instances.

        Args:
            other (LogicalLayer): The other LogicalLayer instance to compare with.

        Returns:
            bool: True if the 'layer' attributes are equal, False otherwise.

        Raises:
            NotImplementedError: If 'other' is not an instance of LogicalLayer.
        """
        if not isinstance(other, type(self)):
            raise NotImplementedError(f"{other} is not a {type(self)}")
        return self.layer == other.layer

    def __hash__(self):
        """Generates a hash value for a LogicalLayer instance.

        This method allows LogicalLayer instances to be used in hash-based data structures such as sets and dictionaries.

        Returns:
            int: The hash value of the layer attribute.
        """
        return hash(self.layer)

    # Boolean AND (&)
    def __and__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean AND (&) operation between two layers.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform AND operation.

        Returns:
            A new DerivedLayer with the AND operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="and")

    # Boolean OR (|, +)
    def __or__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean OR (|) operation between two layers.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform OR operation.

        Returns:
            A new DerivedLayer with the OR operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="or")

    def __add__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean OR (+) operation between two derived layers.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform OR operation.

        Returns:
            A new DerivedLayer with the AND operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="or")

    # Boolean XOR (^)
    def __xor__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean XOR (^) operation between two derived layers.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform XOR operation.

        Returns:
            A new DerivedLayer with the XOR operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="xor")

    # Boolean NOT (-)
    def __sub__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean NOT (-) operation on a derived layer.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform NOT operation.

        Returns:
            A new DerivedLayer with the NOT operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="not")


class DerivedLayer(BaseModel):
    """Physical "derived layer", resulting from a combination of GDS design layers. Can be used by renderers and simulators.

    Overloads operators for simpler expressions.

    Attributes:
        input_layer1: primary layer comprising the derived layer. Can be a GDS design layer (kf.LayerEnum, tuple[int, int]), or another derived layer.
        input_layer2: secondary layer comprising the derived layer. Can be a GDS design layer (kf.LayerEnum, tuple[int, int]), or another derived layer.
        operation: operation to perform between layer1 and layer2. One of "and", "or", "xor", or "not" or associated symbols.
    """

    layer1: LogicalLayer | DerivedLayer | int
    layer2: LogicalLayer | DerivedLayer | int
    operation: Literal["and", "&", "or", "|", "xor", "^", "not", "-"]

    # Boolean AND (&)
    def __and__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean AND (&) operation between two layers.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform AND operation.

        Returns:
            A new DerivedLayer with the AND operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="and")

    # Boolean OR (|, +)
    def __or__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean OR (|) operation between two layers.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform OR operation.

        Returns:
            A new DerivedLayer with the OR operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="or")

    def __add__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean OR (+) operation between two derived layers.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform OR operation.

        Returns:
            A new DerivedLayer with the AND operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="or")

    # Boolean XOR (^)
    def __xor__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean XOR (^) operation between two derived layers.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform XOR operation.

        Returns:
            A new DerivedLayer with the XOR operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="xor")

    # Boolean NOT (-)
    def __sub__(self, other: LogicalLayer | DerivedLayer) -> DerivedLayer:
        """Represents boolean NOT (-) operation on a derived layer.

        Args:
            other (LogicalLayer | DerivedLayer): Another Layer object to perform NOT operation.

        Returns:
            A new DerivedLayer with the NOT operation logged.
        """
        return DerivedLayer(layer1=self, layer2=other, operation="not")


class LayerLevel(BaseModel):
    """Level for 3D LayerStack.

    Parameters:

        # Identification
            name: str
            layer: LogicalLayer or DerivedLayer. DerivedLayers can be composed of operations consisting of multiple other GDSLayers or other DerivedLayers.
            derived_layer: if the layer is derived, LogicalLayer to assign to the derived layer.

        # Extrusion
            thickness: layer thickness in um.
            thickness_tolerance: layer thickness tolerance in um.
            zmin: height position where material starts in um.
            zmin_tolerance: layer height tolerance in um.

            One of:
                sidewall_angle: in degrees with respect to normal.
                sidewall_angle_tolerance: in degrees.
                width_to_z: if sidewall_angle, reference z-position (0 --> zmin, 1 --> zmin + thickness, 0.5 in the middle).

                bias: shrink/grow of the level compared to the mask

                z_to_bias: most generic way to specify an extrusion.\
                    Two tuples of the same length specifying the shrink/grow (float) to apply between zmin (0) and zmin + thickness (1)\
                    I.e. [[z1, z2, ..., zN], [bias1, bias2, ..., biasN]]\
                    Defaults no buffering [[0, 1], [0, 0]].
                    NOTE: A dict might be more expressive.

        # Rendering
            mesh_order: lower mesh order (e.g. 1) will have priority over higher \
                    mesh order (e.g. 2) in the regions where materials overlap.
            material: used in the klayout script

        # Other
            info: all other rendering and simulation metadata should go here.
    """

    # ID
    name: str | None = None
    layer: LogicalLayer | DerivedLayer | None = None
    derived_layer: LogicalLayer | None = None

    # Extrusion rules
    thickness: float
    thickness_tolerance: float | None = None
    zmin: float
    zmin_tolerance: float | None = None
    sidewall_angle: float = 0.0
    sidewall_angle_tolerance: float | None = None
    width_to_z: float = 0.0
    z_to_bias: tuple[list[float], list[float]] | None = None
    bias: tuple[float, float] | float | None = None

    # Rendering
    mesh_order: int = 3
    material: str | None = None

    # Other
    info: dict[str, Any] = Field(default_factory=dict)

    @property
    def bounds(self) -> tuple[float, float]:
        """Calculates and returns the bounds of the layer level in the z-direction.

        Returns:
            tuple: A tuple containing the minimum and maximum z-values of the layer level.
        """
        return tuple(sorted([self.zmin, self.zmin + self.thickness]))


class LayerStack(BaseModel):
    """For simulation and 3D rendering. Captures design intent of the chip layers after fabrication.

    Parameters:
        layers: dict of layer_levels.
    """

    layers: dict[str, LayerLevel] = Field(
        default_factory=dict,
        description="dict of layer_levels",
    )

    def model_copy(self) -> LayerStack:
        """Returns a copy of the LayerStack."""
        return LayerStack.model_validate_json(self.model_dump_json())

    def __init__(self, **data: Any) -> None:
        """Add LayerLevels automatically for subclassed LayerStacks."""
        super().__init__(**data)

        for field in self.model_dump():
            val = getattr(self, field)
            if isinstance(val, LayerLevel):
                self.layers[field] = val

    def pprint(self) -> None:
        console = Console()
        table = Table(show_header=True, header_style="bold")
        keys = ["layer", "thickness", "material", "sidewall_angle"]

        for key in ["name"] + keys:
            table.add_column(key)

        for layer_name, layer in self.layers.items():
            port_dict = dict(layer)
            row = [layer_name] + [str(port_dict.get(key, "")) for key in keys]
            table.add_row(*row)

        console.print(table)

    def get_layer_to_thickness(self) -> dict[tuple[int, int], float]:
        """Returns layer tuple to thickness (um)."""
        layer_to_thickness = {}

        for level in self.layers.values():
            layer = level.layer

            if layer and level.thickness:
                layer_to_thickness[layer] = level.thickness
            elif hasattr(level, "operator"):
                layer_to_thickness[level.layer] = level.thickness

        return layer_to_thickness

    def get_component_with_derived_layers(self, component, **kwargs):
        """Returns component with derived layers."""
        return get_component_with_derived_layers(
            component=component, layer_stack=self, **kwargs
        )

    def get_layer_to_zmin(self) -> dict[tuple[int, int], float]:
        """Returns layer tuple to z min position (um)."""
        return {
            level.layer: level.zmin for level in self.layers.values() if level.thickness
        }

    def get_layer_to_material(self) -> dict[tuple[int, int], str]:
        """Returns layer tuple to material name."""
        return {
            level.layer: level.material
            for level in self.layers.values()
            if level.thickness
        }

    def get_layer_to_sidewall_angle(self) -> dict[tuple[int, int], str]:
        """Returns layer tuple to material name."""
        return {
            level.layer: level.sidewall_angle
            for level in self.layers.values()
            if level.thickness
        }

    def get_layer_to_info(self) -> dict[tuple[int, int], dict]:
        """Returns layer tuple to info dict."""
        return {level.layer: level.info for level in self.layers.values()}

    def get_layer_to_layername(self) -> dict[tuple[int, int], str]:
        """Returns layer tuple to layername."""
        d = defaultdict(list)
        for level_name, level in self.layers.items():
            d[level.layer].append(level_name)

        return d

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {level_name: dict(level) for level_name, level in self.layers.items()}

    def __getitem__(self, key) -> LayerLevel:
        """Access layer stack elements."""
        if key not in self.layers:
            layers = list(self.layers.keys())
            raise ValueError(f"{key!r} not in {layers}")

        return self.layers[key]

    def get_klayout_3d_script(
        self,
        layer_views: LayerViews | None = None,
        dbu: float | None = 0.001,
    ) -> str:
        """Returns script for 2.5D view in KLayout.

        You can include this information in your tech.lyt

        Args:
            layer_views: optional layer_views.
            dbu: Optional database unit. Defaults to 1nm.
        """
        layers = self.layers or {}

        unetched_layers = [
            layer_name
            for layer_name, level in layers.items()
            if level.layer and level.layer_type == "grow"
        ]
        etch_layers = [
            layer_name
            for layer_name, level in layers.items()
            if level.layer and level.layer_type == "etch"
        ]

        # remove all etched layers from the grown layers
        unetched_layers_dict = defaultdict(list)
        for layer_name in etch_layers:
            level = layers[layer_name]
            into = level.into or []
            for layer_name_etched in into:
                unetched_layers_dict[layer_name_etched].append(layer_name)
                if layer_name_etched in unetched_layers:
                    unetched_layers.remove(layer_name_etched)

        # define layers
        out = "\n".join(
            [
                f"{layer_name} = input({level.layer[0]}, {level.layer[1]})"
                for layer_name, level in layers.items()
                if level.layer
            ]
        )
        out += "\n"
        out += "\n"

        # define unetched layers
        for layer_name_etched, etching_layers in unetched_layers_dict.items():
            etching_layers = " - ".join(etching_layers)
            out += f"unetched_{layer_name_etched} = {layer_name_etched} - {etching_layers}\n"

        out += "\n"

        # define slabs
        for layer_name, level in layers.items():
            if level.layer_type == "etch":
                into = level.into or []
                for i, layer1 in enumerate(into):
                    out += f"slab_{layer1}_{layer_name}_{i} = {layer1} &amp; {layer_name}\n"

        out += "\n"

        for layer_name, level in layers.items():
            layer = level.layer
            zmin = level.zmin
            zmax = zmin + level.thickness
            if dbu:
                rnd_pl = len(str(dbu).split(".")[-1])
                zmin = round(zmin, rnd_pl)
                zmax = round(zmax, rnd_pl)

            if layer is None:
                continue

            elif level.layer_type == "etch":
                name = f"{layer_name}: {level.material}"

                into = level.into or []
                for i, layer1 in enumerate(into):
                    unetched_level = layers[layer1]
                    unetched_zmin = unetched_level.zmin
                    unetched_zmax = unetched_zmin + unetched_level.thickness

                    # slab
                    slab_layer_name = f"slab_{layer1}_{layer_name}_{i}"
                    slab_zmin = unetched_level.zmin
                    slab_zmax = unetched_zmax - level.thickness
                    name = f"{slab_layer_name}: {level.material} {layer[0]}/{layer[1]}"
                    txt = (
                        f"z("
                        f"{slab_layer_name}, "
                        f"zstart: {slab_zmin}, "
                        f"zstop: {slab_zmax}, "
                        f"name: '{name}'"
                    )
                    if layer_views:
                        txt += ", "
                        props = layer_views.get_from_tuple(layer)
                        if hasattr(props, "color"):
                            if props.color.fill == props.color.frame:
                                txt += f"color: {props.color.fill}"
                            else:
                                txt += (
                                    f"fill: {props.color.fill}, "
                                    f"frame: {props.color.frame}"
                                )
                    txt += ")"
                    out += f"{txt}\n"

            elif layer_name in unetched_layers:
                name = f"{layer_name}: {level.material} {layer[0]}/{layer[1]}"

                txt = (
                    f"z("
                    f"{layer_name}, "
                    f"zstart: {zmin}, "
                    f"zstop: {zmax}, "
                    f"name: '{name}'"
                )
                if layer_views:
                    txt += ", "
                    props = layer_views.get_from_tuple(layer)
                    if hasattr(props, "color"):
                        if props.color.fill == props.color.frame:
                            txt += f"color: {props.color.fill}"
                        else:
                            txt += (
                                f"fill: {props.color.fill}, "
                                f"frame: {props.color.frame}"
                            )

                txt += ")"
                out += f"{txt}\n"

        out += "\n"

        for layer_name in unetched_layers_dict:
            unetched_level = self.layers[layer_name]
            layer = unetched_level.layer

            unetched_zmin = unetched_level.zmin
            unetched_zmax = unetched_zmin + unetched_level.thickness
            name = f"{slab_layer_name}: {unetched_level.material}"

            unetched_layer_name = f"unetched_{layer_name}"
            name = f"{unetched_layer_name}: {unetched_level.material} {layer[0]}/{layer[1]}"
            txt = (
                f"z("
                f"{unetched_layer_name}, "
                f"zstart: {unetched_zmin}, "
                f"zstop: {unetched_zmax}, "
                f"name: '{name}'"
            )
            if layer_views:
                txt += ", "
                props = layer_views.get_from_tuple(layer)
                if hasattr(props, "color"):
                    if props.color.fill == props.color.frame:
                        txt += f"color: {props.color.fill}"
                    else:
                        txt += (
                            f"fill: {props.color.fill}, " f"frame: {props.color.frame}"
                        )
            txt += ")"
            out += f"{txt}\n"

        return out

    def filtered(self, layers) -> LayerStack:
        """Returns filtered layerstack, given layer specs."""
        return LayerStack(
            layers={k: self.layers[k] for k in layers if k in self.layers}
        )

    def z_offset(self, dz) -> LayerStack:
        """Translates the z-coordinates of the layerstack."""
        layers = self.layers or {}
        for layer in layers.values():
            layer.zmin += dz

        return self

    def invert_zaxis(self) -> LayerStack:
        """Flips the zmin values about the origin."""
        layers = self.layers or {}
        for layer in layers.values():
            layer.zmin *= -1

        return self


def get_shapes_from_arbitrary_layer(
    layer: DerivedLayer | LogicalLayer,
    component: Component,
) -> kf.kdb.Region:
    """Recursively evaluate the derived layer operations on the component.

    Arguments:
        layer: derived layer whose shapes to add.
        component: to read polygons from,

    Returns:
        r: KLayout region corresponding to the component's derived layer.
    """
    from gdsfactory.pdk import get_layer

    # Component polygons
    polygons_per_layer = component.get_polygons()

    # If we hit a LogicalLayer, retrieve the polygons
    if isinstance(layer, LogicalLayer):
        layer_index = get_layer(layer.layer)
        polygons = polygons_per_layer[layer_index]
        return kf.kdb.Region(polygons)
    elif isinstance(layer, DerivedLayer):
        # Recurse through derived layer1 if needed
        if isinstance(layer.layer1, LogicalLayer):
            layer_index = get_layer(layer.layer1.layer)
            polygons = polygons_per_layer[layer_index]
            r1 = kf.kdb.Region(polygons)
        elif isinstance(layer.layer1, DerivedLayer):
            r1 = get_shapes_from_arbitrary_layer(layer.layer1, component)

        # Recurse through derived layer2 if needed
        if isinstance(layer.layer2, LogicalLayer):
            layer_index = get_layer(layer.layer2.layer)
            polygons = polygons_per_layer[layer_index]
            r2 = kf.kdb.Region(polygons)
        elif isinstance(layer.layer2, DerivedLayer):
            r2 = get_shapes_from_arbitrary_layer(layer.layer2, component)

        # Get new region from boolean operation
        return gf.component.boolean_operations[layer.operation](r1, r2)

    else:
        raise ValueError("layer must be one of LogicalLayer or DerivedLayer")


def get_component_with_derived_layers(component, layer_stack: LayerStack) -> Component:
    """Returns a component with derived layers.

    Args:
        component: Component to get derived layers for.
        layer_stack: Layer stack to get derived layers from.
    """
    from gdsfactory.pdk import get_layer

    component_derived = Component()

    for layer_name, level in layer_stack.layers.items():
        if isinstance(level.layer, LogicalLayer):
            derived_layer_index = get_layer(level.layer.layer)
        elif isinstance(level.layer, DerivedLayer):
            if level.derived_layer is not None:
                derived_layer_index = get_layer(level.derived_layer.layer)
            else:
                raise ValueError(
                    f"Error at LayerLevel {layer_name}: derived_layer must be provided if the level's layer is a DerivedLayer"
                )
        else:
            raise ValueError("layer must be one of LogicalLayer or DerivedLayer")

        shapes = get_shapes_from_arbitrary_layer(layer=level.layer, component=component)
        component_derived.shapes(derived_layer_index).insert(shapes)

    component_derived.add_ports(component.ports)
    return component_derived


if __name__ == "__main__":
    # For now, make regular layers trivial DerivedLayers
    # This might be automatable during LayerStack instantiation, or we could modify the Layer object in LayerMap too

    from gdsfactory.generic_tech import LAYER

    layer1 = LogicalLayer(layer=(2, 0))
    layer2 = LogicalLayer(layer=LAYER.WG)

    ls = LayerStack(
        layers={
            "layerlevel_layer1": LayerLevel(layer=layer1, thickness=10, zmin=0),
            "layerlevel_layer2": LayerLevel(layer=layer2, thickness=10, zmin=10),
            "layerlevel_and_layer": LayerLevel(
                layer=layer1 & layer2,
                thickness=10,
                zmin=0,
                derived_layer=LogicalLayer(layer=(3, 0)),
            ),
            "layerlevel_xor_layer": LayerLevel(
                layer=layer1 ^ layer2,
                thickness=10,
                zmin=0,
                derived_layer=LogicalLayer(layer=(4, 0)),
            ),
            "layerlevel_not_layer": LayerLevel(
                layer=layer1 - layer2,
                thickness=10,
                zmin=0,
                derived_layer=LogicalLayer(layer=(5, 0)),
            ),
            "layerlevel_or_layer": LayerLevel(
                layer=layer1 | layer2,
                thickness=10,
                zmin=0,
                derived_layer=LogicalLayer(layer=(6, 0)),
            ),
            "layerlevel_composed_layer": LayerLevel(
                layer=layer1 - (layer1 & layer2),
                thickness=10,
                zmin=0,
                derived_layer=LogicalLayer(layer=(7, 0)),
            ),
        }
    )

    # Test with simple component
    import gdsfactory as gf

    c = gf.Component()

    rect1 = c << gf.components.rectangle(size=(10, 10), layer=(1, 0))
    rect2 = c << gf.components.rectangle(size=(10, 10), layer=(2, 0))
    rect2.dmove((5, 5))
    c.show()

    c = get_component_with_derived_layers(c, ls)
    c.show()

    # LAYER_STACK.get_klayout_3d_script()
