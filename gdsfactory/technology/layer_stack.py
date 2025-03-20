from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

import kfactory as kf
from kfactory.layer import LayerEnum
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.table import Table

import gdsfactory as gf
from gdsfactory.technology.layer_views import LayerViews
from gdsfactory.typings import LayerSpec

if TYPE_CHECKING:
    from gdsfactory.component import Component

T = TypeVar("T", bound="AbstractLayer")


class AbstractLayer(BaseModel):
    """Generic design layer.

    Attributes:
        sizings_xoffsets: sequence of xoffset sizings to apply to this Logical or Derived layer.
        sizings_yoffsets: sequence of yoffset sizings to apply to this Logical or Derived layer.
        sizings_modes: sequence of sizing modes to apply to this Logical or Derived layer.
    """

    sizings_xoffsets: Sequence[int] = (0,)
    sizings_yoffsets: Sequence[int] = (0,)
    sizings_modes: Sequence[int] = (2,)

    def _perform_operation(
        self, other: AbstractLayer, operation: Literal["and", "or", "xor", "not"]
    ) -> DerivedLayer:
        if isinstance(other, DerivedLayer | LogicalLayer) and isinstance(
            self, DerivedLayer | LogicalLayer
        ):
            return DerivedLayer(layer1=self, layer2=other, operation=operation)
        raise ValueError(f"{other} is not a DerivedLayer or LogicalLayer")

    # Boolean AND (&)
    def __and__(self, other: AbstractLayer) -> DerivedLayer:
        """Represents boolean AND (&) operation between two layers.

        Args:
            other (AbstractLayer): Another Layer object to perform AND operation.

        Returns:
            A new DerivedLayer with the AND operation logged.
        """
        return self._perform_operation(other, "and")

    # Boolean OR (|, +)
    def __or__(self, other: AbstractLayer) -> DerivedLayer:
        """Represents boolean OR (|) operation between two layers.

        Args:
            other (AbstractLayer): Another Layer object to perform OR operation.

        Returns:
            A new DerivedLayer with the OR operation logged.
        """
        return self._perform_operation(other, "or")

    def __add__(self, other: AbstractLayer) -> DerivedLayer:
        """Represents boolean OR (+) operation between two derived layers.

        Args:
            other (AbstractLayer): Another Layer object to perform OR operation.

        Returns:
            A new DerivedLayer with the AND operation logged.
        """
        return self._perform_operation(other, "or")

    # Boolean XOR (^)
    def __xor__(self, other: AbstractLayer) -> DerivedLayer:
        """Represents boolean XOR (^) operation between two derived layers.

        Args:
            other (AbstractLayer): Another Layer object to perform XOR operation.

        Returns:
            A new DerivedLayer with the XOR operation logged.
        """
        return self._perform_operation(other, "xor")

    # Boolean NOT (-)
    def __sub__(self, other: AbstractLayer) -> DerivedLayer:
        """Represents boolean NOT (-) operation on a derived layer.

        Args:
            other (AbstractLayer): Another Layer object to perform NOT operation.

        Returns:
            A new DerivedLayer with the NOT operation logged.
        """
        return self._perform_operation(other, "not")

    def sized(
        self: T,
        xoffset: int | tuple[int, ...],
        yoffset: int | tuple[int, ...] | None = None,
        mode: int | tuple[int, ...] | None = None,
    ) -> T:
        """Accumulates a list of sizing operations for the layer by the provided offset (in dbu).

        Args:
            xoffset (int | tuple): number of dbu units to buffer by. Can be a tuple for sequential sizing operations.
            yoffset (int | tuple): number of dbu units to buffer by in the y-direction. If not specified, uses xfactor. Can be a tuple for sequential sizing operations.
            mode (int | tuple): mode of the sizing operation(s). Can be a tuple for sequential sizing operations.
        """
        #  Validate inputs
        xoffset_list: list[int]
        if isinstance(xoffset, int):
            xoffset_list = [xoffset]
        else:
            xoffset_list = list(xoffset)
        yoffset_list: list[int]
        if isinstance(yoffset, tuple):
            if len(yoffset) != len(xoffset_list):
                raise ValueError(
                    "If yoffset is provided as a tuple, length must be equal to xoffset!"
                )
            else:
                yoffset_list = list(yoffset)
        elif yoffset is None:
            yoffset_list = xoffset_list
        else:
            yoffset_list = [yoffset] * len(xoffset_list)

        mode_list: list[int]
        if isinstance(mode, tuple):
            if len(mode) != len(xoffset_list):
                raise ValueError(
                    "If mode is provided as a tuple, length must be equal to xoffset!"
                )
            else:
                mode_list = list(mode)
        elif mode is None:
            mode_list = [2] * len(xoffset_list)
        else:
            mode_list = [mode] * len(xoffset_list)

        # Accumulate
        sizings_xoffsets = list(self.sizings_xoffsets) + xoffset_list
        sizings_yoffsets = list(self.sizings_yoffsets) + yoffset_list
        sizings_modes = list(self.sizings_modes) + mode_list

        # Return a copy of the layer with updated sizings
        current_layer_attributes = self.__dict__.copy()
        current_layer_attributes["sizings_xoffsets"] = sizings_xoffsets
        current_layer_attributes["sizings_yoffsets"] = sizings_yoffsets
        current_layer_attributes["sizings_modes"] = sizings_modes
        return self.__class__(**current_layer_attributes)


class LogicalLayer(AbstractLayer):
    """GDS design layer."""

    layer: LayerSpec

    def __eq__(self, other: object) -> bool:
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

    def __hash__(self) -> int:
        """Generates a hash value for a LogicalLayer instance.

        This method allows LogicalLayer instances to be used in hash-based data structures such as sets and dictionaries.

        Returns:
            int: The hash value of the layer attribute.
        """
        return hash(self.layer)

    def get_shapes(self, component: "Component") -> kf.kdb.Region:
        """Return the shapes of the component argument corresponding to this layer.

        Arguments:
            component: Component from which to extract shapes on this layer.

        Returns:
            kf.kdb.Region: A region of polygons on this layer.
        """
        from gdsfactory.pdk import get_layer

        polygons_per_layer = component.get_polygons()
        layer_index = get_layer(self.layer)
        polygons = (
            polygons_per_layer[layer_index] if layer_index in polygons_per_layer else []
        )
        region = kf.kdb.Region(polygons)
        if not (
            all(v == 0 for v in self.sizings_xoffsets)
            and all(v == 0 for v in self.sizings_yoffsets)
        ):
            for xoffset, yoffset, mode in zip(
                self.sizings_xoffsets, self.sizings_yoffsets, self.sizings_modes
            ):
                region = region.sized(xoffset, yoffset, mode)
        return region

    def __repr__(self) -> str:
        """Print text representation."""
        return f"{self.layer}"

    __str__ = __repr__


class DerivedLayer(AbstractLayer):
    """Physical "derived layer", resulting from a combination of GDS design layers. Can be used by renderers and simulators.

    Overloads operators for simpler expressions.

    Attributes:
        input_layer1: primary layer comprising the derived layer. Can be a GDS design layer (kf.kcell.LayerEnum , tuple[int, int]), or another derived layer.
        input_layer2: secondary layer comprising the derived layer. Can be a GDS design layer (kf.kcell.LayerEnum , tuple[int, int]), or another derived layer.
        operation: operation to perform between layer1 and layer2. One of "and", "or", "xor", or "not" or associated symbols.
    """

    layer1: DerivedLayer | LogicalLayer
    layer2: DerivedLayer | LogicalLayer
    operation: Literal["and", "&", "or", "|", "xor", "^", "not", "-"]

    def __hash__(self) -> int:
        """Generates a hash value for a LogicalLayer instance.

        This method allows LogicalLayer instances to be used in hash-based data structures such as sets and dictionaries.

        Returns:
            int: The hash value of the layer attribute.
        """
        return hash((self.layer1.__hash__(), self.layer2.__hash__(), self.operation))

    @property
    def keyword_to_symbol(self) -> dict[str, str]:
        return {
            "and": "&",
            "or": "|",
            "xor": "^",
            "not": "-",
        }

    @property
    def symbol_to_keyword(self) -> dict[str, str]:
        return {
            "&": "and",
            "|": "or",
            "^": "xor",
            "-": "not",
        }

    def get_symbol(self) -> str:
        if self.operation in self.keyword_to_symbol:
            return self.keyword_to_symbol[self.operation]
        else:
            return self.operation

    def get_shapes(self, component: "Component") -> kf.kdb.Region:
        """Return the shapes of the component argument corresponding to this layer.

        Arguments:
            component: Component from which to extract shapes on this layer.

        Returns:
            kf.kdb.Region: A region of polygons on this layer.
        """
        from gdsfactory.component import boolean_operations

        r1 = self.layer1.get_shapes(component)
        r2 = self.layer2.get_shapes(component)
        region = boolean_operations[self.operation](r1, r2)
        if not (
            all(v == 0 for v in self.sizings_xoffsets)
            and all(v == 0 for v in self.sizings_yoffsets)
        ):
            for xoffset, yoffset, mode in zip(
                self.sizings_xoffsets, self.sizings_yoffsets, self.sizings_modes
            ):
                region = region.sized(xoffset, yoffset, mode)
        return region

    def __repr__(self) -> str:
        """Print text representation."""
        return f"({self.layer1} {self.get_symbol()} {self.layer2})"

    __str__ = __repr__


BroadLayer: TypeAlias = (
    LogicalLayer | DerivedLayer | int | str | tuple[int, int] | LayerEnum
)


class LayerLevel(BaseModel):
    """Level for 3D LayerStack.

    Parameters:
        name: str
        layer: LogicalLayer or DerivedLayer. DerivedLayers can be composed of operations consisting of multiple other GDSLayers or other DerivedLayers.
        derived_layer: if the layer is derived, LogicalLayer to assign to the derived layer.
        thickness: layer thickness in um.
        thickness_tolerance: layer thickness tolerance in um.
        width_tolerance: layer width tolerance in um.
        zmin: height position where material starts in um.
        zmin_tolerance: layer height tolerance in um.
        sidewall_angle: in degrees with respect to normal.
        sidewall_angle_tolerance: in degrees.
        width_to_z: if sidewall_angle, reference z-position (0 --> zmin, 1 --> zmin + thickness, 0.5 in the middle).
        bias: shrink/grow of the level compared to the mask
        z_to_bias: most generic way to specify an extrusion.\
            Two tuples of the same length specifying the shrink/grow (float) to apply between zmin (0) and zmin + thickness (1)\
            I.e. [[z1, z2, ..., zN], [bias1, bias2, ..., biasN]]\
                    Defaults no buffering [[0, 1], [0, 0]].
                    NOTE: A dict might be more expressive.
        mesh_order: lower mesh order (e.g. 1) will have priority over higher mesh order (e.g. 2) in the regions where materials overlap.
        material: used in the klayout script
        info: all other rendering and simulation metadata should go here.
    """

    # ID
    name: str | None = None
    layer: BroadLayer
    derived_layer: LogicalLayer | None = None

    # Extrusion rules
    thickness: float
    thickness_tolerance: float | None = None
    width_tolerance: float | None = None
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

    @field_validator("layer")
    @classmethod
    def check_layer(
        cls, layer: BroadLayer | int | str | tuple[int, int] | LayerEnum
    ) -> LogicalLayer | DerivedLayer:
        if isinstance(layer, int | str | tuple | LayerEnum):
            layer = gf.get_layer(layer)
            return LogicalLayer(layer=layer)

        return layer

    @property
    def bounds(self) -> tuple[float, float]:
        """Calculates and returns the bounds of the layer level in the z-direction.

        Returns:
            tuple: A tuple containing the minimum and maximum z-values of the layer level.
        """
        z_values = [self.zmin, self.zmin + self.thickness]
        z_values.sort()
        return z_values[0], z_values[1]


class LayerStack(BaseModel):
    """For simulation and 3D rendering. Captures design intent of the chip layers after fabrication.

    Parameters:
        layers: dict of layer_levels.
    """

    layers: dict[str, LayerLevel] = Field(
        default_factory=dict,
        description="dict of layer_levels",
    )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> LayerStack:
        """Returns a copy of the LayerStack."""
        return super().model_copy(update=update, deep=True)

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

        for key in ["name", *keys]:
            table.add_column(key)

        for layer_name, layer in self.layers.items():
            port_dict = dict(layer)
            row = [layer_name] + [str(port_dict.get(key, "")) for key in keys]
            table.add_row(*row)

        console.print(table)

    def get_layer_to_thickness(self) -> dict[BroadLayer, float]:
        """Returns layer tuple to thickness (um)."""
        layer_to_thickness: dict[BroadLayer, float] = {}

        for level in self.layers.values():
            layer = level.layer

            if (layer and level.thickness) or hasattr(level, "operator"):
                layer_to_thickness[layer] = level.thickness

        return layer_to_thickness

    def get_component_with_derived_layers(
        self, component: "Component", **kwargs: Any
    ) -> "Component":
        """Returns component with derived layers."""
        return get_component_with_derived_layers(
            component=component, layer_stack=self, **kwargs
        )

    def get_layer_to_zmin(self) -> dict[BroadLayer, float]:
        """Returns layer tuple to z min position (um)."""
        return {
            level.layer: level.zmin for level in self.layers.values() if level.thickness
        }

    def get_layer_to_material(self) -> dict[BroadLayer, str | None]:
        """Returns layer tuple to material name."""
        return {
            level.layer: level.material
            for level in self.layers.values()
            if level.thickness
        }

    def get_layer_to_sidewall_angle(self) -> dict[BroadLayer, float]:
        """Returns layer tuple to material name."""
        return {
            level.layer: level.sidewall_angle
            for level in self.layers.values()
            if level.thickness
        }

    def get_layer_to_info(self) -> dict[BroadLayer, dict[str, Any]]:
        """Returns layer tuple to info dict."""
        return {level.layer: level.info for level in self.layers.values()}

    def get_layer_to_layername(self) -> dict[BroadLayer, list[str]]:
        """Returns layer tuple to layername."""
        d: dict[BroadLayer, list[str]] = defaultdict(list)
        for level_name, level in self.layers.items():
            d[level.layer].append(level_name)

        return d

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {level_name: dict(level) for level_name, level in self.layers.items()}

    def __getitem__(self, key: str) -> LayerLevel:
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
        from gdsfactory.pdk import get_layer_views

        layers = self.layers or {}
        layer_views = layer_views or get_layer_views()

        # Collect etch layers and unetched layers
        etch_layers = [
            layer_name
            for layer_name, level in layers.items()
            if isinstance(level.layer, DerivedLayer)
        ]

        unetched_layers = [
            layer_name
            for layer_name, level in layers.items()
            if isinstance(level.layer, LogicalLayer)
        ]
        from gdsfactory.pdk import get_layer, get_layer_tuple

        # Define input layers
        out = "\n".join(
            [
                f"{layer_name} = input({(__layer := get_layer_tuple(level.derived_layer.layer))[0]}, {__layer[1]})"
                for layer_name, level in layers.items()
                if level.derived_layer
            ]
        )
        out += "\n\n"

        # Remove all etched layers from the grown layers
        unetched_layers_dict: dict[BroadLayer, list[str]] = defaultdict(list)
        for layer_name in etch_layers:
            level = layers[layer_name]
            if level.derived_layer:
                unetched_layers_dict[level.derived_layer.layer].append(layer_name)
                if level.derived_layer.layer in unetched_layers:
                    unetched_layers.remove(level.derived_layer.layer)  # type: ignore[arg-type]

        # Define layers
        out += "\n".join(
            [
                f"{layer_name} = input({level.layer.layer[0]}, {level.layer.layer[1]})"  # type: ignore[index,union-attr]
                for layer_name, level in layers.items()
                if hasattr(level.layer, "layer")
            ]
        )
        out += "\n\n"

        # Define unetched layers
        for layer_name_etched, etching_layers in unetched_layers_dict.items():
            etching_layers_str = " - ".join(etching_layers)
            out += f"unetched_{layer_name_etched} = {layer_name_etched} - {etching_layers_str}\n"

        out += "\n"

        # Define slabs
        for layer_name, level in layers.items():
            if level.derived_layer:
                _layer_from_derived = get_layer(level.derived_layer.layer)
                out += f"slab_{_layer_from_derived}_{layer_name} = {_layer_from_derived} & {layer_name}\n"

        out += "\n"

        for layer_name, level in layers.items():
            layer = level.layer
            zmin = level.zmin
            zmax = zmin + level.thickness
            if dbu:
                rnd_pl = len(str(dbu).split(".")[-1])
                zmin = round(zmin, rnd_pl)
                zmax = round(zmax, rnd_pl)

            elif level.derived_layer:
                derived_layer = level.derived_layer.layer
                derived_layer_layer = get_layer_tuple(derived_layer)
                slab_layer_name = f"slab_{layer}_{layer_name}"
                slab_zmin = zmin
                slab_zmax = zmax - level.thickness
                name = f"{slab_layer_name}: {level.material} {derived_layer_layer[0]}/{derived_layer_layer[1]}"
                txt = (
                    f"z("
                    f"{slab_layer_name}, "
                    f"zstart: {slab_zmin}, "
                    f"zstop: {slab_zmax}, "
                    f"name: '{name}'"
                )
                if layer_views:
                    txt += ", "
                    if layer in layer_views:  # type: ignore[operator]
                        props = layer_views.get_from_tuple(layer)  # type: ignore[arg-type]
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
                # TODO: Reimplement this
                layer_tuple = get_layer_tuple(layer.layer)  # type: ignore[union-attr]
                name = (
                    f"{layer_name}: {level.material} {layer_tuple[0]}/{layer_tuple[1]}"
                )
                txt = f"z({layer_name}, zstart: {zmin}, zstop: {zmax}, name: '{name}'"
                if layer_views:
                    txt += ", "
                    props = layer_views.get_from_tuple(get_layer_tuple(layer_tuple))
                    if hasattr(props, "color"):
                        if props.color.fill == props.color.frame:
                            txt += f"color: {props.color.fill}"
                        else:
                            txt += (
                                f"fill: {props.color.fill}, frame: {props.color.frame}"
                            )

                txt += ")"
                out += f"{txt}\n"

        return out

    def filtered(self, layers: list[str]) -> LayerStack:
        """Returns filtered layerstack, given layer specs."""
        return LayerStack(
            layers={k: self.layers[k] for k in layers if k in self.layers}
        )

    def z_offset(self, dz: float) -> LayerStack:
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


def get_component_with_derived_layers(
    component: "Component", layer_stack: LayerStack
) -> "Component":
    """Returns a component with derived layers.

    Args:
        component: Component to get derived layers for.
        layer_stack: Layer stack to get derived layers from.
    """
    from gdsfactory.component import Component
    from gdsfactory.pdk import get_layer

    component_derived = Component()

    for level in layer_stack.layers.values():
        if level.derived_layer is None:
            if isinstance(level.layer, LogicalLayer):
                derived_layer_index = get_layer(level.layer.layer)
            else:
                raise ValueError(
                    "If derived_layer is not provided, the LayerLevel layer must be a LogicalLayer"
                )
        else:
            derived_layer_index = get_layer(level.derived_layer.layer)
        if isinstance(level.layer, AbstractLayer):
            shapes = level.layer.get_shapes(component=component)
            component_derived.shapes(derived_layer_index).insert(shapes)

    component_derived.add_ports(component.ports)
    return component_derived


if __name__ == "__main__":
    # For now, make regular layers trivial DerivedLayers
    # This might be automatable during LayerStack instantiation, or we could modify the Layer object in LayerMap too

    layer1 = LogicalLayer(layer=(1, 0))
    layer2 = LogicalLayer(layer=(2, 0))
    layer1_sized = LogicalLayer(layer=(1, 0)).sized(10000)
    layer1_sized_asymmetric = LogicalLayer(layer=(1, 0)).sized(0, 50000)

    layer3 = LogicalLayer(layer=(3, 0))
    layer3_sequence = LogicalLayer(layer=(3, 0)).sized(2000, 2000).sized(-1000, -1000)
    layer3_sequence_list = LogicalLayer(layer=(3, 0)).sized((2000, 2000))
    layer3_sequence_lists = LogicalLayer(layer=(3, 0)).sized((0, 0), (5000, 1000))

    ls = LayerStack(
        layers={
            "layerlevel_layer1": LayerLevel(layer=layer1, thickness=10, zmin=0),
            "layerlevel_layer1_sized": LayerLevel(
                layer=layer1_sized, thickness=10, zmin=0
            ),
            "layerlevel_layer1_asymmetric": LayerLevel(
                layer=layer1_sized_asymmetric, thickness=10, zmin=0
            ),
            "layerlevel_layer1_to_layer2_derived": LayerLevel(
                layer=layer1_sized, thickness=10, zmin=0, derived_layer=layer2
            ),
            "layerlevel_layer3": LayerLevel(layer=layer3, thickness=10, zmin=0),
            "layer3_sequence": LayerLevel(
                layer=layer3_sequence,
                thickness=10,
                zmin=0,
                derived_layer=LogicalLayer(layer=(4, 0)),
            ),
            "layer3_sequence_list": LayerLevel(
                layer=layer3_sequence_list,
                thickness=10,
                zmin=0,
                derived_layer=LogicalLayer(layer=(5, 0)),
            ),
            "layer3_sequence_lists": LayerLevel(
                layer=layer3_sequence_lists,
                thickness=10,
                zmin=0,
                derived_layer=LogicalLayer(layer=(6, 0)),
            ),
        }
    )

    # Test with simple component
    import gdsfactory as gf

    c = gf.Component()

    rect1 = c << gf.components.rectangle(size=(10, 10), layer=(1, 0))
    rect2 = c << gf.components.rectangle(size=(10, 10), layer=(3, 0))
    rect2.move((30, 30))
    # c.show()

    # import gdsfactory as gf

    # c = gf.Component()

    # rect1 = c << gf.components.rectangle(size=(10, 10), layer=(1, 0))
    # rect2 = c << gf.components.rectangle(size=(10, 10), layer=(2, 0))
    # rect2.move((5, 5))
    # c.show()

    c = get_component_with_derived_layers(c, ls)
    c.show()

    # s = ls.get_klayout_3d_script()
    # print(s)

    res = ls.get_klayout_3d_script()
