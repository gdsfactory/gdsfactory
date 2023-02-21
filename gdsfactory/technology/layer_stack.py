from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import Literal
from pydantic import BaseModel, Field

from gdsfactory.technology.layer_views import LayerViews


class LayerLevel(BaseModel):
    """Level for 3D LayerStack.

    Parameters:
        layer: (GDSII Layer number, GDSII datatype).
        thickness: layer thickness in um.
        thickness_tolerance: layer thickness tolerance in um.
        zmin: height position where material starts in um.
        material: material name.
        sidewall_angle: in degrees with respect to normal.
        width_to_z: if sidewall_angle, relative z-position
            (0 --> zmin, 1 --> zmin + thickness).
        z_to_bias: parametrizes shrinking/expansion of the design GDS layer
            when extruding from zmin (0) to zmin + thickness (1).
            Defaults no buffering [[0, 1], [0, 0]].
        info: simulation_info and other types of metadata.
            mesh_order: lower mesh order (1) will have priority over higher
                mesh order (2) in the regions where materials overlap.
            refractive_index: refractive_index
                can be int, complex or function that depends on wavelength (um).
            type: grow, etch, implant, or background.
            mode: octagon, taper, round.
                https://gdsfactory.github.io/klayout_pyxs/DocGrow.html
            into: etch into another layer.
                https://gdsfactory.github.io/klayout_pyxs/DocGrow.html
            doping_concentration: for implants.
            resistivity: for metals.
            bias: in um for the etch.
    """

    layer: Optional[Tuple[int, int]]
    thickness: float
    thickness_tolerance: Optional[float] = None
    zmin: float
    material: Optional[str] = None
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    z_to_bias: Optional[Tuple[List[float], List[float]]] = None
    info: Dict[str, Any] = {}


class DerivedLayerLevel(LayerLevel):
    layer: Optional[Tuple[int, int]]
    layer1: Optional[Tuple[int, int]]
    layer2: Optional[Tuple[int, int]]
    operator: Literal["-", "+", "&", "|"]

    """Level for 3D LayerStack.

    layer = layer1 operator layer2

    Parameters:
        layer: (GDSII Layer number, GDSII datatype) for operation result.
        layer1: (GDSII Layer number, GDSII datatype).
        layer2: (GDSII Layer number, GDSII datatype).
        operator: can be
            - not
            & and
            | + or
        thickness: layer thickness in um.
        thickness_tolerance: layer thickness tolerance in um.
        zmin: height position where material starts in um.
        material: material name.
        sidewall_angle: in degrees with respect to normal.
        width_to_z: if sidewall_angle, relative z-position
            (0 --> zmin, 1 --> zmin + thickness).
        z_to_bias: parametrizes shrinking/expansion of the design GDS layer
            when extruding from zmin (0) to zmin + thickness (1).
            Defaults no buffering [[0, 1], [0, 0]].
        info: simulation_info and other types of metadata.
            mesh_order: lower mesh order (1) will have priority over higher
                mesh order (2) in the regions where materials overlap.
            refractive_index: refractive_index
                can be int, complex or function that depends on wavelength (um).
            type: grow, etch, implant, or background.
            mode: octagon, taper, round.
                https://gdsfactory.github.io/klayout_pyxs/DocGrow.html
            into: etch into another layer.
                https://gdsfactory.github.io/klayout_pyxs/DocGrow.html
            doping_concentration: for implants.
            resistivity: for metals.
            bias: in um for the etch.
    """


class LayerStack(BaseModel):
    """For simulation and 3D rendering.

    Parameters:
        layers: dict of layer_levels.
    """

    layers: Optional[Dict[str, LayerLevel]] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        """Add LayerLevels automatically for subclassed LayerStacks."""
        super().__init__(**data)

        for field in self.dict():
            val = getattr(self, field)
            if isinstance(val, LayerLevel):
                self.layers[field] = val

    def get_layer_to_thickness(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to thickness (um)."""
        layer_to_thickness = {}

        for level in self.layers.values():
            layer = level.layer

            if layer and level.thickness:
                layer_to_thickness[layer] = level.thickness
            elif hasattr(level, "operator"):
                layer_to_thickness[level.layer] = level.thickness

        return layer_to_thickness

    def get_component_with_derived_layers(self, component):
        """Returns component with derived layers."""
        import gdstk

        component_layers = component.get_layers()

        non_derived_layers = []
        derived_levels = []

        for level in self.layers.values():
            layer = level.layer

            if layer and level.thickness:
                if hasattr(level, "operator"):
                    derived_levels.append(level)
                elif layer in component_layers:
                    non_derived_layers.append(layer)

        component_derived = component.extract(layers=non_derived_layers)

        for derived_level in derived_levels:
            if derived_level.operator == "-":
                operation = "not"
            elif derived_level.operator == "&":
                operation = "and"
            elif derived_level.operator in ["|", "+"]:
                operation = "or"

            gds_layer, gds_datatype = derived_level.layer
            A_polys = component.get_polygons(by_spec=derived_level.layer1)
            B_polys = component.get_polygons(by_spec=derived_level.layer2)
            p = gdstk.boolean(
                operand1=A_polys,
                operand2=B_polys,
                operation=operation,
                layer=gds_layer,
                datatype=gds_datatype,
            )
            component_derived.add(p)
        component_derived.add_ports(component.ports)
        component_derived.name = f"{component.name}_derived_layers"
        return component_derived

    def get_layer_to_zmin(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to z min position (um)."""
        return {
            level.layer: level.zmin for level in self.layers.values() if level.thickness
        }

    def get_layer_to_material(self) -> Dict[Tuple[int, int], str]:
        """Returns layer tuple to material name."""
        return {
            level.layer: level.material
            for level in self.layers.values()
            if level.thickness
        }

    def get_layer_to_sidewall_angle(self) -> Dict[Tuple[int, int], str]:
        """Returns layer tuple to material name."""
        return {
            level.layer: level.sidewall_angle
            for level in self.layers.values()
            if level.thickness
        }

    def get_layer_to_info(self) -> Dict[Tuple[int, int], Dict]:
        """Returns layer tuple to info dict."""
        return {level.layer: level.info for level in self.layers.values()}

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {level_name: dict(level) for level_name, level in self.layers.items()}

    def __getitem__(self, key) -> LayerLevel:
        """Access layer stack elements."""
        if key not in self.layers:
            layers = list(self.layers.keys())
            raise ValueError(f"{key!r} not in {layers}")

        return self.layers[key]

    def get_klayout_3d_script(
        self,
        layer_views: Optional[LayerViews] = None,
        dbu: Optional[float] = 0.001,
    ) -> str:
        """Returns script for 2.5D view in KLayout.

        You can include this information in your tech.lyt

        Args:
            layer_views: optional layer_views.
            dbu: Optional database unit. Defaults to 1nm.
        """
        out = ""

        # define non derived layers
        out = "\n".join(
            [
                f"{layer_name} = input({level.layer[0]}, {level.layer[1]})"
                for layer_name, level in self.layers.items()
                if not hasattr(level, "operator") and level.layer
            ]
        )
        out += "\n"

        # define derived layers
        for layer_name, level in self.layers.items():
            if hasattr(level, "operator"):
                out += f"{layer_name} = input({level.layer1[0]}, {level.layer1[1]}) {level.operator} input({level.layer2[0]}, {level.layer2[1]})\n"

        out += "\n"

        for layer_name, level in self.layers.items():
            layer = level.layer
            zmin = level.zmin
            zmax = zmin + level.thickness

            if layer:
                name = f"{layer_name}: {level.material} {layer[0]}/{layer[1]}"

            elif hasattr(level, "operator"):
                name = f"{layer_name}: {level.material}"

            else:
                continue

            if dbu:
                rnd_pl = len(str(dbu).split(".")[-1])
                zmin = round(zmin, rnd_pl)
                zmax = round(zmax, rnd_pl)

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
                if props.color.fill == props.color.frame:
                    txt += f"color: {props.color.fill}"
                else:
                    txt += f"fill: {props.color.fill}, " f"frame: {props.color.frame}"

            txt += ")"
            out += f"{txt}\n"

        return out


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.generic_tech import LAYER_STACK

    component = c = gf.components.grating_coupler_elliptical_trenches()

    # script = LAYER_STACK.get_klayout_3d_script()
    # print(script)

    layer_stack = LAYER_STACK
    layer_to_thickness = layer_stack.get_layer_to_thickness(component)

    c2 = layer_stack.get_component_with_derived_layers(component)
    c2.show(show_ports=True)

    # import pathlib
    # filepath = pathlib.Path(
    #     "/home/jmatres/gdslib/sp/temp/write_sparameters_meep_mpi.json"
    # )
    # ls_json = filepath.read_bytes()
    # ls2 = LayerStack.parse_raw(ls_json)
