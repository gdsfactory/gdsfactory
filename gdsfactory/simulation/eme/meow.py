from itertools import permutations

import meow as mw
import numpy as np
import pandas as pd

import gdsfactory as gf
from gdsfactory.pdk import _ACTIVE_PDK
from gdsfactory.simulation.gmsh.parse_layerstack import list_unique_layerstack_z
from gdsfactory.tech import LAYER, LayerStack


class MEOW:
    def __init__(
        self,
        component,
        layerstack,
        wavelength=1.55,
        num_modes: int = 4,
        num_cells: int = 10,
        horiz_buffer: float = 2.0,
        horiz_offset: float = 0,
        horiz_res: int = 100,
        vert_buffer: float = 2.0,
        vert_offset: float = 0,
        vert_res: int = 100,
    ) -> None:
        """Computes multimode 2-port S-parameters for a gdsfactory component, assuming port 1 is at the left boundary and port 2 at the right boundary.

        Note coordinate systems:
            gdsfactory uses x,y in the plane to represent components, with the layerstack existing in z
            meow uses x,y to represent a cross-section, with propagation in the z-direction
            hence we have [x,y,z] <--> [y,z,x] for gdsfactory <--> meow

        Arguments:
            component: gdsfactory component
            layerstack: gdsfactory layerstack
            num_modes: number of modes to compute for the eigenmode expansion
            wavelength: wavelength in microns (for FDE, and for material properties)
            temperature: temperature in C (for material properties)
            num_cells: number of component slices along the propagation direction for the EME
            xbuffer: extra size of the horizontal cross-sectional simulation region (meow coordinates) w.r.t. bbox. Default 2.
            xoffset: center of the horizontal cross-sectional simulation region  (meow coordinates)
            xres: number of mesh points for the horizontal cross-sectional simulation region  (meow coordinates)
            ybuffer: extra size of the vertical cross-sectional simulation region. Default is 0, resulting in total layerstack thickness.  (meow coordinates)
            yoffset: center of the vertical cross-sectional simulation region  (meow coordinates)
            yres: number of mesh points for the vertical cross-sectional simulation region  (meow coordinates)
            validate_component: whether raise errors if the component ports are of the wrong number and orientation. Default True.

        Returns:
            S-parameters in form o1@0,o2@0 at wavelength
        """
        # Validate component
        self.validate_component(component)

        # Save parameters
        self.wavelength = wavelength
        self.num_modes = num_modes
        self.num_cells = num_cells
        self.temperature = 25  # unused for now
        self.material_to_color = {
            "si": (0.9, 0, 0, 0.9),
            "sio2": (0.9, 0.9, 0.9, 0.9),
            "sin": (0.0, 0.9, 0.0, 0.9),
        }

        # Process simulation bounds
        self.horiz_span = np.diff(component.bbox[:, 1])[0] + horiz_buffer
        zs = list_unique_layerstack_z(layerstack)
        self.vert_span = np.max(zs) - np.min(zs) + vert_buffer
        self.horiz_offset = horiz_offset
        self.horiz_res = horiz_res
        self.vert_offset = vert_offset
        self.vert_res = vert_res

        # Setup simulation
        self.component, self.layerstack = self.add_global_layers(component, layerstack)
        self.extrusion_rules = self.layerstack_to_extrusion()
        self.structs = mw.extrude_gds(self.component, self.extrusion_rules)
        self.cells = self.get_eme_cells()
        self.env = mw.Environment(wl=self.wavelength, T=self.temperature)
        self.css = [mw.CrossSection(cell=cell, env=self.env) for cell in self.cells]
        self.modes = [None] * num_cells
        self.S = None
        self.port_map = None

    def gf_material_to_meow_material(
        self, material_name="si", wavelengths=None, color=None
    ):
        wavelengths = wavelengths or np.linspace(1.5, 1.6, 101)
        color = color or (0.9, 0.9, 0.9, 0.9)
        ns = _ACTIVE_PDK.materials_index[material_name](wavelengths)
        if ns.dtype in [np.float64, np.float32]:
            nr = ns
            ni = np.zeros_like(ns)
        else:
            nr = np.real(ns)
            ni = np.imag(ns)
        df = pd.DataFrame({"wl": wavelengths, "nr": nr, "ni": ni})
        return mw.Material.from_df(
            material_name,
            df,
            meta={"color": color},
        )

    def add_global_layers(
        self, component, layerstack, buffer_y=1, global_layer_index=10000
    ):
        """Adds bbox polygons for global layers.

        LAYER.WAFER layers are represented as polygons of size [bbox.x, xspan (meow coords)]

        Arguments:
            component: gdsfactory component
            layerstack: gdsfactory LayerStack
            xspan: from eme setup
            global_layer_index: int, layer index at which to starting adding the global layers.
                    Default 10000 with +1 increments to avoid clashing with physical layers.

        """
        c = gf.Component()
        c.add_ref(component)
        bbox = component.bbox
        for _layername, layer in layerstack.layers.items():
            if layer.layer == LAYER.WAFER:
                c.add_ref(
                    gf.components.bbox(
                        bbox=(
                            (bbox[0, 0], bbox[0, 1] - buffer_y),
                            (bbox[1, 0], bbox[1, 1] + buffer_y),
                        ),
                        layer=(global_layer_index, 0),
                    )
                )
                layer.layer = (global_layer_index, 0)
                global_layer_index += 1

        return c, layerstack

    def layerstack_to_extrusion(self):
        """Convert LayerStack to meow extrusions."""
        extrusions = {}
        for _layername, layer in self.layerstack.layers.items():
            if layer.layer not in extrusions.keys():
                extrusions[layer.layer] = []
            extrusions[layer.layer].append(
                mw.GdsExtrusionRule(
                    material=self.gf_material_to_meow_material(
                        layer.material,
                        np.array([self.wavelength]),
                        color=self.material_to_color.get(layer.material),
                    ),
                    h_min=layer.zmin,
                    h_max=layer.zmin + layer.thickness,
                    mesh_order=layer.info["mesh_order"],
                )
            )
        return extrusions

    def get_eme_cells(
        self,
        num_cells: int = 10,
        xspan: float = 2.0,
        xoffset: float = 0,
        xres: int = 100,
        yspan: float = 2.0,
        yoffset: float = 0,
        yres: int = 100,
    ):
        """Get meow cells from extruded component.

        Arguments:
            structs: meow structs
            extrusion rules: from layerstack_to_extrusion
            num_cells: number of slices
            xspan: size of the horizontal cross-sectional simulation region
            xoffset: center of the horizontal cross-sectional simulation region
            xres: number of mesh points for the horizontal cross-sectional simulation region
            yspan: size of the vertical cross-sectional simulation region
            yoffset: center of the vertical cross-sectional simulation region
            yres: number of mesh points for the vertical cross-sectional simulation region
        """
        bbox = self.component.bbox
        Ls = [np.diff(bbox[:, 0]).item() / num_cells for _ in range(num_cells)]
        return mw.create_cells(
            structures=self.structs,
            mesh=mw.Mesh2d(
                x=np.linspace(
                    self.horiz_offset - self.horiz_span / 2,
                    self.horiz_offset + self.horiz_span / 2,
                    self.horiz_res,
                ),
                y=np.linspace(
                    self.vert_offset - self.vert_span / 2,
                    self.vert_offset + self.vert_span / 2,
                    self.vert_res,
                ),
            ),
            Ls=Ls,
        )

    def plot_structure(self, scale=(1, 1, 0.2)):
        return mw.visualize(self.structs, scale=scale)

    def plot_cross_section(self, xs_num):
        env = mw.Environment(wl=self.wavelength, T=self.temperature)
        css = [mw.CrossSection(cell=cell, env=env) for cell in self.cells]
        return mw.visualize(css[xs_num])

    def plot_mode(self, xs_num, mode_num):
        if self.modes[xs_num] is None:
            self.modes[xs_num] = self.compute_mode(xs_num)
        return mw.visualize(self.modes[xs_num][mode_num])

    def get_port_map(self):
        if self.port_map is None:
            self.compute_sparameters()
        return self.port_map

    def plot_Sparams(self):
        if self.S is None:
            self.compute_sparameters()
        return mw.visualize(self.S)

    def validate_component(self, component):
        optical_ports = [
            port
            for portname, port in component.ports.items()
            if port.port_type == "optical"
        ]
        if len(optical_ports) != 2:
            raise ValueError(
                "Component provided to MEOW does not have exactly 2 optical ports."
            )
        elif component.ports["o1"].orientation != 180:
            raise ValueError("Component port o1 does not face westward (180 deg).")
        elif component.ports["o2"].orientation != 0:
            raise ValueError("Component port o2 does not face eastward (0 deg).")

    def compute_mode(self, xs_num):
        return mw.compute_modes(self.css[xs_num], num_modes=self.num_modes)

    def compute_all_modes(self):
        for xs_num in range(self.num_cells):
            if self.modes[xs_num] is None:
                self.modes[xs_num] = self.compute_mode(xs_num)

    def compute_sparameters(self):
        # Compute EME
        self.compute_all_modes()
        if self.S is None or self.port_map is None:
            self.S, self.port_map = mw.compute_s_matrix(self.modes)

        # Convert coefficients to existing format
        meow_to_gf_keys = {
            "left": "o1",
            "right": "o2",
        }
        sp = {}
        for port1, port2 in permutations(self.port_map.values(), 2):
            value = self.S[port1, port2]
            meow_key1 = [k for k, v in self.port_map.items() if v == port1][0]
            meow_port1, meow_mode1 = meow_key1.split("@")
            meow_key2 = [k for k, v in self.port_map.items() if v == port2][0]
            meow_port2, meow_mode2 = meow_key2.split("@")
            sp[
                f"{meow_to_gf_keys[meow_port1]}@{meow_mode1},{meow_to_gf_keys[meow_port2]}@{meow_mode2}"
            ] = value
        sp["wavelengths"] = self.wavelength

        return sp


if __name__ == "__main__":

    c = gf.components.taper_cross_section_linear()
    c.show()

    from gdsfactory.tech import get_layer_stack_generic

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "box",
                "clad",
            )
        }
    )

    sp = MEOW(component=c, layerstack=filtered_layerstack, wavelength=1.55)

    import pprint

    pprint.pprint(sp.compute_sparameters())
