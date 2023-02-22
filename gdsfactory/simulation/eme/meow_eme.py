import pathlib
import time
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import meow as mw
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import gdsfactory as gf
from gdsfactory.config import logger
from gdsfactory.generic_tech import LAYER
from gdsfactory.pdk import get_active_pdk
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_meow as get_sparameters_path,
)
from gdsfactory.simulation.gmsh.parse_layerstack import list_unique_layerstack_z
from gdsfactory.technology import LayerStack
from gdsfactory.typings import PathType

ColorRGB = Tuple[float, float, float]

material_to_color_default = {
    "si": (0.9, 0, 0, 0.9),
    "sio2": (0.9, 0.9, 0.9, 0.9),
    "sin": (0.0, 0.9, 0.0, 0.9),
}


class MEOW:
    def __init__(
        self,
        component: gf.Component,
        layerstack,
        wavelength: float = 1.55,
        temperature: float = 25.0,
        num_modes: int = 4,
        cell_length: float = 0.5,
        spacing_x: float = 2.0,
        center_x: float = 0,
        resolution_x: int = 100,
        spacing_y: float = 2.0,
        center_y: float = 0,
        resolution_y: int = 100,
        material_to_color: Dict[str, ColorRGB] = material_to_color_default,
        dirpath: Optional[PathType] = None,
        filepath: Optional[PathType] = None,
        overwrite: bool = False,
    ) -> None:
        """Computes multimode 2-port S-parameters for a gdsfactory component.

        assumes port 1 is at the left boundary and port 2 at the right boundary.

        Note coordinate systems:
            gdsfactory uses x,y in the plane to represent components, with the layerstack existing in z
            meow uses x,y to represent a cross-section, with propagation in the z-direction
            hence we have [x,y,z] <--> [y,z,x] for gdsfactory <--> meow

        Arguments:
            component: gdsfactory component.
            layerstack: gdsfactory layerstack.
            wavelength: wavelength in microns (for FDE, and for material properties).
            temperature: temperature in C (for material properties). Unused now.
            num_modes: number of modes to compute for the eigenmode expansion.
            cell_length: in un.
            spacing_x: at beginning and end of the simulation region.
            center_x: in um.
            resolution_x: pixels in horizontal region.
            spacing_y: at the beginning and end of simulation region.
            center_y: in um.
            resolution_y: pixels in vertical direction.
            material_to_color: dict of materials colors for struct plot
            dirpath: directory to store Sparameters.
            filepath: to store pandas Dataframe with Sparameters in npz format.
                Defaults to dirpath/component_.npz.
            overwrite: overwrites stored Sparameter npz results.

        Returns:
            S-parameters in form o1@0,o2@0 at wavelength.

        ::

            cross_section view:
               ________________________________
              |                                |
              |                                |
              | spacing_x            spacing_x |  spacing_y
              |<--------->           <-------->|
              |          ___________   _ _ _   |
              |         |           |          |
              |         |           |_ _ _ _ _ |_ center_y
              |         |           |          |
              |         |___________|          |
              |               |                |
              |                                |
              |               |                |  spacing_y
              |                                |
              |_______________|________________|
                          center_x

            top side view:
               ________________________________
              |                                |
              |                                |
              |cell_length                     |
              |<-------->                      |
              |_____________________ __________|
              |         |           |          |
              |         |           |          |
              | cell0   |  cell1    |  cell2   |
              |_________|___________|__________|
              |                                |
              |                                |
              |                                |
              |                                |
              |________________________________|
        """
        # Validate component
        self.validate_component(component)

        # Save parameters
        self.wavelength = wavelength
        self.num_modes = num_modes
        self.temperature = temperature  # unused for now
        self.material_to_color = material_to_color

        # Process simulation bounds
        self.span_x = np.diff(component.bbox[:, 1])[0] + spacing_x
        self.num_cells = int(self.span_x / cell_length)
        zs = list_unique_layerstack_z(layerstack)
        self.span_y = np.max(zs) - np.min(zs) + spacing_y
        self.center_x = center_x
        self.resolution_x = resolution_x
        self.center_y = center_y
        self.resolution_y = resolution_y

        # Setup simulation
        self.component, self.layerstack = self.add_global_layers(component, layerstack)
        self.extrusion_rules = self.layerstack_to_extrusion()
        self.structs = mw.extrude_gds(self.component, self.extrusion_rules)
        self.cells = self.create_cells(cell_length=cell_length)
        self.env = mw.Environment(wl=self.wavelength, T=self.temperature)
        self.css = [mw.CrossSection(cell=cell, env=self.env) for cell in self.cells]
        self.modes_per_cell = [None] * self.num_cells
        self.S = None
        self.port_map = None

        # Cache
        sim_settings = dict(
            wavelength=wavelength,
            temperature=temperature,
            num_modes=num_modes,
            cell_length=cell_length,
            spacing_x=spacing_x,
            center_x=center_x,
            resolution_x=resolution_x,
            spacing_y=spacing_y,
            center_y=center_y,
            resolution_y=resolution_y,
        )

        filepath = filepath or get_sparameters_path(
            component=component,
            dirpath=dirpath,
            layer_stack=layerstack,
            **sim_settings,
        )

        sim_settings = sim_settings.copy()
        sim_settings["layer_stack"] = layerstack.to_dict()
        sim_settings["component"] = component.to_dict()
        self.sim_settings = sim_settings
        self.filepath = pathlib.Path(filepath)
        self.filepath_sim_settings = filepath.with_suffix(".yml")
        self.overwrite = overwrite

    def gf_material_to_meow_material(
        self, material_name: str = "si", wavelengths=None, color=None
    ):
        """Converts a gdsfactory material into a MEOW material."""
        wavelengths = wavelengths or np.linspace(1.5, 1.6, 101)
        color = color or (0.9, 0.9, 0.9, 0.9)
        PDK = get_active_pdk()
        ns = PDK.materials_index[material_name](wavelengths)
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
        self,
        component,
        layerstack,
        buffer_y: float = 1,
        global_layer_index: int = 10000,
    ):
        """Adds bbox polygons for global layers.

        LAYER.WAFER layers are represented as polygons of size [bbox.x, xspan (meow coords)]

        Arguments:
            component: gdsfactory component.
            layerstack: gdsfactory LayerStack.
            xspan: from eme setup.
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
                    mesh_order=layer.mesh_order,
                )
            )
        return extrusions

    def create_cells(self, cell_length: float = 1.0) -> List[mw.Cell]:
        """Get meow cells from extruded component.

        Args:
            cell_length: in um.
        """
        self.cell_length = cell_length
        length = self.component.xsize  # in the propagation direction

        self.num_cells = int(length / cell_length)
        Ls = np.array([length / self.num_cells for _ in range(self.num_cells)])

        return mw.create_cells(
            structures=self.structs,
            mesh=mw.Mesh2d(
                x=np.linspace(
                    self.center_x - self.span_x / 2,
                    self.center_x + self.span_x / 2,
                    self.resolution_x,
                ),
                y=np.linspace(
                    self.center_y - self.span_y / 2,
                    self.center_y + self.span_y / 2,
                    self.resolution_y,
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
        if self.modes_per_cell[xs_num] is None:
            self.modes_per_cell[xs_num] = self.compute_mode(xs_num)
        return mw.visualize(self.modes_per_cell[xs_num][mode_num])

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

    def compute_all_modes(self) -> None:
        self.modes_per_cell = []
        for cs in tqdm(self.css):
            modes_in_cs = mw.compute_modes(cs, num_modes=self.num_modes)
            self.modes_per_cell.append(modes_in_cs)

    def compute_sparameters(self) -> Dict[str, np.ndarray]:
        """Returns Sparameters using EME."""
        if self.filepath.exists():
            if not self.overwrite:
                logger.info(f"Simulation loaded from {self.filepath!r}")
                return dict(np.load(self.filepath))
            else:
                self.filepath.unlink()

        start = time.time()

        self.compute_all_modes()

        if self.S is None or self.port_map is None:
            self.S, self.port_map = mw.compute_s_matrix(self.modes_per_cell)

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

        np.savez_compressed(self.filepath, **sp)

        end = time.time()

        self.sim_settings.update(compute_time_seconds=end - start)
        self.sim_settings.update(compute_time_minutes=(end - start) / 60)
        logger.info(f"Write simulation results to {self.filepath!r}")
        self.filepath_sim_settings.write_text(OmegaConf.to_yaml(self.sim_settings))
        logger.info(f"Write simulation settings to {self.filepath_sim_settings!r}")

        return sp


if __name__ == "__main__":
    c = gf.components.taper(length=10, width2=2)
    c.show()

    from gdsfactory.pdk import get_layer_stack

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "slab90",
                "core",
                "box",
                "clad",
            )
        }
    )
    m = MEOW(
        component=c, layerstack=filtered_layerstack, wavelength=1.55, overwrite=False
    )
    print(len(m.cells))

    import pprint

    pprint.pprint(m.compute_sparameters())
